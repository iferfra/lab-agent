"""LLM-as-a-judge. Each metric gets its own short prompt and a yes/no decision
per item, which is easier for the judge (and for a human) to check than a
single 0-2 score.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from eval.common import eval_llm

# Keeps the judge prompt within the model's context window
MAX_CONTEXT_CHARS = 30_000


# --- Fact recall: does the answer contain each required fact? ----------------


class FactCheck(BaseModel):
    fact: str
    present: bool


class FactRecall(BaseModel):
    checks: list[FactCheck]


FACT_RECALL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You check whether an answer contains a list of required facts.
For each fact, say if the answer states it (paraphrases count). A fact that is
only implied, or that the answer gets wrong, is not present. Return one check
per fact, in the same order.""",
        ),
        (
            "human",
            "Question: {question}\n\nRequired facts:\n{facts}\n\nAnswer:\n{answer}",
        ),
    ]
)


def judge_fact_recall(question: str, facts: list[str], answer: str) -> FactRecall:
    chain = FACT_RECALL_PROMPT | eval_llm().with_structured_output(FactRecall)
    return chain.invoke(
        {
            "question": question,
            "facts": "\n".join(f"- {fact}" for fact in facts),
            "answer": answer,
        }
    )


# --- Faithfulness: is every claim backed by the retrieved context? -----------


class Claim(BaseModel):
    claim: str
    supported: bool


class Faithfulness(BaseModel):
    claims: list[Claim]

    @property
    def score(self) -> float:
        # An answer that makes no factual claims (e.g. "I could not find it") invents nothing
        if not self.claims:
            return 1.0
        return sum(c.supported for c in self.claims) / len(self.claims)


FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You check an answer for hallucinations.
1. Split the answer into its factual claims about the papers (skip greetings,
   formatting and statements that something could not be found).
2. For each claim, say if the retrieved context supports it.
Judge only against the context below, not against your own knowledge.""",
        ),
        ("human", "Retrieved context:\n{context}\n\nAnswer:\n{answer}"),
    ]
)


def judge_faithfulness(answer: str, contexts: list[str]) -> Faithfulness:
    context = (
        "\n\n---\n\n".join(contexts)[:MAX_CONTEXT_CHARS] or "(nothing was retrieved)"
    )
    chain = FAITHFULNESS_PROMPT | eval_llm().with_structured_output(Faithfulness)
    return chain.invoke({"context": context, "answer": answer})


# --- Abstention: does the agent admit it does not know? ----------------------


class Abstention(BaseModel):
    abstained: bool
    reason: str


ABSTENTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """The question below cannot be answered from the papers. Decide whether the
answer says so (abstained=true) or presents a made-up answer as fact
(abstained=false). Mentioning related information is fine as long as the
answer is clear that the specific detail is not available.""",
        ),
        ("human", "Question: {question}\n\nAnswer:\n{answer}"),
    ]
)


def judge_abstention(question: str, answer: str) -> Abstention:
    chain = ABSTENTION_PROMPT | eval_llm().with_structured_output(Abstention)
    return chain.invoke({"question": question, "answer": answer})
