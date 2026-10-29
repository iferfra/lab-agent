from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END

from src.agent.llm import agent_llm
from src.agent.state import AgentState

MAX_MODEL_CALLS = 5

SYSTEM_PROMPT = """You are an academic research assistant using a Neo4j
knowledge graph and semantic search over scientific paper text.

Your goal is to answer research questions accurately using retrieved evidence.

TOOL SELECTION
- Use search_knowledge_graph for paper metadata, authorship, and graph
  relationships.
- Use search_chunks_semantically for specific claims, technical details,
  methods, limitations, experimental results, cited authors, and evidence
  from the original paper text.
- Use find_paper_entities to list the methods a paper proposes or compares
  with, its datasets, metrics or tasks, or to find which paper uses a given
  dataset or method.
- When the question names a paper, pass its title as paper_title to
  search_chunks_semantically. You do not need search_knowledge_graph just to
  find the paper.
- Select tools according to the information required by the question.
- Do not call a tool when the available evidence is already sufficient.
- Do not use graph metadata as evidence for technical claims that require
  support from the original paper text.

EVIDENCE AND RETRIEVAL
- Base factual claims on evidence returned by the tools.
- Prefer passages that directly address the question over passages that
  discuss only the same general topic.
- Assess whether the retrieved evidence actually answers the specific
  question, not merely whether it is relevant to the paper.
- If the evidence is relevant but insufficient, use an appropriate search
  tool to look for more specific evidence when possible.
- Do not assume that a fact is absent merely because it does not appear in
  a retrieved passage or summary.
- Never invent facts, citations, quotations, numerical results, or evidence.
- Do not claim that a tool found information it did not return.
- Distinguish explicit statements in the paper from your own inferences.

HYBRID QUESTIONS
- Use both graph evidence and original text evidence when both are relevant.
- A paper's actual authors are not necessarily the authors cited in its text.
- Verify cited names, technical claims, and specific details against source
  chunks.
- Use graph evidence for graph-related facts and text evidence for
  paper-specific claims.
- Combine sources only when they provide complementary information.
- Do not force a hybrid answer when one source is not needed.
- A summary omitting a fact does not prove that the fact is absent.

ANSWERING
- Identify exactly what the question asks for before composing the answer.
- Answer the specific question directly in the first sentence.
- Address every part of the question.
- Distinguish between related concepts, such as a method's motivation,
  its limitations, its assumptions, and its experimental results.
- Include only details that help answer the question or clarify the evidence.
- Do not replace a specific answer with a general explanation of the topic.
- Preserve important conditions, limitations, and qualifications.
- If the available evidence does not support a definitive answer, explain
  what could not be verified instead of guessing.
- Use clear, concise Markdown.
- Never expose internal tool calls, code variables, or raw JSON.
"""


def make_call_model(tools: list[BaseTool]):
    llm_with_tools = agent_llm.bind_tools(tools)

    def call_model(state: AgentState):
        messages = list(state["messages"])
        model_calls = state.get("model_calls", 0) + 1

        if not any(
            isinstance(message, SystemMessage)
            and "academic research assistant" in str(message.content).lower()
            for message in messages
        ):
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "tools": [call["name"] for call in response.tool_calls],
            "model_calls": model_calls,
        }

    return call_model


FINAL_ANSWER_PROMPT = """You have no tool calls left. Answer the question
below using only the evidence that was retrieved. If the evidence is not
enough, say what you could not verify.

Question:
{question}

Retrieved evidence:
{evidence}"""


def final_answer(state: AgentState):
    """Answer with the evidence gathered so far when the tool budget runs out.

    The history is passed as plain text: Bedrock rejects a conversation that
    contains tool calls unless tools are bound, and here the model must not
    call any more tools.
    """
    question = next(m.text for m in state["messages"] if isinstance(m, HumanMessage))
    evidence = "\n\n---\n\n".join(
        m.text for m in state["messages"] if isinstance(m, ToolMessage)
    )
    prompt = FINAL_ANSWER_PROMPT.format(
        question=question, evidence=evidence or "(nothing was retrieved)"
    )

    response = agent_llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )

    return {"messages": [response], "model_calls": state.get("model_calls", 0) + 1}


def should_continue(state: AgentState) -> Literal["tools", "final_answer", "__end__"]:
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return END

    # The model still wants tools: allow them while the budget lasts, otherwise
    # make it answer instead of ending with an empty message.
    if state.get("model_calls", 0) < MAX_MODEL_CALLS:
        return "tools"
    return "final_answer"
