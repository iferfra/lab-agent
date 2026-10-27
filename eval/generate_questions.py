# src/evaluation/generate_questions.py

import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from eval.llm import eval_llm

load_dotenv()

# Configuration
INPUT_PATH = Path(os.getenv("DATA_DIR", "data/") + "eval/source_bundle.json")
OUTPUT_PATH = Path(os.getenv("DATA_DIR", "data/") + "eval/candidates.jsonl")

CATEGORY = "unanswerable"  # hybrid, semantic, graph, unanswerable
NUM_QUESTIONS = 20
DIFFICULTY = "medium"  # easy, medium, hard

Category = Literal["hybrid", "semantic", "graph", "unanswerable"]
Difficulty = Literal["easy", "medium", "hard"]


class RequiredFact(BaseModel):
    id: str
    description: str
    source_type: Literal["graph", "semantic", "cross_source", "corpus_gap"]


class Evidence(BaseModel):
    source_type: Literal["graph", "chunk", "corpus_gap"]
    source_id: str
    paper: str = ""
    supports: list[str] = Field(default_factory=list)


class QuestionCandidate(BaseModel):
    category: Category
    question: str
    answerable: bool
    expected_answer: str
    required_facts: list[RequiredFact]
    expected_tools: list[str]
    evidence: list[Evidence]
    difficulty: Difficulty
    unanswerable_reason: str | None = None


llm = eval_llm

QUESTION_ANGLES = {
    "hybrid": [
        "Identify papers by a specific author and compare their methods.",
        "Connect a paper's authors to its experimental datasets.",
        "Compare evaluation benchmarks across two related papers.",
        "Identify a research contribution and explain its limitations.",
        "Trace a relationship between two papers and compare their findings.",
        "Compare the experimental setup of two papers on different tasks.",
        "Identify which paper supports a claim and retrieve the supporting evidence.",
        "Connect a researcher's publications to the methods described in them.",
    ],
    "semantic": [
        "Ask about a paper's limitations.",
        "Ask about the experimental setup.",
        "Ask about datasets and evaluation metrics.",
        "Ask about the proposed methodology.",
        "Ask about ablation studies or comparisons.",
        "Ask about future work.",
    ],
    "graph": [
        "Ask about authorship relationships.",
        "Ask about publication years.",
        "Ask which papers share an author.",
        "Ask about relationships between papers and research topics.",
    ],
    "unanswerable": [
        # "Ask about a specific training detail not reported in the corpus.",
        # "Ask for a numerical result absent from the available sources.",
        # "Ask about an experimental setting that is explicitly undocumented.",
        "Ask anything that is not in the resources."
    ],
}

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You design evaluation benchmarks for a Graph RAG system over scientific
papers. Generate exactly one question using only the provided sources.
Never invent facts, paper titles, authors, source IDs, or evidence.

Categories:
- graph: requires graph metadata or relationships.
- semantic: requires understanding the content of chunks.
- hybrid: requires both graph relationships and chunk content.
- unanswerable: requests information explicitly documented in known_gaps.

STRICT HYBRID QUESTION RULES:

1. A hybrid question must require BOTH graph relationships and semantic
   content to produce a complete answer.

2. Graph evidence must contribute a necessary fact, such as identifying
   the correct paper, author relationship, or connection between papers.

3. Semantic evidence must contribute a necessary fact, such as a method,
   dataset, experimental result, or limitation.

4. Merely mentioning an author or paper in the question does not make
   the question hybrid.

5. If the question can be answered completely using only semantic
   evidence, classify it as semantic.

6. If the question can be answered completely using only graph facts,
   classify it as graph.

7. The expected answer must explicitly address every part of the question.

8. Never use vague answers such as "and others", "different methods",
   or "the paper uses various techniques".

9. If the sources do not support a specific answer, do not generate
   the question.

Prefer questions covering different combinations of:
- authorship and research methods;
- paper relationships and experimental datasets;
- research contributions and limitations;
- publication metadata and evaluation results;
- shared authors and differences between experimental setups.

Do not repeatedly ask about the same paper pair and the same type of fact
unless the available evidence leaves no meaningful alternative.   

Rules:
1. Every answerable question must be supported by the sources.
2. Give each required fact a unique ID, such as fact_1.
3. Every evidence source_id must exist in the provided sources.
4. Evidence.supports must reference required fact IDs.
5. A hybrid question must genuinely require both graph and chunk evidence.
6. Avoid duplicates and trivial questions.
7. An unanswerable question must use a known_gaps entry, have
   answerable=false, and include an unanswerable_reason.
8. All other categories must have answerable=true.
9. Use these exact expected_tools:
   graph: ["search_knowledge_graph"]
   semantic: ["search_chunks_semantically"]
   hybrid: ["search_knowledge_graph", "search_chunks_semantically"]
   unanswerable: ["search_knowledge_graph", "search_chunks_semantically"]
10. Use source_type="cross_source" for facts requiring both sources.
11. If the available sources are insufficient, do not invent a question.
        """,
        ),
        (
            "human",
            """
Requested category: {category}
Requested difficulty: {difficulty}
Required question angle: {angle}

Available sources:
{sources}

Previously generated questions:
{existing_questions}

Generate one question that follows the requested angle.
Do not reuse the same paper combination, target fact, or question structure
as a previous question when alternatives exist.
If the sources do not support this angle, do not invent information.
""",
        ),
    ]
)

question_chain = prompt | llm.with_structured_output(QuestionCandidate)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    if CATEGORY not in {"hybrid", "semantic", "graph", "unanswerable"}:
        raise ValueError(f"Invalid CATEGORY: {CATEGORY}")

    if DIFFICULTY not in {"easy", "medium", "hard"}:
        raise ValueError(f"Invalid DIFFICULTY: {DIFFICULTY}")

    if not os.getenv("EVALUATION_LLM"):
        raise ValueError("EVALUATION_LLM is not configured in .env")

    sources = load_json(INPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = load_jsonl(OUTPUT_PATH)
    existing_questions = [item["question"] for item in existing if "question" in item]

    prefix = CATEGORY
    next_id = 1 + sum(
        1 for item in existing if item.get("id", "").startswith(f"{prefix}_")
    )
    angles = QUESTION_ANGLES[CATEGORY]
    with OUTPUT_PATH.open("a", encoding="utf-8") as file:
        for i in range(NUM_QUESTIONS):
            angle = angles[i % len(angles)]

            try:
                candidate = question_chain.invoke(
                    {
                        "category": CATEGORY,
                        "difficulty": DIFFICULTY,
                        "angle": angle,
                        "sources": json.dumps(sources, ensure_ascii=False, indent=2),
                        "existing_questions": json.dumps(
                            existing_questions[-100:],
                            ensure_ascii=False,
                        ),
                    }
                )
            except Exception as e:  # noqa: BLE001
                print(f"Exception: {e}")
                continue

            def normalize_question(question: str) -> str:
                return " ".join(question.lower().strip().split())

            # After candidate = question_chain.invoke(...)
            new_question = normalize_question(candidate.question)

            if any(normalize_question(q) == new_question for q in existing_questions):
                print("Duplicate question skipped.")
                continue

            record = candidate.model_dump()
            record["id"] = f"{prefix}_{next_id:03d}"
            record["validation_status"] = "pending"

            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()

            existing_questions.append(record["question"])

            print(f"Generated {record['id']}: {record['question']}")
            next_id += 1


if __name__ == "__main__":
    main()
