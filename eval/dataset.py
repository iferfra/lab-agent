"""Build the benchmark: generate candidate questions with an LLM, then validate them.

    python -m eval.dataset generate --category hybrid -n 10
    python -m eval.dataset validate

candidates.jsonl works as an inbox: `validate` moves the valid candidates to
questions.jsonl and leaves the rejected ones in the inbox, so you can fix or
delete them. Review questions.jsonl by hand before committing it: the automatic
checks catch broken references, not bad questions.
"""

import argparse
import json
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from eval.common import (
    CANDIDATES_PATH,
    QUESTIONS_PATH,
    SOURCES_PATH,
    eval_llm,
    read_json,
    read_jsonl,
    write_jsonl,
)

Category = Literal["graph", "semantic", "hybrid", "unanswerable"]

# What kind of source each agent tool reads. Questions say which sources they
# need, not which tool, so the agent may pick any tool of the right kind.
TOOL_SOURCE = {
    "search_knowledge_graph": "graph",
    "find_paper_entities": "graph",
    "search_chunks_semantically": "text",
}

EXPECTED_SOURCES: dict[str, set[str]] = {
    "graph": {"graph"},
    "semantic": {"text"},
    "hybrid": {"graph", "text"},
    "unanswerable": {"graph", "text"},
}


class QuestionDraft(BaseModel):
    category: Category
    difficulty: Literal["easy", "medium", "hard"]
    question: str
    expected_answer: str
    required_facts: list[str] = Field(
        description="Self-contained statements a correct answer must include. Empty for unanswerable questions."
    )
    evidence_ids: list[str] = Field(
        description="IDs of the sources that support the answer."
    )


class Question(QuestionDraft):
    id: str

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"

    @property
    def expected_sources(self) -> set[str]:
        return EXPECTED_SOURCES[self.category]


class Sources:
    """The facts, chunks and known gaps that questions are generated from."""

    def __init__(self, data: dict):
        self.data = data
        self.graph_ids = {fact["source_id"] for fact in data["graph_facts"]}
        self.chunk_ids = {chunk["source_id"] for chunk in data["chunks"]}
        self.gap_ids = {gap["source_id"] for gap in data["known_gaps"]}

    @classmethod
    def load(cls) -> "Sources":
        return cls(read_json(SOURCES_PATH))


def find_problems(question: Question, sources: Sources) -> list[str]:
    problems = []
    evidence = set(question.evidence_ids)

    unknown = evidence - sources.graph_ids - sources.chunk_ids - sources.gap_ids
    if unknown:
        problems.append(f"unknown evidence ids: {sorted(unknown)}")

    has_graph = bool(evidence & sources.graph_ids)
    has_chunk = bool(evidence & sources.chunk_ids)
    has_gap = bool(evidence & sources.gap_ids)

    match question.category:
        case "graph" if not has_graph:
            problems.append("graph question without graph evidence")
        case "semantic" if not has_chunk:
            problems.append("semantic question without chunk evidence")
        case "hybrid" if not (has_graph and has_chunk):
            problems.append("hybrid question needs both graph and chunk evidence")
        case "unanswerable" if not has_gap:
            problems.append("unanswerable question must point to a known gap")

    if question.answerable and not question.required_facts:
        problems.append("answerable question without required facts")

    return problems


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


# --- Generation --------------------------------------------------------------

ANGLES: dict[str, list[str]] = {
    "graph": [
        "the authors of a paper",
        "the authors shared by both papers",
        "the method a paper proposes or the methods it builds on",
        "the datasets or metrics a paper uses",
        "the methods a paper compares against",
    ],
    "semantic": [
        "how the proposed method works",
        "an experimental result",
        "a training or evaluation detail",
        "a limitation or conclusion of the paper",
    ],
    "hybrid": [
        "a detail of the method proposed by a given author's paper",
        "a result on one of the datasets the graph links to a paper",
        "compare the two papers that share an author",
        "why a paper uses one of the datasets or methods the graph links to it",
    ],
    "unanswerable": [
        "a detail listed in known_gaps that sounds plausible for the paper",
        "a known gap asked as if the information existed",
    ],
}

GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You write benchmark questions for a Graph RAG assistant over scientific papers.
Use only the provided sources. Never invent papers, authors, numbers or source ids.

Categories:
- graph: answerable only with graph facts (authors, venue, code, and the
  methods, datasets, metrics and tasks linked to each paper).
- semantic: answerable only with the text of a chunk.
- hybrid: needs a graph fact AND a chunk. Just naming a paper in the question
  does not make it hybrid.
- unanswerable: asks for something listed in known_gaps.

Write required_facts as complete statements that can be checked against an
answer, e.g. "Luis Baumela is an author of the paper", not "Authorship of the
paper". evidence_ids must list the source_id of every source you used, copied
exactly (chunk ids are 32 characters long).""",
        ),
        (
            "human",
            """Write {count} different {category} questions of {difficulty} difficulty.
Spread them over both papers and use a different angle for each one:
{angles}

Sources:
{sources}

Questions that already exist (do not repeat them):
{existing}""",
        ),
    ]
)


class QuestionDrafts(BaseModel):
    questions: list[QuestionDraft]


def generate(category: str, difficulty: str, count: int) -> None:
    """Generate all the questions of a category in one call, so the model sees
    the ones it already wrote and does not repeat itself."""
    sources = Sources.load()
    chain = GENERATION_PROMPT | eval_llm().with_structured_output(QuestionDrafts)

    candidates = read_jsonl(CANDIDATES_PATH)
    questions = read_jsonl(QUESTIONS_PATH)
    existing = [q["question"] for q in questions + candidates]
    next_number = 1 + max(
        (
            int(q["id"].rsplit("_", 1)[1])
            for q in questions + candidates
            if q["category"] == category
        ),
        default=0,
    )

    drafts = chain.invoke(
        {
            "count": count,
            "category": category,
            "difficulty": difficulty,
            "angles": "\n".join(f"- {angle}" for angle in ANGLES[category]),
            "sources": json.dumps(sources.data, ensure_ascii=False, indent=1),
            "existing": "\n".join(existing) or "(none)",
        }
    )

    for draft in drafts.questions:
        if normalize(draft.question) in {normalize(q) for q in existing}:
            print(f"skipped duplicate: {draft.question}")
            continue

        question = Question(id=f"{category}_{next_number:03d}", **draft.model_dump())
        candidates.append(question.model_dump())
        existing.append(question.question)
        next_number += 1
        print(f"{question.id}: {question.question}")

    write_jsonl(CANDIDATES_PATH, candidates)


# --- Validation --------------------------------------------------------------


def validate() -> None:
    sources = Sources.load()
    accepted = read_jsonl(QUESTIONS_PATH)
    seen = {normalize(q["question"]) for q in accepted}

    still_pending = []
    for raw in read_jsonl(CANDIDATES_PATH):
        question = Question.model_validate(raw)
        problems = find_problems(question, sources)
        if normalize(question.question) in seen:
            problems.append("duplicate question")

        if problems:
            print(f"✗ {question.id}: {'; '.join(problems)}")
            still_pending.append(raw)
            continue

        print(f"✓ {question.id}: {question.question}")
        accepted.append(question.model_dump())
        seen.add(normalize(question.question))

    write_jsonl(QUESTIONS_PATH, accepted)
    write_jsonl(CANDIDATES_PATH, still_pending)
    print(
        f"\n{len(accepted)} questions in {QUESTIONS_PATH.name}, "
        f"{len(still_pending)} rejected left in {CANDIDATES_PATH.name}"
    )


def load_questions() -> list[Question]:
    return [Question.model_validate(q) for q in read_jsonl(QUESTIONS_PATH)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    gen = commands.add_parser("generate", help="generate candidate questions")
    gen.add_argument("--category", choices=list(EXPECTED_SOURCES), required=True)
    gen.add_argument(
        "--difficulty", choices=["easy", "medium", "hard"], default="medium"
    )
    gen.add_argument("-n", "--count", type=int, default=5)

    commands.add_parser("validate", help="move the valid candidates to questions.jsonl")

    args = parser.parse_args()
    if args.command == "generate":
        generate(args.category, args.difficulty, args.count)
    else:
        validate()
