# src/evaluation/validate_questions.py

import json
import re
from pathlib import Path

# Configuration
INPUT_PATH = Path("data/eval/candidates.jsonl")
SOURCES_PATH = Path("data/eval/source_bundle.json")
VALIDATED_PATH = Path("data/eval/validated_candidates.jsonl")
REJECTED_PATH = Path("data/eval/rejected_questions.jsonl")

EXPECTED_TOOLS = {
    "graph": {"search_knowledge_graph"},
    "semantic": {"search_chunks_semantically"},
    "hybrid": {
        "search_knowledge_graph",
        "search_chunks_semantically",
    },
    "unanswerable": {
        "search_knowledge_graph",
        "search_chunks_semantically",
    },
}

VALID_CATEGORIES = set(EXPECTED_TOOLS)
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def validate_question(item: dict, source_ids: dict) -> list[str]:
    errors = []

    required_fields = {
        "id",
        "category",
        "question",
        "answerable",
        "expected_answer",
        "required_facts",
        "expected_tools",
        "evidence",
        "difficulty",
    }

    missing = required_fields - item.keys()
    if missing:
        return [f"Missing fields: {sorted(missing)}"]

    category = item["category"]

    if category not in VALID_CATEGORIES:
        return [f"Invalid category: {category}"]

    if not isinstance(item["question"], str) or not item["question"].strip():
        errors.append("Question is empty.")

    if (
        not isinstance(item["expected_answer"], str)
        or not item["expected_answer"].strip()
    ):
        errors.append("Expected answer is empty.")

    if item["difficulty"] not in VALID_DIFFICULTIES:
        errors.append("Invalid difficulty.")

    if not isinstance(item["answerable"], bool):
        errors.append("'answerable' must be a boolean.")
    elif category == "unanswerable" and item["answerable"]:
        errors.append("Unanswerable questions must have answerable=false.")
    elif category != "unanswerable" and not item["answerable"]:
        errors.append("This category requires answerable=true.")

    if not isinstance(item["expected_tools"], list):
        errors.append("expected_tools must be a list.")
    elif set(item["expected_tools"]) != EXPECTED_TOOLS[category]:
        errors.append(
            f"Incorrect tools for {category}. "
            f"Expected: {sorted(EXPECTED_TOOLS[category])}"
        )

    # Validate required facts
    facts = item["required_facts"]
    if not isinstance(facts, list) or not facts:
        errors.append("At least one required fact is needed.")
        facts = []

    fact_ids = [fact.get("id") for fact in facts]

    if len(fact_ids) != len(set(fact_ids)):
        errors.append("Duplicate required fact IDs.")

    valid_fact_ids = set(fact_ids)

    for fact in facts:
        if (
            not isinstance(fact.get("description"), str)
            or not fact["description"].strip()
        ):
            errors.append(f"Missing description for fact {fact.get('id')}.")

        if fact.get("source_type") not in {
            "graph",
            "semantic",
            "cross_source",
            "corpus_gap",
        }:
            errors.append(f"Invalid source_type for fact {fact.get('id')}.")

    # Validate evidence
    evidence = item["evidence"]
    if not isinstance(evidence, list) or not evidence:
        errors.append("At least one evidence item is needed.")
        evidence = []

    supported_fact_ids = set()
    evidence_types = set()

    for ev in evidence:
        source_type = ev.get("source_type")
        source_id = ev.get("source_id")

        if source_type not in source_ids:
            errors.append(f"Invalid evidence type: {source_type}")
            continue

        if not source_id or source_id not in source_ids[source_type]:
            errors.append(f"Evidence source does not exist: {source_type}/{source_id}")

        evidence_types.add(source_type)

        supports = ev.get("supports", [])
        if not isinstance(supports, list):
            errors.append(f"Evidence supports must be a list: {source_id}")
            continue

        for fact_id in supports:
            if fact_id not in valid_fact_ids:
                errors.append(f"Evidence references unknown fact: {fact_id}")
            else:
                supported_fact_ids.add(fact_id)

    for fact_id in valid_fact_ids - supported_fact_ids:
        errors.append(f"Fact has no linked evidence: {fact_id}")

    # Basic category/evidence consistency
    if category == "graph" and "graph" not in evidence_types:
        errors.append("Graph questions require graph evidence.")

    if category == "semantic" and "chunk" not in evidence_types:
        errors.append("Semantic questions require chunk evidence.")

    if category == "hybrid":
        if "graph" not in evidence_types or "chunk" not in evidence_types:
            errors.append("Hybrid questions require graph and chunk evidence.")

        fact_types = {fact.get("source_type") for fact in facts}
        if not {"graph", "semantic"}.issubset(fact_types):
            errors.append("Hybrid questions require graph and semantic required facts.")

    if category == "unanswerable":
        reason = item.get("unanswerable_reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append("Missing unanswerable_reason.")

        if "corpus_gap" not in evidence_types:
            errors.append("Unanswerable questions require known-gap evidence.")

    return errors


def main():
    candidates = read_jsonl(INPUT_PATH)
    sources = read_json(SOURCES_PATH)

    source_ids = {
        "graph": {
            item["source_id"]
            for item in sources.get("graph_facts", [])
            if item.get("source_id")
        },
        "chunk": {
            item.get("source_id", item.get("chunk_id"))
            for item in sources.get("chunks", [])
            if item.get("source_id", item.get("chunk_id"))
        },
        "corpus_gap": {
            item["source_id"]
            for item in sources.get("known_gaps", [])
            if item.get("source_id")
        },
    }

    accepted = []
    rejected = []
    seen_ids = set()
    seen_questions = set()

    for item in candidates:
        errors = validate_question(item, source_ids)

        question_id = item.get("id")
        question_text = item.get("question", "")

        if question_id in seen_ids:
            errors.append(f"Duplicate question ID: {question_id}")
        else:
            seen_ids.add(question_id)

        normalized_question = normalize(question_text)

        if normalized_question in seen_questions:
            errors.append("Duplicate question text.")
        else:
            seen_questions.add(normalized_question)

        if errors:
            rejected.append(
                {
                    "id": question_id,
                    "question": question_text,
                    "errors": errors,
                }
            )
        else:
            item["validation_status"] = "needs_review"
            accepted.append(item)

    for path, records in [
        (VALIDATED_PATH, accepted),
        (REJECTED_PATH, rejected),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Candidates: {len(candidates)}")
    print(f"Passed automatic validation: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Review candidates: {VALIDATED_PATH}")
    print(f"Validation errors: {REJECTED_PATH}")


if __name__ == "__main__":
    main()
