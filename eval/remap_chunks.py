"""Point the dataset at the new chunk ids after the papers are re-ingested.

Chunk ids are hashes of the chunk text, so they change whenever the chunking
changes. For every chunk in sources.json this finds the new chunk that
contains most of its words, and replaces the old id with the new
one in sources.json, the questions and the results of earlier runs (so those
runs can still be scored).

    python -m eval.remap_chunks          # show the mapping
    python -m eval.remap_chunks --write  # apply it

Matches below 80% overlap are flagged: check those questions by hand.
"""

import argparse
import re
from difflib import SequenceMatcher

from langchain_neo4j import Neo4jGraph

from eval.common import (
    CANDIDATES_PATH,
    QUESTIONS_PATH,
    RUNS_DIR,
    SOURCES_PATH,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)

CHUNKS_QUERY = """
MATCH (chunk:Document)
RETURN chunk.id AS id, chunk.text AS text
"""


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def overlap(old_text: str, new_text: str) -> float:
    """Fraction of the old chunk's words found, in order, in the new chunk."""
    old_words = words(old_text)
    matcher = SequenceMatcher(None, old_words, words(new_text), autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(old_words)


def replace_ids(ids: list[str], new_ids: dict[str, str]) -> list[str]:
    return [new_ids.get(chunk_id, chunk_id) for chunk_id in ids]


def main(write: bool) -> None:
    sources = read_json(SOURCES_PATH)
    new_chunks = Neo4jGraph().query(CHUNKS_QUERY)

    new_ids = {}
    for chunk in sources["chunks"]:
        best = max(new_chunks, key=lambda c: overlap(chunk["text"], c["text"]))
        score = overlap(chunk["text"], best["text"])

        flag = "  <-- check" if score < 0.8 else ""
        print(f"{chunk['source_id']} -> {best['id']}  {score:.0%}{flag}")

        new_ids[chunk["source_id"]] = best["id"]
        chunk["source_id"] = best["id"]
        chunk["text"] = best["text"]

    if not write:
        print("\nDry run. Use --write to apply the mapping.")
        return

    write_json(SOURCES_PATH, sources)

    for path in (QUESTIONS_PATH, CANDIDATES_PATH):
        questions = read_jsonl(path)
        for question in questions:
            question["evidence_ids"] = replace_ids(question["evidence_ids"], new_ids)
        write_jsonl(path, questions)

    for results_path in RUNS_DIR.glob("*/results.jsonl"):
        results = read_jsonl(results_path)
        for result in results:
            result["retrieved_chunk_ids"] = replace_ids(
                result["retrieved_chunk_ids"], new_ids
            )
        write_jsonl(results_path, results)

    print("\nUpdated sources.json, the questions and the runs' results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--write", action="store_true", help="apply the mapping")
    args = parser.parse_args()
    main(args.write)
