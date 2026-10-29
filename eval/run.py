"""Run the agent over every benchmark question and save what it did.

    python -m eval.run --variant hybrid
    python -m eval.run --variant vector --name vector_baseline

The variant decides which tools the agent gets, so the same questions can be
used to compare graph-only, vector-only and hybrid retrieval.
"""

import argparse
import re
import subprocess
import time
from datetime import UTC, datetime

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_neo4j import Neo4jGraph

from eval.common import RUNS_DIR, write_json, write_jsonl
from eval.dataset import load_questions
from src.agent.llm import agent_llm
from src.agent.tools import (
    find_paper_entities,
    initialize_graph,
    search_chunks_semantically,
    search_knowledge_graph,
)
from src.agent.workflow import build_agent

VARIANTS = {
    "hybrid": [search_knowledge_graph, find_paper_entities, search_chunks_semantically],
    "vector": [search_chunks_semantically],
    "graph": [search_knowledge_graph, find_paper_entities],
}

# search_chunks_semantically prints one "Chunk ID: <md5>" line per retrieved chunk
CHUNK_ID_PATTERN = re.compile(r"Chunk ID: (\w+)")


def summarize_messages(messages: list) -> dict:
    """Pull the answer, the tools used and the retrieved context out of the agent's messages."""
    tools_used: list[str] = []
    contexts: list[str] = []
    answer = ""

    for message in messages:
        if isinstance(message, AIMessage):
            tools_used += [call["name"] for call in message.tool_calls]
            if not message.tool_calls:
                answer = message.text
        elif isinstance(message, ToolMessage):
            contexts.append(message.text)

    chunk_ids = dict.fromkeys(CHUNK_ID_PATTERN.findall("\n".join(contexts)))
    return {
        "answer": answer,
        "tools_used": list(dict.fromkeys(tools_used)),
        "retrieved_chunk_ids": list(chunk_ids),
        "contexts": contexts,
    }


def run_question(agent, question: str) -> dict:
    start = time.perf_counter()
    messages = []
    error = None

    # The callback also counts tokens spent inside tools (e.g. Cypher generation)
    with get_usage_metadata_callback() as usage:
        try:
            for state in agent.stream(
                {"messages": [HumanMessage(question)]}, stream_mode="values"
            ):
                messages = state["messages"]
        except Exception as exc:  # noqa: BLE001 - one failing question must not stop the run
            error = f"{type(exc).__name__}: {exc}"

    tokens = usage.usage_metadata.values()
    return {
        **summarize_messages(messages),
        "error": error,
        "latency_s": round(time.perf_counter() - start, 2),
        "model_calls": sum(isinstance(m, AIMessage) for m in messages),
        "input_tokens": sum(t["input_tokens"] for t in tokens),
        "output_tokens": sum(t["output_tokens"] for t in tokens),
    }


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def main(variant: str, name: str | None, limit: int | None) -> None:
    questions = load_questions()[:limit]
    if not questions:
        raise SystemExit(
            "questions.jsonl is empty: generate and validate questions first"
        )
    run_dir = RUNS_DIR / (name or f"{datetime.now(UTC):%Y%m%d_%H%M}_{variant}")

    initialize_graph(Neo4jGraph())
    agent = build_agent(VARIANTS[variant])

    write_json(
        run_dir / "config.json",
        {
            "variant": variant,
            "tools": [tool.name for tool in VARIANTS[variant]],
            "agent_model": agent_llm.model_id,
            "questions": len(questions),
            "git_commit": git_commit(),
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )

    results = []
    for i, question in enumerate(questions, 1):
        result = {
            "id": question.id,
            "question": question.question,
            **run_question(agent, question.question),
        }
        results.append(result)
        status = (
            result["error"] or f"{result['latency_s']}s, tools={result['tools_used']}"
        )
        print(f"[{i}/{len(questions)}] {question.id}: {status}")

        # Save after every question so a crash halfway keeps the finished ones
        write_jsonl(run_dir / "results.jsonl", results)

    print(f"\nSaved to {run_dir}. Next: python -m eval.evaluate {run_dir.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", choices=list(VARIANTS), default="hybrid")
    parser.add_argument("--name", help="run folder name (default: <date>_<variant>)")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    args = parser.parse_args()
    main(args.variant, args.name, args.limit)
