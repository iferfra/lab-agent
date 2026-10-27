import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_neo4j import Neo4jGraph

# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = Path("data/eval/validated_candidates.jsonl")
OUTPUT_PATH = Path("data/eval/run_results.jsonl")

MAX_MODEL_CALLS = 5
SAVE_AGENT_CONFIG = True

AGENT_CONFIG = {
    "model": os.getenv("AGENT_LLM", "not_recorded"),
    "max_model_calls": MAX_MODEL_CALLS,
}

# Replace this import with the actual location of your compiled graph.
from src.agent.tools import initialize_graph
from src.agent.workflow import agent

initialize_graph(Neo4jGraph())

# --------------------------------------------------
# Utility functions
# --------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def get_message_content(message) -> str:
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    return json.dumps(content, ensure_ascii=False, default=str)


def serialize_tool_call(tool_call: dict) -> dict:
    return {
        "id": tool_call.get("id"),
        "name": tool_call.get("name"),
        "args": tool_call.get("args", {}),
        "type": tool_call.get("type", "tool_call"),
    }


def extract_execution_data(state: dict) -> dict:
    messages = state.get("messages", [])

    answer = None
    tool_calls = []
    tool_results = []
    tools_used = []

    for message in messages:
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", []):
                serialized_call = serialize_tool_call(call)
                tool_calls.append(serialized_call)

                name = serialized_call["name"]
                if name and name not in tools_used:
                    tools_used.append(name)

            # Keep the last assistant message that contains a final answer.
            if not getattr(message, "tool_calls", []):
                content = get_message_content(message)
                if content.strip():
                    answer = content

        elif isinstance(message, ToolMessage):
            tool_results.append(
                {
                    "tool_call_id": getattr(message, "tool_call_id", None),
                    "name": getattr(message, "name", None),
                    "content": get_message_content(message),
                    "status": getattr(message, "status", None),
                }
            )

    retrieved_contexts = [
        result["content"] for result in tool_results if result["content"]
    ]

    model_calls = state.get("model_calls")

    if model_calls is None:
        # Fallback if the state does not expose its model-call counter.
        model_calls = sum(isinstance(message, AIMessage) for message in messages)

    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "tools_used": tools_used,
        "retrieved_contexts": retrieved_contexts,
        "model_calls": model_calls,
    }


# --------------------------------------------------
# Benchmark execution
# --------------------------------------------------


def run_benchmark() -> None:
    questions = load_jsonl(INPUT_PATH)

    run_id = (
        datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    )

    # Start a fresh output file for this run.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("", encoding="utf-8")

    print(f"Run ID: {run_id}")
    print(f"Questions: {len(questions)}")

    for index, item in enumerate(questions, start=1):
        question_id = item.get("id", f"question_{index:03d}")
        question = item["question"]

        print(f"[{index}/{len(questions)}] {question_id}")

        final_state = {}

        record = {
            "run_id": run_id,
            "id": question_id,
            "question": question,
            "answer": None,
            "tool_calls": [],
            "error": None,
            "tool_results": [],
            "tools_used": [],
            "retrieved_contexts": [],
            "model_calls": 0,
        }

        if SAVE_AGENT_CONFIG:
            record["agent_config"] = AGENT_CONFIG.copy()

        try:
            # Streaming preserves the latest state if execution fails
            # after some intermediate steps.
            for state in agent.stream(
                {"messages": [HumanMessage(content=question)]},
                stream_mode="values",
            ):
                final_state = state

            execution_data = extract_execution_data(final_state)
            record.update(execution_data)

        except Exception as exc:  # noqa: BLE001
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

            # Preserve any intermediate messages available.
            if final_state:
                record.update(extract_execution_data(final_state))

        append_jsonl(OUTPUT_PATH, record)

        if record["error"]:
            print(f"  Error: {record['error']['type']}")
        else:
            print(f"  Tools used: {record['tools_used']}")
            print(f"  Model calls: {record['model_calls']}")

    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_benchmark()
