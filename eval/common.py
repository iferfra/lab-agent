import json
import os
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

load_dotenv()

EVAL_DIR = Path(__file__).parent
DATASET_DIR = EVAL_DIR / "dataset"
RUNS_DIR = EVAL_DIR / "runs"

SOURCES_PATH = DATASET_DIR / "sources.json"
CANDIDATES_PATH = DATASET_DIR / "candidates.jsonl"
QUESTIONS_PATH = DATASET_DIR / "questions.jsonl"

EVAL_MODEL = os.getenv("EVALUATION_LLM")


@cache
def eval_llm() -> ChatBedrockConverse:
    """LLM used to generate questions and to judge answers."""
    if not EVAL_MODEL:
        raise RuntimeError("Set EVALUATION_LLM in .env")
    return ChatBedrockConverse(
        model_id=EVAL_MODEL, region_name="us-east-1", temperature=0.0
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
