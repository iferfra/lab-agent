"""Metrics computed directly from the run, without an LLM."""

from statistics import mean, median


def context_recall(
    expected_chunk_ids: set[str], retrieved_chunk_ids: list[str]
) -> float | None:
    """Fraction of the evidence chunks that the agent actually retrieved.

    This is a strict lower bound: another chunk may hold the same information.
    Returns None when the question has no chunk evidence (e.g. graph questions).
    """
    if not expected_chunk_ids:
        return None
    return len(expected_chunk_ids & set(retrieved_chunk_ids)) / len(expected_chunk_ids)


def sources_match(expected_sources: set[str], sources_used: set[str]) -> bool:
    """True when the agent read exactly the sources (graph, text) the question
    needs: no more, no less."""
    return expected_sources == sources_used


def average(values: list[float | bool | None]) -> float | None:
    """Mean of the values that apply, ignoring None (metric not applicable)."""
    present = [float(v) for v in values if v is not None]
    return round(mean(present), 3) if present else None


def summarize(scores: list[dict]) -> dict:
    """Aggregate per-question scores (a non-empty list) into the run's numbers."""
    return {
        "questions": len(scores),
        "fact_recall": average([s["fact_recall"] for s in scores]),
        "faithfulness": average([s["faithfulness"] for s in scores]),
        "abstention": average([s["abstained"] for s in scores]),
        "context_recall": average([s["context_recall"] for s in scores]),
        "source_accuracy": average([s["sources_match"] for s in scores]),
        "errors": sum(s["error"] is not None for s in scores),
        "latency_s_median": round(median(s["latency_s"] for s in scores), 1),
        "tokens_mean": round(
            mean(s["input_tokens"] + s["output_tokens"] for s in scores)
        ),
    }
