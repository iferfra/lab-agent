"""Score a run: deterministic metrics plus the LLM judge.

    python -m eval.evaluate 20261028_1200_hybrid
    python -m eval.evaluate 20261028_1200_hybrid --no-judge   # free, no LLM calls

Writes scores.jsonl (one line per question, with the judge's reasoning) and
summary.json (overall and per category) into the run folder.
"""

import argparse

from eval.common import RUNS_DIR, read_jsonl, write_json, write_jsonl
from eval.dataset import TOOL_SOURCE, Question, Sources, load_questions
from eval.judge import judge_abstention, judge_fact_recall, judge_faithfulness
from eval.metrics import context_recall, sources_match, summarize


def score(question: Question, result: dict, sources: Sources, use_judge: bool) -> dict:
    answer = result["answer"]
    scores = {
        "id": question.id,
        "category": question.category,
        "context_recall": context_recall(
            set(question.evidence_ids) & sources.chunk_ids,
            result["retrieved_chunk_ids"],
        ),
        "sources_match": sources_match(
            question.expected_sources,
            {TOOL_SOURCE[t] for t in result["tools_used"] if t in TOOL_SOURCE},
        ),
        "fact_recall": None,
        "faithfulness": None,
        "abstained": None,
        "notes": {},
        "error": result["error"],
        "latency_s": result["latency_s"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }
    if not use_judge:
        return scores

    if not answer:
        # No answer is a failure for every judged metric
        scores["fact_recall"] = 0.0 if question.answerable else None
        scores["abstained"] = None if question.answerable else False
        return scores

    try:
        if question.answerable:
            recall = judge_fact_recall(
                question.question, question.required_facts, answer
            )
            scores["fact_recall"] = sum(c.present for c in recall.checks) / len(
                recall.checks
            )
            scores["notes"]["missing_facts"] = [
                c.fact for c in recall.checks if not c.present
            ]
        else:
            abstention = judge_abstention(question.question, answer)
            scores["abstained"] = abstention.abstained
            scores["notes"]["abstention"] = abstention.reason

        faithfulness = judge_faithfulness(answer, result["contexts"])
        scores["faithfulness"] = round(faithfulness.score, 3)
        scores["notes"]["unsupported_claims"] = [
            c.claim for c in faithfulness.claims if not c.supported
        ]
    except Exception as exc:  # noqa: BLE001 - keep the deterministic metrics if the judge fails
        scores["notes"]["judge_error"] = f"{type(exc).__name__}: {exc}"

    return scores


def print_table(summary: dict) -> None:
    rows = {"all": summary["overall"], **summary["by_category"]}
    metrics = [
        "questions",
        "fact_recall",
        "faithfulness",
        "abstention",
        "context_recall",
        "source_accuracy",
    ]
    print(f"{'':<14}" + "".join(f"{m:>16}" for m in metrics))
    for name, row in rows.items():
        cells = [
            "—"
            if row[m] is None
            else f"{row[m]:.2f}"
            if isinstance(row[m], float)
            else str(row[m])
            for m in metrics
        ]
        print(f"{name:<14}" + "".join(f"{c:>16}" for c in cells))


def main(run_name: str, use_judge: bool) -> None:
    run_dir = RUNS_DIR / run_name
    results = read_jsonl(run_dir / "results.jsonl")
    if not results:
        available = sorted(d.name for d in RUNS_DIR.iterdir() if d.is_dir())
        raise SystemExit(f"No results in {run_dir}. Available runs: {available}")

    questions = {q.id: q for q in load_questions()}
    sources = Sources.load()

    # A regenerated dataset can reuse ids for different questions
    outdated = [
        r["id"]
        for r in results
        if r["id"] not in questions or r.get("question") != questions[r["id"]].question
    ]
    if outdated:
        raise SystemExit(
            f"{run_name} was run on other questions than the current dataset "
            f"({len(outdated)} differ, e.g. {outdated[0]}). Run the agent again."
        )

    all_scores = []
    for result in results:
        all_scores.append(score(questions[result["id"]], result, sources, use_judge))
        print(f"scored {result['id']}")

    categories = sorted({s["category"] for s in all_scores})
    summary = {
        "overall": summarize(all_scores),
        "by_category": {
            c: summarize([s for s in all_scores if s["category"] == c])
            for c in categories
        },
    }

    write_jsonl(run_dir / "scores.jsonl", all_scores)
    write_json(run_dir / "summary.json", summary)
    print()
    print_table(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", help="folder name inside eval/runs/")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="only compute the metrics that need no LLM",
    )
    args = parser.parse_args()
    main(args.run, use_judge=not args.no_judge)
