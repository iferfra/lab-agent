"""Compare every evaluated run in a single Markdown table.

    python -m eval.report

Prints the table and saves it to eval/results.md.
"""

from eval.common import EVAL_DIR, RUNS_DIR, read_json

COLUMNS = {
    "variant": "Variant",
    "fact_recall": "Fact recall",
    "faithfulness": "Faithfulness",
    "abstention": "Abstention",
    "context_recall": "Context recall",
    "source_accuracy": "Source accuracy",
    "latency_s_median": "Latency (s, median)",
    "tokens_mean": "Tokens / question",
}

RATIOS = {
    "fact_recall",
    "faithfulness",
    "abstention",
    "context_recall",
    "source_accuracy",
}


def format_cell(key: str, value) -> str:
    if value is None:
        return "—"
    if key in RATIOS:
        return f"{value:.0%}"
    return str(value)


def main() -> None:
    rows = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not (run_dir / "summary.json").exists():
            continue
        config = read_json(run_dir / "config.json")
        summary = read_json(run_dir / "summary.json")["overall"]
        rows.append({"run": run_dir.name, "variant": config["variant"], **summary})

    header = ["Run", *COLUMNS.values()]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *(
            "| "
            + " | ".join(
                [row["run"], *(format_cell(key, row.get(key)) for key in COLUMNS)]
            )
            + " |"
            for row in rows
        ),
    ]
    table = "\n".join(lines)

    (EVAL_DIR / "results.md").write_text(
        f"# Evaluation results\n\n{table}\n", encoding="utf-8"
    )
    print(table)


if __name__ == "__main__":
    main()
