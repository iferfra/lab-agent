import json
import os
from pathlib import Path
from statistics import mean
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from eval.llm import eval_llm

load_dotenv()

# Files
BENCHMARK_PATH = Path(
    os.getenv("DATA_DIR", "data/") + "eval/validated_candidates.jsonl"
)
RUN_RESULTS_PATH = Path(os.getenv("DATA_DIR", "data/") + "eval/run_results.jsonl")
RESULTS_PATH = Path(os.getenv("DATA_DIR", "data/") + "eval/evaluation_results.jsonl")
SUMMARY_PATH = Path(os.getenv("DATA_DIR", "data/") + "eval/evaluation_summary.json")

# Bedrock
EVALUATION_MODEL = os.getenv("EVALUATION_LLM")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


# ------------------------------------------------------------------
# 1. Define the evaluation scores
# ------------------------------------------------------------------

Score = Literal[0, 1, 2]


class Evaluation(BaseModel):
    answer_correctness: Score
    faithfulness: Score
    evidence_coverage: Score
    context_precision: Score
    context_recall: Score
    tool_selection_accuracy: Score
    graph_fact_accuracy: Score | None
    cross_source_attribution: Score | None
    abstention_correct: Literal[0, 1] | None
    reason: str = Field(description="Short explanation of the scores")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an impartial evaluator of a Graph RAG research assistant.

Evaluate the agent using only the supplied benchmark item and execution
results. Do not assume that a claim is true merely because it sounds
plausible. Do not penalize valid paraphrases or differences in wording.

GENERAL SCORING RULES
For metrics scored from 0 to 2:
- 0 = incorrect, absent, or substantially unsupported.
- 1 = partially correct, partially supported, or incomplete.
- 2 = correct, sufficiently supported, and complete.

Evaluate each metric independently. A low score on one metric must not
automatically determine the score of another metric.

ANSWER CORRECTNESS
- Compare the meaning of the answer with the expected answer and required
  facts, not their exact wording.
- Accept factually equivalent paraphrases and alternative explanations
  supported by the supplied evidence.
- Check whether the answer addresses the specific question asked.
- A relevant discussion of the general topic is not sufficient if it
  misses the specific fact requested.
- Do not penalize additional relevant information unless it is incorrect,
  misleading, or contradicts the reference.
- Score 0 when the central answer is incorrect or missing.
- Score 1 when the answer is partly correct but misses an important detail.
- Score 2 when the answer correctly and sufficiently addresses the question.

FAITHFULNESS
- Evaluate whether factual claims in the answer are supported by the
  retrieved contexts and tool results supplied in the execution.
- Do not use disagreement with the expected answer as sufficient reason
  to assign a low faithfulness score.
- A response can be faithful to the retrieved evidence but incomplete
  or incorrect as an answer to the question.
- A response can match the expected answer but still be unfaithful if
  its claims are not supported by the retrieved evidence.
- Penalize unsupported claims, fabricated details, and contradictions
  with the supplied evidence.
- Distinguish unsupported claims from claims that are simply not addressed
  by the available evidence.
- Do not assume that information absent from the retrieved contexts is
  false.
- If evidence is insufficient to verify a claim, explain this limitation.

EVIDENCE COVERAGE
- Evaluate whether the final answer includes the required facts listed
  in the benchmark.
- Assess each required fact individually.
- Do not award full coverage merely because the retrieved contexts contain
  the required facts if the final answer omits them.
- Accept semantically equivalent statements.

CONTEXT PRECISION
- Evaluate how much of the retrieved context is relevant to answering
  the specific question.
- Distinguish direct evidence from passages that only discuss the same
  general topic.
- Do not assume that a high similarity score guarantees relevance.

CONTEXT RECALL
- Evaluate whether the retrieved contexts contain the evidence needed to
  answer the question and support the required facts.
- If the benchmark identifies expected evidence, compare the retrieved
  contexts with that evidence.
- Do not evaluate context recall based on whether the final answer
  successfully uses the retrieved evidence.
- If the supplied execution does not contain enough information to assess
  retrieval completeness, use the most defensible score and explain the
  limitation in the reason.

TOOL SELECTION ACCURACY
- Evaluate whether the selected tools were appropriate for the information
  required by the question.
- Consider the expected tools, the actual tool calls, and the evidence
  obtained.
- An additional tool call is not automatically wrong if it provides
  useful information.
- Penalize unnecessary tool calls when they add no relevant evidence.
- Do not confuse tool selection quality with answer correctness.

GRAPH FACT ACCURACY
- Evaluate claims about graph entities, metadata, authorship, and
  relationships against the graph tool results.
- Do not score this metric based on claims supported only by paper text.
- Return null when the answer contains no relevant graph-derived claims
  or the metric is not applicable.

CROSS-SOURCE ATTRIBUTION
- When graph and semantic sources are both relevant, assess whether the
  answer combines them correctly and attributes claims to the appropriate
  source.
- Do not require both sources when the question does not need both.
- Do not penalize a semantic-only answer merely because it does not use
  graph evidence when graph evidence is unnecessary.
- Return null when cross-source attribution is not applicable.

ABSTENTION
- For unanswerable questions, abstention_correct = 1 if the agent
  appropriately acknowledges that the answer cannot be established from
  the available evidence.
- Set abstention_correct = 0 if the agent presents an unsupported answer
  as established fact.
- For answerable questions, abstention_correct must be null.
- Do not treat an appropriate, qualified answer as an incorrect abstention.

RUNTIME ERRORS
- Do not confuse a missing answer with a runtime error.
- Evaluate the answer actually produced, even if some tool calls failed.
- Mention runtime errors when they materially affect the evaluation.

EVALUATION REASON
- Provide a concise, evidence-based explanation for the assigned scores.
- Identify the specific required fact that is missing or incorrect when
  applicable.
- For faithfulness, identify unsupported or contradicted claims rather
  than merely stating that the answer differs from the reference.
- Do not claim that a fact is present in a retrieved context unless it
  can be found in the supplied execution results.
- If the supplied information is insufficient to make a reliable
  judgment, explicitly state that limitation.
""",
        ),
        (
            "human",
            """
BENCHMARK ITEM:
{benchmark}

AGENT EXECUTION:
{execution}
""",
        ),
    ]
)


if not EVALUATION_MODEL:
    raise ValueError("Set EVALUATION_LLM in your .env file.")

llm = eval_llm

judge = prompt | llm.with_structured_output(Evaluation)


# ------------------------------------------------------------------
# 2. Read the JSONL files
# ------------------------------------------------------------------


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


benchmark = read_jsonl(BENCHMARK_PATH)
runs = read_jsonl(RUN_RESULTS_PATH)

# Match each execution to its benchmark question.
benchmark_by_id = {item["id"]: item for item in benchmark}


# ------------------------------------------------------------------
# 3. Evaluate each question
# ------------------------------------------------------------------

results = []

for run in runs:
    question_id = run["id"]
    item = benchmark_by_id.get(question_id)

    if item is None:
        print(f"Skipping {question_id}: not found in benchmark.")
        continue

    if item["category"] == "unanswerable":
        print("Debug unanswerable")

    expected_tools = set(item.get("expected_tools", []))

    # tools_used is preferred because it usually contains tool names.
    # Fall back to tool_calls if necessary.
    actual_tools = run.get("tools_used") or run.get("tool_calls") or []

    # Convert tool-call dictionaries to tool names, if needed.
    actual_tools = {
        tool if isinstance(tool, str) else tool.get("name") or tool.get("tool_name")
        for tool in actual_tools
    }
    actual_tools.discard(None)

    # Deterministic tool metrics.
    true_positives = len(expected_tools & actual_tools)
    false_positives = len(actual_tools - expected_tools)
    false_negatives = len(expected_tools - actual_tools)

    tool_precision = (
        true_positives / len(actual_tools)
        if actual_tools
        else (1.0 if not expected_tools else 0.0)
    )
    tool_recall = (
        true_positives / len(expected_tools)
        if expected_tools
        else (1.0 if not actual_tools else 0.0)
    )
    tool_f1 = (
        2 * tool_precision * tool_recall / (tool_precision + tool_recall)
        if tool_precision + tool_recall
        else 0.0
    )

    tool_exact_match = expected_tools == actual_tools

    # Give the judge the reference and the actual execution evidence.
    benchmark_for_judge = {
        "id": item["id"],
        "category": item.get("category"),
        "question": item["question"],
        "answerable": item.get("answerable"),
        "expected_answer": item.get("expected_answer"),
        "required_facts": item.get("required_facts", []),
        "expected_tools": item.get("expected_tools", []),
        "evidence": item.get("evidence", []),
    }

    execution_for_judge = {
        "answer": run.get("answer"),
        "tools_used": run.get("tools_used"),
        "tool_calls": run.get("tool_calls"),
        "tool_results": run.get("tool_results"),
        "retrieved_contexts": run.get("retrieved_contexts"),
        "runtime_error": run.get("error"),
    }

    result = {
        "run_id": run.get("run_id"),
        "id": question_id,
        "category": item.get("category"),
        "answerable": item.get("answerable"),
        "answer": run.get("answer"),
        "runtime_error": run.get("error"),
        "expected_tools": sorted(expected_tools),
        "actual_tools": sorted(actual_tools),
        "tool_precision": tool_precision,
        "tool_recall": tool_recall,
        "tool_f1": tool_f1,
        "tool_exact_match": tool_exact_match,
        "model_calls": run.get("model_calls"),
        # These remain null unless the runner records them.
        "latency_seconds": run.get("latency_seconds", run.get("duration_seconds")),
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }

    # Extract token usage if the runner has recorded it.
    usage = run.get("token_usage") or run.get("usage_metadata") or {}

    if isinstance(usage, dict):
        result["input_tokens"] = usage.get("input_tokens", usage.get("prompt_tokens"))
        result["output_tokens"] = usage.get(
            "output_tokens", usage.get("completion_tokens")
        )
        result["total_tokens"] = usage.get("total_tokens")

        if result["total_tokens"] is None and (
            result["input_tokens"] is not None or result["output_tokens"] is not None
        ):
            result["total_tokens"] = (result["input_tokens"] or 0) + (
                result["output_tokens"] or 0
            )

    try:
        evaluation = judge.invoke(
            {
                "benchmark": json.dumps(benchmark_for_judge, ensure_ascii=False),
                "execution": json.dumps(execution_for_judge, ensure_ascii=False)[
                    :30000
                ],
            }
        )

        scores = evaluation.model_dump()

        # Abstention only applies to unanswerable questions.
        if item.get("answerable") is True:
            scores["abstention_correct"] = None

        result["llm_evaluation"] = scores

    except Exception as error:  # noqa: BLE001
        # Keep the deterministic metrics even if the judge fails.
        result["evaluation_error"] = str(error)

    results.append(result)
    print(f"Evaluated {question_id}")


# ------------------------------------------------------------------
# 4. Save detailed results
# ------------------------------------------------------------------

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

with RESULTS_PATH.open("w", encoding="utf-8") as file:
    for result in results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------
# 5. Calculate summary metrics
# ------------------------------------------------------------------

score_names = [
    "answer_correctness",
    "faithfulness",
    "evidence_coverage",
    "context_precision",
    "context_recall",
    "tool_selection_accuracy",
    "graph_fact_accuracy",
    "cross_source_attribution",
]

summary = {
    "total_questions": len(results),
    "llm_metrics": {},
    "tool_metrics": {},
    "abstention": {},
    "runtime": {},
    "by_category": {},
}

# LLM metrics: mean score on 0-2 scale and normalized 0-1 scale.
for name in score_names:
    scores = [
        result["llm_evaluation"][name]
        for result in results
        if result.get("llm_evaluation")
        and result["llm_evaluation"].get(name) is not None
    ]

    summary["llm_metrics"][name] = {
        "mean_0_to_2": mean(scores) if scores else None,
        "mean_0_to_1": mean(scores) / 2 if scores else None,
        "count": len(scores),
    }

# Deterministic tool metrics.
for name in ["tool_precision", "tool_recall", "tool_f1", "tool_exact_match"]:
    values = [result[name] for result in results if result.get(name) is not None]
    summary["tool_metrics"][name] = mean(values) if values else None

# Abstention accuracy: unanswerable questions only.
abstention_scores = [
    result["llm_evaluation"]["abstention_correct"]
    for result in results
    if result.get("answerable") is False
    and result.get("llm_evaluation")
    and result["llm_evaluation"].get("abstention_correct") is not None
]

summary["abstention"] = {
    "accuracy": mean(abstention_scores) if abstention_scores else None,
    "evaluated_questions": len(abstention_scores),
    "total_unanswerable_questions": sum(
        result.get("answerable") is False for result in results
    ),
}

# Runtime metrics are null when the runner did not record them.
for name in [
    "latency_seconds",
    "model_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
]:
    values = [
        result[name]
        for result in results
        if isinstance(result.get(name), (int, float))
        and not isinstance(result.get(name), bool)
    ]
    summary["runtime"][name] = mean(values) if values else None

# Repeat the main metrics for each question category.
for category in sorted({result.get("category", "unknown") for result in results}):
    category_results = [
        result for result in results if result.get("category", "unknown") == category
    ]

    category_metrics = {}

    for name in score_names:
        values = [
            result["llm_evaluation"][name]
            for result in category_results
            if result.get("llm_evaluation")
            and result["llm_evaluation"].get(name) is not None
        ]
        category_metrics[name] = mean(values) / 2 if values else None

    category_metrics["tool_f1"] = (
        mean([result["tool_f1"] for result in category_results])
        if category_results
        else None
    )

    summary["by_category"][category] = {
        "count": len(category_results),
        "metrics_0_to_1": category_metrics,
    }


# ------------------------------------------------------------------
# 6. Save and print summary
# ------------------------------------------------------------------

with SUMMARY_PATH.open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2, ensure_ascii=False)

print("\nEvaluation completed.")
print(f"Detailed results: {RESULTS_PATH}")
print(f"Summary: {SUMMARY_PATH}")
print(json.dumps(summary, indent=2, ensure_ascii=False))
