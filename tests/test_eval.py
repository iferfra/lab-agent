from langchain_core.messages import AIMessage, ToolMessage

from eval.dataset import Question, Sources, find_problems
from eval.metrics import average, context_recall, sources_match
from eval.run import summarize_messages

SOURCES = Sources(
    {
        "graph_facts": [{"source_id": "graph_001"}],
        "chunks": [{"source_id": "chunk_a"}, {"source_id": "chunk_b"}],
        "known_gaps": [{"source_id": "gap_001"}],
    }
)


def make_question(
    category: str, evidence_ids: list[str], facts: list[str] | None = None
) -> Question:
    return Question(
        id="q",
        category=category,
        difficulty="easy",
        question="?",
        expected_answer="!",
        required_facts=["a fact"] if facts is None else facts,
        evidence_ids=evidence_ids,
    )


def test_context_recall_counts_retrieved_evidence():
    assert context_recall({"chunk_a", "chunk_b"}, ["chunk_a", "other"]) == 0.5


def test_context_recall_not_applicable_without_chunk_evidence():
    assert context_recall(set(), ["chunk_a"]) is None


def test_sources_match_requires_exact_set():
    assert sources_match({"graph"}, {"graph"})
    assert not sources_match({"graph"}, {"graph", "text"})
    assert not sources_match({"graph", "text"}, {"graph"})


def test_average_skips_not_applicable_values():
    assert average([1.0, None, 0.0, True]) == 0.667
    assert average([None]) is None


def test_valid_hybrid_question_has_no_problems():
    assert (
        find_problems(make_question("hybrid", ["graph_001", "chunk_a"]), SOURCES) == []
    )


def test_hybrid_question_needs_both_kinds_of_evidence():
    assert find_problems(make_question("hybrid", ["chunk_a"]), SOURCES)


def test_unknown_evidence_is_rejected():
    problems = find_problems(make_question("semantic", ["chunk_a", "made_up"]), SOURCES)
    assert any("made_up" in p for p in problems)


def test_unanswerable_question_points_to_a_gap():
    assert (
        find_problems(make_question("unanswerable", ["gap_001"], facts=[]), SOURCES)
        == []
    )
    assert find_problems(make_question("unanswerable", ["chunk_a"], facts=[]), SOURCES)


def test_summarize_messages_extracts_answer_tools_and_chunks():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_chunks_semantically", "args": {}, "id": "1"},
                {"name": "search_chunks_semantically", "args": {}, "id": "2"},
            ],
        ),
        ToolMessage(
            content="Chunk ID: abc\ntext\n---\nChunk ID: def\ntext", tool_call_id="1"
        ),
        ToolMessage(content="Chunk ID: abc\ntext", tool_call_id="2"),
        AIMessage(content="The answer."),
    ]
    summary = summarize_messages(messages)

    assert summary["answer"] == "The answer."
    assert summary["tools_used"] == ["search_chunks_semantically"]
    assert summary["retrieved_chunk_ids"] == ["abc", "def"]
