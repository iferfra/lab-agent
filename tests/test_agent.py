from langchain_core.messages import AIMessage
from langgraph.graph import END

from src.agent.nodes import MAX_MODEL_CALLS, should_continue

TOOL_CALL = {"name": "search_chunks_semantically", "args": {}, "id": "1"}


def test_ends_when_the_model_answers():
    state = {"messages": [AIMessage(content="Answer")], "model_calls": 1}
    assert should_continue(state) == END


def test_runs_tools_while_budget_remains():
    state = {
        "messages": [AIMessage(content="", tool_calls=[TOOL_CALL])],
        "model_calls": 1,
    }
    assert should_continue(state) == "tools"


def test_forces_a_final_answer_when_budget_is_exhausted():
    state = {
        "messages": [AIMessage(content="", tool_calls=[TOOL_CALL])],
        "model_calls": MAX_MODEL_CALLS,
    }
    assert should_continue(state) == "final_answer"
