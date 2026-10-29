from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.nodes import final_answer, make_call_model, should_continue
from src.agent.state import AgentState
from src.agent.tools import ALL_TOOLS


def build_agent(tools: list[BaseTool] = ALL_TOOLS):
    """Build the agent graph. Passing a subset of tools is used for ablations."""
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", make_call_model(tools))
    workflow.add_node("tools", ToolNode(tools=tools))
    workflow.add_node("final_answer", final_answer)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "final_answer": "final_answer", END: END},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("final_answer", END)

    return workflow.compile()


agent = build_agent()
