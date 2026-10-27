from langgraph.graph import END, START, StateGraph

from src.agent.nodes import call_model, should_continue, tool_node
from src.agent.state import AgentState

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

agent = workflow.compile()
