from typing import Literal

from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from src.agent.llm import agent_llm
from src.agent.state import AgentState
from src.agent.tools import ALL_TOOLS

agent_llm_tools = agent_llm.bind_tools(ALL_TOOLS)


tool_node = ToolNode(tools=ALL_TOOLS)


def call_model(state: AgentState):
    messages = state["messages"]
    response = agent_llm_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools"

    return END
