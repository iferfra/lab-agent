from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from src.agent.llm import agent_llm
from src.agent.state import AgentState
from src.agent.tools import (
    ALL_TOOLS,
)

agent_llm_tools = agent_llm.bind_tools(ALL_TOOLS)
tool_node = ToolNode(tools=ALL_TOOLS)

MAX_MODEL_CALLS = 5

SYSTEM_PROMPT = """You are an academic research assistant using a Neo4j
knowledge graph and semantic search over scientific paper text.

TOOL SELECTION
- Use search_knowledge_graph for paper metadata, authorship, and graph
  relationships.
- Use search_chunks_semantically for specific claims, technical details,
  cited authors, references, and evidence from the original paper text.


HYBRID QUESTIONS
- Use both graph evidence and original text evidence when both are relevant.
- A paper's actual authors are not necessarily the authors cited in its text.
- Verify cited names and specific claims against source chunks.
- A summary omitting a fact does not prove that the fact is absent.
- Do not claim that information is missing until relevant source evidence
  has been searched.
- Never invent facts or claim that a tool found evidence it did not return.

ANSWERING
- Base technical claims on retrieved evidence.
- Answer every part of the question.
- If evidence is insufficient, state what could not be verified.
- Use clear, concise Markdown.
- Never expose internal tool calls, code variables, or raw JSON.
"""


def call_model(state: AgentState):
    messages = list(state["messages"])
    model_calls = state.get("model_calls", 0) + 1

    if not any(
        isinstance(message, SystemMessage)
        and "academic research assistant" in str(message.content).lower()
        for message in messages
    ):
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

    response = agent_llm_tools.invoke(messages)

    return {
        "messages": [response],
        "tools": [call["name"] for call in response.tool_calls],
        "model_calls": model_calls,
    }


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]

    # Continue tool execution only while the model-call budget remains.
    if last_message.tool_calls and state.get("model_calls", 0) < MAX_MODEL_CALLS:
        return "tools"

    return END
