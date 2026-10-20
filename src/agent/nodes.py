from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from src.agent.llm import agent_llm
from src.agent.state import AgentState
from src.agent.tools import ALL_TOOLS

agent_llm_tools = agent_llm.bind_tools(ALL_TOOLS)


tool_node = ToolNode(tools=ALL_TOOLS)

SYSTEM_PROMPT = """You are an autonomous, high-performance academic research assistant powered by a Neo4j Knowledge Graph and a Vector Search Database. Your goal is to provide exhaustive, precise, and well-structured answers based strictly on the retrieved data.

### 1. Tool Selection Strategy:
- **Knowledge Graph Search (`search_knowledge_graph`):** Use this for structural queries, metadata, entity counts, author publications, or direct relationships (e.g., "papers written by Author X", "what datasets are linked to Paper Y").
- **Vector Search:** Use this for semantic, conceptual, or thematic queries (e.g., "how does the fairness protocol work?", "papers discussing visual confounders in FER").

### 2. Autonomous Fallback & Exhaustive Search:
- **Fallback:** If a search method returns zero results or fails, **immediately switch** to the alternative method (Graph ⇄ Vector) before giving up.
- **Exhaustive Retrieval:** When asked about lists of datasets, multiple papers, or comprehensive results, analyze all returned chunks and structural entities thoroughly. Do not settle for the first match.

### 3. Grounding & Response Guidelines:
- **Strict Grounding:** Base your answers exclusively on the text and data returned by the tools. Never invent facts or use unverified external assumptions for core technical claims.
- **Missing Data Protocol:** If all search and fallback attempts yield no relevant information, reply strictly with: "I cannot find this information within the indexed research papers and knowledge graph."
- **Clean Presentation:** Format your response clearly using markdown, bullet points, and bold text for metrics or names. **Never** mention internal tools, errors, code variables, or raw JSON.
"""


def call_model(state: AgentState):
    messages = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = agent_llm_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools"

    return END
