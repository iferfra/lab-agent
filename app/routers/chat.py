from fastapi import APIRouter, Request

from app.schemas.chat import ChatRequest
from src.agent.workflow import agent

chat_router = APIRouter()


@chat_router.post("/chat")
async def chat(request: Request, body: ChatRequest):

    config = {
        "configurable": {
            "neo4j_graph": request.app.state.neo4j_graph,
        }
    }

    result = agent.invoke({"messages": body.query}, config=config)

    return {
        "query": body.query,
        "response": result.get("messages")[-1].text,
    }
