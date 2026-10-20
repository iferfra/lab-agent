import os

import chainlit as cl
import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(
        content=(
            "👋 Hello! I am your **GraphRAG** academic assistant (connected to"
            " Neo4j and AWS Bedrock). Ask me about papers, authors, or metrics."
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="🔍 Querying the knowledge graph...")
    await msg.send()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(API_URL, json={"query": message.content})

            if response.status_code == 200:
                data = response.json()
                msg.content = data.get("response", "No response from the agent.")
            else:
                msg.content = (
                    f"❌ FastAPI server error ({response.status_code}): {response.text}"
                )

    except httpx.ConnectError:
        msg.content = "❌ Could not connect to the API. Make sure FastAPI is running on port 8000."
    except Exception as e:  # noqa: BLE001
        msg.content = f"❌ An unexpected error occurred: {e!s}"

    await msg.update()
