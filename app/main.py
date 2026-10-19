import asyncio
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_neo4j import Neo4jGraph

import src.agent.tools as tool_module
from app.routers.chat import chat_router
from app.routers.ingest import ingest_router
from src.graph import init_neo4j_schema
from src.ingestion import ingestion_queue_worker

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.neo4j_graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
    )

    init_neo4j_schema(app.state.neo4j_graph, embedding_dimension=1024)
    tool_module.graph = app.state.neo4j_graph

    worker_task = asyncio.create_task(ingestion_queue_worker(app))
    await asyncio.sleep(0)

    yield

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        print("Worker stopped correctly.")


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(chat_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
