import asyncio
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.routers.ingest import ingest_router, queue_worker

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Logic before app startup
    worker_task = asyncio.create_task(queue_worker())  # Ingestion queue worker
    await asyncio.sleep(0)  # To let the event loop start the worker

    ingestion_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
    )
    chunk_embedder = BedrockEmbeddings(
        region_name="us-east-1", model_id="amazon.titan-embed-text-v2:0", normalize=True
    )
    vector_store = Chroma(
        embedding_function=chunk_embedder, persist_directory="./chroma_db"
    )
    yield {
        "ingestion_text_splitter": ingestion_text_splitter,
        "embedder": chunk_embedder,
        "vector_store": vector_store,
    }
    # Logic before app stop
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
