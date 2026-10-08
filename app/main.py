from contextlib import asynccontextmanager
import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.ingest import queue_worker, ingest_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Logic before app startup
    worker_task = asyncio.create_task(queue_worker())  # Ingestion queue worker
    await asyncio.sleep(0)
    yield
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
