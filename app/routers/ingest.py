import asyncio
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.dependencies import get_text_splitter, get_vector_store
from src.ingestion import chunk_index_document, pdf_to_md, validate_pdf

ingest_router = APIRouter()

UPLOAD_DIR = Path("data/docs/")

ingestion_queue = asyncio.Queue()
ingestion_status = {}


def db_ingestion(
    file_path: Path,
    md_path: Path,
    text_splitter: RecursiveCharacterTextSplitter,
    vector_store: Chroma,
):
    pdf_to_md(file_path, md_path)
    chunk_index_document(text_splitter, vector_store, md_path)


async def queue_worker():
    while True:
        (
            file_path,
            md_path,
            filename,
            text_splitter,
            vector_store,
        ) = await ingestion_queue.get()
        try:
            ingestion_status[filename] = "processing"
            await asyncio.to_thread(
                db_ingestion, file_path, md_path, text_splitter, vector_store
            )

            ingestion_status[filename] = "completed"
        except Exception:  # noqa: BLE001
            ingestion_status[filename] = "failed"
            file_path.unlink(missing_ok=True)
            md_path.unlink(missing_ok=True)
            print(f"Deleted files for: {filename}")
        finally:
            ingestion_queue.task_done()


@ingest_router.post("/ingest")
async def upload_ingest(
    files: list[UploadFile],
    text_splitter: Annotated[
        RecursiveCharacterTextSplitter, Depends(get_text_splitter)
    ],
    vector_store: Annotated[Chroma, Depends(get_vector_store)],
):

    for file in files:
        if await validate_pdf(file):
            file_path = UPLOAD_DIR / Path("pdf") / file.filename
            if not file_path.is_file():
                file_path.parent.mkdir(exist_ok=True, parents=True)

                async with aiofiles.open(file_path, "wb") as buffer:
                    while chunk := await file.read(64 * 1024):
                        await buffer.write(chunk)

                md_path = (
                    UPLOAD_DIR
                    / Path("md")
                    / Path(file.filename.rsplit(".", 1)[0] + ".md")
                )
                md_path.parent.mkdir(exist_ok=True, parents=True)

                ingestion_status[file.filename] = "pending"
                await ingestion_queue.put(
                    (file_path, md_path, file.filename, text_splitter, vector_store)
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"{file.filename} already uploaded and indexed",
                )

        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{file.filename} couldn't be uploaded: It's not a pdf",
            )

    return {
        "status": "Upload done correctly. Ingestion started for all files.'",
        "filenames": [file.filename for file in files],
    }


@ingest_router.get("/ingest/status")
async def get_ingest_status():
    return ingestion_status
