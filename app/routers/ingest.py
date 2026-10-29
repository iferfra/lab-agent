import hashlib
import os
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import APIRouter, Form, HTTPException, UploadFile, status

from src.ingestion import ingestion_queue, ingestion_status

ingest_router = APIRouter()
UPLOAD_DIR = Path(os.getenv("DATA_DIR", "data/") + "docs")


async def calculate_file_hash(file: UploadFile) -> str:
    sha256 = hashlib.sha256()

    await file.seek(0)

    while chunk := await file.read(64 * 1024):
        sha256.update(chunk)

    await file.seek(0)

    return sha256.hexdigest()


async def validate_pdf(file: UploadFile) -> bool:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Not permitted extension. File '{file.filename}' must be .pdf",
        )

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mime type is not application/pdf",
        )
    try:
        first_bytes = await file.read(5)
        await file.seek(0)
        if first_bytes != b"%PDF-":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Content validation failed.",
            )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intern error processing the file: {e!s}",
        )
    return True


@ingest_router.post("/ingest/paper")
async def upload_ingest_paper(
    file: UploadFile,
    paper_title: Annotated[str | None, Form()] = None,
    authors: Annotated[list[str] | None, Form()] = None,
    doi: Annotated[str | None, Form()] = None,
    venue: Annotated[str | None, Form()] = None,
    year: Annotated[int | None, Form()] = None,
    abstract: Annotated[str | None, Form()] = None,
    repository_link: Annotated[str | None, Form()] = None,
):
    if await validate_pdf(file):
        file_hash = await calculate_file_hash(file)
        file_path = UPLOAD_DIR / "pdf" / f"{file_hash}.pdf"

        # The same PDF always has the same hash, so it is only indexed once
        if not file_path.is_file():
            file_path.parent.mkdir(exist_ok=True, parents=True)

            async with aiofiles.open(file_path, "wb") as buffer:
                while chunk := await file.read(64 * 1024):
                    await buffer.write(chunk)

            md_path = UPLOAD_DIR / Path("md") / f"{file_hash}.md"
            md_path.parent.mkdir(exist_ok=True, parents=True)

            file_metadata = {
                "paper_id": file_hash,
                "file_name": file.filename,
                "file_path": file_path,
                "md_path": md_path,
            }

            form_metadata = {
                "paper_title": paper_title,
                "authors": authors,
                "doi": doi,
                "venue": venue,
                "year": year,
                "abstract": abstract,
                "repository_link": repository_link,
            }

            file_metadata.update(
                {k: v for k, v in form_metadata.items() if v is not None}
            )

            ingestion_status[file.filename] = "pending"

            await ingestion_queue.put(file_metadata)
    else:
        raise HTTPException(
            status_code=400, detail=f"{file.filename} is not a valid PDF"
        )

    return {"status": "Ingestion started", "filename": [file.filename]}


@ingest_router.get("/ingest/status")
async def get_ingest_status():
    return ingestion_status
