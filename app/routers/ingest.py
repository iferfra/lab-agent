import aiofiles
from pathlib import Path
import asyncio
from fastapi import APIRouter, HTTPException, UploadFile, status, BackgroundTasks
import pymupdf4llm

ingest_router = APIRouter()

UPLOAD_DIR = Path("data/docs/")

ingestion_queue = asyncio.Queue()
ingestion_status = {} 

def pdf_to_md(pdf_path: Path, output_path: Path):
    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_text)
    except Exception as e:
        print(f"Error processing file: {e}")

def db_ingestion(file_path: Path, md_path:Path):
    pdf_to_md(file_path, md_path)

async def queue_worker():
    while True:

        file_path, md_path, filename = await ingestion_queue.get()
        try:
            ingestion_status[filename] = "processing"
            await asyncio.to_thread(db_ingestion, file_path, md_path)
            
            ingestion_status[filename] = "completed"
        except Exception:
            ingestion_status[filename] = "failed"
        finally:
            ingestion_queue.task_done()


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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intern error processing the file: {e!s}",
        )
    return True


@ingest_router.post("/ingest")
async def upload_ingest(files: list[UploadFile]):

    for file in files:
        if await validate_pdf(file):
            file_path = UPLOAD_DIR / Path("pdf") / file.filename
            file_path.parent.mkdir(exist_ok=True, parents=True)

            async with aiofiles.open(file_path, "wb") as buffer:
                while chunk := await file.read(64 * 1024):  
                    await buffer.write(chunk)

            md_path = UPLOAD_DIR / Path("md") /  Path(file.filename.rsplit('.', 1)[0] + ".md")
            md_path.parent.mkdir(exist_ok=True, parents=True)

            ingestion_status[file.filename] = "pending"
            await ingestion_queue.put((file_path, md_path, file.filename))

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
