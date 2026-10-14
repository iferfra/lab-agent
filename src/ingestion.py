import hashlib
from pathlib import Path

import pymupdf4llm
from fastapi import HTTPException, UploadFile, status
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.graph import build_upload_graph


class IndexingProcessError(Exception):
    """Exception raised when the ingestion fails. Doing rollback of the chunks."""

    def __init__(self, message, added_ids):
        super().__init__(message)
        self.added_ids = added_ids


class DocumentAlreadyIndexedError(Exception):
    """Exception raised when all chunks have already been processed (repeated document)"""


def pdf_to_md(pdf_path: Path, output_path: Path):
    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"Error processing file: {e}")


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


def chunk_index_document(
    text_splitter: RecursiveCharacterTextSplitter,
    vector_store: Chroma,
    document_path: Path,
):
    ids = []
    try:
        loader = UnstructuredMarkdownLoader(document_path)
        docs = loader.load()
        splits = text_splitter.split_documents(docs)

        # We use hash id to prevent duplicated chunks in db (based in chunk content)

        for split in splits:
            unique_string = split.page_content
            chunk_id = hashlib.md5(unique_string.encode("utf-8")).hexdigest()
            ids.append(chunk_id)

        existing_data = vector_store.get(ids=ids)
        existing_ids = set(existing_data.get("ids", []))

        if len(existing_ids) == len(splits):
            print(
                f"Document '{document_path.name}' hasn't changed (all chunks exist in the DB)."
            )
            raise DocumentAlreadyIndexedError(
                f"Document {document_path.name} already indexed."
            )

        # Add chunks to the vector store
        added_ids = vector_store.add_documents(documents=splits, ids=ids)
        print(f"Indexed {len(splits)} chunks.")

        # Add chunks to the knowledge graph
        build_upload_graph(splits)
        return added_ids

    except DocumentAlreadyIndexedError:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"Error processing the file: {e}")
        raise IndexingProcessError(f"Error in pipeline: {e}", added_ids=added_ids)
