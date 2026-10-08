from pathlib import Path

import pymupdf4llm
from fastapi import HTTPException, UploadFile, status
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def pdf_to_md(pdf_path: Path, output_path: Path):
    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_text)
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
    try:
        loader = UnstructuredMarkdownLoader(document_path)
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        vector_store.add_documents(documents=splits)
        print(f"Indexed {len(splits)} chunks.")

    except Exception as e:
        print(e)
        raise
