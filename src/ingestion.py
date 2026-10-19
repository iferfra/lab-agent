import asyncio
import os
from pathlib import Path

import aiofiles
import pymupdf4llm
from langchain_aws import ChatBedrockConverse
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from src.graph import build_upload_graph

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=int(os.getenv("CHUNK_SIZE", "1500")),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
)

extract_llm = ChatBedrockConverse(
    model_id="us.amazon.nova-lite-v1:0",
    region_name="us-east-1",
    temperature=0.0,
)

ingestion_queue = asyncio.Queue()
ingestion_status = {}


class ExtractMetadata(BaseModel):
    """Metadata to be extracted from the text provided."""

    year: str = Field(description="Year when the paper was published.")
    paper_title: str = Field(
        description="Paper title. Well formatted as it appears in the text."
    )
    authors: list[str] = Field(
        description="List of authors of the paper. Well formatted."
    )
    doi: str = Field(description="DOI of the paper.")
    venue: str = Field(description="Venue where the paper was published.")
    abstract: str = Field(description="Complete abstract of the paper.")
    repository_link: str = Field(description="Link of the repository. Ex: GitHub")


def pdf_to_md(pdf_path: Path, output_path: Path):
    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"Error processing file: {e}")
        return False


async def extract_metadata(file_metadata) -> dict:

    async with aiofiles.open(file_metadata["md_path"], "r", encoding="utf-8") as f:
        paper_text = await f.read(4000)

    extraction_llm = extract_llm.with_structured_output(ExtractMetadata)

    messages = [
        SystemMessage(
            content=(
                "You are an expert scientific metadata extraction assistant. "
                "Analyze the provided text excerpt from a research paper and extract the requested metadata accurately. "
                "If a field cannot be found in the text, leave it empty or omit it."
            )
        ),
        HumanMessage(
            content=(
                "Please extract the metadata from the following beginning section of the paper:\n\n"
                f"{paper_text}"
            )
        ),
    ]

    extracted_metadata = extraction_llm.invoke(messages)

    for metadata_field in extracted_metadata.model_dump():
        if metadata_field not in file_metadata:
            file_metadata[metadata_field] = extracted_metadata.model_dump()[
                metadata_field
            ]
    return file_metadata


async def ingest_document(neo4j_graph, file_metadata: Path):

    try:
        file_metadata = await extract_metadata(file_metadata=file_metadata)
        loader = UnstructuredMarkdownLoader(file_metadata["md_path"])
        docs = loader.load()
        splits = text_splitter.split_documents(docs)

        build_upload_graph(neo4j_graph, splits, file_metadata)

    except Exception as e:  # noqa:BLE001
        raise Exception(f"IndexingProcessError: {e}")  # noqa: TRY002


async def ingestion_queue_worker(app):
    while True:
        file_metadata = await ingestion_queue.get()
        neo4j_graph = app.state.neo4j_graph

        try:
            ingestion_status[file_metadata["file_name"]] = "processing"
            converted = await asyncio.to_thread(
                pdf_to_md, file_metadata["file_path"], file_metadata["md_path"]
            )

            if converted:
                await ingest_document(neo4j_graph, file_metadata)
            ingestion_status[file_metadata["file_name"]] = "completed"

        except Exception:  # noqa: BLE001
            ingestion_status[file_metadata["file_name"]] = "failed"
            file_metadata["file_path"].unlink(missing_ok=True)
            file_metadata["md_path"].unlink(missing_ok=True)
        finally:
            ingestion_queue.task_done()
