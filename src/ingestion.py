import asyncio
import os

import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.graph.graph import build_upload_graph, delete_paper

# First split the paper by its section headings (the "#" heading is usually the
# title), then split long sections by size.
section_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("##", "section"), ("###", "subsection")]
)
size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=int(os.getenv("CHUNK_SIZE", "1500")),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
)

# The bibliography only adds noise (venue names extracted as benchmarks...)
# and extraction cost, so it is not indexed.
SKIPPED_SECTIONS = {"references", "bibliography"}

ingestion_queue = asyncio.Queue()
ingestion_status = {}


def split_into_sections(paper_text: str) -> list[Document]:
    """The sections of the paper, without the bibliography."""
    sections = []

    for section in section_splitter.split_text(paper_text):
        headings = [h.strip("* ") for h in section.metadata.values()]
        if any(h.lower() in SKIPPED_SECTIONS for h in headings):
            continue
        section.metadata = {"section": " > ".join(headings)}
        sections.append(section)

    return sections


def split_into_chunks(sections: list[Document]) -> list[Document]:
    """Split long sections, keeping the section each chunk comes from."""
    return size_splitter.split_documents(sections)


def ingest_document(neo4j_graph, file_metadata: dict) -> None:
    """Convert a PDF to Markdown and index it in the graph.

    Synchronous on purpose: the worker runs it in a separate thread so that
    the slow LLM calls do not block the API.
    """
    markdown = pymupdf4llm.to_markdown(str(file_metadata["file_path"]))
    file_metadata["md_path"].write_text(markdown, encoding="utf-8")

    sections = split_into_sections(markdown)
    paper_text = "\n\n".join(
        f"## {s.metadata['section']}\n\n{s.page_content}" for s in sections
    )
    build_upload_graph(
        neo4j_graph, paper_text, split_into_chunks(sections), file_metadata
    )


async def ingestion_queue_worker(app):
    while True:
        file_metadata = await ingestion_queue.get()
        neo4j_graph = app.state.neo4j_graph
        file_name = file_metadata["file_name"]

        try:
            ingestion_status[file_name] = "processing"
            await asyncio.to_thread(ingest_document, neo4j_graph, file_metadata)
            ingestion_status[file_name] = "completed"

        except Exception as e:  # noqa: BLE001
            print(f"Ingestion of {file_name} failed: {e!r}")
            ingestion_status[file_name] = "failed"
            # Leave no half-indexed paper behind so it can be uploaded again
            delete_paper(neo4j_graph, file_metadata["paper_id"])
            file_metadata["file_path"].unlink(missing_ok=True)
            file_metadata["md_path"].unlink(missing_ok=True)
        finally:
            ingestion_queue.task_done()
