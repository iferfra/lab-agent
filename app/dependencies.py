from fastapi import Request
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_text_splitter(request: Request) -> RecursiveCharacterTextSplitter:
    return request.state.ingestion_text_splitter


def get_chunk_embedder(request: Request) -> OllamaEmbeddings:
    return request.state.chunk_embedder


def get_vector_store(request: Request) -> Chroma:
    return request.state.vector_store
