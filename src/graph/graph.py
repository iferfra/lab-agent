import hashlib
import os

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_community.graphs.graph_document import (
    GraphDocument,
    Node,
    Relationship,
)
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph

from src.graph.extraction_schemas import ChunkExtraction

load_dotenv()


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Extract scientific entities and relationships from the text.

            ENTITY RULES
            - Method: A specifically identifiable model, architecture,
              algorithm, or named technical approach.
            - Dataset: A specifically identifiable data collection.
            - Metric: A named quantitative evaluation measure.
            - Benchmark: An identifiable evaluation suite or protocol.
            - Task: A specific scientific problem.
            - Concept: A meaningful scientific concept, mechanism, or theory.

            GENERAL RULES
            - Extract only information explicitly supported by the text.
            - Preserve the entity mention as written.
            - Do not extract generic phrases such as "our method",
              "training data", "performance", or "proposed approach".
            - Do not invent entities from domain knowledge.
            - Do not infer relationships from co-occurrence alone.
            - Every entity and relationship must have textual evidence.
            - Evidence must be copied verbatim from the input.
            - Return empty lists when no valid entities or relationships
              can be extracted.
            """,
        ),
        (
            "human",
            "Extract knowledge from this chunk:\n\n{chunk_text}",
        ),
    ]
)

embedder = BedrockEmbeddings(
    region_name="us-east-1", model_id="amazon.titan-embed-text-v2:0", normalize=True
)

# graph_llm = ChatBedrockConverse(
#     model_id="us.anthropic.claude-sonnet-4-6",
#     region_name="us-east-1",
#     temperature=0.0,
# )

# price_1m_token_input = 3
# price_1m_token_output = 15

graph_llm = ChatBedrockConverse(
    model_id=os.getenv("GRAPH_LLM"),
    region_name="us-east-1",
    temperature=0.0,
)

price_1m_token_input = 0.33
price_1m_token_output = 2.75


structured_llm = graph_llm.with_structured_output(
    ChunkExtraction,
    method="function_calling",
    include_raw=True,
)


extractor = EXTRACTION_PROMPT | structured_llm

total_price = 0


def extract_chunk(chunk_text: str) -> ChunkExtraction:
    global total_price  # Permite modificar la variable externa

    response_dict = extractor.invoke({"chunk_text": chunk_text})

    raw_message = response_dict.get("raw")
    parsed_result = response_dict.get("parsed")

    if isinstance(raw_message, AIMessage) and hasattr(raw_message, "usage_metadata"):
        usage = raw_message.usage_metadata
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # print("Input tokens:", input_tokens)
        # print("Output tokens:", output_tokens)

        price_chunk = (input_tokens * (price_1m_token_input / 1e6)) + (
            output_tokens * (price_1m_token_output / 1e6)
        )
        # print(f"Costo del chunk: ${price_chunk:.6f}")
        total_price += price_chunk
    else:
        print("No se encontraron metadatos de uso.")

    if not isinstance(parsed_result, ChunkExtraction):
        raise TypeError(f"Expected ChunkExtraction, got {type(parsed_result).__name__}")

    return parsed_result


def to_graph_document(
    extraction: ChunkExtraction,
    source: Document,
) -> GraphDocument:
    nodes_by_mention = {}

    for entity in extraction.entities:
        key = (entity.mention.strip(), entity.entity_type)

        if key not in nodes_by_mention:
            nodes_by_mention[key] = Node(
                id=entity.mention.strip(),
                type=entity.entity_type,
                properties={
                    "name": entity.mention.strip(),
                    "evidence": entity.evidence,
                },
            )

    relationships = []

    for rel in extraction.relationships:
        source_candidates = [
            node
            for (mention, _), node in nodes_by_mention.items()
            if mention == rel.source_mention.strip()
        ]
        target_candidates = [
            node
            for (mention, _), node in nodes_by_mention.items()
            if mention == rel.target_mention.strip()
        ]

        if len(source_candidates) != 1 or len(target_candidates) != 1:
            continue

        relationships.append(
            Relationship(
                source=source_candidates[0],
                target=target_candidates[0],
                type=rel.relationship_type,
                properties={
                    "evidence": rel.evidence,
                },
            )
        )

    return GraphDocument(
        nodes=list(nodes_by_mention.values()),
        relationships=relationships,
        source=source,
    )


def build_upload_graph(graph: Neo4jGraph, chunks: list[Document], file_metadata: dict):

    paper_id = hashlib.md5(
        file_metadata.get("file_name", "unknown").encode("utf-8")
    ).hexdigest()
    create_paper_query = """
    MERGE (p:Paper {id: $paper_id})
    SET p.title = $title,
        p.doi = $doi,
        p.venue = $venue,
        p.year = $year,
        p.file_name = $file_name,
        p.repository_link = $repository_link
    """
    graph.query(
        create_paper_query,
        params={
            "paper_id": paper_id,
            "title": file_metadata.get("paper_title"),
            "doi": file_metadata.get("doi"),
            "venue": file_metadata.get("venue"),
            "year": file_metadata.get("year"),
            "file_name": file_metadata.get("file_name"),
            "repository_link": file_metadata.get("repository_link"),
        },
    )
    authors = file_metadata.get("authors")
    if authors:
        authors_query = """
        MATCH (p:Paper {id: $paper_id})
        UNWIND $authors AS author_name
        MERGE (a:Author {name: author_name})
        MERGE (a)-[:WROTE]->(p)
        """
        graph.query(authors_query, params={"paper_id": paper_id, "authors": authors})

    graph_documents = []
    chunk_embeddings_list = []
    skipped_count = 0

    for i, chunk in enumerate(chunks, 1):
        try:
            unique_string = chunk.page_content
            chunk_id = hashlib.md5(unique_string.encode("utf-8")).hexdigest()

            chunk.metadata["id"] = chunk_id

            check_query = (
                "MATCH (d:Document {id: $chunk_id}) RETURN count(d) > 0 AS exists"
            )
            res = graph.query(check_query, params={"chunk_id": chunk_id})

            if res and res[0]["exists"]:
                skipped_count += 1
                continue

            doc = Document(page_content=chunk.page_content, metadata={"id": chunk_id})

            doc_embedding = embedder.embed_documents([chunk.page_content])[0]
            chunk_embeddings_list.append({"id": chunk_id, "embedding": doc_embedding})

            # print(f"Chunk text:\n{chunk.page_content[:2000]}")
            extraction = extract_chunk(chunk.page_content)

            if extraction is None:
                print("Extractor returned None. Skipping chunk.")
                continue

            # print(f"Extracted entities: {extraction.entities}")
            # print(f"Extracted relationships: {extraction.relationships}")

            g_doc = to_graph_document(
                extraction=extraction,
                source=doc,
            )
            graph_documents.append(g_doc)

        except Exception as e:  # noqa: BLE001
            print(f"\nError in chunk {i}: {e}. Skipping chunk...")

    if graph_documents:
        print(f"Uploading {len(graph_documents)} graph blocks to Neo4j...")

        graph.add_graph_documents(graph_documents, include_source=True)

        print("Attaching vector embeddings and linking chunks to the Paper...")

        link_chunks_query = """
        UNWIND $updates AS row
        MATCH (d:Document {id: row.id})
        MATCH (p:Paper {id: $paper_id})
        SET d.embedding = row.embedding
        MERGE (d)-[:PART_OF]->(p)
        """

        graph.query(
            link_chunks_query,
            params={"updates": chunk_embeddings_list, "paper_id": paper_id},
        )
        print("Graph ingestion and metadata linking complete.")
        print(f"\nDocument total price: {total_price!s} $")
    else:
        print("\nNo entities or relations could be extracted.")
        raise Exception  # noqa: TRY002


def init_neo4j_schema(graph, embedding_dimension: int = 1024):
    """
    Inicializa automáticamente restricciones e índices vectoriales en Neo4j
    al arrancar la aplicación. Es seguro ejecutarlo múltiples veces.
    """
    setup_queries = [
        """
        CREATE CONSTRAINT unique_node_id IF NOT EXISTS 
        FOR (n:Entity) REQUIRE n.id IS UNIQUE
        """,
        f"""
        CREATE VECTOR INDEX documentVectorIndex IF NOT EXISTS
        FOR (d:Document) 
        ON (d.embedding)
        OPTIONS {{ indexConfig: {{
          `vector.dimensions`: {embedding_dimension},
          `vector.similarity_function`: 'cosine'
        }}}}
        """,
    ]

    print("Checking and initializing Neo4j schema...")
    for query in setup_queries:
        try:
            graph.query(query.strip())
        except Exception as e:  # noqa:BLE001
            print(f"Schema notice (safe to ignore if already present): {e}")
    print("Neo4j schema is ready.")
