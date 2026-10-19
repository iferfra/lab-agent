import hashlib

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

load_dotenv()


ALLOWED_NODES = [
    "Concept",
    "Method",
    "Dataset",
    "Metric",
    "Benchmark",
    "Task",
]

ALLOWED_RELATIONSHIPS = [
    "USES",
    "SOLVES",
    "TRAINED_ON",
    "EVALUATED_ON",
    "OUTPERFORMS",
    "HAS_METRIC",
]


ADDITIONAL_INSTRUCTIONS = """
Extract scientific knowledge graph entities and relationships from the text chunk.

ALLOWED NODES:
- Concept: Theories, principles, mechanisms.
- Method: Algorithms, models, architectures.
- Dataset: Data collections. Must be specific datasets. Ex: 'Coco', 'Multi-PIE', 'RAF-DB'.
- Metric: Measurement units. Ex: 'Accuracy', 'Recall', 'Fairness GAP'.
- Benchmark: Evaluation suites.
- Task: Problems addressed.

ALLOWED RELATIONSHIPS: USES, SOLVES, TRAINED_ON, EVALUATED_ON, OUTPERFORMS, HAS_METRIC.

RULES:
1. Strict schema compliance only.
2. NO authors, paper titles, institutions, URLs, or generic text ("paper", "table", "dataset").
3. Use canonical, concise names. If no technical entities are found, return empty.
"""

embedder = BedrockEmbeddings(
    region_name="us-east-1", model_id="amazon.titan-embed-text-v2:0", normalize=True
)

graph_llm = ChatBedrockConverse(
    model_id="us.meta.llama3-3-70b-instruct-v1:0",
    region_name="us-east-1",
    temperature=0.0,
)

graph_transformer = LLMGraphTransformer(
    llm=graph_llm,
    strict_mode=True,
    allowed_nodes=ALLOWED_NODES,
    allowed_relationships=ALLOWED_RELATIONSHIPS,
    additional_instructions=ADDITIONAL_INSTRUCTIONS,
    node_properties=True,
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

            g_doc = graph_transformer.convert_to_graph_documents([doc])[0]

            entities = g_doc.nodes
            if not entities:
                continue

            entity_payload = [{"name": e.id, "type": e.type} for e in entities]

            query = """
            UNWIND $entities AS ent
            MATCH (n)
            WHERE any(label IN labels(n) WHERE label = ent.type)
            WITH ent, n, apoc.text.levenshteinSimilarity(n.id, ent.name) AS similarity
            WHERE similarity > 0.85
            RETURN ent.name AS original, n.id AS match
            ORDER BY similarity DESC
            """

            results = graph.query(query, params={"entities": entity_payload})

            name_mapping = {row["original"]: row["match"] for row in results}

            for entity in entities:
                if entity.id in name_mapping:
                    entity.id = name_mapping[entity.id]

            for rel in g_doc.relationships:
                if rel.source.id in name_mapping:
                    rel.source.id = name_mapping[rel.source.id]
                if rel.target.id in name_mapping:
                    rel.target.id = name_mapping[rel.target.id]

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
