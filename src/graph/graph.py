"""Turn a paper into the Neo4j knowledge graph (schema in ontology.py).

Extraction happens in two steps:
1. The whole paper is read once to get its metadata and main entities, each
   with a canonical name and its aliases ("FER" -> "facial expression
   recognition", "our model" -> the proposed method). These become the
   relationships of the Paper node.
2. Each chunk is linked to the entities it mentions, using those canonical
   names so that every spelling of an entity ends up in the same node.
"""

import hashlib
import os
import re

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_community.graphs.graph_document import (
    GraphDocument,
    Node,
    Relationship,
)
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph

from src.graph.extraction_schemas import ChunkMentions, PaperExtraction
from src.graph.ontology import PAPER_RELATIONSHIPS
from src.graph.validation import appears_in, is_specific_name, validate_evidence

load_dotenv()


PAPER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You read research papers and extract their metadata and main
            entities for a knowledge graph.

            RULES
            - Use only what the paper states. Never add entities from your own
              knowledge.
            - List each entity once, under its full name, and put abbreviations
              and other spellings in its aliases. Example: name "facial
              expression recognition", aliases ["FER"].
            - Add the phrases the paper uses for its own contribution ("our
              method", "our model"...) as aliases of the proposed method.
            - Only specific, named things. No generic phrases ("neural
              network", "training data", "input image") and no math symbols or
              variables.
            - Leave a metadata field empty if the paper does not state it.
            """,
        ),
        ("human", "Paper:\n\n{paper_text}"),
    ]
)

CHUNK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Find the entities mentioned in a chunk of the paper "{title}".

            Known entities of this paper:
            {known_entities}

            RULES
            - If a mention refers to a known entity, return the known name and
              type, also when the chunk uses an alias or says "our method".
            - Other entities can be added only if they are specifically named
              methods, datasets, metrics or tasks, such as a cited method.
            - Never return generic phrases ("neural network", "encoder",
              "input image") or math symbols and variables ("d", "θ", "x^b").
            - evidence: the sentence where the entity appears, copied verbatim.
            - Return an empty list when nothing qualifies.
            """,
        ),
        ("human", "Chunk:\n\n{chunk_text}"),
    ]
)

# Parallel chunk extraction calls; higher values get throttled by Bedrock
MAX_CONCURRENCY = 4

embedder = BedrockEmbeddings(
    region_name="us-east-1", model_id="amazon.titan-embed-text-v2:0", normalize=True
)

# Reading a whole paper is a single, harder call, so it gets a stronger model;
# the many per-chunk calls use the cheaper one.
paper_llm = ChatBedrockConverse(
    model_id=os.getenv("PAPER_LLM"),
    region_name="us-east-1",
    temperature=0.0,
    max_tokens=16000,
)
graph_llm = ChatBedrockConverse(
    model_id=os.getenv("GRAPH_LLM"),
    region_name="us-east-1",
    temperature=0.0,
)

paper_extractor = PAPER_PROMPT | paper_llm.with_structured_output(
    PaperExtraction, method="function_calling"
)
mention_extractor = CHUNK_PROMPT | graph_llm.with_structured_output(
    ChunkMentions, method="function_calling"
)

METADATA_FIELDS = ["paper_title", "authors", "year", "venue", "doi", "repository_link"]


def entity_key(name: str) -> str:
    """Normalize a name so that "Aff-Wild2" and "AffWild2" are one entity."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def paper_metadata(
    paper: PaperExtraction, file_metadata: dict, paper_text: str
) -> dict:
    """Extracted metadata, overridden by the values the user filled in."""
    extracted = {
        field: value
        for field, value in paper.model_dump(include=set(METADATA_FIELDS)).items()
        if value not in (None, "", "null")
    }
    # The model tends to guess these two when the paper does not state them
    for field in ("year", "doi"):
        if field in extracted and not appears_in(str(extracted[field]), paper_text):
            del extracted[field]

    from_user = {f: file_metadata[f] for f in METADATA_FIELDS if f in file_metadata}
    return extracted | from_user


def clean_name(name: str) -> str:
    """Remove citation markers and HTML: "Multi-PIE [7]" -> "Multi-PIE"."""
    return re.sub(r"\[[^\]]*\]|<[^>]+>", "", name).strip()


def paper_entities(paper: PaperExtraction, paper_text: str) -> list[tuple[str, Node]]:
    """(relationship from the Paper, entity node) for every main entity whose
    name or one of its aliases really appears in the paper."""
    entities = []
    label_of = {}  # an entity keeps the first label it was given

    for field, (relationship, label) in PAPER_RELATIONSHIPS.items():
        for entity in getattr(paper, field):
            name = clean_name(entity.name)
            key = entity_key(name)
            aliases = list({clean_name(a) for a in entity.aliases} - {"", name})
            if not is_specific_name(name) or label_of.get(key, label) != label:
                continue
            if not any(appears_in(n, paper_text) for n in [name, *aliases]):
                continue

            label_of[key] = label
            node = Node(
                id=key, type=label, properties={"name": name, "aliases": aliases}
            )
            entities.append((relationship, node))

    return entities


def describe_entities(nodes: list[Node]) -> str:
    """The known entities as a list for the chunk prompt."""
    lines = []
    for node in nodes:
        aliases = node.properties["aliases"]
        also = f" (also: {', '.join(aliases)})" if aliases else ""
        lines.append(f"- {node.type}: {node.properties['name']}{also}")
    return "\n".join(lines)


def to_chunk_graph_document(
    extraction: ChunkMentions, chunk: Document, known: dict[str, Node]
) -> GraphDocument:
    """Link the chunk to the entities it mentions.

    Mentions of a known entity (by name or alias) go to its node. New entities
    are kept only if they are specific, and never as Concepts: those come only
    from the paper-level extraction.
    """
    nodes = {}

    for mention in extraction.mentions:
        if not validate_evidence(mention.evidence, chunk.page_content):
            continue

        node = known.get(entity_key(mention.name))
        if node is None:
            if mention.entity_type == "Concept" or not is_specific_name(mention.name):
                continue
            node = Node(
                id=entity_key(mention.name),
                type=mention.entity_type,
                properties={"name": mention.name.strip()},
            )
        nodes[(node.id, node.type)] = node

    return GraphDocument(nodes=list(nodes.values()), relationships=[], source=chunk)


def to_paper_graph_document(
    paper_id: str, properties: dict, authors: list[str], entities: list
) -> GraphDocument:
    properties = {k: v for k, v in properties.items() if v is not None}
    paper = Node(id=paper_id, type="Paper", properties=properties)
    authors = [Node(id=a, type="Author", properties={"name": a}) for a in authors]

    relationships = [
        Relationship(source=a, target=paper, type="WROTE") for a in authors
    ]
    relationships += [
        Relationship(source=paper, target=node, type=relationship)
        for relationship, node in entities
    ]

    return GraphDocument(
        nodes=[paper, *authors, *(node for _, node in entities)],
        relationships=relationships,
        # Required by GraphDocument, but only stored with include_source=True
        source=Document(page_content=""),
    )


LINK_CHUNKS_QUERY = """
UNWIND $chunks AS chunk
MATCH (d:Document {id: chunk.id})
MATCH (p:Paper {id: $paper_id})
SET d.embedding = chunk.embedding
MERGE (d)-[:PART_OF]->(p)
"""


def build_upload_graph(
    graph: Neo4jGraph,
    paper_text: str,
    chunks: list[Document],
    file_metadata: dict,
) -> None:
    paper_id = file_metadata["paper_id"]

    for chunk in chunks:
        chunk.metadata["id"] = hashlib.md5(
            f"{paper_id}{chunk.page_content}".encode()
        ).hexdigest()

    # 1. The LLM work
    with get_usage_metadata_callback() as usage:
        paper = paper_extractor.invoke({"paper_text": paper_text})
        if paper is None:
            raise RuntimeError("The model did not return the paper extraction")

        metadata = paper_metadata(paper, file_metadata, paper_text)
        title = metadata.get("paper_title") or file_metadata["file_name"]
        entities = paper_entities(paper, paper_text)
        print(f"Paper entities: {[n.properties['name'] for _, n in entities]}")

        # One node per entity, reachable from its name and from every alias
        unique_nodes = list({(n.id, n.type): n for _, n in entities}.values())
        known = {}
        for node in unique_nodes:
            for name in [node.properties["name"], *node.properties["aliases"]]:
                known.setdefault(entity_key(name), node)

        extractions = mention_extractor.batch(
            [
                {
                    "title": title,
                    "known_entities": describe_entities(unique_nodes),
                    "chunk_text": chunk.page_content,
                }
                for chunk in chunks
            ],
            config={"max_concurrency": MAX_CONCURRENCY},
            return_exceptions=True,
        )

    # The embedded text starts with the paper and section so that a chunk like
    # "we train on it for 10 epochs" can still be matched to its topic.
    embeddings = embedder.embed_documents(
        [f"{title} > {c.metadata['section']}\n\n{c.page_content}" for c in chunks]
    )

    # A chunk fails with an exception, or with None when the model answers in
    # plain text instead of calling the extraction function.
    chunk_documents = []
    for chunk, extraction in zip(chunks, extractions, strict=True):
        if not isinstance(extraction, ChunkMentions):
            print(f"Extraction failed for chunk {chunk.metadata['id']}: {extraction!r}")
            extraction = ChunkMentions()
        chunk_documents.append(to_chunk_graph_document(extraction, chunk, known))

    tokens = {"input_tokens": 0, "output_tokens": 0}
    for model_usage in usage.usage_metadata.values():
        tokens["input_tokens"] += model_usage["input_tokens"]
        tokens["output_tokens"] += model_usage["output_tokens"]
    mentions = sum(len(d.nodes) for d in chunk_documents)
    print(f"{len(chunks)} chunks, {mentions} mentions, tokens: {tokens}")

    # 2. Write everything to Neo4j. The paper goes first: nodes only get their
    # properties when they are created, and the paper-level entities carry
    # the canonical names and aliases.
    paper_properties = {
        "title": title,
        "doi": metadata.get("doi"),
        "venue": metadata.get("venue"),
        "year": metadata.get("year"),
        "repository_link": metadata.get("repository_link"),
        "file_name": file_metadata["file_name"],
        "ingestion_input_tokens": tokens["input_tokens"],
        "ingestion_output_tokens": tokens["output_tokens"],
    }
    graph.add_graph_documents(
        [
            to_paper_graph_document(
                paper_id, paper_properties, metadata.get("authors", []), entities
            )
        ]
    )
    graph.add_graph_documents(chunk_documents, include_source=True)
    graph.query(
        LINK_CHUNKS_QUERY,
        params={
            "paper_id": paper_id,
            "chunks": [
                {"id": chunk.metadata["id"], "embedding": embedding}
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ],
        },
    )

    # The Cypher tool generates queries from this cached schema
    graph.refresh_schema()


DELETE_PAPER_QUERY = """
MATCH (p:Paper {id: $paper_id})
OPTIONAL MATCH (chunk:Document)-[:PART_OF]->(p)
DETACH DELETE chunk, p
WITH count(*) AS _
// Entities and authors that are no longer connected to any paper or chunk
MATCH (n)
WHERE (n:Author OR n:Method OR n:Dataset OR n:Metric OR n:Task OR n:Concept)
  AND NOT (n)--(:Paper) AND NOT (n)--(:Document)
DETACH DELETE n
"""


def delete_paper(graph: Neo4jGraph, paper_id: str) -> None:
    """Remove a paper, its chunks, and the entities and authors only it had."""
    graph.query(DELETE_PAPER_QUERY, params={"paper_id": paper_id})
    graph.refresh_schema()


def init_neo4j_schema(graph, embedding_dimension: int = 1024):
    """
    Create the Neo4j constraints and vector indexes on application startup.
    Safe to run multiple times.
    """
    setup_queries = [
        # Uniqueness constraints also create the indexes that MERGE uses
        "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (n:Paper) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (n:Author) REQUIRE n.name IS UNIQUE",
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
