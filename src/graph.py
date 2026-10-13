import os

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

load_dotenv()

ALLOWED_NODES = [
    "Person",
    "Organization",
    "AI Model",
    "Dataset",
    "Paper Reference",
    "Metric",
    "Hardware",
    "Loss Function",
]

ALLOWED_RELATIONSHIPS = [
    "USED_IN",
    "AUTHORED",
    "RESEARCHES",
    "AFFILIATED_WITH",
    "APPEARS_IN",
    "ACHIEVES",  # (AI Model) -[:ACHIEVES {value: "85.4%"}]-> (Metric)
    "TRAINED_ON",  # (AI Model) -[:TRAINED_ON]-> (Dataset)
    "USES_LOSS",  # (AI Model) -[:USES_LOSS]-> (Loss Function)
    "BENCHMARKED_ON",  # (AI Model) -[:BENCHMARKED_ON]-> (Hardware)
]

NODE_PROPERTIES = [
    "name",
    "source",
    "authors",
    "venue",
    "title",
    "parameters",
    "framework",
    "modality",
    "value",
]

ADDITIONAL_INSTRUCTIONS = """
Extract only explicitly stated entities (no assumptions/hallucinations).
1. Title Case names. Use hyphens for technical terms ('RAF-DB').
2. Person: Full names. BAN 'et al.' or group suffixes.
3. Organization: Fully expanded ('Google').
4. AI Model: ML/CV methods ('Transformers', 'YOLO'). Properties: 'parameters', 'framework'.
5. Dataset: Specific benchmarks only ('ImageNet'). BAN generics. Property: 'modality'.
6. Metric: Performance measures ('mAP', 'Accuracy'). Property: 'value'.
7. Hardware: Compute units ('NVIDIA A100', 'RTX 3090').
8. Loss Function: Optimization objectives ('Cross-Entropy').
9. Paper Reference properties required: authors, title, venue. Do not create entities for number references, example: [23].
"""


def build_upload_graph(chunks):
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    neo_db = os.getenv("NEO4J_DATABASE")

    graph = Neo4jGraph(url=uri, username=username, password=password, database=neo_db)

    llm = ChatBedrockConverse(
        model_id="us.amazon.nova-2-lite-v1:0",
        region_name="us-east-1",
        temperature=0.0,
    )

    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS,
        additional_instructions=ADDITIONAL_INSTRUCTIONS,
        node_properties=NODE_PROPERTIES,
    )

    graph_documents = []

    for i, chunk in enumerate(chunks, 1):
        try:
            g_doc = llm_transformer.convert_to_graph_documents([chunk])
            if not g_doc:
                continue

            entities = g_doc[0].nodes
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

            for rel in g_doc[0].relationships:
                if rel.source.id in name_mapping:
                    rel.source.id = name_mapping[rel.source.id]
                if rel.target.id in name_mapping:
                    rel.target.id = name_mapping[rel.target.id]

            graph_documents.extend(g_doc)

        except Exception as e:  # noqa: BLE001
            print(f"\nError in chunk {i}: {e}. Skipping chunk...")
            # raise

    if graph_documents:
        print(f"Uploading {len(graph_documents)} graph blocks to Neo4j...")
        graph.add_graph_documents(graph_documents, include_source=True)
        print("Upload complete.")
    else:
        print("\nNo entities or relations could be extracted.")
        raise Exception  # noqa: TRY002


def rollback_neo4j(file_path: str):
    try:
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        neo_db = os.getenv("NEO4J_DATABASE")

        graph = Neo4jGraph(
            url=uri, username=username, password=password, database=neo_db
        )

        query = """
        MATCH (d:Document)
        WHERE d.source = $source_path OR d.id = $source_path
        OPTIONAL MATCH (d)-[:MENTIONS]->(e)
        DETACH DELETE d
        WITH e
        WHERE e IS NOT NULL AND size((e)--()) = 0
        DELETE e
        """
        graph.query(query, params={"source_path": str(file_path)})
        print(f"Rollback Neo4j: Graph cleaned for {file_path}")
    except Exception as e:  # noqa: BLE001
        print(f"Error while doing rollback in Neo4j: {e}")
