from langchain_core.tools import tool
from langchain_neo4j import GraphCypherQAChain

from src.agent.llm import agent_llm, embedder

graph = None


@tool
def search_knowledge_graph(query: str) -> str:
    """Useful for structural queries, metadata, entity counts, or direct relationships

    between graph entities (such as authors, papers, datasets, metrics, or institutions)
    by automatically generating and executing Cypher queries.

    Args:
        query: The natural language question regarding the graph structure.
    """
    if graph is None:
        return "Error: Graph database client is not initialized."

    try:
        # Initialize LangChain's GraphCypherQAChain for Text-to-Cypher translation
        chain = GraphCypherQAChain.from_llm(
            llm=agent_llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True,  # Required for executing custom generated Cypher
        )
        response = chain.invoke({"query": query})
        return response.get("result", "No results found in the graph.")
    except Exception as e:  # noqa:BLE001
        return f"Error executing knowledge graph search: {e!s}"


@tool
def search_chunks_semantically(query: str) -> str:
    """Useful for searching conceptual information, methodologies, or deep scientific

    findings across text chunks (:Document nodes) of papers using vector similarity
    search against the 'documentVectorIndex'.

    Args:
        query: The conceptual or technical question to search in text embeddings.
    """
    if graph is None:
        return "Error: Database client is not initialized."

    try:
        # 1. Generate embedding for the search query
        query_vector = embedder.embed_query(query)

        # 2. Query the Neo4j vector index created in init_neo4j_schema
        vector_query = """
        CALL db.index.vector.queryNodes('documentVectorIndex', 5, $embedding)
        YIELD node, score
        RETURN node.text AS text, score
        """
        results = graph.query(vector_query, params={"embedding": query_vector})

        if not results:
            return "No relevant text chunks found via semantic search."

        formatted_chunks = []
        for r in results:
            formatted_chunks.append(f"Relevance Score: {r['score']:.4f}\n{r['text']}")

        return "\n\n---\n\n".join(formatted_chunks)
    except Exception as e:  # noqa:BLE001
        return f"Error during semantic chunk search: {e!s}"


@tool
def get_paper_summary_and_chunks(query: str) -> str:
    """Useful when you have a specific paper title or paper ID and need to retrieve

    its associated text chunks (:Document connected via :PART_OF) from the graph
    to perform a deep analysis, extract conclusions, or generate a comprehensive summary.

    Args:
        query: The title, part of the title, or unique paper ID.
    """
    if graph is None:
        return "Error: Database client is not initialized."

    try:
        # Match paper and its connected chunks using your exact schema relationship: (:Document)-[:PART_OF]->(:Paper)
        cypher_query = """
        MATCH (p:Paper)
        WHERE toLower(p.title) CONTAINS toLower($query) OR p.id = $query
        MATCH (d:Document)-[:PART_OF]->(p)
        RETURN p.title AS title, p.year AS year, p.venue AS venue, collect(d.text) AS chunks
        LIMIT 1
        """
        result = graph.query(cypher_query, params={"query": query})

        if not result:
            return f"No paper found matching '{query}'."

        paper_data = result[0]
        paper_title = paper_data.get("title", "Unknown Title")
        year = paper_data.get("year", "N/A")
        venue = paper_data.get("venue", "N/A")
        chunks = paper_data.get("chunks", [])

        if not chunks:
            return (
                f"Found paper '{paper_title}', but no associated text chunks were"
                " found."
            )

        # Combine key chunks (limiting to avoid token limits)
        combined_text = "\n".join(chunks[:8])

        prompt = (
            f"Provide a clear, structured scientific summary of the following"
            f" text chunks belonging to the paper '{paper_title}' (Published in"
            f" {year}, Venue: {venue}):\n\n{combined_text}"
        )
        summary_response = agent_llm.invoke(prompt)

        return (
            f"### Paper Summary: {paper_title} ({year})\nVenue: {venue}\n\n"
            f"{summary_response.content}"
        )
    except Exception as e:  # noqa:BLE001
        return f"Error retrieving paper summary and chunks: {e!s}"


@tool
def find_entity_connections(query: str) -> str:
    """Useful for finding indirect paths or relationships between entities (such as authors,

    papers, concepts, datasets, or methods) within the knowledge graph network.

    Args:
        query: Description of the entities or relationship path to find (e.g., 'Connection

          between author X and dataset Y').
    """
    if graph is None:
        return "Error: Database client is not initialized."

    try:
        # Leverages GraphCypherQAChain to translate path/connection requests into accurate Cypher path queries
        chain = GraphCypherQAChain.from_llm(
            llm=agent_llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True,
        )
        prompt = (
            f"Find the relationship path or connection in the graph regarding:"
            f" {query}. Return the connecting nodes and relationships."
        )
        response = chain.invoke({"query": prompt})
        return response.get("result", "No entity connections found matching the query.")
    except Exception as e:  # noqa:BLE001
        return f"Error finding entity connections: {e!s}"


ALL_TOOLS = [
    search_knowledge_graph,
    search_chunks_semantically,
    get_paper_summary_and_chunks,
    find_entity_connections,
]
