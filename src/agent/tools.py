from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_neo4j import GraphCypherQAChain

from src.agent.llm import agent_llm, embedder

graph = None


def initialize_graph(neo4j_graph):
    global graph
    graph = neo4j_graph


CYPHER_PROMPT_TEMPLATE = """You are an expert Neo4j Cypher translator.
Given an input question, create a syntactically correct Cypher query to run.

### CRITICAL RULES FOR CYPHER GENERATION:
1. NEVER use `RETURN p` or return raw nodes alone. 
2. ALWAYS project explicit fields with aliases, including connecting entities (like Author names) so the context is preserved in the output.
   - ❌ FORBIDDEN: `MATCH (a:Author)-[:WROTE]->(p:Paper) RETURN p`
   - ✅ MANDATORY: `MATCH (a:Author)-[:WROTE]->(p:Paper) RETURN a.name AS author_name, p.title AS title, p.id AS paper_id, p.year AS year`

Schema:
{schema}

Question: {question}
CypherQuery:"""

CUSTOM_CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_PROMPT_TEMPLATE
)


@tool
def search_knowledge_graph(query: str) -> str:
    """Retrieve structured facts from the Neo4j knowledge graph.

    Use for paper metadata, author identities, authorship relationships,
    entity relationships, and graph-level questions.

    Do not use this tool as the only source for details cited inside
    paper text, such as related-work references or specific claims.
    """
    if graph is None:
        return "Error: Graph database client is not initialized."

    try:
        # 2. Inicializamos la cadena inyectando el prompt personalizado
        chain = GraphCypherQAChain.from_llm(
            llm=agent_llm,
            graph=graph,
            verbose=True,
            validate_cypher=True,
            allow_dangerous_requests=True,
            cypher_prompt=CUSTOM_CYPHER_PROMPT,  # 👈 AQUÍ FORZAMOS EL RETURN CORRECTO
            return_intermediate_steps=True,
        )
        response = chain.invoke({"query": query})

        # 3. Extraemos el resultado crudo de Neo4j para dárselo limpio al agente
        intermediate = response.get("intermediate_steps", [])
        if len(intermediate) >= 2 and "context" in intermediate[1]:
            context_data = intermediate[1]["context"]
            if context_data:
                return str(context_data)

        final_result = response.get("result")
        if final_result and "I don't know" not in str(final_result):
            return str(final_result)

        return "No records found in the knowledge graph matching this query."

    except Exception as e:  # noqa: BLE001
        return f"Error executing knowledge graph search: {e!s}"


@tool
def search_chunks_semantically(query: str) -> str:
    """Search the original text chunks of research papers using vector similarity.

    Use this tool to verify specific scientific claims, technical details,
    experimental results, named entities, cited authors, and references
    such as 'Wu et al.'.

    Use it whenever a question asks for evidence from a paper's text.
    Do not conclude that a fact is absent based only on a paper summary.
    """

    if graph is None:
        return "Error: Database client is not initialized."

    try:
        query_vector = embedder.embed_query(query)

        vector_query = """
CALL db.index.vector.queryNodes('documentVectorIndex', 5, $embedding)
YIELD node, score
RETURN node.id AS chunk_id, node.text AS text, score
ORDER BY score DESC
"""

        results = graph.query(
            vector_query,
            params={"embedding": query_vector},
        )

        if not results:
            return "No relevant text chunks found via semantic search."

        formatted_chunks = []

        for r in results:
            formatted_chunks.append(
                f"Chunk ID: {r['chunk_id']}\n"
                f"Relevance Score: {r['score']:.4f}\n"
                f"{r['text']}"
            )

        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:  # noqa: BLE001
        return f"Error during semantic chunk search: {e!s}"


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
    # find_entity_connections,
]
