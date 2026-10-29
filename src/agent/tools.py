from typing import Literal

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
3. NEVER return `Document` text or embeddings. This tool answers questions about
   metadata and relationships; paper text is retrieved by a different tool.
4. Entities hang from the Paper: PROPOSES, BUILDS_ON and COMPARES_WITH (Method),
   ADDRESSES (Task), USES_DATASET (Dataset), MEASURES_WITH (Metric) and
   STUDIES (Concept). Entities have a `name` and a list of `aliases`.
5. Match titles and names with `toLower(...) CONTAINS toLower(...)`, never with
   exact equality, since the question may use a partial title or an abbreviation.

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
        # The custom prompt forces Cypher queries to return explicit fields
        chain = GraphCypherQAChain.from_llm(
            llm=agent_llm,
            graph=graph,
            verbose=True,
            validate_cypher=True,
            allow_dangerous_requests=True,
            cypher_prompt=CUSTOM_CYPHER_PROMPT,
            return_intermediate_steps=True,
        )
        response = chain.invoke({"query": query})

        # Return the raw Neo4j rows so the agent sees the actual data
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


CHUNKS_PER_SEARCH = 10

# The vector index cannot filter by paper, so when a paper is given we fetch
# more candidates and keep the best ones that belong to it (via PART_OF).
CANDIDATES_WHEN_FILTERING = 100

VECTOR_SEARCH_QUERY = """
CALL db.index.vector.queryNodes('documentVectorIndex', $candidates, $embedding)
YIELD node, score
MATCH (node)-[:PART_OF]->(paper:Paper)
WHERE $paper_title IS NULL OR toLower(paper.title) CONTAINS toLower($paper_title)
RETURN node.id AS chunk_id, paper.title AS paper, node.section AS section,
       node.text AS text, score
ORDER BY score DESC
LIMIT $limit
"""


@tool
def search_chunks_semantically(query: str, paper_title: str | None = None) -> str:
    """Search the original text chunks of research papers using vector similarity.

    Use this tool to verify specific scientific claims, technical details,
    experimental results, named entities, cited authors, and references
    such as 'Wu et al.'.

    Use it whenever a question asks for evidence from a paper's text.
    Do not conclude that a fact is absent based only on a paper summary.

    Args:
        query: What to look for in the papers' text.
        paper_title: Optional title (or part of it) of a paper to restrict the
            search to. When the question names a paper, pass it here instead
            of looking the paper up in the knowledge graph first.
    """

    if graph is None:
        return "Error: Database client is not initialized."

    try:
        results = graph.query(
            VECTOR_SEARCH_QUERY,
            params={
                "embedding": embedder.embed_query(query),
                "paper_title": paper_title,
                "candidates": CANDIDATES_WHEN_FILTERING
                if paper_title
                else CHUNKS_PER_SEARCH,
                "limit": CHUNKS_PER_SEARCH,
            },
        )

        if not results:
            if paper_title:
                return (
                    f"No text chunks found for a paper matching '{paper_title}'. "
                    "Check the exact title with search_knowledge_graph."
                )
            return "No relevant text chunks found via semantic search."

        formatted_chunks = []

        for r in results:
            formatted_chunks.append(
                f"Chunk ID: {r['chunk_id']}\n"
                f"Paper: {r['paper']}\n"
                f"Section: {r['section']}\n"
                f"Relevance Score: {r['score']:.4f}\n"
                f"{r['text']}"
            )

        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:  # noqa: BLE001
        return f"Error during semantic chunk search: {e!s}"


EntityType = Literal["Method", "Dataset", "Metric", "Task", "Concept"]

# An entity belongs to a paper when the paper has a relationship with it
# (PROPOSES, USES_DATASET...) or when one of its chunks mentions it, e.g. a
# method cited in the related work.
ENTITY_QUERY = """
MATCH (paper:Paper), (entity)
WHERE $entity_type IN labels(entity)
  AND ($paper_title IS NULL OR toLower(paper.title) CONTAINS toLower($paper_title))
  AND ($name IS NULL
       OR any(n IN [entity.name] + coalesce(entity.aliases, [])
              WHERE toLower(n) CONTAINS toLower($name)))
OPTIONAL MATCH (paper)-[role]->(entity)
WITH paper, entity, collect(type(role)) AS roles,
     COUNT { (paper)<-[:PART_OF]-(:Document)-[:MENTIONS]->(entity) } AS mentions
WHERE size(roles) > 0 OR mentions > 0
RETURN paper.title AS paper, entity.name AS entity, roles, mentions
ORDER BY paper, size(roles) DESC, mentions DESC
LIMIT $limit
"""

MAX_ENTITIES = 30


@tool
def find_paper_entities(
    entity_type: EntityType,
    paper_title: str | None = None,
    name: str | None = None,
) -> str:
    """List the methods, datasets, metrics, tasks or concepts of the papers.

    Each result says how the paper relates to the entity: PROPOSES (its own
    method), BUILDS_ON, COMPARES_WITH (baselines), USES_DATASET, ADDRESSES,
    MEASURES_WITH or STUDIES; entities without a relationship are only
    mentioned in the text, e.g. cited works. Use it for questions such as
    "which datasets does paper X use?" (pass paper_title) or "which paper uses
    HPatches?" (pass name). Use search_chunks_semantically afterwards if you
    need details about a specific entity.

    Args:
        entity_type: Kind of entity to list.
        paper_title: Optional title (or part of it) of the paper to look in.
        name: Optional name, abbreviation or part of it of the entity.
    """
    if graph is None:
        return "Error: Graph database client is not initialized."

    try:
        rows = graph.query(
            ENTITY_QUERY,
            params={
                "entity_type": entity_type,
                "paper_title": paper_title,
                "name": name,
                "limit": MAX_ENTITIES,
            },
        )
    except Exception as e:  # noqa: BLE001
        return f"Error searching entities: {e!s}"

    if not rows:
        return f"No {entity_type} entities found for these filters."

    lines = []
    for row in rows:
        roles = ", ".join(row["roles"]) or "only mentioned"
        lines.append(
            f"- [{row['paper']}] {row['entity']}: {roles} "
            f"(mentioned in {row['mentions']} chunks)"
        )
    return "\n".join(lines)


ALL_TOOLS = [
    search_knowledge_graph,
    search_chunks_semantically,
    find_paper_entities,
]
