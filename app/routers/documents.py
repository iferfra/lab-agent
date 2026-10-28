from fastapi import APIRouter, Request

document_router = APIRouter()


@document_router.get("/documents")
async def get_documents(request: Request):

    neo4j_graph = request.app.state.neo4j_graph
    cypher_query = """
        MATCH (p:Paper)
        OPTIONAL MATCH (a:Author)-[:WROTE]->(p)

        RETURN
            p.id AS id,
            p.title AS title,
            p.year AS year,
            p.venue AS venue,
            collect(DISTINCT a.name) AS authors

        ORDER BY title ASC
    """
    documents_result = neo4j_graph.query(cypher_query)
    return documents_result
