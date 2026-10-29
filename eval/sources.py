"""Build dataset/sources.json, the material benchmark questions are written from.

    python -m eval.sources

- graph facts: sentences built from the knowledge graph (authors, venue, and
  the methods, datasets, metrics... linked to each paper).
- chunks: for each paper, the chunks that mention most of its main entities,
  at most one per section, so the questions cover the whole paper.
- known gaps: things the papers do not say, for unanswerable questions. They
  cannot come from the graph, so they are written by hand in known_gaps.json.

Run it again after re-ingesting the papers, then regenerate the questions.
"""

from langchain_neo4j import Neo4jGraph

from eval.common import DATASET_DIR, SOURCES_PATH, read_json, write_json

CHUNKS_PER_PAPER = 8
MIN_CHUNK_LENGTH = 400  # shorter chunks rarely hold a full fact

# How each relationship of a Paper reads as a sentence
FACT_TEMPLATES = {
    "PROPOSES": "'{paper}' proposes {names}.",
    "BUILDS_ON": "The method of '{paper}' builds on {names}.",
    "COMPARES_WITH": "'{paper}' compares its method with {names}.",
    "ADDRESSES": "'{paper}' addresses {names}.",
    "USES_DATASET": "'{paper}' uses the datasets {names}.",
    "MEASURES_WITH": "'{paper}' reports the metrics {names}.",
    "STUDIES": "'{paper}' studies {names}.",
}

PAPERS_QUERY = """
MATCH (p:Paper)
OPTIONAL MATCH (a:Author)-[:WROTE]->(p)
RETURN p.id AS id, p.title AS title, p.venue AS venue, p.year AS year,
       p.repository_link AS repository, collect(a.name) AS authors
ORDER BY title
"""

RELATIONSHIPS_QUERY = """
MATCH (p:Paper {id: $paper_id})-[r]->(e)
RETURN type(r) AS relationship, collect(e.name) AS names
"""

SHARED_AUTHORS_QUERY = """
MATCH (a:Author)-[:WROTE]->(p:Paper)
WITH a, collect(p.title) AS papers
WHERE size(papers) > 1
RETURN a.name AS author, papers
ORDER BY author
"""

# Chunks ranked by how many of the paper's main entities they mention
CHUNKS_QUERY = """
MATCH (c:Document)-[:PART_OF]->(p:Paper {id: $paper_id})
WHERE size(c.text) >= $min_length
OPTIONAL MATCH (c)-[:MENTIONS]->(e)<--(p)
RETURN c.id AS id, c.section AS section, c.text AS text,
       count(DISTINCT e) AS main_entities
ORDER BY main_entities DESC, id
"""


def join_names(names: list[str]) -> str:
    names = sorted(names)
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def paper_facts(graph: Neo4jGraph, paper: dict) -> list[dict]:
    prefix = f"graph_{paper['id'][:6]}"
    title = paper["title"]
    facts = [
        (
            f"{prefix}_authors",
            f"The authors of '{title}' are {join_names(paper['authors'])}.",
        )
    ]
    if paper["venue"]:
        facts.append(
            (f"{prefix}_venue", f"'{title}' was published at {paper['venue']}.")
        )
    if paper["year"]:
        facts.append((f"{prefix}_year", f"'{title}' was published in {paper['year']}."))
    if paper["repository"]:
        facts.append(
            (f"{prefix}_code", f"The code of '{title}' is at {paper['repository']}.")
        )

    for row in graph.query(RELATIONSHIPS_QUERY, params={"paper_id": paper["id"]}):
        template = FACT_TEMPLATES.get(row["relationship"])
        if template:
            facts.append(
                (
                    f"{prefix}_{row['relationship'].lower()}",
                    template.format(paper=title, names=join_names(row["names"])),
                )
            )

    return [{"source_id": i, "fact": fact, "paper": title} for i, fact in facts]


def shared_author_facts(graph: Neo4jGraph) -> list[dict]:
    return [
        {
            "source_id": f"graph_shared_{i}",
            "fact": f"{row['author']} is an author of both "
            + " and ".join(f"'{title}'" for title in sorted(row["papers"]))
            + ".",
            "paper": None,
        }
        for i, row in enumerate(graph.query(SHARED_AUTHORS_QUERY), 1)
    ]


def paper_chunks(graph: Neo4jGraph, paper: dict) -> list[dict]:
    rows = graph.query(
        CHUNKS_QUERY,
        params={"paper_id": paper["id"], "min_length": MIN_CHUNK_LENGTH},
    )

    chunks, sections = [], set()
    for row in rows:
        if row["section"] in sections:
            continue
        sections.add(row["section"])
        chunks.append(
            {
                "source_id": row["id"],
                "paper": paper["title"],
                "section": row["section"],
                "text": row["text"],
            }
        )
        if len(chunks) == CHUNKS_PER_PAPER:
            break

    return chunks


def main() -> None:
    graph = Neo4jGraph()
    papers = graph.query(PAPERS_QUERY)

    sources = {
        "graph_facts": [f for p in papers for f in paper_facts(graph, p)]
        + shared_author_facts(graph),
        "chunks": [c for p in papers for c in paper_chunks(graph, p)],
        "known_gaps": read_json(DATASET_DIR / "known_gaps.json"),
    }
    write_json(SOURCES_PATH, sources)

    print(
        f"{len(sources['graph_facts'])} graph facts, {len(sources['chunks'])} chunks, "
        f"{len(sources['known_gaps'])} known gaps -> {SOURCES_PATH}"
    )


if __name__ == "__main__":
    main()
