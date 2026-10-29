"""Schema of the knowledge graph.

(Author)-[:WROTE]->(Paper)<-[:PART_OF]-(Document)-[:MENTIONS]->(entity)

(Paper)-[:PROPOSES]->(Method)       its own contribution
(Paper)-[:BUILDS_ON]->(Method)      existing methods it reuses
(Paper)-[:COMPARES_WITH]->(Method)  baselines in its experiments
(Paper)-[:ADDRESSES]->(Task)
(Paper)-[:USES_DATASET]->(Dataset)
(Paper)-[:MEASURES_WITH]->(Metric)
(Paper)-[:STUDIES]->(Concept)
"""

# Field of PaperExtraction -> (relationship from the Paper, label of the entity)
PAPER_RELATIONSHIPS = {
    "proposed_methods": ("PROPOSES", "Method"),
    "base_methods": ("BUILDS_ON", "Method"),
    "baseline_methods": ("COMPARES_WITH", "Method"),
    "tasks": ("ADDRESSES", "Task"),
    "datasets": ("USES_DATASET", "Dataset"),
    "metrics": ("MEASURES_WITH", "Metric"),
    "concepts": ("STUDIES", "Concept"),
}
