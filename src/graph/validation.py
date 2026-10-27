ALLOWED_ENDPOINTS = {
    "ADDRESSES_TASK": {("Method", "Task")},
    "TRAINED_ON": {("Method", "Dataset")},
    "EVALUATED_ON": {
        ("Method", "Dataset"),
        ("Method", "Benchmark"),
    },
    "USES_METHOD": {("Method", "Method")},
    "INSTANCE_OF": {
        ("Method", "Concept"),
        ("Concept", "Concept"),
    },
}


def validate_evidence(evidence: str, chunk_text: str) -> bool:
    return bool(evidence.strip()) and evidence.strip() in chunk_text


def validate_relationship_types(
    source_type: str,
    relationship_type: str,
    target_type: str,
) -> bool:
    allowed_pairs = ALLOWED_ENDPOINTS.get(relationship_type, set())
    return (source_type, target_type) in allowed_pairs
