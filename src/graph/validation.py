import re


def normalize_text(text: str) -> str:
    """Ignore Markdown emphasis and whitespace, which the LLM rarely copies."""
    return " ".join(text.replace("*", "").replace("_", "").split())


def validate_evidence(evidence: str, chunk_text: str) -> bool:
    evidence = normalize_text(evidence)
    return bool(evidence) and evidence in normalize_text(chunk_text)


def appears_in(name: str, text: str) -> bool:
    return normalize_text(name).lower() in normalize_text(text).lower()


def is_specific_name(name: str) -> bool:
    """Reject math symbols and variables such as "d", "θ_1" or "x^b"."""
    return len(re.findall(r"[A-Za-z]", name)) >= 3
