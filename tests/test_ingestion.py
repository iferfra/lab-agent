from langchain_core.documents import Document

from src.graph.extraction_schemas import (
    ChunkMentions,
    Mention,
    PaperEntity,
    PaperExtraction,
)
from src.graph.graph import (
    entity_key,
    paper_entities,
    paper_metadata,
    to_chunk_graph_document,
)
from src.graph.validation import is_specific_name, validate_evidence
from src.ingestion import split_into_chunks, split_into_sections

PAPER = """# **A Paper**

## **1. Introduction**

We train our model on COCO for facial expression recognition (FER).

### **1.1. Details**

More details.

## **References**

[1] A. Author. European Conference on Computer Vision.
"""


def test_chunks_keep_their_section_and_skip_references():
    chunks = split_into_chunks(split_into_sections(PAPER))

    sections = [chunk.metadata["section"] for chunk in chunks]
    assert sections == ["", "1. Introduction", "1. Introduction > 1.1. Details"]
    assert not any("European Conference" in c.page_content for c in chunks)


def test_entity_key_merges_spelling_variants():
    assert entity_key("Aff-Wild2") == entity_key("AffWild2") == entity_key("aff wild2")


def test_math_symbols_are_not_entities():
    assert not any(is_specific_name(name) for name in ["d", "θ_1", "x^b", "J^"])
    assert all(is_specific_name(name) for name in ["FER", "HPatches", "mAP"])


def test_evidence_ignores_markdown_and_whitespace():
    chunk = "We evaluate on **RAF-DB**\nand AffectNet+."
    assert validate_evidence("We evaluate on RAF-DB and AffectNet+.", chunk)
    assert not validate_evidence("We evaluate on CelebA.", chunk)


def test_user_metadata_overrides_extracted_and_empty_values_are_dropped():
    paper = PaperExtraction(paper_title="Extracted title", venue="", year=2024)
    text = "A paper published in 2024."

    metadata = paper_metadata(
        paper, {"paper_title": "User title", "file_name": "x"}, text
    )

    assert metadata == {"paper_title": "User title", "year": 2024, "authors": []}


def test_metadata_not_stated_in_the_paper_is_dropped():
    paper = PaperExtraction(paper_title="A Paper", year=2023, doi="10.1/abc")

    assert paper_metadata(paper, {}, "No dates here.") == {
        "paper_title": "A Paper",
        "authors": [],
    }


def test_paper_entities_must_appear_in_the_paper():
    paper = PaperExtraction(
        paper_title="A Paper",
        tasks=[PaperEntity(name="facial expression recognition", aliases=["FER"])],
        datasets=[PaperEntity(name="COCO"), PaperEntity(name="ImageNet")],
    )

    entities = paper_entities(paper, PAPER)

    assert [(rel, node.id) for rel, node in entities] == [
        ("ADDRESSES", "facialexpressionrecognition"),
        ("USES_DATASET", "coco"),
    ]


def test_aliases_are_cleaned_and_an_entity_keeps_one_label():
    paper = PaperExtraction(
        paper_title="A Paper",
        tasks=[PaperEntity(name="facial expression recognition", aliases=["FER [3]"])],
        concepts=[PaperEntity(name="Facial expression recognition")],
    )

    entities = paper_entities(paper, PAPER)

    assert len(entities) == 1
    assert entities[0][1].properties["aliases"] == ["FER"]


def test_chunk_mentions_use_the_canonical_entity():
    chunk = Document(page_content="Results for FER and the encoder on COCO.")
    _, fer = paper_entities(
        PaperExtraction(
            paper_title="A Paper",
            tasks=[PaperEntity(name="facial expression recognition", aliases=["FER"])],
        ),
        PAPER,
    )[0]
    known = {entity_key("facial expression recognition"): fer, entity_key("FER"): fer}
    extraction = ChunkMentions(
        mentions=[
            Mention(name="FER", entity_type="Concept", evidence="Results for FER"),
            Mention(name="COCO", entity_type="Dataset", evidence="on COCO"),
            # New concepts only come from the paper-level extraction
            Mention(name="encoder", entity_type="Concept", evidence="the encoder"),
            # Evidence not in the chunk
            Mention(name="ImageNet", entity_type="Dataset", evidence="on ImageNet"),
        ]
    )

    graph_document = to_chunk_graph_document(extraction, chunk, known)

    assert [(node.type, node.id) for node in graph_document.nodes] == [
        ("Task", "facialexpressionrecognition"),
        ("Dataset", "coco"),
    ]
