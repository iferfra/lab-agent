from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal["Method", "Dataset", "Metric", "Task", "Concept"]


class PaperEntity(BaseModel):
    name: str = Field(description="Full name, as written in the paper.")
    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Other ways the paper refers to it: abbreviations, other spellings, "
            "or phrases like 'our method' for the paper's own contribution."
        ),
    )


class PaperExtraction(BaseModel):
    """Metadata and main entities of a research paper."""

    paper_title: str = Field(description="Title of the paper.")
    authors: list[str] = Field(default_factory=list, description="Author names.")
    year: int | None = Field(default=None, description="Publication year.")
    venue: str | None = Field(default=None, description="Conference or journal.")
    doi: str | None = Field(default=None, description="DOI of the paper.")
    repository_link: str | None = Field(
        default=None, description="URL of the code repository, e.g. GitHub."
    )

    proposed_methods: list[PaperEntity] = Field(
        default_factory=list,
        description="Methods, models or protocols that this paper contributes. "
        "Usually only one.",
    )
    base_methods: list[PaperEntity] = Field(
        default_factory=list,
        description="Existing methods the proposal is built on or reuses, such as "
        "backbones, teacher models or components.",
    )
    baseline_methods: list[PaperEntity] = Field(
        default_factory=list,
        description="Named methods the proposal is compared against, such as "
        "'SuperPoint'. Never descriptions such as 'previous approaches'.",
    )
    tasks: list[PaperEntity] = Field(
        default_factory=list,
        description="Scientific problems the paper solves, not application areas "
        "only mentioned as motivation.",
    )
    datasets: list[PaperEntity] = Field(
        default_factory=list,
        description="Datasets or benchmarks used for training or evaluation.",
    )
    metrics: list[PaperEntity] = Field(
        default_factory=list, description="Evaluation metrics the paper reports."
    )
    concepts: list[PaperEntity] = Field(
        default_factory=list,
        description="At most 10 central concepts or factors the paper studies "
        "that are not methods, tasks, datasets or metrics.",
    )


class Mention(BaseModel):
    name: str = Field(description="Name of the entity.")
    entity_type: EntityType
    evidence: str = Field(
        description="Sentence of the chunk where it appears, copied verbatim."
    )


class ChunkMentions(BaseModel):
    """Entities mentioned in a chunk of a paper."""

    mentions: list[Mention] = Field(default_factory=list)
