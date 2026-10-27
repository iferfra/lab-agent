from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EntityType = Literal[
    "Method",
    "Dataset",
    "Metric",
    "Benchmark",
    "Task",
    "Concept",
]

RelationshipType = Literal[
    "ADDRESSES_TASK",
    "TRAINED_ON",
    "EVALUATED_ON",
    "USES_METHOD",
    "INSTANCE_OF",
]


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mention: str = Field(min_length=1)
    entity_type: EntityType
    evidence: str = Field(min_length=1)


class ExtractedRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_mention: str = Field(min_length=1)
    target_mention: str = Field(min_length=1)
    relationship_type: RelationshipType
    evidence: str = Field(min_length=1)


class ChunkExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
