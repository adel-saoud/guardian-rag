"""Domain models for the ingestion pipeline.

A document is split into parent chunks (used for generation context) and
child chunks (used for retrieval and PII-precision tracking). Every child
carries its parent's id so retrieval can hand off the larger context window
to the generator without an extra database round-trip.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from guardian_rag.pii.models import RedactionMap


class Document(BaseModel):
    """A document about to enter the ingestion pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    text: str
    source: str = "unknown"
    metadata: dict[str, str] = Field(default_factory=lambda: {})


class ParentChunk(BaseModel):
    """Larger context window. Stored as Qdrant payload, never embedded.

    Generation pulls the parent text for citation. The order field is
    stable within a document so the dashboard can render chunks in the
    right sequence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    document_id: str
    text: str
    order: int = Field(ge=0)


class ChildChunk(BaseModel):
    """Smaller chunk that gets embedded for retrieval.

    Each child stores `parent_id` so the hybrid retriever can fetch the
    enclosing parent for generation. `order` is the child's position within
    its parent, which makes overlap debugging straightforward.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    document_id: str
    parent_id: str
    text: str
    order: int = Field(ge=0)


class IngestionResult(BaseModel):
    """What the pipeline returns after redacting and chunking one document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    redacted_text: str
    redaction_map: RedactionMap
    parent_chunks: list[ParentChunk]
    child_chunks: list[ChildChunk]

    @property
    def n_entities(self) -> int:
        return len(self.redaction_map.entities)
