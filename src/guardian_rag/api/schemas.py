"""Wire models for the FastAPI surface.

Every request and response goes through a frozen pydantic model. Inner
domain types (PiiEntity, RedactionMap) are reused directly rather than
duplicated.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from guardian_rag.pii.models import PiiEntity, RedactionMap


class IngestRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    text: str
    source: str = "unknown"


class IngestResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    n_entities: int
    n_parents: int
    n_children: int
    redacted_preview: str
    """First 240 characters of the redacted text. Full text not returned."""


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    document_id: str
    parent_id: str
    score: float
    preview: str


class QueryRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    top_k: int = Field(default=8, ge=1, le=50)
    document_id: str | None = None


class QueryResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    answer: str
    audit_clean: bool
    audit_fail_closed: bool
    citations: list[Citation] = Field(default_factory=lambda: [])
    audit_entities: list[PiiEntity] = Field(default_factory=lambda: [])


class AuditResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    redaction_map: RedactionMap


class StatsResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_documents: int
    n_entities_total: int
    n_leaks_caught: int
    entity_type_breakdown: dict[str, int]
    vector_store_size: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
