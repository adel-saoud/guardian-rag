"""Wire models for the FastAPI surface.

Every request and response goes through a frozen pydantic model. Inner
domain types (PiiEntity, RedactionMap) are reused directly rather than
duplicated.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from guardian_rag.pii.models import PiiEntity, RedactionMap


class IngestRequest(BaseModel):
    """Body of `POST /v1/ingest`. `document_id` also names the redaction map file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    text: str
    source: str = "unknown"


class IngestResponse(BaseModel):
    """Counts and a short redacted preview. The full redacted text is never returned."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    n_entities: int
    n_parents: int
    n_children: int
    n_quarantined_parents: int
    redacted_preview: str


class Citation(BaseModel):
    """One retrieved child chunk that fed the answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    document_id: str
    parent_id: str
    score: float
    preview: str


class QueryRequest(BaseModel):
    """Body of `POST /v1/query`. `document_id` restricts retrieval to one document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    top_k: int = Field(default=8, ge=1, le=50)
    document_id: str | None = None


class QueryResponse(BaseModel):
    """Audited answer plus the screening and audit outcome for this request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    answer: str
    audit_clean: bool
    audit_fail_closed: bool
    injection_blocked: bool = False
    injection_dropped_parents: int = 0
    citations: list[Citation] = Field(default_factory=lambda: [])
    audit_entities: list[PiiEntity] = Field(default_factory=lambda: [])


class AuditResponse(BaseModel):
    """Stored redaction map for one document, original values included."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    redaction_map: RedactionMap


class StatsResponse(BaseModel):
    """Counters since process start. `vector_store_size` is the Qdrant point count."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_documents: int
    n_entities_total: int
    n_leaks_caught: int
    n_injections_blocked: int
    n_parents_quarantined: int
    entity_type_breakdown: dict[str, int]
    vector_store_size: int


class HealthResponse(BaseModel):
    """Liveness payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
