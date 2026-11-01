"""POST /v1/ingest, detect + redact + chunk + store in one shot."""

from __future__ import annotations

from fastapi import APIRouter

from guardian_rag.api.dependencies import PipelineDep, RetrieverDep, StatsDep
from guardian_rag.api.schemas import IngestRequest, IngestResponse
from guardian_rag.ingestion.models import Document

router = APIRouter(prefix="/v1", tags=["ingest"])


_REDACTED_PREVIEW_CHARS = 240
"""Characters of redacted text echoed back, so a caller can eyeball the placeholders."""


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    body: IngestRequest,
    pipeline: PipelineDep,
    retriever: RetrieverDep,
    stats: StatsDep,
) -> IngestResponse:
    """Ingest one document and rebuild the BM25 index so the next query sees it."""
    result = pipeline.ingest(
        Document(
            document_id=body.document_id,
            text=body.text,
            source=body.source,
        )
    )
    retriever.refresh()
    stats.record_ingest(result.redaction_map.entities)
    stats.record_quarantine(result.n_quarantined_parents)
    return IngestResponse(
        document_id=result.document_id,
        n_entities=result.n_entities,
        n_parents=len(result.parent_chunks),
        n_children=len(result.child_chunks),
        n_quarantined_parents=result.n_quarantined_parents,
        redacted_preview=result.redacted_text[:_REDACTED_PREVIEW_CHARS],
    )
