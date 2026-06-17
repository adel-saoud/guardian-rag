"""POST /v1/ingest, detect + redact + chunk + store in one shot."""

from __future__ import annotations

from fastapi import APIRouter

from guardian_rag.api.dependencies import PipelineDep, RetrieverDep, StatsDep
from guardian_rag.api.schemas import IngestRequest, IngestResponse
from guardian_rag.ingestion.models import Document

router = APIRouter(prefix="/v1", tags=["ingest"])


_PREVIEW_CHARS = 240


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    body: IngestRequest,
    pipeline: PipelineDep,
    retriever: RetrieverDep,
    stats: StatsDep,
) -> IngestResponse:
    result = pipeline.ingest(
        Document(
            document_id=body.document_id,
            text=body.text,
            source=body.source,
        )
    )
    # New documents in the vector store mean the BM25 index is stale.
    retriever.refresh()
    stats.record_ingest(result.redaction_map.entities)
    return IngestResponse(
        document_id=result.document_id,
        n_entities=result.n_entities,
        n_parents=len(result.parent_chunks),
        n_children=len(result.child_chunks),
        redacted_preview=result.redacted_text[:_PREVIEW_CHARS],
    )
