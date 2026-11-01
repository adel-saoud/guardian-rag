"""POST /v1/query, the full RAG path with post-generation audit.

Pipeline (per the architecture diagram):
    1. Redact the query before embedding.
    2. Screen the redacted query for prompt injection (fail-closed). A
       flagged query is refused before retrieval or generation.
    3. Hybrid retrieve top-K children + their parents, reranked when enabled.
    4. Drop parents quarantined at ingestion, and re-screen parents whose
       stored verdict came from a different screener.
    5. Generate a grounded answer.
    6. Post-generation PII audit (fail-closed). Always the last stage.
    7. Set X-Guardian-* response headers.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Response

from guardian_rag.api.dependencies import (
    AuditorDep,
    ChunkScreenerDep,
    DetectorDep,
    GeneratorDep,
    QueryScreenerDep,
    RedactorDep,
    RetrieverDep,
    StatsDep,
)
from guardian_rag.api.schemas import Citation, QueryRequest, QueryResponse
from guardian_rag.injection.screener import screen_parents_for_generation
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.redactor import Redactor

logger = structlog.get_logger(__name__)


router = APIRouter(prefix="/v1", tags=["query"])


_CITATION_PREVIEW_CHARS = 200
"""Characters of child text returned per citation, enough to recognise the passage."""

INJECTION_REFUSAL_TEXT = "This request was blocked by the prompt-injection screen."
"""Answer returned for a flagged query. Fixed text, so it needs no PII audit."""


def _redact_query(text: str, detector: PiiDetector, redactor: Redactor) -> str:
    """Run the same detector + redactor over the query.

    Any PII the user typed (deliberately or otherwise) is replaced with
    typed placeholders before the query ever reaches the embedder or
    the LLM. The redaction map is discarded; query-side maps are not
    persisted.
    """
    entities = detector.detect(text)
    if not entities:
        return text
    redacted, _map = redactor.redact(text, entities, "query")
    return redacted


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    response: Response,
    detector: DetectorDep,
    redactor: RedactorDep,
    retriever: RetrieverDep,
    generator: GeneratorDep,
    auditor: AuditorDep,
    query_screener: QueryScreenerDep,
    chunk_screener: ChunkScreenerDep,
    stats: StatsDep,
) -> QueryResponse:
    """Answer one query. Every path that returns generated text goes through the audit."""
    request_id = f"chatcmpl-{uuid.uuid4()}"
    safe_query = _redact_query(body.query, detector, redactor)
    response.headers["X-Guardian-Request-Id"] = request_id

    query_verdict = query_screener.screen([safe_query])[0]
    if query_verdict.flagged:
        stats.record_injection_blocked()
        response.headers["X-Guardian-Injection"] = "blocked"
        logger.warning(
            "query.injection.blocked",
            request_id=request_id,
            reason=query_verdict.reason,
            score=query_verdict.score,
        )
        return QueryResponse(
            request_id=request_id,
            answer=INJECTION_REFUSAL_TEXT,
            audit_clean=True,
            audit_fail_closed=False,
            injection_blocked=True,
        )

    retrieval = retriever.retrieve(
        safe_query,
        top_k=body.top_k,
        document_id=body.document_id,
    )
    parents, dropped = screen_parents_for_generation(chunk_screener, retrieval.parents)
    stats.record_quarantine(len(dropped))
    kept_parent_ids = {parent.chunk_id for parent in parents}
    children = [hit for hit in retrieval.children if hit.parent_id in kept_parent_ids]

    grounded = await generator.generate(safe_query, parents)
    audit = auditor.audit(grounded.answer)

    if not audit.clean:
        stats.record_leak(audit.entities)

    response.headers["X-Guardian-Pii-Audit"] = "clean" if audit.clean else "redacted"
    response.headers["X-Guardian-Injection"] = "context_dropped" if dropped else "clean"
    response.headers["X-Guardian-Chunks-Used"] = str(len(children))

    logger.info(
        "query.complete",
        request_id=request_id,
        clean=audit.clean,
        fail_closed=audit.fail_closed,
        n_children=len(children),
        n_parents=len(parents),
        n_parents_dropped=len(dropped),
    )

    return QueryResponse(
        request_id=request_id,
        answer=audit.answer,
        audit_clean=audit.clean,
        audit_fail_closed=audit.fail_closed,
        injection_dropped_parents=len(dropped),
        audit_entities=audit.entities,
        citations=[
            Citation(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                parent_id=hit.parent_id,
                score=hit.score,
                preview=hit.text[:_CITATION_PREVIEW_CHARS],
            )
            for hit in children
        ],
    )
