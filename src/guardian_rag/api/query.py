"""POST /v1/query, the full RAG path with post-generation audit.

Pipeline (per the architecture diagram):
    1. Redact the query before embedding.
    2. Hybrid retrieve top-K children + their parents.
    3. Generate a grounded answer.
    4. Post-generation PII audit (fail-closed).
    5. Set X-Guardian-* response headers.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Response

from guardian_rag.api.dependencies import (
    AuditorDep,
    DetectorDep,
    GeneratorDep,
    RedactorDep,
    RetrieverDep,
    StatsDep,
)
from guardian_rag.api.schemas import Citation, QueryRequest, QueryResponse
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.redactor import Redactor

logger = structlog.get_logger(__name__)


router = APIRouter(prefix="/v1", tags=["query"])


_PREVIEW_CHARS = 200


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
    stats: StatsDep,
) -> QueryResponse:
    request_id = f"chatcmpl-{uuid.uuid4()}"
    safe_query = _redact_query(body.query, detector, redactor)

    retrieval = retriever.retrieve(
        safe_query,
        top_k=body.top_k,
        document_id=body.document_id,
    )
    grounded = await generator.generate(safe_query, retrieval.parents)
    audit = auditor.audit(grounded.answer)

    if not audit.clean:
        stats.record_leak(audit.entities)

    response.headers["X-Guardian-Pii-Audit"] = "clean" if audit.clean else "redacted"
    response.headers["X-Guardian-Chunks-Used"] = str(len(retrieval.children))
    response.headers["X-Guardian-Faithfulness"] = ""
    response.headers["X-Guardian-Request-Id"] = request_id

    logger.info(
        "query.complete",
        request_id=request_id,
        clean=audit.clean,
        fail_closed=audit.fail_closed,
        n_children=len(retrieval.children),
        n_parents=len(retrieval.parents),
    )

    return QueryResponse(
        request_id=request_id,
        answer=audit.answer,
        audit_clean=audit.clean,
        audit_fail_closed=audit.fail_closed,
        audit_entities=audit.entities,
        citations=[
            Citation(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                parent_id=hit.parent_id,
                score=hit.score,
                preview=hit.text[:_PREVIEW_CHARS],
            )
            for hit in retrieval.children
        ],
    )
