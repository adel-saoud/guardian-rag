"""GET /v1/audit/{document_id}: return the per-document PII audit record.

It returns every entity Guardian detected and the resulting placeholder
map, so a reviewer can see exactly what was redacted before the document
entered the vector store. The map holds original PII values: expose this
route only to people allowed to see them.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from guardian_rag.api.dependencies import RedactionStoreDep
from guardian_rag.api.schemas import AuditResponse

router = APIRouter(prefix="/v1", tags=["audit"])


@router.get("/audit/{document_id}", response_model=AuditResponse)
def audit(document_id: str, store: RedactionStoreDep) -> AuditResponse:
    """Return the stored redaction map; 404 if unknown, 400 for a path-like id."""
    try:
        redaction_map = store.load(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"unknown document_id: {document_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AuditResponse(redaction_map=redaction_map)
