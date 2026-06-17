"""GET /v1/stats, aggregate PII protection counters."""

from __future__ import annotations

from fastapi import APIRouter

from guardian_rag.api.dependencies import StatsDep, StoreDep
from guardian_rag.api.schemas import StatsResponse

router = APIRouter(prefix="/v1", tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
def stats(stats: StatsDep, store: StoreDep) -> StatsResponse:
    return StatsResponse(
        n_documents=stats.n_documents,
        n_entities_total=stats.n_entities_total,
        n_leaks_caught=stats.n_leaks_caught,
        entity_type_breakdown=dict(stats.entity_type_breakdown),
        vector_store_size=store.count(),
    )
