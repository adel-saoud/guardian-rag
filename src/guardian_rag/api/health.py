"""Liveness probe."""

from __future__ import annotations

from fastapi import APIRouter

from guardian_rag.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness only: answers once the lifespan has finished loading models."""
    return HealthResponse(status="ok")
