"""Liveness probe."""

from __future__ import annotations

from fastapi import APIRouter

from guardian_rag.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
