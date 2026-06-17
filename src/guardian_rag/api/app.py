"""FastAPI app factory.

PR 1 ships a minimal placeholder so `guardian_rag.main` can resolve. The
lifespan, dependency container, and routers land in the API + dashboard PR.
"""

from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Return a bare FastAPI app. Filled in later."""
    return FastAPI(
        title="Guardian RAG",
        version="0.1.0",
        description="Privacy-safe RAG with two-stage PII detection and post-generation audit.",
    )
