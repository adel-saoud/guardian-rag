"""FastAPI app factory + lifespan wiring.

Production code calls `create_app()`. Tests call `build_app(state)` with a
pre-built `AppState` containing fakes, skipping the lifespan handler.

Heavy ML imports (GLiNER, sentence-transformers) happen only inside the
lifespan so importing `guardian_rag.api` stays cheap for tools, scripts,
and tests.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI
from qdrant_client import QdrantClient

from guardian_rag.api.audit import router as audit_router
from guardian_rag.api.dependencies import AppState, StatsCollector
from guardian_rag.api.health import router as health_router
from guardian_rag.api.ingest import router as ingest_router
from guardian_rag.api.query import router as query_router
from guardian_rag.api.stats import router as stats_router
from guardian_rag.config import Settings, get_settings
from guardian_rag.generation.generator import Generator
from guardian_rag.generation.ollama import OllamaClient
from guardian_rag.ingestion.chunker import HierarchicalChunker
from guardian_rag.ingestion.pipeline import IngestionPipeline
from guardian_rag.ingestion.redaction_store import RedactionMapStore
from guardian_rag.pii.audit import PostGenerationAuditor
from guardian_rag.pii.deberta_validator import DebertaValidator
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.gliner_recognizer import GlinerRecognizer
from guardian_rag.pii.presidio_engine import PresidioEngine
from guardian_rag.pii.redactor import Redactor
from guardian_rag.retrieval.embedder import Embedder
from guardian_rag.retrieval.hybrid import HybridRetriever
from guardian_rag.retrieval.store import QdrantStore

logger = structlog.get_logger(__name__)


def _include_routers(app: FastAPI) -> None:
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(query_router)
    app.include_router(audit_router)
    app.include_router(stats_router)


def build_app(state: AppState) -> FastAPI:
    """Build an app bound to a pre-constructed `AppState`, used by tests."""
    app = FastAPI(
        title="Guardian RAG",
        version="0.1.0",
        description="Privacy-safe RAG with two-stage PII detection and post-generation audit.",
    )
    app.state.guardian = state
    _include_routers(app)
    return app


def create_app() -> FastAPI:
    """Production entry point, wires up real collaborators via lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        settings = get_settings()
        logger.info(
            "app.lifespan.startup",
            ollama=settings.ollama_base_url,
            qdrant=settings.qdrant_url,
        )

        # Heavy ML backends imported lazily.
        from guardian_rag.pii._real_backends import (
            RealDebertaValidatorBackend,
            RealGlinerBackend,
        )
        from guardian_rag.retrieval._real_backends import RealEmbedderBackend

        presidio = PresidioEngine(language=settings.languages[0])
        gliner = GlinerRecognizer(
            backend=RealGlinerBackend(settings.gliner_model),
            threshold=settings.gliner_threshold,
        )
        validator = DebertaValidator(
            backend=RealDebertaValidatorBackend(settings.deberta_model),
            score_threshold=settings.deberta_threshold,
        )
        detector = PiiDetector(presidio=presidio, gliner=gliner, validator=validator)
        redactor = Redactor()
        redaction_store = RedactionMapStore(settings.redaction_map_dir)

        embedder = Embedder(
            backend=RealEmbedderBackend(settings.embedding_model),
            dim=settings.embedding_dim,
        )
        qdrant_client = _build_qdrant_client(settings)
        store = QdrantStore(
            client=qdrant_client,
            embedder=embedder,
            collection_name=settings.qdrant_collection,
        )
        chunker = HierarchicalChunker(
            parent_tokens=settings.parent_chunk_tokens,
            child_tokens=settings.child_chunk_tokens,
            child_overlap=settings.child_chunk_overlap,
        )
        pipeline = IngestionPipeline(
            detector=detector,
            redactor=redactor,
            chunker=chunker,
            redaction_store=redaction_store,
            vector_sink=store,
        )
        retriever = HybridRetriever(store, bm25_weight=settings.bm25_weight)
        retriever.refresh()

        async with httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=httpx.Timeout(settings.ollama_timeout_s),
        ) as http:
            ollama = OllamaClient(http=http)
            generator = Generator(ollama=ollama, model=settings.ollama_model)
            auditor = PostGenerationAuditor(
                detector=detector,
                redactor=redactor,
                fail_closed=settings.audit_fail_closed,
            )
            app.state.guardian = AppState(
                settings=settings,
                detector=detector,
                redactor=redactor,
                pipeline=pipeline,
                redaction_store=redaction_store,
                store=store,
                retriever=retriever,
                generator=generator,
                auditor=auditor,
                stats=StatsCollector(),
            )
            try:
                yield
            finally:
                logger.info("app.lifespan.shutdown")

    app = FastAPI(
        title="Guardian RAG",
        version="0.1.0",
        description="Privacy-safe RAG with two-stage PII detection and post-generation audit.",
        lifespan=lifespan,
    )
    _include_routers(app)
    return app


def _build_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url.startswith(":memory"):
        return QdrantClient(location=":memory:")
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


__all__ = ["build_app", "create_app"]
