"""FastAPI app factory + lifespan wiring.

Production code calls `create_app()`. Tests call `build_app(state)` with a
pre-built `AppState` containing fakes, skipping the lifespan handler.

Heavy ML imports (GLiNER, sentence-transformers, the injection
classifier) happen only inside the lifespan so importing
`guardian_rag.api` stays cheap for tools, scripts, and tests.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import cache

import httpx
import structlog
from fastapi import FastAPI
from qdrant_client import QdrantClient

from guardian_rag import __version__
from guardian_rag.api.audit import router as audit_router
from guardian_rag.api.dependencies import AppState, StatsCollector
from guardian_rag.api.health import router as health_router
from guardian_rag.api.ingest import router as ingest_router
from guardian_rag.api.query import router as query_router
from guardian_rag.api.stats import router as stats_router
from guardian_rag.config import get_settings
from guardian_rag.generation.generator import Generator
from guardian_rag.generation.ollama import OllamaClient
from guardian_rag.ingestion.chunker import HierarchicalChunker
from guardian_rag.ingestion.pipeline import IngestionPipeline
from guardian_rag.ingestion.redaction_store import RedactionMapStore
from guardian_rag.injection.screener import ClassifierScreener, build_screener
from guardian_rag.pii.audit import PostGenerationAuditor
from guardian_rag.pii.deberta_validator import DebertaValidator
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.gliner_recognizer import GlinerRecognizer
from guardian_rag.pii.presidio_engine import PresidioEngine
from guardian_rag.pii.redactor import Redactor
from guardian_rag.retrieval.embedder import Embedder
from guardian_rag.retrieval.hybrid import HybridRetriever
from guardian_rag.retrieval.reranker import Reranker
from guardian_rag.retrieval.store import QdrantStore
from guardian_rag.service_versions import (
    MIN_OLLAMA_VERSION,
    MIN_QDRANT_VERSION,
    require_min_version,
)

logger = structlog.get_logger(__name__)


def _new_app(lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None) -> FastAPI:
    """Create the FastAPI app with every router, shared by tests and production."""
    app = FastAPI(
        title="Guardian RAG",
        version=__version__,
        description=(
            "Privacy-safe RAG with three-stage PII detection, prompt-injection screening, "
            "and a post-generation audit."
        ),
        lifespan=lifespan,
    )
    for router in (health_router, ingest_router, query_router, audit_router, stats_router):
        app.include_router(router)
    return app


def build_app(state: AppState) -> FastAPI:
    """Build an app bound to a pre-constructed `AppState`, skipping model loads (tests)."""
    app = _new_app(lifespan=None)
    app.state.guardian = state
    return app


def create_app() -> FastAPI:
    """Production entry point: the lifespan loads models and connects to Qdrant and Ollama."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Build every collaborator once at startup; close the Ollama client on shutdown."""
        settings = get_settings()
        logger.info(
            "app.lifespan.startup",
            ollama=settings.ollama_base_url,
            qdrant=settings.qdrant_url,
        )

        # Check service versions first, so an old server fails before the slow model loads.
        qdrant = QdrantClient(location=settings.qdrant_url, api_key=settings.qdrant_api_key)
        require_min_version("Qdrant", qdrant.info().version, MIN_QDRANT_VERSION)
        async with httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=httpx.Timeout(settings.ollama_timeout_s),
        ) as http:
            ollama = OllamaClient(http=http, think=settings.ollama_think)
            require_min_version("Ollama", await ollama.version(), MIN_OLLAMA_VERSION)

            from guardian_rag.injection._real_backends import RealInjectionClassifierBackend
            from guardian_rag.pii._real_backends import (
                RealDebertaValidatorBackend,
                RealGlinerBackend,
            )
            from guardian_rag.retrieval._real_backends import (
                RealCrossEncoderBackend,
                RealEmbedderBackend,
            )

            presidio = PresidioEngine()
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

            @cache
            def injection_classifier() -> ClassifierScreener:
                """Cached so the query and chunk screeners share one model load."""
                return ClassifierScreener(
                    backend=RealInjectionClassifierBackend(
                        settings.injection_model, settings.injection_label
                    ),
                    model_name=settings.injection_model,
                    threshold=settings.injection_threshold,
                )

            query_screener = build_screener(settings.query_injection_screener, injection_classifier)
            chunk_screener = build_screener(settings.chunk_injection_screener, injection_classifier)

            embedder = Embedder(
                backend=RealEmbedderBackend(settings.embedding_model),
                dim=settings.embedding_dim,
            )
            store = QdrantStore(
                client=qdrant,
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
                screener=chunk_screener,
            )
            reranker = (
                Reranker(RealCrossEncoderBackend(settings.reranker_model), settings.reranker_model)
                if settings.reranker_enabled
                else None
            )
            retriever = HybridRetriever(
                store,
                bm25_weight=settings.bm25_weight,
                reranker=reranker,
                rerank_candidates=settings.rerank_candidates,
            )
            retriever.refresh()

            generator = Generator(ollama=ollama, model=settings.ollama_model)
            auditor = PostGenerationAuditor(
                detector=detector,
                redactor=redactor,
                fail_closed=settings.audit_fail_closed,
            )
            app.state.guardian = AppState(
                detector=detector,
                redactor=redactor,
                pipeline=pipeline,
                redaction_store=redaction_store,
                store=store,
                retriever=retriever,
                generator=generator,
                auditor=auditor,
                query_screener=query_screener,
                chunk_screener=chunk_screener,
                stats=StatsCollector(),
            )
            try:
                yield
            finally:
                logger.info("app.lifespan.shutdown")

    return _new_app(lifespan=lifespan)


__all__ = ["build_app", "create_app"]
