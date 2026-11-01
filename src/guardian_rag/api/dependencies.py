"""Dependency injection for the FastAPI surface.

The lifespan handler builds an `AppState` and stashes it on
`app.state.guardian`. Routes pull individual collaborators via
`Annotated[T, Depends(...)]` aliases below, which lets tests override
any single piece with `app.dependency_overrides[get_x] = fake_x`.
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, Request

from guardian_rag.generation.generator import Generator
from guardian_rag.ingestion.pipeline import IngestionPipeline
from guardian_rag.ingestion.redaction_store import RedactionMapStore
from guardian_rag.injection.screener import InjectionScreener
from guardian_rag.pii.audit import PostGenerationAuditor
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.models import PiiEntity
from guardian_rag.pii.redactor import Redactor
from guardian_rag.retrieval.hybrid import HybridRetriever
from guardian_rag.retrieval.store import QdrantStore


@dataclass
class StatsCollector:
    """Thread-safe in-memory counters powering `/v1/stats`.

    Resets on process restart. Durable metrics would need a time-series
    sink (Prometheus export is on the roadmap).
    """

    n_documents: int = 0
    n_entities_total: int = 0
    n_leaks_caught: int = 0
    n_injections_blocked: int = 0
    n_parents_quarantined: int = 0
    entity_type_breakdown: Counter[str] = field(default_factory=lambda: Counter())
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_ingest(self, entities: list[PiiEntity]) -> None:
        """Count one ingested document and the entities redacted from it."""
        with self._lock:
            self.n_documents += 1
            self.n_entities_total += len(entities)
            for entity in entities:
                self.entity_type_breakdown[entity.entity_type.value] += 1

    def record_injection_blocked(self) -> None:
        """Count one query refused by the injection screen."""
        with self._lock:
            self.n_injections_blocked += 1

    def record_quarantine(self, n_parents: int) -> None:
        """Count parents quarantined at ingestion or dropped from a query's context."""
        with self._lock:
            self.n_parents_quarantined += n_parents

    def record_leak(self, entities: list[PiiEntity]) -> None:
        """Count one answer the post-generation audit had to redact."""
        with self._lock:
            if entities:
                self.n_leaks_caught += 1
                for entity in entities:
                    self.entity_type_breakdown[entity.entity_type.value] += 1


@dataclass
class AppState:
    """Long-lived collaborators built at lifespan startup, shared by every request."""

    detector: PiiDetector
    redactor: Redactor
    pipeline: IngestionPipeline
    redaction_store: RedactionMapStore
    store: QdrantStore
    retriever: HybridRetriever
    generator: Generator
    auditor: PostGenerationAuditor
    query_screener: InjectionScreener
    chunk_screener: InjectionScreener
    stats: StatsCollector


def get_state(request: Request) -> AppState:
    """Return the `AppState` the lifespan (or `build_app`) attached to the app."""
    state = request.app.state.guardian
    if not isinstance(state, AppState):
        raise RuntimeError("app.state.guardian is not set; create the app with create_app()")
    return state


def get_detector(state: Annotated[AppState, Depends(get_state)]) -> PiiDetector:
    return state.detector


def get_redactor(state: Annotated[AppState, Depends(get_state)]) -> Redactor:
    return state.redactor


def get_pipeline(state: Annotated[AppState, Depends(get_state)]) -> IngestionPipeline:
    return state.pipeline


def get_redaction_store(
    state: Annotated[AppState, Depends(get_state)],
) -> RedactionMapStore:
    return state.redaction_store


def get_store(state: Annotated[AppState, Depends(get_state)]) -> QdrantStore:
    return state.store


def get_retriever(state: Annotated[AppState, Depends(get_state)]) -> HybridRetriever:
    return state.retriever


def get_generator(state: Annotated[AppState, Depends(get_state)]) -> Generator:
    return state.generator


def get_auditor(state: Annotated[AppState, Depends(get_state)]) -> PostGenerationAuditor:
    return state.auditor


def get_query_screener(state: Annotated[AppState, Depends(get_state)]) -> InjectionScreener:
    return state.query_screener


def get_chunk_screener(state: Annotated[AppState, Depends(get_state)]) -> InjectionScreener:
    return state.chunk_screener


def get_stats(state: Annotated[AppState, Depends(get_state)]) -> StatsCollector:
    return state.stats


DetectorDep = Annotated[PiiDetector, Depends(get_detector)]
RedactorDep = Annotated[Redactor, Depends(get_redactor)]
PipelineDep = Annotated[IngestionPipeline, Depends(get_pipeline)]
RedactionStoreDep = Annotated[RedactionMapStore, Depends(get_redaction_store)]
StoreDep = Annotated[QdrantStore, Depends(get_store)]
RetrieverDep = Annotated[HybridRetriever, Depends(get_retriever)]
GeneratorDep = Annotated[Generator, Depends(get_generator)]
AuditorDep = Annotated[PostGenerationAuditor, Depends(get_auditor)]
QueryScreenerDep = Annotated[InjectionScreener, Depends(get_query_screener)]
ChunkScreenerDep = Annotated[InjectionScreener, Depends(get_chunk_screener)]
StatsDep = Annotated[StatsCollector, Depends(get_stats)]
