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

from guardian_rag.config import Settings
from guardian_rag.generation.generator import Generator
from guardian_rag.ingestion.pipeline import IngestionPipeline
from guardian_rag.ingestion.redaction_store import RedactionMapStore
from guardian_rag.pii.audit import PostGenerationAuditor
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.models import PiiEntity
from guardian_rag.pii.redactor import Redactor
from guardian_rag.retrieval.hybrid import HybridRetriever
from guardian_rag.retrieval.store import QdrantStore


@dataclass
class StatsCollector:
    """Thread-safe in-memory counters powering `/v1/stats`.

    Resets on process restart. The portfolio demo doesn't need durable
    metrics; production would back this with a real time-series sink.
    """

    n_documents: int = 0
    n_entities_total: int = 0
    n_leaks_caught: int = 0
    entity_type_breakdown: Counter[str] = field(default_factory=lambda: Counter())
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_ingest(self, entities: list[PiiEntity]) -> None:
        with self._lock:
            self.n_documents += 1
            self.n_entities_total += len(entities)
            for entity in entities:
                self.entity_type_breakdown[entity.entity_type.value] += 1

    def record_leak(self, entities: list[PiiEntity]) -> None:
        with self._lock:
            if entities:
                self.n_leaks_caught += 1
                for entity in entities:
                    self.entity_type_breakdown[entity.entity_type.value] += 1


@dataclass
class AppState:
    """Long-lived collaborators built at lifespan startup."""

    settings: Settings
    detector: PiiDetector
    redactor: Redactor
    pipeline: IngestionPipeline
    redaction_store: RedactionMapStore
    store: QdrantStore
    retriever: HybridRetriever
    generator: Generator
    auditor: PostGenerationAuditor
    stats: StatsCollector


def get_state(request: Request) -> AppState:
    state = request.app.state.guardian
    assert isinstance(state, AppState)
    return state


def get_settings(state: Annotated[AppState, Depends(get_state)]) -> Settings:
    return state.settings


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


def get_stats(state: Annotated[AppState, Depends(get_state)]) -> StatsCollector:
    return state.stats


SettingsDep = Annotated[Settings, Depends(get_settings)]
DetectorDep = Annotated[PiiDetector, Depends(get_detector)]
RedactorDep = Annotated[Redactor, Depends(get_redactor)]
PipelineDep = Annotated[IngestionPipeline, Depends(get_pipeline)]
RedactionStoreDep = Annotated[RedactionMapStore, Depends(get_redaction_store)]
StoreDep = Annotated[QdrantStore, Depends(get_store)]
RetrieverDep = Annotated[HybridRetriever, Depends(get_retriever)]
GeneratorDep = Annotated[Generator, Depends(get_generator)]
AuditorDep = Annotated[PostGenerationAuditor, Depends(get_auditor)]
StatsDep = Annotated[StatsCollector, Depends(get_stats)]
