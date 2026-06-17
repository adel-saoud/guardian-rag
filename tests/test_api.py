"""End-to-end API tests with a pre-built AppState and fakes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from guardian_rag.api.app import build_app
from guardian_rag.api.dependencies import AppState, StatsCollector
from guardian_rag.config import Settings
from guardian_rag.generation.citation import GroundedAnswer
from guardian_rag.ingestion.chunker import HierarchicalChunker
from guardian_rag.ingestion.models import ParentChunk
from guardian_rag.ingestion.pipeline import IngestionPipeline
from guardian_rag.ingestion.redaction_store import RedactionMapStore
from guardian_rag.pii.audit import PostGenerationAuditor
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.presidio_engine import PresidioEngine
from guardian_rag.pii.redactor import Redactor
from guardian_rag.retrieval.embedder import Embedder
from guardian_rag.retrieval.hybrid import HybridRetriever
from guardian_rag.retrieval.models import RetrievalHit, RetrievalResult
from guardian_rag.retrieval.store import QdrantStore


@dataclass
class HashEmbedder:
    """Deterministic embedder for hermetic tests."""

    dim: int = 16

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            h = abs(hash(text))
            v = [((h >> i) & 0xFF) / 255.0 for i in range(self.dim)]
            vectors.append(v)
        return vectors


@dataclass
class FakeGenerator:
    """Mimics the Generator surface without making network calls."""

    canned: str = "Email contact is [EMAIL_1] [1]."
    model: str = "fake"
    calls: list[tuple[str, list[ParentChunk]]] = field(default_factory=list)

    async def generate(self, query: str, parents: list[ParentChunk]) -> GroundedAnswer:
        self.calls.append((query, list(parents)))
        if not parents:
            return GroundedAnswer(answer="I do not know based on the provided context.")
        return GroundedAnswer(answer=self.canned, cited_parent_ids=[parents[0].chunk_id])


@dataclass
class FakeRetriever:
    """Retriever stub returning canned parents."""

    parents: list[ParentChunk] = field(default_factory=list)

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        document_id: str | None = None,
    ) -> RetrievalResult:
        del top_k, document_id
        hits = [
            RetrievalHit(
                chunk_id=str(uuid.uuid4()),
                document_id="doc-1",
                parent_id=parent.chunk_id,
                text=parent.text[:120],
                order=0,
                score=0.9,
            )
            for parent in self.parents
        ]
        return RetrievalResult(query=query, children=hits, parents=self.parents)

    def refresh(self) -> None:  # used by the ingest route
        pass


def _build_state(
    tmp_path: Path,
    generator_text: str = "Email contact is [EMAIL_1] [1].",
) -> AppState:
    settings = Settings(
        redaction_map_dir=str(tmp_path / "redaction_maps"),
        qdrant_url=":memory:",
        ollama_base_url="http://ollama",
    )
    detector = PiiDetector(presidio=PresidioEngine())
    redactor = Redactor()
    redaction_store = RedactionMapStore(settings.redaction_map_dir)
    embedder = Embedder(HashEmbedder(dim=16), dim=16)
    client = QdrantClient(location=":memory:")
    store = QdrantStore(
        client=client,
        embedder=embedder,
        collection_name="api_t",
        recreate=True,
    )
    pipeline = IngestionPipeline(
        detector=detector,
        redactor=redactor,
        chunker=HierarchicalChunker(parent_tokens=128, child_tokens=64, child_overlap=8),
        redaction_store=redaction_store,
        vector_sink=store,
    )
    auditor = PostGenerationAuditor(detector=detector, redactor=redactor)
    return AppState(
        settings=settings,
        detector=detector,
        redactor=redactor,
        pipeline=pipeline,
        redaction_store=redaction_store,
        store=store,
        retriever=HybridRetriever(store),  # real, will be overridden below in some tests
        generator=FakeGenerator(canned=generator_text),  # type: ignore[arg-type]
        auditor=auditor,
        stats=StatsCollector(),
    )


@pytest.fixture
def state(tmp_path: Path) -> AppState:
    return _build_state(tmp_path)


@pytest.fixture
def client(state: AppState) -> TestClient:
    return TestClient(build_app(state))


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_redacts_and_returns_counts(client: TestClient, state: AppState):
    body = {
        "document_id": "doc-1",
        "text": "Email alice@example.com for next steps.",
        "source": "test",
    }
    response = client.post("/v1/ingest", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "doc-1"
    assert data["n_entities"] >= 1
    assert "[EMAIL_1]" in data["redacted_preview"]
    assert "alice@example.com" not in data["redacted_preview"]
    assert state.stats.n_documents == 1
    # The redaction map is persisted to the on-disk store.
    assert state.redaction_store.exists("doc-1")


def test_audit_returns_redaction_map(client: TestClient):
    client.post(
        "/v1/ingest",
        json={"document_id": "doc-2", "text": "Reach jane@example.com today."},
    )
    response = client.get("/v1/audit/doc-2")
    assert response.status_code == 200
    payload = response.json()
    assert payload["redaction_map"]["document_id"] == "doc-2"
    assert "[EMAIL_1]" in payload["redaction_map"]["placeholders"]


def test_audit_missing_document_returns_404(client: TestClient):
    response = client.get("/v1/audit/never-ingested")
    assert response.status_code == 404


def test_audit_rejects_unsafe_id(client: TestClient):
    response = client.get("/v1/audit/..%2Fescape")
    # The unsafe id check raises ValueError → 400.
    assert response.status_code in {400, 404}


def test_query_sets_audit_headers_and_returns_citations(
    tmp_path: Path,
):
    state = _build_state(tmp_path, generator_text="Contact is alice@example.com [1].")
    parent = ParentChunk(
        chunk_id=str(uuid.uuid4()),
        document_id="doc-1",
        text="Reach alice@example.com please.",
        order=0,
    )
    state.retriever = FakeRetriever(parents=[parent])  # type: ignore[assignment]
    app = build_app(state)
    client = TestClient(app)

    response = client.post("/v1/query", json={"query": "Who do I contact?"})
    assert response.status_code == 200
    # The generator's answer leaked an email; the audit must catch it.
    assert response.headers["X-Guardian-Pii-Audit"] == "redacted"
    assert response.headers["X-Guardian-Chunks-Used"] == "1"
    assert response.headers["X-Guardian-Request-Id"].startswith("chatcmpl-")
    data = response.json()
    assert "alice@example.com" not in data["answer"]
    assert data["audit_clean"] is False
    assert data["citations"]


def test_query_redacts_pii_in_the_query_before_embedding(tmp_path: Path):
    state = _build_state(tmp_path)
    parent = ParentChunk(
        chunk_id=str(uuid.uuid4()),
        document_id="doc-1",
        text="Generic context.",
        order=0,
    )
    state.retriever = FakeRetriever(parents=[parent])  # type: ignore[assignment]
    app = build_app(state)
    client = TestClient(app)

    response = client.post(
        "/v1/query",
        json={"query": "Find docs about alice@example.com"},
    )
    assert response.status_code == 200
    fake = state.generator
    # The generator should have received the query with the email replaced.
    assert isinstance(fake, FakeGenerator)
    last_query, _ = fake.calls[-1]
    assert "alice@example.com" not in last_query
    assert "[EMAIL_1]" in last_query


def test_stats_aggregates_counts(client: TestClient):
    client.post(
        "/v1/ingest",
        json={"document_id": "s1", "text": "alice@example.com and bob@example.com"},
    )
    response = client.get("/v1/stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["n_documents"] == 1
    assert payload["n_entities_total"] >= 2
    assert payload["entity_type_breakdown"]["EMAIL"] >= 2
