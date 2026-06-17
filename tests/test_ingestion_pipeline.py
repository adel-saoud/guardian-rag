"""End-to-end ingestion pipeline tests with mocked PII detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from guardian_rag.ingestion.chunker import HierarchicalChunker
from guardian_rag.ingestion.models import ChildChunk, Document, ParentChunk
from guardian_rag.ingestion.pipeline import IngestionPipeline, NullVectorSink
from guardian_rag.ingestion.redaction_store import RedactionMapStore
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.gliner_recognizer import GlinerRecognizer, GlinerSpan
from guardian_rag.pii.presidio_engine import PresidioEngine
from guardian_rag.pii.redactor import Redactor


@dataclass
class FakeGliner:
    spans: list[GlinerSpan] = field(default_factory=list)

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float,
    ) -> list[GlinerSpan]:
        del text, labels, threshold
        return list(self.spans)


@dataclass
class RecordingSink:
    parents: list[ParentChunk] = field(default_factory=list)
    children: list[ChildChunk] = field(default_factory=list)

    def upsert(
        self,
        parent_chunks: list[ParentChunk],
        child_chunks: list[ChildChunk],
    ) -> None:
        self.parents.extend(parent_chunks)
        self.children.extend(child_chunks)


def _pipeline(
    tmp_path: Path,
    gliner: FakeGliner | None = None,
    sink: object | None = None,
) -> IngestionPipeline:
    detector = PiiDetector(
        presidio=PresidioEngine(),
        gliner=GlinerRecognizer(gliner if gliner is not None else FakeGliner()),
    )
    return IngestionPipeline(
        detector=detector,
        redactor=Redactor(),
        chunker=HierarchicalChunker(parent_tokens=64, child_tokens=24, child_overlap=4),
        redaction_store=RedactionMapStore(tmp_path / "redaction_maps"),
        vector_sink=sink,  # type: ignore[arg-type]
    )


def test_pipeline_redacts_and_chunks(tmp_path: Path):
    pipeline = _pipeline(
        tmp_path,
        gliner=FakeGliner(
            spans=[
                {"start": 0, "end": 10, "text": "John Smith", "label": "person", "score": 0.95},
            ]
        ),
    )
    doc = Document(
        document_id="hr-policy-1",
        text="John Smith joined HR on 2024-01-01. His email is john.smith@example.com.",
        source="hr",
    )
    result = pipeline.ingest(doc)

    assert "[PERSON_1]" in result.redacted_text
    assert "John Smith" not in result.redacted_text
    assert "[EMAIL_1]" in result.redacted_text
    assert result.n_entities >= 2
    assert result.parent_chunks
    assert result.child_chunks
    parent_ids = {p.chunk_id for p in result.parent_chunks}
    assert all(c.parent_id in parent_ids for c in result.child_chunks)


def test_pipeline_persists_redaction_map(tmp_path: Path):
    pipeline = _pipeline(tmp_path)
    doc = Document(
        document_id="hr-policy-2",
        text="Send updates to alice@example.com.",
    )
    pipeline.ingest(doc)
    store = RedactionMapStore(tmp_path / "redaction_maps")
    assert store.exists("hr-policy-2")
    loaded = store.load("hr-policy-2")
    assert any(e.text == "alice@example.com" for e in loaded.entities)


def test_pipeline_routes_clean_chunks_to_sink(tmp_path: Path):
    sink = RecordingSink()
    pipeline = _pipeline(tmp_path, sink=sink)
    doc = Document(
        document_id="hr-policy-3",
        text="Contact dpo@example.com for any data subject request.",
    )
    pipeline.ingest(doc)
    assert sink.parents
    # Nothing the sink received should contain the raw PII.
    for parent in sink.parents:
        assert "dpo@example.com" not in parent.text
    for child in sink.children:
        assert "dpo@example.com" not in child.text


def test_null_sink_collects_chunks(tmp_path: Path):
    sink = NullVectorSink()
    pipeline = _pipeline(tmp_path, sink=sink)
    pipeline.ingest(Document(document_id="doc-x", text="Hello world."))
    assert sink.parents
