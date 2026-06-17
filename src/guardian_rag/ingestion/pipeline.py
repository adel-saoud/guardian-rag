"""End-to-end ingestion pipeline: detect, redact, chunk, persist redaction map.

Embedding and vector-store writes are exposed via a `Protocol` so Layer 3
can plug in the real Qdrant adapter and Layer 2 tests can use a stub. The
pipeline orchestrates the steps in the only safe order:

    1. Detect PII on the raw document.
    2. Redact spans with typed placeholders.
    3. Persist the redaction map separately (never embedded).
    4. Chunk the redacted text into parents + children.
    5. Hand parent + child chunks to the vector sink.

Only clean (redacted) text reaches the vector sink. The original document
never leaves this function.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog

from guardian_rag.ingestion.chunker import HierarchicalChunker
from guardian_rag.ingestion.models import ChildChunk, Document, IngestionResult, ParentChunk
from guardian_rag.ingestion.redaction_store import RedactionMapStore
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.redactor import Redactor

logger = structlog.get_logger(__name__)


@runtime_checkable
class VectorSink(Protocol):
    """Where chunks land after redaction and chunking.

    Layer 3 supplies a Qdrant-backed implementation. Tests can supply a
    no-op or in-memory fake without depending on Qdrant.
    """

    def upsert(
        self,
        parent_chunks: list[ParentChunk],
        child_chunks: list[ChildChunk],
    ) -> None:
        """Persist parents (as payload) and children (as vectors)."""
        ...


class NullVectorSink:
    """No-op sink, useful for Layer 2 tests and dry-run ingestion."""

    def __init__(self) -> None:
        self.parents: list[ParentChunk] = []
        self.children: list[ChildChunk] = []

    def upsert(
        self,
        parent_chunks: list[ParentChunk],
        child_chunks: list[ChildChunk],
    ) -> None:
        self.parents.extend(parent_chunks)
        self.children.extend(child_chunks)


class IngestionPipeline:
    """Glue the PII detector, redactor, chunker, and vector sink together.

    Parameters
    ----------
    detector:
        The three-stage PII detector. Required.
    redactor:
        Typed-placeholder redactor. Required.
    chunker:
        Hierarchical chunker. Required.
    redaction_store:
        Filesystem store for redaction maps. Required.
    vector_sink:
        Anything implementing `VectorSink`. Default `NullVectorSink` so the
        pipeline can run dry (useful for the eval harness).
    """

    def __init__(
        self,
        detector: PiiDetector,
        redactor: Redactor,
        chunker: HierarchicalChunker,
        redaction_store: RedactionMapStore,
        vector_sink: VectorSink | None = None,
    ) -> None:
        self._detector = detector
        self._redactor = redactor
        self._chunker = chunker
        self._redaction_store = redaction_store
        self._vector_sink = vector_sink if vector_sink is not None else NullVectorSink()

    def ingest(self, document: Document) -> IngestionResult:
        """Run the full pipeline over one document."""
        entities = self._detector.detect(document.text)
        redacted_text, redaction_map = self._redactor.redact(
            document.text,
            entities,
            document.document_id,
        )
        self._redaction_store.save(redaction_map)

        parents, children = self._chunker.split(redacted_text, document.document_id)
        self._vector_sink.upsert(parents, children)

        logger.info(
            "ingestion.complete",
            document_id=document.document_id,
            source=document.source,
            n_entities=len(entities),
            n_parents=len(parents),
            n_children=len(children),
        )

        return IngestionResult(
            document_id=document.document_id,
            redacted_text=redacted_text,
            redaction_map=redaction_map,
            parent_chunks=parents,
            child_chunks=children,
        )
