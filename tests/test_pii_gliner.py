"""GLiNER recogniser tests, using a deterministic fake backend."""

from __future__ import annotations

from dataclasses import dataclass, field

from guardian_rag.pii.gliner_recognizer import (
    DEFAULT_PII_LABELS,
    GlinerRecognizer,
    GlinerSpan,
)
from guardian_rag.pii.models import PiiEntityType, PiiStage


@dataclass
class FakeBackend:
    """Deterministic backend that replays canned spans regardless of input."""

    spans: list[GlinerSpan] = field(default_factory=list)
    calls: list[tuple[str, tuple[str, ...], float]] = field(default_factory=list)

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float,
    ) -> list[GlinerSpan]:
        self.calls.append((text, tuple(labels), threshold))
        return list(self.spans)


def test_maps_person_label_to_domain_entity():
    text = "John Smith reported to HR."
    backend = FakeBackend(
        spans=[
            {"start": 0, "end": 10, "text": "John Smith", "label": "person", "score": 0.92},
        ]
    )
    rec = GlinerRecognizer(backend)
    entities = rec.detect(text)
    assert len(entities) == 1
    assert entities[0].entity_type is PiiEntityType.PERSON
    assert entities[0].stage is PiiStage.GLINER
    assert entities[0].text == "John Smith"


def test_drops_unknown_labels():
    text = "Some text"
    backend = FakeBackend(
        spans=[
            {"start": 0, "end": 4, "text": "Some", "label": "unknown_label", "score": 0.9},
        ]
    )
    rec = GlinerRecognizer(backend)
    assert rec.detect(text) == []


def test_default_labels_match_domain_map():
    backend = FakeBackend()
    rec = GlinerRecognizer(backend)
    rec.detect("text")
    assert backend.calls[0][1] == DEFAULT_PII_LABELS


def test_empty_text_short_circuits():
    backend = FakeBackend(
        spans=[
            {"start": 0, "end": 4, "text": "Some", "label": "person", "score": 0.9},
        ]
    )
    rec = GlinerRecognizer(backend)
    assert rec.detect("") == []
    assert backend.calls == []


def test_threshold_propagates_to_backend():
    backend = FakeBackend()
    rec = GlinerRecognizer(backend, threshold=0.42)
    rec.detect("anything")
    assert backend.calls[0][2] == 0.42


def test_entities_are_sorted_by_start():
    text = "A B C D E F"
    backend = FakeBackend(
        spans=[
            {"start": 6, "end": 7, "text": "D", "label": "person", "score": 0.9},
            {"start": 0, "end": 1, "text": "A", "label": "person", "score": 0.9},
        ]
    )
    rec = GlinerRecognizer(backend)
    entities = rec.detect(text)
    assert [e.start for e in entities] == [0, 6]
