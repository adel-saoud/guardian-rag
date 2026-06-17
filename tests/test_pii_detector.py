"""Orchestrator tests for the three-stage PII detector."""

from __future__ import annotations

from dataclasses import dataclass, field

from guardian_rag.pii.deberta_validator import DebertaValidator
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.gliner_recognizer import GlinerRecognizer, GlinerSpan
from guardian_rag.pii.models import PiiEntityType, PiiStage
from guardian_rag.pii.presidio_engine import PresidioEngine


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
class FakeDeberta:
    score_value: float = 0.9

    def score(self, text: str, hypothesis: str) -> float:
        del text, hypothesis
        return self.score_value


def test_detector_returns_presidio_and_gliner_entities_sorted():
    text = "John Smith emailed alice@example.com about the project."
    presidio = PresidioEngine()
    gliner = GlinerRecognizer(
        FakeGliner(
            spans=[
                {"start": 0, "end": 10, "text": "John Smith", "label": "person", "score": 0.95},
            ]
        )
    )
    detector = PiiDetector(presidio=presidio, gliner=gliner)
    entities = detector.detect(text)

    types = {e.entity_type for e in entities}
    assert PiiEntityType.PERSON in types
    assert PiiEntityType.EMAIL in types
    starts = [e.start for e in entities]
    assert starts == sorted(starts)


def test_detector_without_gliner_returns_only_presidio():
    text = "Email: jane@example.com"
    detector = PiiDetector(presidio=PresidioEngine())
    entities = detector.detect(text)
    assert all(e.stage is PiiStage.PRESIDIO for e in entities)


def test_detector_with_validator_filters_low_confidence_gliner_spans():
    text = "John works at Acme."
    gliner = GlinerRecognizer(
        FakeGliner(
            spans=[
                # Low GLiNER confidence: will be sent to DeBERTa.
                {"start": 0, "end": 4, "text": "John", "label": "person", "score": 0.40},
            ]
        )
    )
    # DeBERTa returns 0.1 -> below accept_threshold of 0.5, so the entity
    # is dropped.
    validator = DebertaValidator(FakeDeberta(score_value=0.1), score_threshold=0.85)
    detector = PiiDetector(presidio=PresidioEngine(), gliner=gliner, validator=validator)
    entities = detector.detect(text)
    assert entities == []


def test_detector_empty_text_short_circuits():
    detector = PiiDetector(presidio=PresidioEngine())
    assert detector.detect("") == []
