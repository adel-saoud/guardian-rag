"""DeBERTa validator tests, using a deterministic fake backend."""

from __future__ import annotations

from dataclasses import dataclass, field

from guardian_rag.pii.deberta_validator import DebertaValidator
from guardian_rag.pii.models import PiiEntity, PiiEntityType, PiiStage


@dataclass
class FakeBackend:
    """Score backend that returns a canned score for every call."""

    score_value: float = 0.9
    calls: list[tuple[str, str]] = field(default_factory=list)

    def score(self, text: str, hypothesis: str) -> float:
        self.calls.append((text, hypothesis))
        return self.score_value


def _entity(score: float, etype: PiiEntityType = PiiEntityType.PERSON) -> PiiEntity:
    return PiiEntity(
        entity_type=etype,
        start=0,
        end=10,
        score=score,
        text="John Smith",
        stage=PiiStage.GLINER,
    )


def test_high_confidence_entities_pass_through_without_validation():
    backend = FakeBackend(score_value=0.0)
    validator = DebertaValidator(backend, score_threshold=0.85)
    survivors = validator.validate([_entity(score=0.95)])
    assert len(survivors) == 1
    assert survivors[0].stage is PiiStage.GLINER
    assert backend.calls == []


def test_low_confidence_entity_accepted_when_deberta_agrees():
    backend = FakeBackend(score_value=0.8)
    validator = DebertaValidator(backend, score_threshold=0.85, accept_threshold=0.5)
    survivors = validator.validate([_entity(score=0.6)])
    assert len(survivors) == 1
    assert survivors[0].stage is PiiStage.GLINER_VALIDATED
    assert survivors[0].score == 0.8


def test_low_confidence_entity_dropped_when_deberta_disagrees():
    backend = FakeBackend(score_value=0.2)
    validator = DebertaValidator(backend, score_threshold=0.85, accept_threshold=0.5)
    assert validator.validate([_entity(score=0.6)]) == []


def test_threshold_bounds():
    import pytest

    backend = FakeBackend()
    with pytest.raises(ValueError, match="score_threshold"):
        DebertaValidator(backend, score_threshold=1.5)
    with pytest.raises(ValueError, match="accept_threshold"):
        DebertaValidator(backend, accept_threshold=-0.1)


def test_every_entity_type_has_a_hypothesis():
    from guardian_rag.pii.deberta_validator import ENTITY_TO_HYPOTHESIS

    for etype in PiiEntityType:
        assert etype.value in ENTITY_TO_HYPOTHESIS
