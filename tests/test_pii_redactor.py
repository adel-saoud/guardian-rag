"""Typed-placeholder redactor tests."""

from __future__ import annotations

from guardian_rag.pii.models import PiiEntity, PiiEntityType, PiiStage
from guardian_rag.pii.redactor import Redactor


def _entity(start: int, end: int, text: str, etype: PiiEntityType, score: float = 0.9) -> PiiEntity:
    return PiiEntity(
        entity_type=etype,
        start=start,
        end=end,
        score=score,
        text=text,
        stage=PiiStage.GLINER,
    )


def test_empty_entities_round_trip_unchanged():
    text = "Nothing to redact here."
    redacted, mapping = Redactor().redact(text, [], "doc-1")
    assert redacted == text
    assert mapping.placeholders == {}
    assert mapping.entities == []


def test_typed_placeholder_format():
    text = "John Smith works here."
    entities = [_entity(0, 10, "John Smith", PiiEntityType.PERSON)]
    redacted, mapping = Redactor().redact(text, entities, "doc-1")
    assert redacted == "[PERSON_1] works here."
    assert mapping.placeholders == {"[PERSON_1]": "John Smith"}


def test_same_span_resolves_to_same_placeholder():
    text = "John Smith met John Smith again."
    entities = [
        _entity(0, 10, "John Smith", PiiEntityType.PERSON),
        _entity(15, 25, "John Smith", PiiEntityType.PERSON),
    ]
    redacted, mapping = Redactor().redact(text, entities, "doc-1")
    assert redacted == "[PERSON_1] met [PERSON_1] again."
    assert mapping.placeholders == {"[PERSON_1]": "John Smith"}


def test_different_spans_get_distinct_indices():
    text = "Alice met Bob today."
    entities = [
        _entity(0, 5, "Alice", PiiEntityType.PERSON),
        _entity(10, 13, "Bob", PiiEntityType.PERSON),
    ]
    redacted, _mapping = Redactor().redact(text, entities, "doc-1")
    assert redacted == "[PERSON_1] met [PERSON_2] today."


def test_different_types_use_separate_index_space():
    text = "Alice emailed alice@ex.com."
    entities = [
        _entity(0, 5, "Alice", PiiEntityType.PERSON),
        _entity(14, 26, "alice@ex.com", PiiEntityType.EMAIL),
    ]
    redacted, _mapping = Redactor().redact(text, entities, "doc-1")
    assert redacted == "[PERSON_1] emailed [EMAIL_1]."


def test_overlapping_entities_dedupe_keeping_higher_score():
    text = "John Smith works here."
    entities = [
        _entity(0, 10, "John Smith", PiiEntityType.PERSON, score=0.6),
        _entity(0, 4, "John", PiiEntityType.PERSON, score=0.9),
    ]
    redacted, _ = Redactor().redact(text, entities, "doc-1")
    # The higher-scoring entity wins, which is the 4-char "John" span.
    assert redacted.startswith("[PERSON_1] Smith")


def test_redaction_preserves_structure_outside_entities():
    text = "Contact: John (HR) john@ex.com, Friday only."
    entities = [
        _entity(9, 13, "John", PiiEntityType.PERSON),
        _entity(19, 30, "john@ex.com", PiiEntityType.EMAIL),
    ]
    redacted, _ = Redactor().redact(text, entities, "doc-1")
    assert "(HR)" in redacted
    assert "Friday only." in redacted
    assert "[PERSON_1]" in redacted
    assert "[EMAIL_1]" in redacted
