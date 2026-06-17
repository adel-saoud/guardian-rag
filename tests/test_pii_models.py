"""Domain-model invariants for the PII pipeline."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guardian_rag.pii.models import (
    AuditResult,
    PiiEntity,
    PiiEntityType,
    PiiStage,
    RedactionMap,
)


def _entity(start: int, end: int, **overrides: object) -> PiiEntity:
    defaults: dict[str, object] = {
        "entity_type": PiiEntityType.PERSON,
        "start": start,
        "end": end,
        "score": 0.9,
        "text": "John",
        "stage": PiiStage.GLINER,
    }
    defaults.update(overrides)
    return PiiEntity(**defaults)  # type: ignore[arg-type]


def test_entity_rejects_inverted_offsets():
    with pytest.raises(ValidationError):
        _entity(10, 5)


def test_entity_rejects_zero_length():
    with pytest.raises(ValidationError):
        _entity(5, 5)


def test_entity_overlap_detection():
    a = _entity(0, 10)
    b = _entity(5, 15)
    c = _entity(20, 30)
    assert a.overlaps(b)
    assert b.overlaps(a)
    assert not a.overlaps(c)


def test_entity_length():
    assert _entity(3, 11).length == 8


def test_redaction_map_reverse_round_trips():
    m = RedactionMap(
        document_id="doc-1",
        placeholders={"[PERSON_1]": "John Smith", "[EMAIL_1]": "john@example.com"},
    )
    rev = m.reverse()
    assert rev["John Smith"] == "[PERSON_1]"
    assert rev["john@example.com"] == "[EMAIL_1]"


def test_audit_result_clean_default():
    res = AuditResult(clean=True, answer="hello world")
    assert res.entities == []
    assert res.fail_closed is False
