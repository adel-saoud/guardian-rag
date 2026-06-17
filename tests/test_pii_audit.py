"""Post-generation audit tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from guardian_rag.pii.audit import FAIL_CLOSED_REPLACEMENT, PostGenerationAuditor
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.gliner_recognizer import GlinerRecognizer, GlinerSpan
from guardian_rag.pii.presidio_engine import PresidioEngine
from guardian_rag.pii.redactor import Redactor


@dataclass
class FakeGliner:
    spans: list[GlinerSpan]

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float,
    ) -> list[GlinerSpan]:
        del text, labels, threshold
        return list(self.spans)


def _auditor(
    gliner_spans: list[GlinerSpan] | None = None,
    fail_closed: bool = True,
) -> PostGenerationAuditor:
    gliner = GlinerRecognizer(FakeGliner(spans=gliner_spans or []))
    detector = PiiDetector(presidio=PresidioEngine(), gliner=gliner)
    return PostGenerationAuditor(detector=detector, redactor=Redactor(), fail_closed=fail_closed)


def test_clean_answer_returns_unchanged():
    auditor = _auditor()
    result = auditor.audit("This is a perfectly safe answer.")
    assert result.clean is True
    assert result.answer == "This is a perfectly safe answer."
    assert result.entities == []
    assert result.fail_closed is False


def test_email_leak_caught_and_redacted():
    auditor = _auditor()
    answer = "Please contact alice@example.com for next steps."
    result = auditor.audit(answer)
    assert result.clean is False
    assert "alice@example.com" not in result.answer
    assert "[EMAIL_1]" in result.answer
    assert any(e.text == "alice@example.com" for e in result.entities)


def test_gliner_only_leak_caught():
    auditor = _auditor(
        gliner_spans=[
            {"start": 0, "end": 10, "text": "John Smith", "label": "person", "score": 0.95},
        ]
    )
    result = auditor.audit("John Smith approved the request.")
    assert result.clean is False
    assert "John Smith" not in result.answer
    assert "[PERSON_1]" in result.answer


def test_detector_exception_fails_closed_by_default():
    class BrokenDetector:
        def detect(self, text: str) -> list[object]:
            del text
            raise RuntimeError("backend down")

    auditor = PostGenerationAuditor(
        detector=BrokenDetector(),  # type: ignore[arg-type]
        redactor=Redactor(),
    )
    result = auditor.audit("anything")
    assert result.clean is False
    assert result.fail_closed is True
    assert result.answer == FAIL_CLOSED_REPLACEMENT
    assert result.entities == []


def test_detector_exception_re_raised_when_fail_closed_off():
    class BrokenDetector:
        def detect(self, text: str) -> list[object]:
            del text
            raise RuntimeError("backend down")

    auditor = PostGenerationAuditor(
        detector=BrokenDetector(),  # type: ignore[arg-type]
        redactor=Redactor(),
        fail_closed=False,
    )
    with pytest.raises(RuntimeError, match="backend down"):
        auditor.audit("anything")


def test_redactor_exception_also_fails_closed():
    class BrokenRedactor:
        def redact(self, *args: object, **kwargs: object) -> tuple[str, object]:
            del args, kwargs
            raise RuntimeError("redactor broken")

    auditor = PostGenerationAuditor(
        detector=PiiDetector(presidio=PresidioEngine()),
        redactor=BrokenRedactor(),  # type: ignore[arg-type]
    )
    result = auditor.audit("Contact alice@example.com please.")
    assert result.clean is False
    assert result.fail_closed is True
    assert result.answer == FAIL_CLOSED_REPLACEMENT


def test_empty_answer_is_clean():
    auditor = _auditor()
    result = auditor.audit("")
    assert result.clean is True
    assert result.answer == ""
