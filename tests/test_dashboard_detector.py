"""The dashboard's detector: the API's three stages, or Presidio alone."""

from __future__ import annotations

from importlib.machinery import ModuleSpec

import pytest

from guardian_rag.config import Settings
from guardian_rag.dashboard import detector as dashboard_detector
from guardian_rag.pii import _real_backends
from guardian_rag.pii.gliner_recognizer import GlinerSpan
from guardian_rag.pii.models import PiiStage

TEXT = "Manager: Bob Chen."


class _FakeGlinerBackend:
    """Finds the one name in `TEXT` with high confidence, so DeBERTa does not fire."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[GlinerSpan]:
        start = text.index("Bob Chen")
        return [
            {"start": start, "end": start + 8, "text": "Bob Chen", "label": "person", "score": 0.99}
        ]


class _FakeDebertaBackend:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def score(self, text: str, hypothesis: str) -> float:
        return 1.0


def _missing(name: str) -> ModuleSpec | None:
    return None


def _present(name: str) -> ModuleSpec | None:
    return ModuleSpec(name, None)


def test_falls_back_to_presidio_without_ml_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dashboard_detector, "find_spec", _missing)
    assert not dashboard_detector.ml_extra_installed()
    entities = dashboard_detector.build_dashboard_detector(Settings()).detect(TEXT)
    assert all(entity.stage is PiiStage.PRESIDIO for entity in entities)


def test_detects_names_with_ml_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dashboard_detector, "find_spec", _present)
    monkeypatch.setattr(_real_backends, "RealGlinerBackend", _FakeGlinerBackend)
    monkeypatch.setattr(_real_backends, "RealDebertaValidatorBackend", _FakeDebertaBackend)
    entities = dashboard_detector.build_dashboard_detector(Settings()).detect(TEXT)
    assert [entity.text for entity in entities] == ["Bob Chen"]
