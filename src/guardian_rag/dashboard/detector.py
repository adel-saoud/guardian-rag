"""The PII detector the dashboard and its screenshot script share.

The dashboard shows the same three stages the API runs, so what a visitor sees
redacted is what Guardian redacts. Without the `ml` extra it falls back to
Presidio alone and says so, instead of failing to start.
"""

from __future__ import annotations

from importlib.util import find_spec

from guardian_rag.config import Settings
from guardian_rag.pii.deberta_validator import DebertaValidator
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.gliner_recognizer import GlinerRecognizer
from guardian_rag.pii.presidio_engine import PresidioEngine


def ml_extra_installed() -> bool:
    """True when `gliner` and `transformers` import, i.e. `uv sync --extra ml` ran."""
    return find_spec("gliner") is not None and find_spec("transformers") is not None


def build_dashboard_detector(settings: Settings) -> PiiDetector:
    """Presidio, GLiNER and the DeBERTa validator as in the API, or Presidio alone."""
    presidio = PresidioEngine()
    if not ml_extra_installed():
        return PiiDetector(presidio=presidio)
    from guardian_rag.pii._real_backends import RealDebertaValidatorBackend, RealGlinerBackend

    return PiiDetector(
        presidio=presidio,
        gliner=GlinerRecognizer(
            backend=RealGlinerBackend(settings.gliner_model),
            threshold=settings.gliner_threshold,
        ),
        validator=DebertaValidator(
            backend=RealDebertaValidatorBackend(settings.deberta_model),
            score_threshold=settings.deberta_threshold,
        ),
    )
