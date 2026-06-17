"""PII detection and redaction pipeline."""

from __future__ import annotations

from guardian_rag.pii.audit import FAIL_CLOSED_REPLACEMENT, PostGenerationAuditor
from guardian_rag.pii.deberta_validator import DebertaValidator, DebertaValidatorBackend
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.gliner_recognizer import (
    DEFAULT_PII_LABELS,
    GlinerBackend,
    GlinerRecognizer,
    GlinerSpan,
)
from guardian_rag.pii.models import (
    AuditResult,
    PiiEntity,
    PiiEntityType,
    PiiStage,
    RedactionMap,
)
from guardian_rag.pii.presidio_engine import PresidioEngine
from guardian_rag.pii.redactor import Redactor

__all__ = [
    "DEFAULT_PII_LABELS",
    "FAIL_CLOSED_REPLACEMENT",
    "AuditResult",
    "DebertaValidator",
    "DebertaValidatorBackend",
    "GlinerBackend",
    "GlinerRecognizer",
    "GlinerSpan",
    "PiiDetector",
    "PiiEntity",
    "PiiEntityType",
    "PiiStage",
    "PostGenerationAuditor",
    "PresidioEngine",
    "RedactionMap",
    "Redactor",
]
