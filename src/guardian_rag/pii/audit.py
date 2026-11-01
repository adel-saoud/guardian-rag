"""Post-generation PII audit.

Most RAG privacy projects stop at ingestion. The LLM can still paraphrase
or infer personal data from clean chunks. This module re-runs the same
three-stage detector over every generated answer before it reaches the
user.

Failure semantics matter here: if the audit raises, we MUST NOT return
the raw answer. The default is fail-closed: an audit exception turns into
a redacted answer with `fail_closed=True`. Setting `fail_closed=False`
re-raises instead, which only makes sense in offline tooling like the
eval harness.
"""

from __future__ import annotations

import structlog

from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.models import AuditResult
from guardian_rag.pii.redactor import Redactor

logger = structlog.get_logger(__name__)


FAIL_CLOSED_REPLACEMENT = "[Audit failed: response redacted for safety]"
"""Returned instead of the answer when the audit itself crashed. Short and generic on purpose."""


class PostGenerationAuditor:
    """Run the PII detector over an LLM answer and redact any leak.

    Parameters
    ----------
    detector:
        The same three-stage detector ingestion uses. Sharing one instance
        means the audit catches exactly what would have been redacted at
        ingest time.
    redactor:
        Typed-placeholder redactor. Same instance as ingestion for
        consistent placeholder labels.
    fail_closed:
        When True (the default), any exception inside `audit()` is caught,
        the answer is replaced with `FAIL_CLOSED_REPLACEMENT`, and the
        result is returned with `fail_closed=True`. When False the
        exception is re-raised, which is useful for offline tooling.
    """

    def __init__(
        self,
        detector: PiiDetector,
        redactor: Redactor,
        fail_closed: bool = True,
    ) -> None:
        self._detector = detector
        self._redactor = redactor
        self._fail_closed = fail_closed

    def audit(self, answer: str, document_id: str = "answer") -> AuditResult:
        """Return an `AuditResult` for one LLM answer.

        On `clean=True` the returned `answer` is identical to the input.
        On `clean=False` it is the redacted version. On `fail_closed=True`
        the input is replaced with a safe placeholder.
        """
        try:
            entities = self._detector.detect(answer)
        except Exception:
            if not self._fail_closed:
                raise
            return _fail_closed(answer, "detect")

        if not entities:
            return AuditResult(clean=True, answer=answer)

        try:
            redacted, _map = self._redactor.redact(answer, entities, document_id)
        except Exception:
            if not self._fail_closed:
                raise
            return _fail_closed(answer, "redact")

        logger.warning(
            "audit.leak_detected",
            n_entities=len(entities),
            entity_types=sorted({e.entity_type.value for e in entities}),
        )
        return AuditResult(clean=False, answer=redacted, entities=entities)


def _fail_closed(answer: str, phase: str) -> AuditResult:
    """Log the audit crash and return the safe replacement, never the raw answer."""
    logger.exception(
        "audit.failed_closed",
        phase=phase,
        answer_preview=answer[:120],
    )
    return AuditResult(
        clean=False,
        answer=FAIL_CLOSED_REPLACEMENT,
        entities=[],
        fail_closed=True,
    )
