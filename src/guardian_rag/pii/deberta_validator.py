"""Stage 3 detector: conditional DeBERTa zero-shot validation.

When GLiNER produces a span whose confidence is below the configured
threshold, we ask a DeBERTa zero-shot NLI model "is this span really a
{label}?". This is the safety net for borderline detections without paying
DeBERTa inference cost on every span.

The backend is exposed as a `Protocol` so tests can drop in a deterministic
fake. The real backend lives in `_real_backends.py`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog

from guardian_rag.pii.models import PiiEntity, PiiStage

logger = structlog.get_logger(__name__)


# Each domain PII type gets a short, human-readable label that DeBERTa NLI
# can score against. The wording matters: "personal name" works better than
# "PERSON" because the model was trained on natural language hypotheses.
ENTITY_TO_HYPOTHESIS: dict[str, str] = {
    "PERSON": "a personal name",
    "EMAIL": "an email address",
    "PHONE": "a phone number",
    "ADDRESS": "a postal address",
    "DATE_OF_BIRTH": "a date of birth",
    "ID_NUMBER": "an identification number",
    "BANK_ACCOUNT": "a bank account number",
    "CREDIT_CARD": "a credit card number",
    "SOCIAL_SECURITY": "a social security number",
    "IP_ADDRESS": "an IP address",
    "URL": "a URL",
    "ORGANIZATION": "an organisation name",
    "LOCATION": "a geographic location",
}


@runtime_checkable
class DebertaValidatorBackend(Protocol):
    """Pluggable DeBERTa backend so tests can avoid loading weights."""

    def score(self, text: str, hypothesis: str) -> float:
        """Probability in [0, 1] that ``hypothesis`` entails ``text``."""
        ...


class DebertaValidator:
    """Validate low-confidence GLiNER spans against a zero-shot NLI model.

    Parameters
    ----------
    backend:
        Anything implementing `DebertaValidatorBackend`.
    score_threshold:
        GLiNER spans with score below this are sent to DeBERTa for
        validation. Spans with score at or above this are accepted as-is.
    accept_threshold:
        DeBERTa probability above which a validated span is kept. Below
        this the span is dropped (treated as a GLiNER false positive).
    """

    def __init__(
        self,
        backend: DebertaValidatorBackend,
        score_threshold: float = 0.85,
        accept_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if not 0.0 <= accept_threshold <= 1.0:
            raise ValueError("accept_threshold must be in [0, 1]")
        self._backend = backend
        self._score_threshold = score_threshold
        self._accept_threshold = accept_threshold

    def validate(self, entities: list[PiiEntity]) -> list[PiiEntity]:
        """Return the subset of ``entities`` that survive validation.

        High-confidence spans (>= score_threshold) pass through unchanged.
        Low-confidence spans are scored by DeBERTa. Spans whose DeBERTa
        probability is below `accept_threshold` are dropped. Surviving
        validated spans have their stage marked GLINER_VALIDATED and their
        score replaced with the DeBERTa probability.
        """
        survivors: list[PiiEntity] = []
        for entity in entities:
            if entity.score >= self._score_threshold:
                survivors.append(entity)
                continue
            hypothesis = ENTITY_TO_HYPOTHESIS.get(entity.entity_type.value)
            if hypothesis is None:
                # No hypothesis means we cannot validate; keep the span to
                # err on the privacy-safe side.
                survivors.append(entity)
                continue
            probability = float(self._backend.score(entity.text, hypothesis))
            if probability < self._accept_threshold:
                logger.debug(
                    "deberta.validator.dropped",
                    text=entity.text,
                    entity_type=entity.entity_type.value,
                    gliner_score=entity.score,
                    deberta_score=probability,
                )
                continue
            survivors.append(
                entity.model_copy(
                    update={
                        "score": probability,
                        "stage": PiiStage.GLINER_VALIDATED,
                    }
                )
            )
        return survivors
