"""Stage 2 detector: GLiNER NER for unstructured PII.

GLiNER is a span-based NER model that beats regex on contextual entities
like names, organisations, and locations. We expose the backend as a
`Protocol` so tests can plug in a deterministic fake without loading the
real model. The real backend lives in `_real_backends.py`.

The recogniser converts GLiNER spans into `PiiEntity` records using the
domain enum, dropping any spans whose label does not map onto a Guardian
PII type.
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

from guardian_rag.pii.models import PiiEntity, PiiEntityType, PiiStage


class GlinerSpan(TypedDict):
    """Wire shape of a single GLiNER prediction.

    Matches the dict the official ``gliner`` package returns from
    `predict_entities`. We use a `TypedDict` here so backends do not have
    to depend on pydantic.
    """

    start: int
    end: int
    text: str
    label: str
    score: float


GLINER_LABEL_TO_DOMAIN: dict[str, PiiEntityType] = {
    "person": PiiEntityType.PERSON,
    "email": PiiEntityType.EMAIL,
    "phone number": PiiEntityType.PHONE,
    "address": PiiEntityType.ADDRESS,
    "date of birth": PiiEntityType.DATE_OF_BIRTH,
    "id number": PiiEntityType.ID_NUMBER,
    "bank account": PiiEntityType.BANK_ACCOUNT,
    "credit card": PiiEntityType.CREDIT_CARD,
    "social security": PiiEntityType.SOCIAL_SECURITY,
    "ip address": PiiEntityType.IP_ADDRESS,
    "url": PiiEntityType.URL,
    "organization": PiiEntityType.ORGANIZATION,
    "location": PiiEntityType.LOCATION,
}
"""GLiNER label (lower case by convention) to domain PII type. Other labels are ignored."""

DEFAULT_PII_LABELS: tuple[str, ...] = tuple(GLINER_LABEL_TO_DOMAIN.keys())
"""Candidate labels sent to GLiNER by default: every label the map can translate."""


@runtime_checkable
class GlinerBackend(Protocol):
    """Pluggable GLiNER backend so tests can avoid downloading weights."""

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float,
    ) -> list[GlinerSpan]:
        """Run GLiNER over ``text`` for the given candidate ``labels``."""
        ...


class GlinerRecognizer:
    """Convert GLiNER predictions into `PiiEntity` records.

    Parameters
    ----------
    backend:
        Anything implementing the `GlinerBackend` protocol.
    labels:
        Candidate PII labels. Defaults to the full Guardian set.
    threshold:
        Minimum score for GLiNER to surface a span. Lower this to be more
        permissive (more entities, more false positives) and raise it to
        be stricter.
    """

    def __init__(
        self,
        backend: GlinerBackend,
        labels: tuple[str, ...] = DEFAULT_PII_LABELS,
        threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._backend = backend
        self._labels = list(labels)
        self._threshold = threshold

    def detect(self, text: str) -> list[PiiEntity]:
        """Return GLiNER spans as `PiiEntity` records in source order, unknown labels dropped."""
        if not text:
            return []
        spans = self._backend.predict_entities(text, self._labels, self._threshold)
        entities: list[PiiEntity] = []
        for span in spans:
            label = span["label"].lower()
            if label not in GLINER_LABEL_TO_DOMAIN:
                continue
            start = int(span["start"])
            end = int(span["end"])
            if end <= start:
                continue
            entities.append(
                PiiEntity(
                    entity_type=GLINER_LABEL_TO_DOMAIN[label],
                    start=start,
                    end=end,
                    score=float(span["score"]),
                    text=text[start:end],
                    stage=PiiStage.GLINER,
                )
            )
        entities.sort(key=lambda e: (e.start, e.end))
        return entities
