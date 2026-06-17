"""Domain models for the PII detection pipeline.

Every cross-module boundary uses frozen pydantic models. Detection stages
attach their identity to each entity so downstream audit reports can show
which stage flagged what.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PiiStage(StrEnum):
    """Which detection stage surfaced a candidate entity."""

    PRESIDIO = "presidio"
    GLINER = "gliner"
    GLINER_VALIDATED = "gliner_validated"


class PiiEntityType(StrEnum):
    """Canonical PII entity types Guardian RAG recognises.

    The string values double as the placeholder label, so a `PERSON` entity
    becomes `[PERSON_1]` after redaction. Adding a new value automatically
    extends placeholder generation.
    """

    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    ADDRESS = "ADDRESS"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    ID_NUMBER = "ID_NUMBER"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    CREDIT_CARD = "CREDIT_CARD"
    SOCIAL_SECURITY = "SOCIAL_SECURITY"
    IP_ADDRESS = "IP_ADDRESS"
    URL = "URL"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"


class PiiEntity(BaseModel):
    """A single PII span detected in source text.

    `start` and `end` are character offsets into the original (unredacted) text.
    `score` is the detector's confidence in [0, 1]. `stage` records which
    detection stage surfaced the entity, which lets the audit report attribute
    detections back to Presidio, GLiNER, or the validator.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_type: PiiEntityType
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    text: str
    stage: PiiStage

    @model_validator(mode="after")
    def _check_offsets(self) -> PiiEntity:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        return self

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: PiiEntity) -> bool:
        """True if this entity shares any character with ``other``."""
        return self.start < other.end and other.start < self.end


class RedactionMap(BaseModel):
    """Mapping between placeholder tokens and the original PII spans.

    Stored separately from the redacted text. Never embedded into the vector
    store. Used to render before/after views in the dashboard and to support
    audit replay.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    placeholders: dict[str, str] = Field(default_factory=lambda: {})
    """Mapping from placeholder token (e.g. ``[PERSON_1]``) to original text."""

    entities: list[PiiEntity] = Field(default_factory=lambda: [])
    """All entities that were redacted, in source-document order."""

    def reverse(self) -> dict[str, str]:
        """Original text -> placeholder. Used for consistency checks."""
        return {v: k for k, v in self.placeholders.items()}


class AuditResult(BaseModel):
    """Outcome of the post-generation PII audit on a single answer.

    `clean=True` means the audit ran successfully and found nothing. A
    `clean=False` answer is always returned redacted, never raw. `entities`
    lists what was found so the dashboard can show the leak.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    clean: bool
    answer: str
    """The user-facing answer. Identical to the input if clean, otherwise redacted."""

    entities: list[PiiEntity] = Field(default_factory=lambda: [])
    fail_closed: bool = False
    """True if the audit raised and the answer was redacted as a safety default."""
