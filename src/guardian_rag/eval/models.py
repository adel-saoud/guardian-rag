"""Data models for the eval harness.

A golden document carries the source text plus a manifest of every planted
PII span. Each span has explicit character offsets so the runner can
compute recall and precision unambiguously, without re-running the
detector against its own output.

Optional QA pairs let the faithfulness judge compare the redacted-context
answer against an unredacted-context baseline.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from guardian_rag.pii.models import PiiEntityType


class GoldenEntity(BaseModel):
    """One planted PII span inside a golden document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    entity_type: PiiEntityType
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_offsets(self) -> GoldenEntity:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")
        return self


class GoldenQA(BaseModel):
    """One question + reference answer pair tied to a golden document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    reference_answer: str
    """The answer a perfect model would produce against the redacted context."""


class GoldenDocument(BaseModel):
    """A synthetic document plus its PII manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    category: str
    """Domain bucket: `hr`, `contracts`, `support`. Used for per-category roll-ups."""

    text: str
    entities: list[GoldenEntity] = Field(default_factory=lambda: [])
    qa_pairs: list[GoldenQA] = Field(default_factory=lambda: [])


class EvalCounts(BaseModel):
    """Raw counts the runner aggregates per category.

    `recall_hits` and `precision_hits` are distinct on purpose. One
    planted entity may be matched by several detected entities (overlap
    can come from multiple recognisers firing on the same span). Counting
    both directions keeps the metrics honest.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    planted: int = 0
    detected: int = 0
    recall_hits: int = 0
    """Planted entities matched by at least one detection."""

    precision_hits: int = 0
    """Detected entities that overlap a planted span."""

    @property
    def recall(self) -> float:
        return self.recall_hits / self.planted if self.planted else 0.0

    @property
    def precision(self) -> float:
        return self.precision_hits / self.detected if self.detected else 0.0


class FaithfulnessResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    question: str
    reference_answer: str
    candidate_answer: str
    verdict: bool
    """True if the judge said the candidate is faithful to the reference."""


class EvalReport(BaseModel):
    """Aggregate eval outcome printed by `scripts/eval.py`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    overall: EvalCounts
    per_category: dict[str, EvalCounts] = Field(default_factory=lambda: {})
    faithfulness: list[FaithfulnessResult] = Field(default_factory=lambda: [])
    leaks_caught: int = 0
    """Post-generation audit captures across all evaluated answers."""

    @property
    def faithfulness_rate(self) -> float:
        if not self.faithfulness:
            return 0.0
        return sum(1 for r in self.faithfulness if r.verdict) / len(self.faithfulness)
