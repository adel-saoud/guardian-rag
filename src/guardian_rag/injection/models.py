"""Injection-screening domain models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InjectionVerdict(BaseModel):
    """Screening result for one text.

    `score` is the screener's confidence that the text carries an
    injection, in [0, 1]. `reason` names what fired (`classifier`,
    `rule:<name>`, or `screener_error` when the screener failed closed).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    flagged: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str | None = None
