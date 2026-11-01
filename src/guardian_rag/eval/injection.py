"""Detection metrics for the prompt-injection screener.

The screener is a binary classifier, so the report is the confusion
matrix plus the two rates that matter operationally: detection rate (how
many injections are caught) and false-positive rate (how many legitimate
queries or chunks are refused or quarantined).
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict


class DetectionCounts(BaseModel):
    """Confusion-matrix counts for one screener on one labelled set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    true_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    false_positives: int = 0

    @property
    def n_positive(self) -> int:
        """Labelled injections in the set."""
        return self.true_positives + self.false_negatives

    @property
    def n_negative(self) -> int:
        """Labelled benign texts in the set."""
        return self.true_negatives + self.false_positives

    @property
    def detection_rate(self) -> float | None:
        """True-positive rate. None when the set has no injections."""
        return self.true_positives / self.n_positive if self.n_positive else None

    @property
    def false_positive_rate(self) -> float | None:
        """Share of benign texts flagged. None when the set has no benign texts."""
        return self.false_positives / self.n_negative if self.n_negative else None


def count_detections(labels: Sequence[bool], flagged: Sequence[bool]) -> DetectionCounts:
    """Build counts from gold labels (True = injection) and screener flags."""
    if len(labels) != len(flagged):
        raise ValueError(f"{len(labels)} labels but {len(flagged)} predictions")
    pairs = list(zip(labels, flagged, strict=True))
    return DetectionCounts(
        true_positives=sum(1 for gold, pred in pairs if gold and pred),
        false_negatives=sum(1 for gold, pred in pairs if gold and not pred),
        true_negatives=sum(1 for gold, pred in pairs if not gold and not pred),
        false_positives=sum(1 for gold, pred in pairs if not gold and pred),
    )
