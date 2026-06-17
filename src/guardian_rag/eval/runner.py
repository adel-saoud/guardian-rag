"""PII recall + precision over the golden corpus.

For each golden document we run the same `PiiDetector` ingestion uses,
then compare detections against the planted manifest. A detection counts
as a match if it overlaps a planted entity by at least one character.

Recall = matched_planted / planted_total
Precision = matched_planted / detected_total

We use overlap rather than exact-offset equality because tokenisers
disagree about whitespace boundaries. A 1-character overlap is enough to
say "we caught this PII".
"""

from __future__ import annotations

from collections import defaultdict

import structlog

from guardian_rag.eval.models import EvalCounts, EvalReport, GoldenDocument, GoldenEntity
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.models import PiiEntity

logger = structlog.get_logger(__name__)


def _overlaps(planted: GoldenEntity, detected: PiiEntity) -> bool:
    return planted.start < detected.end and detected.start < planted.end


def _evaluate_document(
    document: GoldenDocument,
    detector: PiiDetector,
) -> tuple[EvalCounts, list[PiiEntity]]:
    detected = detector.detect(document.text)
    recall_hits = sum(1 for p in document.entities if any(_overlaps(p, d) for d in detected))
    precision_hits = sum(1 for d in detected if any(_overlaps(p, d) for p in document.entities))
    counts = EvalCounts(
        planted=len(document.entities),
        detected=len(detected),
        recall_hits=recall_hits,
        precision_hits=precision_hits,
    )
    return counts, detected


def run_pii_eval(
    corpus: list[GoldenDocument],
    detector: PiiDetector,
) -> EvalReport:
    """Run the detector over every document and aggregate counts."""
    by_category_planted: dict[str, int] = defaultdict(int)
    by_category_detected: dict[str, int] = defaultdict(int)
    by_category_recall: dict[str, int] = defaultdict(int)
    by_category_precision: dict[str, int] = defaultdict(int)
    total_planted = 0
    total_detected = 0
    total_recall = 0
    total_precision = 0

    for document in corpus:
        counts, _detected = _evaluate_document(document, detector)
        by_category_planted[document.category] += counts.planted
        by_category_detected[document.category] += counts.detected
        by_category_recall[document.category] += counts.recall_hits
        by_category_precision[document.category] += counts.precision_hits
        total_planted += counts.planted
        total_detected += counts.detected
        total_recall += counts.recall_hits
        total_precision += counts.precision_hits

    per_category = {
        cat: EvalCounts(
            planted=by_category_planted[cat],
            detected=by_category_detected[cat],
            recall_hits=by_category_recall[cat],
            precision_hits=by_category_precision[cat],
        )
        for cat in by_category_planted
    }
    overall = EvalCounts(
        planted=total_planted,
        detected=total_detected,
        recall_hits=total_recall,
        precision_hits=total_precision,
    )
    logger.info(
        "eval.pii.complete",
        recall=overall.recall,
        precision=overall.precision,
        n_docs=len(corpus),
    )
    return EvalReport(overall=overall, per_category=per_category)
