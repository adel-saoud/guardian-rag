"""Eval-runner tests.

We test the recall/precision math against deterministic detector outputs,
not against the real detector. The Presidio-backed end-to-end is exercised
by `test_eval_corpus_smoke` so we know the harness pairs up with the
production detector at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from guardian_rag.eval.models import GoldenDocument, GoldenEntity
from guardian_rag.eval.runner import run_pii_eval
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.models import PiiEntity, PiiEntityType, PiiStage
from guardian_rag.pii.presidio_engine import PresidioEngine


@dataclass
class FixedDetector:
    entities_by_text: dict[str, list[PiiEntity]]

    def detect(self, text: str) -> list[PiiEntity]:
        return list(self.entities_by_text.get(text, []))


def _planted(start: int, end: int, etype: PiiEntityType = PiiEntityType.PERSON) -> GoldenEntity:
    return GoldenEntity(text="x" * (end - start), entity_type=etype, start=start, end=end)


def _detected(start: int, end: int) -> PiiEntity:
    return PiiEntity(
        entity_type=PiiEntityType.PERSON,
        start=start,
        end=end,
        score=0.9,
        text="x" * (end - start),
        stage=PiiStage.GLINER,
    )


def test_perfect_recall_and_precision():
    doc = GoldenDocument(
        document_id="d1",
        category="hr",
        text="full text",
        entities=[_planted(0, 5)],
    )
    detector = FixedDetector({"full text": [_detected(0, 5)]})
    report = run_pii_eval([doc], detector)  # type: ignore[arg-type]
    assert report.overall.recall == 1.0
    assert report.overall.precision == 1.0


def test_recall_drops_when_planted_missed():
    doc = GoldenDocument(
        document_id="d1",
        category="hr",
        text="t",
        entities=[_planted(0, 5), _planted(10, 15)],
    )
    detector = FixedDetector({"t": [_detected(0, 5)]})
    report = run_pii_eval([doc], detector)  # type: ignore[arg-type]
    assert report.overall.recall == 0.5
    assert report.overall.precision == 1.0


def test_precision_drops_with_false_positives():
    doc = GoldenDocument(
        document_id="d1",
        category="hr",
        text="t",
        entities=[_planted(0, 5)],
    )
    detector = FixedDetector(
        {"t": [_detected(0, 5), _detected(50, 55)]},
    )
    report = run_pii_eval([doc], detector)  # type: ignore[arg-type]
    assert report.overall.recall == 1.0
    assert report.overall.precision == 0.5


def test_overlap_counts_as_match():
    doc = GoldenDocument(
        document_id="d1",
        category="hr",
        text="t",
        entities=[_planted(10, 20)],
    )
    detector = FixedDetector({"t": [_detected(15, 25)]})
    report = run_pii_eval([doc], detector)  # type: ignore[arg-type]
    assert report.overall.recall_hits == 1
    assert report.overall.precision_hits == 1


def test_per_category_aggregation():
    docs = [
        GoldenDocument(document_id="h", category="hr", text="t1", entities=[_planted(0, 5)]),
        GoldenDocument(document_id="c", category="contracts", text="t2", entities=[_planted(0, 5)]),
    ]
    detector = FixedDetector({"t1": [_detected(0, 5)], "t2": []})
    report = run_pii_eval(docs, detector)  # type: ignore[arg-type]
    assert report.per_category["hr"].recall == 1.0
    assert report.per_category["contracts"].recall == 0.0


def test_eval_corpus_smoke(golden_corpus_dir: Path):
    """Sanity check: Presidio on the real corpus catches the structured PII."""
    from guardian_rag.eval.loader import load_golden_corpus

    corpus = load_golden_corpus(golden_corpus_dir)
    assert len(corpus) == 30
    presidio = PiiDetector(presidio=PresidioEngine())
    report = run_pii_eval(corpus, presidio)
    # Structured PII (email, phone, IBAN, card, IP) makes up most of the
    # planted spans. Presidio alone should comfortably exceed 50% recall.
    assert report.overall.recall >= 0.5
    assert report.overall.precision >= 0.85
