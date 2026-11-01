"""Faithfulness eval: redacted versus original answers, and judge agreement with labels."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from guardian_rag.eval.faithfulness import (
    FaithfulnessReport,
    judge_agreement,
    load_judge_labels,
    run_faithfulness_eval,
)
from guardian_rag.eval.judge import JUDGE_SYSTEM_PROMPT, FaithfulnessJudge
from guardian_rag.eval.models import GoldenDocument, GoldenEntity, GoldenQA
from guardian_rag.generation.generator import Generator
from guardian_rag.generation.ollama import OllamaClient
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.models import PiiEntityType
from guardian_rag.pii.presidio_engine import PresidioEngine
from guardian_rag.pii.redactor import Redactor

LABELS = Path(__file__).resolve().parent.parent / "eval" / "faithfulness" / "judge_labels.jsonl"


def _ollama() -> OllamaClient:
    """Generator answers say which context they saw; the judge accepts only originals."""

    def handler(request: httpx.Request) -> httpx.Response:
        messages = json.loads(request.content)["messages"]
        if messages[0]["content"] == JUDGE_SYSTEM_PROMPT:
            reply = "yes" if "from original" in messages[1]["content"] else "no"
        else:
            redacted = "[EMAIL_1]" in messages[-1]["content"]
            reply = f"Answer from {'redacted' if redacted else 'original'} [1]"
        return httpx.Response(
            200, json={"model": "m", "message": {"role": "assistant", "content": reply}}
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama")
    return OllamaClient(http=http, think=False)


async def test_run_faithfulness_eval_judges_both_answers():
    text = "Appeals can be sent to hiro.tanaka@example.com within 14 days."
    doc = GoldenDocument(
        document_id="hr-005",
        category="hr",
        text=text,
        entities=[
            GoldenEntity(
                text="hiro.tanaka@example.com", entity_type=PiiEntityType.EMAIL, start=23, end=46
            )
        ],
        qa_pairs=[GoldenQA(question="How long to appeal?", reference_answer="Within 14 days.")],
    )
    ollama = _ollama()

    report = await run_faithfulness_eval(
        [doc],
        detector=PiiDetector(presidio=PresidioEngine()),
        redactor=Redactor(),
        generator=Generator(ollama=ollama, model="m"),
        judge=FaithfulnessJudge(ollama=ollama, model="m"),
    )

    assert report.n == 1
    case = report.cases[0]
    assert case.original_answer.startswith("Answer from original")
    assert case.redacted_answer.startswith("Answer from redacted")
    assert report.original_rate == 1.0
    assert report.redacted_rate == 0.0


def test_judge_agreement_counts_and_kappa():
    labels = [True, True, False, False]
    verdicts = [True, False, False, True]
    result = judge_agreement(labels, verdicts)
    assert (result.true_positives, result.false_negatives) == (1, 1)
    assert (result.true_negatives, result.false_positives) == (1, 1)
    assert result.agreement == 0.5
    assert result.cohen_kappa == pytest.approx(0.0)


def test_judge_agreement_perfect_and_single_class():
    assert judge_agreement([True, False], [True, False]).cohen_kappa == pytest.approx(1.0)
    assert judge_agreement([True, True], [True, True]).cohen_kappa is None


def test_judge_agreement_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        judge_agreement([True], [])


def test_report_rates_are_zero_without_cases():
    report = FaithfulnessReport(cases=[])
    assert (report.n, report.original_rate, report.redacted_rate) == (0, 0.0, 0.0)


def test_committed_judge_labels_are_balanced_and_reviewed():
    labels = load_judge_labels(LABELS)
    assert len(labels) == 30
    assert {label.labelled_by for label in labels} == {"maintainer"}
    assert 10 <= sum(label.label for label in labels) <= 20
