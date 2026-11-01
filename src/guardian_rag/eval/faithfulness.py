"""Does redaction hurt answer quality? Measured, not assumed.

Each golden QA pair is answered twice by the same generator: once from the
original document and once from the document after PII detection and
redaction. The judge scores both answers against the reference answer.
The gap between the two rates is the cost of redaction on answer quality.
QA pairs ask about facts that are not PII (a rating, a deadline, a
company), so a drop means redaction damaged the context, not that a name
was hidden on purpose.

A small local judge is noisy, so its verdicts are only worth reporting
next to its agreement with hand labels (`judge_agreement`). Both numbers
are informational and never gate a build.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from guardian_rag.eval.judge import FaithfulnessJudge
from guardian_rag.eval.models import GoldenDocument
from guardian_rag.generation.generator import Generator
from guardian_rag.ingestion.models import ParentChunk
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.redactor import Redactor


class JudgeLabel(BaseModel):
    """One hand-labelled judge case: is `candidate_answer` equivalent to the reference?"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    document_id: str
    question: str
    reference_answer: str
    candidate_answer: str
    label: bool
    labelled_by: str
    """`maintainer-draft` until the maintainer has reviewed the label."""


class JudgeAgreement(BaseModel):
    """Judge verdicts compared with hand labels; positive means "equivalent"."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    agreement: float
    cohen_kappa: float | None
    """Agreement corrected for chance; None when it is undefined (one class only)."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


class FaithfulnessCase(BaseModel):
    """Both answers to one QA pair and the judge verdict on each."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    question: str
    reference_answer: str
    original_answer: str
    redacted_answer: str
    original_verdict: bool
    redacted_verdict: bool


class FaithfulnessReport(BaseModel):
    """Judge pass rates with original and redacted context, over the same QA pairs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: list[FaithfulnessCase]

    @property
    def n(self) -> int:
        """Number of QA pairs scored."""
        return len(self.cases)

    @property
    def original_rate(self) -> float:
        """Share of answers from the original document judged equivalent to the reference."""
        return sum(c.original_verdict for c in self.cases) / self.n if self.cases else 0.0

    @property
    def redacted_rate(self) -> float:
        """Share of answers from the redacted document judged equivalent to the reference."""
        return sum(c.redacted_verdict for c in self.cases) / self.n if self.cases else 0.0


def load_judge_labels(path: Path) -> list[JudgeLabel]:
    """Read one `JudgeLabel` per non-empty JSONL line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [JudgeLabel.model_validate(json.loads(line)) for line in lines if line.strip()]


def judge_agreement(labels: list[bool], verdicts: list[bool]) -> JudgeAgreement:
    """Agreement rate and Cohen's kappa between hand labels and judge verdicts."""
    if len(labels) != len(verdicts):
        raise ValueError("labels and verdicts must have the same length")
    n = len(labels)
    tp = sum(lab and ver for lab, ver in zip(labels, verdicts, strict=True))
    tn = sum(not lab and not ver for lab, ver in zip(labels, verdicts, strict=True))
    fp = sum(not lab and ver for lab, ver in zip(labels, verdicts, strict=True))
    fn = n - tp - tn - fp
    observed = (tp + tn) / n if n else 0.0
    kappa: float | None = None
    if n:
        expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
        kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return JudgeAgreement(
        n=n,
        agreement=observed,
        cohen_kappa=kappa,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


def _single_parent(document_id: str, text: str) -> list[ParentChunk]:
    """Golden documents are a few sentences, so each fits in one parent chunk."""
    return [
        ParentChunk(chunk_id=f"{document_id}-parent", document_id=document_id, text=text, order=0)
    ]


async def run_faithfulness_eval(
    corpus: list[GoldenDocument],
    *,
    detector: PiiDetector,
    redactor: Redactor,
    generator: Generator,
    judge: FaithfulnessJudge,
) -> FaithfulnessReport:
    """Answer every QA pair from the original and the redacted text, then judge both.

    Retrieval is skipped on purpose: each answer is generated from its own
    document, so the only difference between the two runs is redaction.
    """
    cases: list[FaithfulnessCase] = []
    for doc in corpus:
        redacted_text, _map = redactor.redact(doc.text, detector.detect(doc.text), doc.document_id)
        for qa in doc.qa_pairs:
            original = await generator.generate(
                qa.question, _single_parent(doc.document_id, doc.text)
            )
            redacted = await generator.generate(
                qa.question, _single_parent(doc.document_id, redacted_text)
            )
            original_verdict = await judge.judge(
                document_id=doc.document_id,
                question=qa.question,
                reference_answer=qa.reference_answer,
                candidate_answer=original.answer,
            )
            redacted_verdict = await judge.judge(
                document_id=doc.document_id,
                question=qa.question,
                reference_answer=qa.reference_answer,
                candidate_answer=redacted.answer,
            )
            cases.append(
                FaithfulnessCase(
                    document_id=doc.document_id,
                    question=qa.question,
                    reference_answer=qa.reference_answer,
                    original_answer=original.answer,
                    redacted_answer=redacted.answer,
                    original_verdict=original_verdict.verdict,
                    redacted_verdict=redacted_verdict.verdict,
                )
            )
    return FaithfulnessReport(cases=cases)
