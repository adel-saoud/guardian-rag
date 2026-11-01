"""Eval harness: golden corpus, PII recall and precision, faithfulness judge."""

from __future__ import annotations

from guardian_rag.eval.judge import FaithfulnessJudge
from guardian_rag.eval.loader import load_golden_corpus
from guardian_rag.eval.models import (
    EvalCounts,
    EvalReport,
    FaithfulnessResult,
    GoldenDocument,
    GoldenEntity,
    GoldenQA,
)
from guardian_rag.eval.runner import run_pii_eval

__all__ = [
    "EvalCounts",
    "EvalReport",
    "FaithfulnessJudge",
    "FaithfulnessResult",
    "GoldenDocument",
    "GoldenEntity",
    "GoldenQA",
    "load_golden_corpus",
    "run_pii_eval",
]
