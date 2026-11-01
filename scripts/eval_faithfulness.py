"""Measure whether redaction hurts answer quality, and how far the judge can be trusted.

Two numbers, both informational:

1. Faithfulness: every golden QA pair is answered from the original and from
   the redacted document by the same Ollama model, and the judge scores both
   answers against the reference. The gap is the cost of redaction.
2. Judge agreement: the judge scores hand-labelled cases
   (`eval/faithfulness/judge_labels.jsonl`) and its verdicts are compared with
   the labels. A low agreement means the faithfulness rates are noise.

Redaction uses the same detector as ingestion: Presidio, GLiNER and the
conditional DeBERTa validator.

Usage:
    ollama pull qwen3.5:4b
    uv sync --extra ml
    uv run python scripts/eval_faithfulness.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
from rich.console import Console

from guardian_rag.config import get_settings
from guardian_rag.eval.faithfulness import (
    FaithfulnessReport,
    JudgeAgreement,
    judge_agreement,
    load_judge_labels,
    run_faithfulness_eval,
)
from guardian_rag.eval.judge import FaithfulnessJudge
from guardian_rag.eval.loader import load_golden_corpus
from guardian_rag.generation.generator import Generator
from guardian_rag.generation.ollama import OllamaClient
from guardian_rag.pii.deberta_validator import DebertaValidator
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.gliner_recognizer import GlinerRecognizer
from guardian_rag.pii.presidio_engine import PresidioEngine
from guardian_rag.pii.redactor import Redactor

ROOT = Path(__file__).resolve().parent.parent
console = Console()


def _build_detector() -> PiiDetector:
    """The ingestion detector: Presidio, GLiNER, and DeBERTa on low-confidence spans."""
    from guardian_rag.pii._real_backends import RealDebertaValidatorBackend, RealGlinerBackend

    settings = get_settings()
    return PiiDetector(
        presidio=PresidioEngine(),
        gliner=GlinerRecognizer(
            backend=RealGlinerBackend(settings.gliner_model),
            threshold=settings.gliner_threshold,
        ),
        validator=DebertaValidator(
            backend=RealDebertaValidatorBackend(settings.deberta_model),
            score_threshold=settings.deberta_threshold,
        ),
    )


async def _run(
    args: argparse.Namespace,
) -> tuple[FaithfulnessReport, JudgeAgreement, dict[str, object]]:
    """Run the faithfulness pass and the judge-agreement pass on one Ollama client."""
    settings = get_settings()
    corpus = load_golden_corpus(args.docs)
    labels = load_judge_labels(args.labels)
    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url,
        timeout=httpx.Timeout(settings.ollama_timeout_s),
    ) as http:
        ollama = OllamaClient(http=http, think=settings.ollama_think)
        judge = FaithfulnessJudge(ollama=ollama, model=args.judge_model)
        report = await run_faithfulness_eval(
            corpus,
            detector=_build_detector(),
            redactor=Redactor(),
            generator=Generator(ollama=ollama, model=args.generator_model, temperature=0.0),
            judge=judge,
        )
        verdicts = [
            (
                await judge.judge(
                    document_id=label.document_id,
                    question=label.question,
                    reference_answer=label.reference_answer,
                    candidate_answer=label.candidate_answer,
                )
            ).verdict
            for label in labels
        ]
    agreement = judge_agreement([label.label for label in labels], verdicts)
    payload: dict[str, object] = {
        "generator_model": args.generator_model,
        "generator_temperature": 0.0,
        "judge_model": args.judge_model,
        "think": settings.ollama_think,
        "faithfulness": {
            "n": report.n,
            "original_rate": report.original_rate,
            "redacted_rate": report.redacted_rate,
            "redacted_minus_original": report.redacted_rate - report.original_rate,
        },
        "judge_agreement": {
            **agreement.model_dump(),
            "labels": str(args.labels.relative_to(ROOT)),
            "labelled_by": sorted({label.labelled_by for label in labels}),
        },
        "cases": [case.model_dump() for case in report.cases],
        "judge_cases": [
            {"case_id": label.case_id, "label": label.label, "verdict": verdict}
            for label, verdict in zip(labels, verdicts, strict=True)
        ],
    }
    return report, agreement, payload


def main() -> int:
    """Write `eval/results/faithfulness.json` and print the headline numbers."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=ROOT / "eval" / "golden")
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "eval" / "faithfulness" / "judge_labels.jsonl"
    )
    parser.add_argument("--generator-model", default=settings.ollama_model)
    parser.add_argument("--judge-model", default=settings.ollama_judge_model)
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "faithfulness.json")
    args = parser.parse_args()

    report, agreement, payload = asyncio.run(_run(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    kappa = "undefined" if agreement.cohen_kappa is None else f"{agreement.cohen_kappa:.2f}"
    console.print(
        f"Faithfulness (n={report.n}): original {report.original_rate:.1%}, "
        f"redacted {report.redacted_rate:.1%}\n"
        f"Judge agreement with labels (n={agreement.n}): {agreement.agreement:.1%}, "
        f"kappa {kappa}"
    )
    console.print(f"[dim]wrote {args.out}[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
