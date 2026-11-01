"""Measure prompt-injection screening: detection rate and false positives.

Three labelled sets:

- `own`: `eval/injection/cases.jsonl`, 60 hand-written cases (20 direct,
  20 indirect injections hidden in HR / contract / support text, 20 benign
  hard negatives that use words like "ignore" or "instructions"). Written
  in redacted form, as the screener sees them in production. Written by
  the same author as the rules, so treat it as a regression set.
- `deepset`: test split of `deepset/prompt-injections` (Apache-2.0),
  downloaded at run time. Not in the classifier's training data per its
  model card, so it is the fairer held-out number. Includes non-English
  rows; a stopword heuristic splits out likely-German rows because the
  default classifier is English-only.
- `golden`: the 30 synthetic golden documents, redacted with their own
  entity spans. All benign; any flag is a false positive at ingestion.

The classifier is also scored at several thresholds (`threshold_sweep` in
the JSON) to check whether any threshold separates benign documents from
injections.

Usage:
    uv sync --extra ml --extra bench
    uv run python scripts/eval_injection.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from guardian_rag.config import Settings
from guardian_rag.eval.injection import DetectionCounts, count_detections
from guardian_rag.eval.loader import golden_entities_as_pii, load_golden_corpus
from guardian_rag.eval.ranking import percentile
from guardian_rag.injection.screener import (
    ClassifierScreener,
    CompositeScreener,
    InjectionScreener,
    RuleScreener,
)
from guardian_rag.pii.redactor import Redactor

console = Console()

ROOT = Path(__file__).resolve().parent.parent
DEEPSET_REVISION = "4f61ecb038e9c3fb77e21034b22511b523772cdd"
DEEPSET_TEST_FILE = "data/test-00000-of-00001-701d16158af87368.parquet"
"""Pinned dataset commit and file, so reruns score the same split."""
THRESHOLD_SWEEP = (0.5, 0.9, 0.99, 0.999)
"""Classifier thresholds re-scored from the same run, to test whether any separates the sets."""
_GERMAN_HINT = re.compile(r"\b(und|der|die|das|ist|nicht|ich|du|sie|wie|was|mit|für)\b", re.I)

Example = tuple[str, bool, str]
"""(text, is_injection, slice)."""


def _own_set() -> list[Example]:
    """The 60 hand-written cases, sliced by `kind` (direct, indirect, benign)."""
    path = ROOT / "eval" / "injection" / "cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [(r["text"], bool(r["label"]), r["kind"]) for r in rows]


def _deepset_set(data_dir: Path) -> list[Example]:
    """`deepset/prompt-injections` test split, sliced into likely English and likely German."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "deepset/prompt-injections",
        DEEPSET_TEST_FILE,
        repo_type="dataset",
        revision=DEEPSET_REVISION,
        local_dir=data_dir / "deepset-prompt-injections",
    )
    examples: list[Example] = []
    for row in pq.read_table(path).to_pylist():
        text = str(row["text"])
        likely_german = len(_GERMAN_HINT.findall(text)) >= 2
        examples.append((text, bool(row["label"]), "likely_de" if likely_german else "likely_en"))
    return examples


def _golden_set() -> list[Example]:
    """Golden documents redacted with their own spans; every one is benign."""
    redactor = Redactor()
    examples: list[Example] = []
    for doc in load_golden_corpus(ROOT / "eval" / "golden"):
        redacted, _ = redactor.redact(doc.text, golden_entities_as_pii(doc), doc.document_id)
        examples.append((redacted, False, doc.category))
    return examples


def _run(
    screener: InjectionScreener, examples: list[Example]
) -> tuple[list[bool], list[float], list[float]]:
    """Screen texts one at a time, as production does, and time each call."""
    flags: list[bool] = []
    scores: list[float] = []
    latency_ms: list[float] = []
    for text, _, _ in examples:
        started = time.perf_counter()
        verdict = screener.screen([text])[0]
        latency_ms.append((time.perf_counter() - started) * 1000)
        flags.append(verdict.flagged)
        scores.append(verdict.score)
    return flags, scores, latency_ms


def _rate(value: float | None) -> str:
    """Percentage, or "n/a" when the set has no texts of that class."""
    return "n/a" if value is None else f"{value:.1%}"


def _summary(counts: DetectionCounts) -> dict[str, float | int | None]:
    """Counts plus rates, as written to the JSON report."""
    return {
        "n_injection": counts.n_positive,
        "n_benign": counts.n_negative,
        "detection_rate": counts.detection_rate,
        "false_positive_rate": counts.false_positive_rate,
        **counts.model_dump(),
    }


def main() -> int:
    """Score every screener on every set and write `eval/results/injection.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "hf")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "injection.json")
    args = parser.parse_args()

    from guardian_rag.injection._real_backends import RealInjectionClassifierBackend

    settings = Settings()
    classifier = ClassifierScreener(
        RealInjectionClassifierBackend(settings.injection_model, settings.injection_label),
        model_name=settings.injection_model,
        threshold=settings.injection_threshold,
    )
    classifier.screen(["warm-up"])
    rules = RuleScreener()
    screeners: dict[str, InjectionScreener] = {
        "rules": rules,
        "classifier": classifier,
        "classifier+rules": CompositeScreener([classifier, rules]),
    }
    datasets = {
        "own": _own_set(),
        "deepset_test": _deepset_set(args.data_dir),
        "golden_corpus": _golden_set(),
    }

    table = Table(title="Guardian RAG · prompt-injection screening eval")
    for column in ("Dataset", "Slice", "Screener", "Injections", "Benign", "Detected", "FPR"):
        table.add_column(column, justify="left" if column in {"Dataset", "Slice"} else "right")

    report: dict[str, object] = {
        "classifier_model": settings.injection_model,
        "classifier_threshold": settings.injection_threshold,
        "results": {},
    }
    results: dict[str, object] = {}
    for dataset_name, examples in datasets.items():
        slices = sorted({s for _, _, s in examples})
        for screener_name, screener in screeners.items():
            flags, scores, latency_ms = _run(screener, examples)
            entry: dict[str, object] = {
                "latency_p50_ms": percentile(latency_ms, 50),
                "latency_p95_ms": percentile(latency_ms, 95),
            }
            if screener_name == "classifier":
                labels = [ex[1] for ex in examples]
                entry["threshold_sweep"] = {
                    str(t): _summary(count_detections(labels, [x >= t for x in scores]))
                    for t in THRESHOLD_SWEEP
                }
            for slice_name in ["all", *slices]:
                idx = [i for i, ex in enumerate(examples) if slice_name in {"all", ex[2]}]
                counts = count_detections([examples[i][1] for i in idx], [flags[i] for i in idx])
                entry[slice_name] = _summary(counts)
                table.add_row(
                    dataset_name,
                    slice_name,
                    screener_name,
                    str(counts.n_positive),
                    str(counts.n_negative),
                    _rate(counts.detection_rate),
                    _rate(counts.false_positive_rate),
                )
            results[f"{dataset_name}/{screener_name}"] = entry
    report["results"] = results
    console.print(table)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    console.print(f"Wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
