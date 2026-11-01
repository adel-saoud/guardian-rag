"""Regenerate every number the README shows, and record how it was produced.

Usage:
    uv sync --all-extras
    ollama pull qwen3.5:4b
    uv run python scripts/reproduce.py                  # all evals, then the manifest
    uv run python scripts/reproduce.py --manifest-only  # rebuild the manifest from eval/results/

Runs the existing eval scripts (it does not duplicate them), each writing its
JSON to `eval/results/`, then writes `results/manifest.json`: date, git commit,
Python and package versions, model ids with their resolved revisions (and the
Ollama digest), seeds, dataset paths with SHA-256, a hardware summary, and
every metric with its sample size. Nothing needs a paid API key: generation
and judging run on a local Ollama, the other models on CPU. Latencies depend
on the hardware recorded in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import httpx

from guardian_rag.config import Settings

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
MANIFEST = ROOT / "results" / "manifest.json"

STEPS: list[list[str]] = [
    ["scripts/eval.py", "--with-gliner", "--out", "eval/results/pii.json"],
    ["scripts/eval_retrieval.py", "--dataset", "scifact"],
    ["scripts/eval_retrieval.py", "--dataset", "golden"],
    ["scripts/eval_retrieval.py", "--dataset", "domain"],
    ["scripts/eval_injection.py"],
    ["scripts/eval_faithfulness.py"],
]
"""Each step is one existing eval script; together they cover every README number."""

PACKAGES = (
    "guardian-rag",
    "presidio-analyzer",
    "spacy",
    "gliner",
    "transformers",
    "torch",
    "sentence-transformers",
    "qdrant-client",
    "rank-bm25",
)

DATASETS: list[tuple[str, str]] = [
    ("eval/golden", "30 synthetic documents, 91 planted PII entities, 30 QA pairs"),
    (
        "eval/faithfulness/judge_labels.jsonl",
        "30 judge cases labelled and reviewed by the maintainer",
    ),
    ("eval/retrieval/golden_queries.jsonl", "golden retrieval smoke queries"),
    ("eval/retrieval/domain", "in-domain topic specs, rendered with seed 0"),
    ("eval/retrieval/domain_queries.jsonl", "in-domain queries"),
    ("eval/injection/cases.jsonl", "own injection set"),
    ("data/beir/BeIR/scifact", "BEIR SciFact corpus and queries, downloaded at a pinned revision"),
    ("data/beir/BeIR/scifact-qrels", "BEIR SciFact test qrels, downloaded at a pinned revision"),
    ("data/hf/deepset-prompt-injections", "deepset/prompt-injections, pinned revision"),
]

SEEDS = {
    "paired_bootstrap_ci": 0,
    "domain_corpus": 0,
    "generator_temperature": 0.0,
    "judge_temperature": 0.0,
}


def _run_steps() -> None:
    """Run each eval script with the project interpreter; stop at the first failure."""
    for step in STEPS:
        print(f"$ uv run python {' '.join(step)}", flush=True)
        subprocess.run([sys.executable, *step], cwd=ROOT, check=True)  # noqa: S603


def _sha256(path: Path) -> str:
    """Hash a file, or every file under a directory in sorted order (skipping caches)."""
    digest = hashlib.sha256()
    files = (
        [path]
        if path.is_file()
        else sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts)
    )
    for file in files:
        digest.update(str(file.relative_to(ROOT)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def _datasets() -> list[dict[str, Any]]:
    """Path, SHA-256 and description of every dataset the evals read."""
    entries: list[dict[str, Any]] = []
    for relative, description in DATASETS:
        path = ROOT / relative
        entries.append(
            {
                "path": relative,
                "sha256": _sha256(path) if path.exists() else None,
                "description": description,
            }
        )
    return entries


def _hf_revision(model_id: str) -> str | None:
    """Commit hash the local Hugging Face cache resolved `main` to, if the model is cached."""
    from huggingface_hub.constants import HF_HUB_CACHE

    ref = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
    return ref.read_text(encoding="utf-8").strip() if ref.exists() else None


def _ollama_digest(settings: Settings, model: str) -> str | None:
    """Digest of `model` from the local Ollama `GET /api/tags`."""
    response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=10.0)
    response.raise_for_status()
    for entry in response.json().get("models", []):
        if entry.get("name") == model or entry.get("model") == model:
            return str(entry.get("digest"))
    return None


def _ollama_version(settings: Settings) -> str:
    """Server version from the local Ollama `GET /api/version`."""
    response = httpx.get(f"{settings.ollama_base_url}/api/version", timeout=10.0)
    response.raise_for_status()
    return str(response.json()["version"])


def _models(settings: Settings, faithfulness: dict[str, Any]) -> list[dict[str, Any]]:
    """Every model behind a README number, with the revision or digest actually used."""
    hf = {
        "pii_ner": settings.gliner_model,
        "pii_validator": settings.deberta_model,
        "embedding": settings.embedding_model,
        "reranker": settings.reranker_model,
        "injection_classifier": settings.injection_model,
    }
    models = [
        {"role": role, "name": name, "source": "huggingface", "revision": _hf_revision(name)}
        for role, name in hf.items()
    ]
    for role in ("generator_model", "judge_model"):
        name = str(faithfulness[role])
        models.append(
            {
                "role": role.removesuffix("_model"),
                "name": name,
                "source": "ollama",
                "ollama_version": _ollama_version(settings),
                "think": faithfulness["think"],
                "digest": _ollama_digest(settings, name),
            }
        )
    return models


def _metric(name: str, suite: str, value: float, n: int, **extra: Any) -> dict[str, Any]:
    return {"name": name, "suite": suite, "value": value, "n": n, **extra}


def _pii_metrics(pii: dict[str, Any]) -> Iterator[dict[str, Any]]:
    slices = {"overall": pii["overall"], **pii["per_category"]}
    for slice_name, counts in slices.items():
        suite = f"pii/{slice_name}"
        yield _metric("recall", suite, counts["recall"], counts["planted"])
        yield _metric("precision", suite, counts["precision"], counts["detected"])


def _retrieval_metrics(dataset: str, result: dict[str, Any]) -> Iterator[dict[str, Any]]:
    n = int(result["n_queries"])
    n_documents = int(result["n_documents"])
    for config, metrics in result["results"].items():
        for name, value in metrics.items():
            yield _metric(name, f"retrieval/{dataset}/{config}", value, n, n_documents=n_documents)
    delta = result["rerank_vs_hybrid"]
    yield _metric(
        "rerank_minus_hybrid_ndcg@10",
        f"retrieval/{dataset}",
        delta["ndcg@10_delta"],
        n,
        n_documents=n_documents,
        ci95=[delta["ci95_low"], delta["ci95_high"]],
    )


def _injection_metrics(injection: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for screener_set, slices in injection["results"].items():
        for slice_name, raw in slices.items():
            # Slices are count dicts; latency floats and threshold sweeps are skipped.
            if not isinstance(raw, dict) or "n_injection" not in raw:
                continue
            counts = cast("dict[str, Any]", raw)
            suite = f"injection/{screener_set}/{slice_name}"
            if counts["detection_rate"] is not None:
                yield _metric(
                    "detection_rate", suite, counts["detection_rate"], counts["n_injection"]
                )
            if counts["false_positive_rate"] is not None:
                yield _metric(
                    "false_positive_rate",
                    suite,
                    counts["false_positive_rate"],
                    counts["n_benign"],
                    false_positives=counts["false_positives"],
                )


def _faithfulness_metrics(faithfulness: dict[str, Any]) -> Iterator[dict[str, Any]]:
    rates = faithfulness["faithfulness"]
    for name in ("original_rate", "redacted_rate", "redacted_minus_original"):
        yield _metric(name, "faithfulness", rates[name], rates["n"])
    agreement = faithfulness["judge_agreement"]
    yield _metric("judge_agreement", "faithfulness/judge", agreement["agreement"], agreement["n"])
    if agreement["cohen_kappa"] is not None:
        yield _metric(
            "judge_cohen_kappa", "faithfulness/judge", agreement["cohen_kappa"], agreement["n"]
        )


def _read(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _git_commit() -> dict[str, Any]:
    """HEAD and whether the tree differs from it; `uncommitted` when there is no clean commit."""

    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    dirty = bool(git("status", "--porcelain"))
    return {"commit": "uncommitted" if dirty else git("rev-parse", "HEAD"), "dirty": dirty}


def _hardware() -> dict[str, Any]:
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = (
            subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],  # noqa: S607
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or cpu
        )
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "accelerator": "none (CPU only)",
    }


def build_manifest() -> dict[str, Any]:
    """Assemble the manifest from the JSON files the eval scripts wrote."""
    settings = Settings()
    faithfulness = _read("faithfulness.json")
    metrics: list[dict[str, Any]] = [
        *_pii_metrics(_read("pii.json")),
        *(
            m
            for d in ("scifact", "golden", "domain")
            for m in _retrieval_metrics(d, _read(f"retrieval_{d}.json"))
        ),
        *_injection_metrics(_read("injection.json")),
        *_faithfulness_metrics(faithfulness),
    ]
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "uv run python scripts/reproduce.py",
        "git": _git_commit(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in PACKAGES},
        "hardware": _hardware(),
        "seeds": SEEDS,
        "datasets": _datasets(),
        "models": _models(settings, faithfulness),
        "results_files": sorted(str(p.relative_to(ROOT)) for p in RESULTS.glob("*.json")),
        "metrics": metrics,
    }


def main() -> int:
    """Run the evals (unless `--manifest-only`) and write `results/manifest.json`."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Skip the evals and rebuild the manifest from eval/results/.",
    )
    args = parser.parse_args()
    if not args.manifest_only:
        _run_steps()
    manifest = build_manifest()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(manifest['metrics'])} metrics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
