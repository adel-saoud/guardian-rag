# Agent Guidelines

## Project

Guardian RAG: privacy-safe RAG pipeline with three-stage PII detection at ingestion, optional cross-encoder reranking (off by default, see ADR 0008), prompt-injection screening on queries and ingested chunks, and a post-generation audit on every answer. EU AI Act Article 10-aligned by design. Runs fully locally with Ollama for generation. Zero API keys for PII detection.

## Part of a system

See the README section of the same name for the five companion repos.

## Repo conventions

- Commits follow Conventional Commits (commitizen hook). The maintainer bumps with `cz bump`, which writes the version and the tag; `CHANGELOG.md` is written by hand.
- Docs live in `docs/`: `architecture.md`, `adr/NNNN-title.md` (MADR 4, indexed in `adr/README.md`; `proposed` until the maintainer accepts the reasoning), `configuration.md`, `api.md`, `project-structure.md`, `roadmap.md`.
- When a setting changes, update `docs/configuration.md` in the same change.
- A number in the README must exist in `results/manifest.json`. Rerun `scripts/reproduce.py` before changing one.

## Principles

1. No em-dashes anywhere in code, comments, or docs. Use commas, colons, semicolons, periods, or "that" clauses.
2. Pyright strict, 0 errors, 0 warnings. Tests get a relaxed executionEnvironment so pytest stubs do not noise the report.
3. Heavy ML imports lazy. Module-level imports must stay light so tools, scripts, and tests are cheap.
4. The post-generation audit must run on every query path. Never let an answer return without it. Audit failures are fail-closed (redact by default).
5. Typed placeholders preserve grammar. Never replace PII with blank tokens, since blanks break retrieval and generation context.
6. Injection screening is fail-closed. A screener error refuses the query or quarantines the parent. `off` is an explicit mode, never a silent fallback.
7. Quarantine is per parent chunk, because the generator reads parents. Never drop only the flagged child.
8. Report only measured numbers. Benchmarks live in `scripts/eval_*.py` and write to `eval/results/`.

## Quick start

```bash
docker compose up
uv run python scripts/demo.py
uv run streamlit run src/guardian_rag/dashboard/app.py
```

## Commands

- `uv sync --extra dev --extra ml --extra dashboard` (Python 3.14 from `.python-version`; floor is 3.12)
- `uv run pre-commit install --hook-type pre-commit --hook-type commit-msg`
- `uv run pre-commit run --all-files` (ruff, pyright, gitleaks; the gate CI runs)
- `uv run pytest` (coverage floor 90%)
- `uv build` (uv_build backend)
- `uv run python scripts/demo.py`
- `uv run python scripts/eval.py`
- `uv sync --extra ml --extra bench && uv run python scripts/eval_retrieval.py` (SciFact, downloaded to `data/`, never committed)
- `uv run python scripts/eval_retrieval.py --dataset domain` (synthetic in-domain set, specs in `eval/retrieval/domain/`)
- `uv run python scripts/eval_injection.py`
- `uv run python scripts/eval_faithfulness.py` (needs Ollama 0.17.5+ with `qwen3.5:4b`)
- `uv run python scripts/reproduce.py` (every eval, then `results/manifest.json`; README numbers come from it)
- `uv run --with playwright python scripts/capture_dashboard.py` (re-captures `docs/dashboard.png` after reproduce)

## Architecture decisions (do not re-debate)

- Three-stage PII detection (Presidio, GLiNER, conditional DeBERTa), see `docs/adr/0010-three-stage-pii-detection.md` (supersedes 0001 to 0003)
- Hierarchical chunking (child 256 / parent 1024), see `docs/adr/0004-hierarchical-chunking.md`
- Typed placeholder redaction, see `docs/adr/0005-typed-placeholders.md`
- Post-generation audit, see `docs/adr/0006-post-generation-audit.md`
- Qdrant over Chroma, see `docs/adr/0007-qdrant-over-chroma.md`
- Cross-encoder reranking, see `docs/adr/0008-cross-encoder-reranking.md`
- Prompt-injection screening, see `docs/adr/0009-prompt-injection-screening.md`
- Faithfulness reported, never gated, see `docs/adr/0011-faithfulness-reported-not-gated.md`
- `qwen3.5:4b` with thinking off, see `docs/adr/0012-qwen35-local-model-thinking-off.md`
- Organisation names redacted, see `docs/adr/0013-redact-organisation-names.md`

## Code style

- Every config value env-driven via `pydantic-settings` with the `GUARDIAN_` prefix
- Every routing decision and PII detection logged via `structlog`
- Pydantic models immutable across boundaries (`frozen=True, extra="forbid"`)
- Backends behind `Protocol` types so tests can inject fakes without touching ML stacks

## Critical gotchas

- ML deps are an optional extra (`uv sync --extra ml`). Tests must mock GLiNER and DeBERTa loads. CI never downloads weights.
- The PII engine is injected via FastAPI `Depends`, never constructed in routes.
- `GUARDIAN_DEBERTA_THRESHOLD` (default 0.85) controls whether the validator fires. Lower the threshold and DeBERTa becomes a hot path. Raise it and the safety net narrows.
- The post-generation audit is mandatory. If the audit raises, the answer is redacted, never returned raw.
- `dashboard/app.py` and the three `_real_backends.py` modules (`pii/`, `retrieval/`, `injection/`) are excluded from pyright. Keep them small so the exclusion does not hide real bugs.

## Design rationale, short form

- Three-stage PII detection: "Presidio handles structured PII through regex with high precision. GLiNER catches unstructured PII like names and organisations through a span-based NER model. DeBERTa fires only when GLiNER is uncertain, so the safety net does not double inference cost."
- Why not just Presidio: "Presidio is a great PII library. It is not a RAG system. Guardian wraps Presidio plus GLiNER, then adds hierarchical chunking, retrieval, generation, and the post-generation audit on top."
- Post-generation audit: "Most RAG privacy projects protect ingestion only. The LLM can still paraphrase or infer PII from clean chunks. Running the same detection pipeline over the answer before it reaches the user closes that gap."
- Hierarchical chunking: "Child chunks at 256 tokens for precise retrieval and sentence-level PII detection. Parent chunks at 1024 tokens for generation context. Both stored in Qdrant. The same pattern is used widely in production RAG."
- Injection threat model: "The LLM only sees redacted text and never the redaction map, so an injection cannot leak original PII. It can hijack the answer. Screening protects answer integrity; the audit still protects PII."
- Ingestion-time screening: "Documents are known at ingestion, so every parent is screened once and quarantined there. Query time only screens the query and any chunk whose stored verdict came from a different screener."

## Out of scope

- Re-identification attacks (this is pseudonymisation, not anonymisation)
- GDPR right-to-erasure (vector store deletion, roadmap)
- Streaming PII detection
- PDF OCR (text PDFs only)
- Prometheus metrics export (roadmap)
