# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-01

The changes below break the 0.1 interface; while the major version is 0 a
breaking change is a minor bump. Removed names have no aliases: an old
variable name now stops startup with the closest valid one.

### Breaking

- **BREAKING:** Python floor raised from 3.11 to 3.12. Development, CI and
  the Docker image run on 3.14; spacy moves to 3.8.16, the first release
  with 3.14 wheels.
- **BREAKING:** an unknown `GUARDIAN_*` variable, in the environment or in
  `.env`, stops startup with `UnknownSettingError` naming the closest
  valid setting. pydantic-settings used to ignore it silently.
- **BREAKING:** startup fails with `IncompatibleServiceError` when Ollama
  is older than 0.17.5 or Qdrant older than 1.17.0, naming the minimum.
- **BREAKING:** the default Ollama model is `qwen3.5:4b` instead of
  `llama3.2:3b`, for generation and for the faithfulness judge
  (`GUARDIAN_OLLAMA_JUDGE_MODEL`). Pull it before upgrading.
- **BREAKING:** `OllamaClient` takes a required `think` argument, sent as
  Ollama's `think` field on every chat (`GUARDIAN_OLLAMA_THINK`, default
  `false`). Qwen3.5 thinks by default, which multiplied output tokens
  about 28 times on a short prompt.
- **BREAKING:** settings removed: `GUARDIAN_TOP_K` and
  `GUARDIAN_PARENT_TOP_K` (unused; the number of chunks per query is the
  `top_k` request field) and `GUARDIAN_LANGUAGES` (stored, never used).
- **BREAKING:** the always-empty `X-Guardian-Faithfulness` response header
  is gone.
- **BREAKING:** the `guardian` console script is gone; it pointed at a
  module that did not exist.
- **BREAKING:** Python API removed: `Document.metadata`, `PiiEntity.length`,
  `RedactionMap.reverse()`, `RedactionMapStore.exists()` and `.delete()`,
  `GroundedAnswer.has_citations` and `.all_citations_grounded`,
  `EvalReport.faithfulness`, `.faithfulness_rate` and `.leaks_caught`,
  and constructor options no caller set (`Redactor(placeholder_format)`,
  `HierarchicalChunker(encoding_name)`,
  `HybridRetriever(candidate_multiplier)`).
- **BREAKING:** the golden corpus QA pairs are new: one per document (30),
  each about a fact that is not PII. The three old pairs are gone.

### Added

- `scripts/eval_faithfulness.py`: every golden QA pair is answered from the
  original and from the redacted document by the same model, and a local
  judge scores both against the reference. Reported with the judge's
  agreement with 30 hand-labelled cases
  (`eval/faithfulness/judge_labels.jsonl`, reviewed by the maintainer). Informational,
  never a gate.
- `scripts/reproduce.py` runs every eval and writes `results/manifest.json`:
  versions, model revisions and the Ollama digest, dataset SHA-256, seeds,
  hardware, and each metric with its sample size. README numbers come from
  it.
- `scripts/capture_dashboard.py` re-captures `docs/dashboard.png` from the
  running dashboard (replaces `scripts/composite_dashboard.py`);
  `--seed-only` and `--compose` cover screenshots taken by hand.
- `scripts/eval.py --out` writes the PII counts as JSON.
- `Generator(temperature=...)`, used by the faithfulness eval at 0.0 so a
  rerun repeats.
- `OllamaClient.version()` (`GET /api/version`) for the startup check.
- The faithfulness judge skips a closed `<think>` trace before its verdict
  and raises `JudgeOutputError` on an unfinished one, instead of reading it
  as a silent "no".
- Dependabot, monthly, grouped per ecosystem (uv, GitHub Actions, pre-commit).
- Commitizen configuration: `cz bump` writes the version and the tag; this changelog stays hand-written.

- Cross-encoder reranking stage after hybrid dense + BM25 retrieval
  (`retrieval/reranker.py`), behind `GUARDIAN_RERANKER_ENABLED`,
  `GUARDIAN_RERANKER_MODEL` and `GUARDIAN_RERANK_CANDIDATES`. A reranker
  failure falls back to the fused order.
- Prompt-injection screening (`injection/`) with a local classifier
  (`protectai/deberta-v3-base-prompt-injection-v2`), regex rules, or both,
  configured separately for queries (`GUARDIAN_QUERY_INJECTION_SCREENER`,
  default `classifier+rules`) and ingested chunks
  (`GUARDIAN_CHUNK_INJECTION_SCREENER`, default `rules`).
  - User queries are screened after PII redaction; a flagged query is
    refused before retrieval and returns `injection_blocked: true` and
    `X-Guardian-Injection: blocked`.
  - Every parent chunk is screened at ingestion. A flagged parent is
    quarantined: stored for audit, excluded from dense search and BM25.
  - Parents screened by a different screener, or ingested before this
    release, are re-screened at query time.
  - Screener errors fail closed.
- Spotlighting: context blocks are wrapped in `<document>` tags and the
  system prompt treats their content as data, never instructions.
- `scripts/eval_retrieval.py` (BEIR SciFact, downloaded at run time, a
  30-query golden smoke set, and `--dataset domain`: 270 synthetic
  multi-chunk HR / contracts / support documents in near-duplicate
  families with 135 hand-written queries, built by
  `guardian_rag.eval.domain_corpus`) and `scripts/eval_injection.py` (own
  60-case set, `deepset/prompt-injections` test split, golden corpus).
- `bench` optional extra (`pyarrow`, `huggingface-hub`) for the benchmarks.
- `/v1/stats` reports `n_injections_blocked` and `n_parents_quarantined`.

### Changed

- The dashboard runs the API's three detection stages (Presidio, GLiNER,
  DeBERTa) instead of Presidio alone, so its redaction explorer and eval
  match what Guardian redacts. Without the `ml` extra it falls back to
  Presidio and shows a warning.
- The faithfulness judge checks that the candidate states the key fact of
  the reference, instead of strict semantic equivalence. The old prompt
  rejected correct answers carrying extra detail and agreed with the hand
  labels on 18 of 30 cases with `llama3.2:3b`.
- `docker-compose.yml` pins `ollama/ollama:0.40.10` and
  `qdrant/qdrant:v1.19.2` instead of `latest`.
- Build backend switched from hatchling to `uv_build`.
- `context/` moved to `docs/`: `docs/architecture.md`, `docs/roadmap.md` and
  the ADRs under `docs/adr/0001-*.md` to `docs/adr/0009-*.md`.
- README follows the shared skeleton. API surface, configuration and
  project structure moved to `docs/api.md`, `docs/configuration.md` and
  `docs/project-structure.md`.
- Pre-commit: ruff 0.17.2 (`ruff-check`), pyright, gitleaks and a
  commitizen commit-msg hook, every hook pinned by commit SHA.
- CI: `contents: read` permissions, actions pinned by SHA, separate
  pre-commit, test and wheel build jobs on Python 3.14.
- `eval.yml` runs on manual trigger only (weekly schedule removed).
- Coverage floor raised from 85% to 90%.
- `guardian_rag.__version__` and the FastAPI app version are read from
  package metadata.
- `scripts/demo.py` redacts the query before screening, retrieval and
  generation, as `/v1/query` does, passes `GUARDIAN_QDRANT_API_KEY`, and
  runs the same service version checks as the app.
- The Qdrant client is built with `QdrantClient(location=...)`, so
  `GUARDIAN_QDRANT_URL` takes a server URL or exactly `:memory:`.
- Parents of a query's hits are read back from Qdrant in one `retrieve`
  call instead of one call per parent.
- Dashboard: the hard-coded "Leaks caught" and empty "Faithfulness"
  metrics are gone.
- `scripts/eval.py` and the README say the PII eval measures Presidio plus
  GLiNER; the DeBERTa validator is not part of it.

- `RetrievalHit` carries `fused_score` and `rerank_score`.
- `ParentChunk` carries `injection_quarantined` and `screener_id`, also
  written to the Qdrant payload.
- Package description and API description now say three-stage PII
  detection.
- `scripts/demo.py` applies the query and chunk injection screens set in
  config.

### Removed

- Unused dependencies: `presidio-anonymizer`, `jinja2`, `markdown-it-py`,
  and the dev tools `respx` and `pytest-mock`.
- `Generator(max_output_tokens)`: no caller set it.
- Roadmap item for a per-request faithfulness score: a judge call per query
  would double latency for a number the offline eval already gives.

### Docs

- Decisions live only in ADRs, in MADR 4 format, indexed in
  `docs/adr/README.md`; `docs/DECISIONS.md` is removed. ADRs 0004 to 0009
  are reformatted with numbers from `results/manifest.json`; 0001 to 0003
  are superseded by 0010 (three-stage detection); 0011 (faithfulness),
  0012 (`qwen3.5:4b`, thinking off) and 0013 (organisation names) are new.
  All are accepted.

## [0.1.0]

- First public release: three-stage PII detection, typed-placeholder
  redaction, hierarchical chunking, hybrid retrieval, Ollama generation,
  post-generation audit, Streamlit dashboard.
