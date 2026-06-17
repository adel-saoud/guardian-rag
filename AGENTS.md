# Agent Guidelines

## Project

Guardian RAG: privacy-safe RAG pipeline with two-stage PII detection at ingestion and a post-generation audit on every answer. EU AI Act Article 10-aligned by design. Runs fully locally with Ollama for generation. Zero API keys for PII detection.

## Companion projects

- [llm-regression-detector](https://github.com/adel-saoud/llm-regression-detector): CI quality gate for LLM systems
- [llm-gateway](https://github.com/adel-saoud/llm-gateway): RAG-aware LLM gateway with cost attribution
- [llm-cost-autopilot](https://github.com/adel-saoud/llm-cost-autopilot): two-stage zero-shot complexity router

Four projects, three axes:

- Cost: gateway tracks where the budget went; autopilot prevents it going to the wrong place.
- Quality: detector catches quality drops when prompts change.
- Privacy: guardian keeps personal data out of both the index and the response.

## Principles

1. No em-dashes anywhere in code, comments, or docs. Use commas, colons, semicolons, periods, or "that" clauses.
2. Pyright strict, 0 errors, 0 warnings. Tests get a relaxed executionEnvironment so pytest stubs do not noise the report.
3. Heavy ML imports lazy. Module-level imports must stay light so tools, scripts, and tests are cheap.
4. The post-generation audit must run on every query path. Never let an answer return without it. Audit failures are fail-closed (redact by default).
5. Typed placeholders preserve grammar. Never replace PII with blank tokens, since blanks break retrieval and generation context.

## Quick start

```bash
docker compose up
uv run python scripts/demo.py
uv run streamlit run src/guardian_rag/dashboard/app.py
```

## Commands

- `uv sync --extra dev --extra ml --extra dashboard`
- `uv run ruff check --fix .`
- `uv run ruff format .`
- `uv run pyright`
- `uv run pytest`
- `uv run python scripts/demo.py`
- `uv run python scripts/eval.py`

## Architecture decisions (do not re-debate)

- Presidio plus GLiNER orchestration, see `context/decisions/001-presidio-gliner-orchestration.md`
- GLiNER over `openai/privacy-filter`, see `context/decisions/002-gliner-over-privacy-filter.md`
- Conditional DeBERTa validation, see `context/decisions/003-conditional-deberta-validation.md`
- Hierarchical chunking (child 256 / parent 1024), see `context/decisions/004-hierarchical-chunking.md`
- Typed placeholder redaction, see `context/decisions/005-typed-placeholders.md`
- Post-generation audit, see `context/decisions/006-post-generation-audit.md`
- Qdrant over Chroma, see `context/decisions/007-qdrant-over-chroma.md`

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
- `dashboard/app.py`, `pii/_real_backends.py`, and `retrieval/_real_backends.py` are excluded from pyright. Keep them small so the exclusion does not hide real bugs.

## Key interview talking points

- Two-stage PII detection: "Presidio handles structured PII through regex with high precision. GLiNER catches unstructured PII like names and organisations through a span-based NER model. DeBERTa fires only when GLiNER is uncertain, so the safety net does not double inference cost."
- Why not just Presidio: "Presidio is a great PII library. It is not a RAG system. Guardian wraps Presidio plus GLiNER, then adds hierarchical chunking, retrieval, generation, and the post-generation audit on top."
- Post-generation audit: "Most RAG privacy projects protect ingestion only. The LLM can still paraphrase or infer PII from clean chunks. Running the same detection pipeline over the answer before it reaches the user closes that gap."
- Hierarchical chunking: "Child chunks at 256 tokens for precise retrieval and sentence-level PII detection. Parent chunks at 1024 tokens for generation context. Both stored in Qdrant. The same pattern is used widely in production RAG."

## Out of scope

- Re-identification attacks (this is pseudonymisation, not anonymisation)
- GDPR right-to-erasure (vector store deletion, roadmap)
- Streaming PII detection
- PDF OCR (text PDFs only)
- Prometheus metrics export (roadmap)
