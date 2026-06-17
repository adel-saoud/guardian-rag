# Architecture

## End-to-end pipeline

```
Ingest path:
  Document -> Presidio (regex) -> GLiNER (NER) -> DeBERTa (conditional)
           -> typed-placeholder redactor -> hierarchical chunker
           -> embedder -> Qdrant (child vectors + parent payload)
           -> RedactionMap persisted to disk

Query path:
  Query -> redact query -> hybrid retrieval (dense + BM25)
        -> fetch parents for retrieved children -> Generator (Ollama)
        -> post-generation audit (same detector + redactor)
        -> response with X-Guardian-Pii-Audit header
```

## Module map

| Module                                 | Role                                                       |
|----------------------------------------|------------------------------------------------------------|
| `guardian_rag/config.py`               | Pydantic `Settings`, env-driven via `GUARDIAN_*`           |
| `guardian_rag/pii/`                    | Three-stage PII detection + typed-placeholder redactor     |
| `guardian_rag/pii/detector.py`         | Orchestrates Presidio + GLiNER + DeBERTa                   |
| `guardian_rag/pii/audit.py`            | Post-generation auditor, fail-closed                       |
| `guardian_rag/pii/_real_backends.py`   | Lazy GLiNER + DeBERTa model loads (pyright-excluded)       |
| `guardian_rag/ingestion/`              | Loader, hierarchical chunker, pipeline, redaction store    |
| `guardian_rag/retrieval/`              | Embedder, Qdrant store, BM25 index, hybrid retriever       |
| `guardian_rag/retrieval/_real_backends.py` | Lazy sentence-transformers backend (pyright-excluded)  |
| `guardian_rag/generation/`             | Ollama client, grounded prompt, citation verification      |
| `guardian_rag/eval/`                   | Golden corpus loader, recall + precision runner, judge     |
| `guardian_rag/api/`                    | FastAPI surface (5 routes), lifespan, dependency container |
| `guardian_rag/dashboard/app.py`        | Streamlit 3-panel dashboard (pyright-excluded)             |

## Boundary contracts

Every cross-module boundary uses frozen `pydantic.BaseModel`s with
`ConfigDict(frozen=True, extra="forbid")`. The boundaries are:

- `PiiEntity`, `RedactionMap`, `AuditResult` (`pii/models.py`)
- `Document`, `ParentChunk`, `ChildChunk`, `IngestionResult` (`ingestion/models.py`)
- `RetrievalHit`, `RetrievalResult` (`retrieval/models.py`)
- `ChatMessage`, `OllamaChatResponse`, `GroundedAnswer` (`generation/`)
- `GoldenDocument`, `GoldenEntity`, `EvalCounts`, `EvalReport` (`eval/models.py`)
- API request/response models (`api/schemas.py`)

## Lazy ML imports

Heavy ML backends (`gliner`, `transformers`, `sentence_transformers`)
are imported **only inside `_real_backends.py` constructors**, never at
module top-level. This means:

- `import guardian_rag.pii` does not load PyTorch.
- Tests substitute deterministic fakes that match the `Protocol` interface.
- CI runs the full suite without downloading weights.
- The FastAPI lifespan is the first place the heavy imports execute.

## Configuration surface

All runtime configuration is environment-driven via `pydantic-settings`
with the `GUARDIAN_` prefix. See `config.py` for the full list. Common
knobs:

| Variable                          | Default                                                    | Purpose                                      |
|-----------------------------------|------------------------------------------------------------|----------------------------------------------|
| `GUARDIAN_GLINER_MODEL`           | `urchade/gliner_multi_pii-v1`                              | GLiNER model id                              |
| `GUARDIAN_DEBERTA_THRESHOLD`      | `0.85`                                                     | Below this, DeBERTa validates GLiNER spans   |
| `GUARDIAN_EMBEDDING_MODEL`        | `sentence-transformers/all-MiniLM-L6-v2`                   | Retrieval embedder                           |
| `GUARDIAN_OLLAMA_MODEL`           | `llama3.2:3b`                                              | Generation model                             |
| `GUARDIAN_QDRANT_URL`             | `http://localhost:6333`                                    | Qdrant base URL                              |
| `GUARDIAN_AUDIT_FAIL_CLOSED`      | `true`                                                     | Redact on audit error vs re-raise            |

## Test strategy

- **Hermetic**, no network, no weights downloaded, no Docker required.
- **Deterministic**, fake backends behind `Protocol` interfaces, fixed
  embeddings via hash functions, in-memory Qdrant.
- **Coverage gate at 85%**, enforced in `pyproject.toml`.
- **130 tests** at the time of writing, ~93% coverage.

## Decision records

See `context/decisions/` for the 7 ADRs covering the major architectural
choices. Start with `001-presidio-gliner-orchestration.md`.
