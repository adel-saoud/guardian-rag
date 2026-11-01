# API

FastAPI app at `guardian_rag.main:app`. Run it with
`uv run uvicorn guardian_rag.main:app`; the OpenAPI page is at `/docs`.
Request and response models live in
[`src/guardian_rag/api/schemas.py`](../src/guardian_rag/api/schemas.py).

| Method | Path | Purpose |
|:--|:--|:--|
| `POST` | `/v1/ingest` | Detect, redact, chunk, screen, embed, store |
| `POST` | `/v1/query` | Query screen, hybrid retrieval, optional rerank, generation, post-generation audit |
| `GET` | `/v1/audit/{document_id}` | Full PII detection record for one document |
| `GET` | `/v1/stats` | Aggregate entity, leak and injection counters |
| `GET` | `/health` | Liveness |

## `POST /v1/ingest`

Body: `document_id`, `text`, optional `source`. The response carries entity,
parent and child counts, `n_quarantined_parents` and the first 240 characters
of the redacted text. The full text is never returned.

## `POST /v1/query`

Body: `query`, optional `top_k` (1 to 50, default 8) and `document_id` to
restrict retrieval to one document.

Response headers:

```
X-Guardian-Pii-Audit:    clean | redacted
X-Guardian-Injection:    clean | context_dropped | blocked
X-Guardian-Chunks-Used:  <n>
X-Guardian-Request-Id:   chatcmpl-<uuid>
```

A blocked query returns only `X-Guardian-Injection` and `X-Guardian-Request-Id`,
with HTTP 200, `injection_blocked: true` and a fixed refusal text, because it
stops before retrieval and generation.
