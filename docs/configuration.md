# Configuration

Every setting is read from the environment by `pydantic-settings`, with the
`GUARDIAN_` prefix, or from a `.env` file in the working directory. The source
of truth is [`src/guardian_rag/config.py`](../src/guardian_rag/config.py).

A `GUARDIAN_*` name that matches no setting below stops startup with
`UnknownSettingError` and the closest valid name, so a typo or a removed
setting never silently falls back to the default.

## PII detection

| Variable | Default | Purpose |
|:--|:--|:--|
| `GUARDIAN_GLINER_MODEL` | `urchade/gliner_multi_pii-v1` | GLiNER model id |
| `GUARDIAN_GLINER_THRESHOLD` | `0.50` | Minimum span score for GLiNER to surface a candidate |
| `GUARDIAN_DEBERTA_MODEL` | `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` | Zero-shot validator for low-confidence spans |
| `GUARDIAN_DEBERTA_THRESHOLD` | `0.85` | GLiNER score below which DeBERTa validates the span |
| `GUARDIAN_AUDIT_FAIL_CLOSED` | `true` | Redact the answer if the post-generation audit raises, instead of re-raising |
| `GUARDIAN_REDACTION_MAP_DIR` | `data/redaction_maps` | Where per-document redaction maps are written |

## Chunking and retrieval

| Variable | Default | Purpose |
|:--|:--|:--|
| `GUARDIAN_CHILD_CHUNK_TOKENS` | `256` | Child chunk size, used for retrieval |
| `GUARDIAN_PARENT_CHUNK_TOKENS` | `1024` | Parent chunk size, used as generation context |
| `GUARDIAN_CHILD_CHUNK_OVERLAP` | `25` | Token overlap between child chunks |
| `GUARDIAN_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Retrieval embedder |
| `GUARDIAN_EMBEDDING_DIM` | `384` | Embedding dimension, must match the model |
| `GUARDIAN_BM25_WEIGHT` | `0.4` | BM25 share in the dense + BM25 fusion |
| `GUARDIAN_RERANKER_ENABLED` | `false` | Cross-encoder rerank of the fused pool ([ADR 0008](adr/0008-cross-encoder-reranking.md)) |
| `GUARDIAN_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Reranker; `BAAI/bge-reranker-v2-m3` for multilingual text |
| `GUARDIAN_RERANK_CANDIDATES` | `30` | Fused candidates handed to the reranker |

The number of chunks returned per query is a request field (`top_k` on
`POST /v1/query`, default 8), not a setting.

## Prompt-injection screening

| Variable | Default | Purpose |
|:--|:--|:--|
| `GUARDIAN_QUERY_INJECTION_SCREENER` | `classifier+rules` | `classifier`, `rules`, `classifier+rules` or `off` |
| `GUARDIAN_CHUNK_INJECTION_SCREENER` | `rules` | Same modes, for parent chunks at ingestion ([ADR 0009](adr/0009-prompt-injection-screening.md)) |
| `GUARDIAN_INJECTION_MODEL` | `protectai/deberta-v3-base-prompt-injection-v2` | Injection classifier (English only) |
| `GUARDIAN_INJECTION_LABEL` | `INJECTION` | Name of the injection class in the classifier's `id2label` |
| `GUARDIAN_INJECTION_THRESHOLD` | `0.5` | Probability at or above which text is flagged |

## Services

| Variable | Default | Purpose |
|:--|:--|:--|
| `GUARDIAN_QDRANT_URL` | `http://localhost:6333` | Qdrant base URL, or `:memory:` for an in-process instance |
| `GUARDIAN_QDRANT_COLLECTION` | `guardian_chunks` | Qdrant collection name |
| `GUARDIAN_QDRANT_API_KEY` | unset | Qdrant API key, if the server needs one |
| `GUARDIAN_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama base URL |
| `GUARDIAN_OLLAMA_MODEL` | `qwen3.5:4b` | Generation model |
| `GUARDIAN_OLLAMA_JUDGE_MODEL` | `qwen3.5:4b` | Faithfulness judge model, used by `scripts/eval_faithfulness.py` |
| `GUARDIAN_OLLAMA_THINK` | `false` | Sent as Ollama's `think` field on every chat. Qwen3.5 thinks by default, which multiplies output tokens and can put a trace before the judge's yes or no |
| `GUARDIAN_OLLAMA_TIMEOUT_S` | `60.0` | Per-request Ollama timeout in seconds |
