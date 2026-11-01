# Roadmap

Items here are explicitly out of scope for the current build. Each entry
notes why it was deferred so the priority can be re-evaluated.

## Detection

- **Keep organisation names as a setting.** Company names are redacted
  today, which costs answers about which company did something (five of
  30 faithfulness questions). A per-type allow-list in the redactor,
  defaulting to redaction, would let a deployment keep them. Deferred
  until a deployment needs it; see `docs/adr/0013-redact-organisation-names.md`.
- **Cross-document entity resolution.** The same person under different
  names (`John Smith`, `J. Smith`, `Smith`) currently resolves to
  separate placeholders. Resolving this needs a coreference layer
  spanning the whole corpus. Out of scope until the eval suite has the
  labelled data to validate it.
- **Streaming PII detection** for documents that arrive as token streams
  rather than complete payloads. Useful for chat-log ingestion. Skipped
  because the current target (HR + contracts + support tickets) is all
  batch ingestion.
- **OCR for scanned PDFs.** `pypdf` handles text PDFs only. Scanned
  documents need Tesseract or similar. Adds large dependencies and a
  per-page latency budget; not justified by the demo corpus.

## Storage

- **GDPR right-to-erasure.** Deleting one user's PII from the vector
  store today requires deleting every child chunk whose payload
  references that user, plus rebuilding the BM25 index. A deletion
  endpoint should batch this.
- **Two-collection Qdrant layout.** The single-collection design
  duplicates parent text across siblings. At ~10K documents the
  duplication starts to matter; split into `guardian_children` +
  `guardian_parents` then.
- **Persistent BM25.** The in-memory `rank_bm25` index rebuilds on every
  process restart. Replace with a persistent inverted index (Tantivy via
  `meilisearch`, or DuckDB FTS) once the corpus passes ~100K children.

## Observability

- **Prometheus metrics export.** The autopilot and gateway companion
  projects ship `/metrics`. Guardian's `StatsCollector` is in-memory
  only. Adding `prometheus-client` is a small change that has not been
  prioritised yet.
- **Per-request tracing.** OpenTelemetry instrumentation on
  `/v1/ingest` and `/v1/query` would measure what the post-generation
  audit costs per query, which no eval records yet.

## Faithfulness

- **Independent judge.** The judge and the generator are the same local
  model (`qwen3.5:4b`). The judge compares answers from one generator on
  original and redacted context, so self-preference mostly cancels out of
  the gap, but not out of the absolute rates. A different-family judge
  would remove it; `GUARDIAN_OLLAMA_JUDGE_MODEL` already allows one.
- **More judge labels.** `eval/faithfulness/judge_labels.jsonl` has 30
  reviewed cases. Grow it so the agreement number has a usable
  confidence interval.
- **Identity questions.** The QA set asks only about non-PII facts.
  Checking that an answer about a person points to the right placeholder
  needs the redaction map in the judge input.

## Eval

- **Calibrated per-domain thresholds.** `GUARDIAN_DEBERTA_THRESHOLD` is
  one global value. HR documents have different false-positive
  distributions than contracts; per-category calibration would push
  precision higher without hurting recall.
- **Production-sized corpus.** 30 synthetic docs is enough for a
  demo. A real deployment needs 1000+ docs and per-domain
  manifests.

## Surface

- **Multipart file upload** for `/v1/ingest`. Today the endpoint takes
  JSON only. Adding `UploadFile` support is a small change for a real
  user demo.
- **Authentication.** No auth shipped. Production deployments should
  pair Guardian with the [`llm-gateway`](https://github.com/adel-saoud/llm-gateway)
  companion project for API-key-scoped quotas.

## Retrieval quality

- **Independent in-domain retrieval labels.** `--dataset domain` (270
  synthetic multi-chunk HR / contracts / support documents, 135 queries)
  ranks configurations where the golden smoke set saturates, but its
  corpus and queries share one author. Queries written by someone else,
  ideally over real redacted documents, are the next step before tuning
  `GUARDIAN_RERANK_CANDIDATES` or the reranker model.
- **Multilingual reranking.** `BAAI/bge-reranker-v2-m3` is the documented
  swap for non-English corpora. Not benchmarked yet.

## Injection screening

- **Multilingual classifier.** The default classifier is English-only.
  `meta-llama/Llama-Prompt-Guard-2-22M` covers 8 languages but is gated
  under the Llama license; it fits `GUARDIAN_INJECTION_MODEL` once its
  label name is set in `GUARDIAN_INJECTION_LABEL` (UNVERIFIED label name).
- **Threshold calibration on held-out data.** The default 0.5 is the
  model card's argmax point, not a value tuned on this domain.
- **Re-screen job.** Changing the screener re-screens stale parents at
  query time, one query at a time. A batch job that re-stamps the whole
  collection would remove that per-query cost.
