# Roadmap

Items here are explicitly out of scope for the current build. Each entry
notes why it was deferred so the priority can be re-evaluated.

## Detection

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
  only. Adding `prometheus-client` is a small change; the brief just
  did not include it in the layer 8 budget.
- **Per-request tracing.** OpenTelemetry instrumentation on
  `/v1/ingest` and `/v1/query` would surface where the ~150ms audit
  budget actually goes.

## Faithfulness

- **`X-Guardian-Faithfulness` populated per-request.** Requires a judge
  call per query, which doubles latency. Either an opt-in flag or an
  async background task.
- **Pairwise judging.** Current judge is binary (yes/no) against a
  reference answer. Pairwise (compare two candidates head-to-head)
  gives tighter signal but costs more.
- **Larger QA set.** Currently 4 QA pairs across 30 docs. The brief
  called for 20.

## Eval

- **Calibrated per-domain thresholds.** `GUARDIAN_DEBERTA_THRESHOLD` is
  one global value. HR documents have different false-positive
  distributions than contracts; per-category calibration would push
  precision higher without hurting recall.
- **Production-sized corpus.** 30 synthetic docs is fine for a
  portfolio demo. A real deployment needs 1000+ docs and per-domain
  manifests.

## Surface

- **Multipart file upload** for `/v1/ingest`. Today the endpoint takes
  JSON only. Adding `UploadFile` support is a small change for a real
  user demo.
- **Authentication.** No auth shipped. Production deployments should
  pair Guardian with the [`llm-gateway`](https://github.com/adel-saoud/llm-gateway)
  companion project for API-key-scoped quotas.
