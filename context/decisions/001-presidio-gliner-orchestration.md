# ADR 001: Presidio + GLiNER orchestration

## Context

Guardian RAG must detect both **structured** PII (emails, phone numbers,
IBANs, credit cards, IP addresses) and **unstructured** PII (people,
organisations, locations). A single ML model is poorly suited to both
classes: regex beats ML on deterministic patterns, ML beats regex on
contextual entities.

## Decision

Run Presidio first (regex recognisers, ~5ms), then GLiNER (span-based
NER, ~50ms). Merge the entity lists, deduplicate overlaps, redact via
typed placeholders.

We use Presidio's `PatternRecognizer` subclasses directly without
loading the full `AnalyzerEngine`. That skips spaCy entirely so Stage 1
stays within its latency budget.

## Consequences

- High structured-PII precision (regex is deterministic).
- High unstructured-PII recall (GLiNER is the current best span model).
- Two collaborators to wire, mocked separately in tests.
- See [[002-gliner-over-privacy-filter]] for why GLiNER, not the
  Presidio + `openai/privacy-filter` combination some references suggest.
- See [[003-conditional-deberta-validation]] for the safety net when
  GLiNER spans land in the low-confidence range.
