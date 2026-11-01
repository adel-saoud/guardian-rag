---
status: superseded by 0010
date: 2026-10-31
---

# Presidio + GLiNER orchestration

Superseded by [0010](0010-three-stage-pii-detection.md), which merges 0001, 0002 and 0003. Kept for history; not updated since.

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
- See [ADR 002](0002-gliner-over-privacy-filter.md) for why GLiNER, not the
  Presidio + `openai/privacy-filter` combination some references suggest.
- See [ADR 003](0003-conditional-deberta-validation.md) for the safety net when
  GLiNER spans land in the low-confidence range.
