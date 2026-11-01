---
status: accepted
date: 2026-10-31
---

# Three-stage PII detection: Presidio, GLiNER, conditional DeBERTa

Supersedes [0001](0001-presidio-gliner-orchestration.md),
[0002](0002-gliner-over-privacy-filter.md) and
[0003](0003-conditional-deberta-validation.md).

## Context and Problem Statement

Every document is redacted before it is chunked, embedded or shown to the
LLM, and every answer is checked by the same detector
([0006](0006-post-generation-audit.md)). The detector must find two kinds
of PII: structured values with a fixed shape (emails, phone numbers,
IBANs, cards, IP addresses) and contextual entities (people,
organisations, locations). Which detectors, in what order, and how is a
borderline span settled?

## Decision Drivers

- Recall first: a missed span reaches the index as plain text.
- CPU only, no API key, no GPU.
- Languages beyond English for contextual entities.

## Considered Options

1. Presidio regex recognisers alone.
2. One NER model for everything.
3. Presidio for structured PII, then an NER model for contextual
   entities: `urchade/gliner_multi_pii-v1` or `openai/privacy-filter`.
4. Option 3 plus a validator on every NER span.
5. Option 3 plus a validator only on low-confidence NER spans.

## Decision Outcome

Chosen option: 5, Presidio then GLiNER, with DeBERTa only on uncertain
GLiNER spans.

- Stage 1: Presidio's pattern recognisers, called directly without the
  `AnalyzerEngine` or an NLP engine (`pii/presidio_engine.py`).
- Stage 2: `urchade/gliner_multi_pii-v1` (`GUARDIAN_GLINER_MODEL`),
  spans at or above `GUARDIAN_GLINER_THRESHOLD` (default 0.50); GLiNER
  labels map to Guardian types in `GLINER_LABEL_TO_DOMAIN`
  (`pii/gliner_recognizer.py`).
- Stage 3: `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`
  (`GUARDIAN_DEBERTA_MODEL`) validates only GLiNER spans scored below
  `GUARDIAN_DEBERTA_THRESHOLD` (default 0.85) and drops those it scores
  below `accept_threshold` (default 0.5, `pii/deberta_validator.py`).
- `pii/detector.py` merges the lists; the redactor resolves overlaps
  ([0005](0005-typed-placeholders.md)).

Draft reasoning, from the superseded ADRs and the architecture doc: regex
is deterministic on fixed-shape values and NER is needed for contextual
ones, so neither alone covers both. GLiNER was preferred over
`openai/privacy-filter` because 0002 recorded it as about 14 times
smaller, multilingual (EN, FR, DE, ES, PT, IT) and pluggable as a
Presidio `EntityRecognizer`, while privacy-filter was English-only. A
validator on every span would roughly double inference cost
([architecture](../architecture.md)), so it runs only where GLiNER is
unsure.

### Consequences

- Measured with Presidio and GLiNER (the DeBERTa stage is not part of this
  eval): 100.0% recall over 91 planted entities, 0.93 precision over 183
  detections (`results/manifest.json`, suite `pii/overall`). Lowest slice
  precision is contracts at 0.83 (`pii/contracts`).
- GLiNER redacts organisation names, which costs answers about companies;
  see [0013](0013-redact-organisation-names.md).
- The audit reuses this detector, so a span GLiNER misses at ingestion it
  can miss again in the answer ([architecture](../architecture.md),
  failure modes).
- Tuning surface: lowering `GUARDIAN_DEBERTA_THRESHOLD` puts DeBERTa on
  the hot path; raising it narrows the safety net.
- Three model collaborators, each behind a Protocol and faked in tests.
- Not measured: per-stage latency and the DeBERTa stage's effect on
  precision.
