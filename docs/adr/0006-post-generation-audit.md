---
status: accepted
date: 2026-10-31
---

# Post-generation PII audit, last stage, fail-closed

## Context and Problem Statement

Redacting at ingestion keeps PII out of the index and the prompt, but an
LLM can still produce PII that was in no retrieved chunk: leaked
placeholders resolved by guessing, common-knowledge inference, or
training-data recall. Should every answer be checked before it leaves,
where in the pipeline, and what happens when the check itself fails?

## Considered Options

1. Trust ingestion redaction; no answer check.
2. Re-run the detector on every answer as the last stage, fail-open on
   errors.
3. Option 2, fail-closed on errors.

## Decision Outcome

Chosen option: 3.

- The same detector and redactor instances used at ingestion
  ([0010](0010-three-stage-pii-detection.md)) run over every generated
  answer (`pii/audit.py`).
- PII found: the leak is logged (entity types and counts), the answer is
  redacted with typed placeholders, and the response carries
  `X-Guardian-Pii-Audit: redacted`.
- Clean: the answer is returned unchanged with
  `X-Guardian-Pii-Audit: clean`.
- Audit error: the answer is replaced by
  `[Audit failed: response redacted for safety]` and `fail_closed=true`
  is set in the body. `GUARDIAN_AUDIT_FAIL_CLOSED=false` re-raises
  instead, for offline tooling only.

Draft reasoning, from the architecture doc: the audit is last so that
nothing after it can change the text the user receives, and fail-closed
means no code path can return an unaudited answer.

### Consequences

- The audit shares the detector's blind spots: a span GLiNER misses at
  ingestion it can miss again in the answer
  ([architecture](../architecture.md), failure modes).
- Not measured: the audit's latency per query. No eval records it; the
  [roadmap](../roadmap.md) tracing item would.
