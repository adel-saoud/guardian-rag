# ADR 006: Post-generation PII audit

## Context

Most RAG privacy projects stop at ingestion. The argument runs: "if the
vector store contains only redacted text, the generated answer is safe."
This is wrong. An LLM can paraphrase, infer, or hallucinate PII that was
**not in any retrieved chunk**, placeholder leak-through, common-knowledge
inference, training-data contamination.

## Decision

Re-run the same three-stage detector over **every generated answer**
before it reaches the user. If PII is detected:

1. Log the leak (entity types + counts).
2. Redact the answer via typed placeholders.
3. Set `X-Guardian-Pii-Audit: redacted` on the response.

If clean:

1. Set `X-Guardian-Pii-Audit: clean`.
2. Return the answer unchanged.

If the audit itself crashes:

1. Return `[Audit failed: response redacted for safety]`.
2. Set `fail_closed=true` in the response body.

## Consequences

- ~150ms added per query for the audit pass on a CPU. Acceptable for HR
  or contract workflows; flagged in the "Honest limitations" README
  section.
- The auditor and ingestion share **the same detector + redactor
  instances** so leaks are caught using identical rules to ingestion.
- See [[001-presidio-gliner-orchestration]] for the detector composition
  reused here.
- Fail-closed by default; `GUARDIAN_AUDIT_FAIL_CLOSED=false` re-raises
  the underlying exception (only useful for offline tooling).
