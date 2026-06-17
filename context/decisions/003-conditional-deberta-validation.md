# ADR 003: Conditional DeBERTa validation

## Context

GLiNER occasionally surfaces low-confidence spans (organisation-like
tokens, ambiguous names). Running DeBERTa zero-shot validation on every
span doubles inference cost. Skipping it entirely leaves a recall ceiling.

## Decision

DeBERTa fires only when GLiNER's span score is below
`GUARDIAN_DEBERTA_THRESHOLD` (default 0.85). Spans at or above the
threshold pass through unchanged.

The validator asks DeBERTa "is this text {hypothesis}?", where
`hypothesis` is a human-readable rendering of the entity type
(`"a personal name"`, `"an email address"`, etc). Spans whose DeBERTa
probability falls below `accept_threshold` (default 0.5) are dropped.

## Consequences

- Conditional cost: DeBERTa only runs on borderline spans, not all spans.
- Mirrors the two-stage routing pattern in
  `llm-cost-autopilot` ([[004-hierarchical-chunking]] for the other
  major two-stage decision).
- Tuning surface: lowering the score threshold turns DeBERTa into a hot
  path; raising it narrows the safety net.
- Future work: replace zero-shot NLI with a small dedicated PII verifier
  once the eval has enough labelled false-positives.
