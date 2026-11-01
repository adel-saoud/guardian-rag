---
status: accepted
date: 2026-10-31
---

# Redact organisation names

## Context and Problem Statement

GLiNER detects organisations and Guardian redacts them as
`[ORGANIZATION_n]` (`GLINER_LABEL_TO_DOMAIN` in
`pii/gliner_recognizer.py`). A company name is often not personal data,
but it can identify a person when combined with other facts (a
one-person consultancy, an employer). The faithfulness eval shows the
cost: answers about which company did something lose the company. Should
organisation names stay redacted?

## Considered Options

1. Redact organisations, as today.
2. Keep organisation names in clear text.
3. Make it a setting, defaulting to redaction.

## Decision Outcome

Chosen option: 1, because Guardian is pseudonymisation and
re-identification by combining quasi-identifiers stays possible, so the
safer default keeps one more quasi-identifier out of the index. Option 3
is on the [roadmap](../roadmap.md) for deployments that need company names.

### Consequences

- Measured (`results/manifest.json`, suite `faithfulness`, n=30): the
  redacted-context rate is 13.3 points below the original-context rate,
  and all five questions that drop are about a company
  (`eval/results/faithfulness.json`, `cases`: contracts-001, -002, -006,
  -008, -010).
- Option 3 would need a per-type allow-list in the redactor and its own
  test; none exists today.
