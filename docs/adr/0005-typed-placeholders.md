---
status: accepted
date: 2026-10-31
---

# Typed placeholders, not blank tokens

## Context and Problem Statement

Blank redaction (`********`) destroys document structure: a chunk that
reads "`******** approved ********`" is useless for retrieval and as
generation context. How should a detected PII span be replaced?

## Considered Options

1. Blank masking (`********` or `[REDACTED]`).
2. Typed, indexed placeholders (`[PERSON_1]`, `[BANK_ACCOUNT_1]`).

## Decision Outcome

Chosen option: 2.

```
"John Smith's IBAN is BE71 0961 2345 6769"
                  becomes
"[PERSON_1]'s IBAN is [BANK_ACCOUNT_1]"
```

- The same source span gets the same placeholder everywhere in a document;
  different spans get different indices, counted per entity type.
- When spans overlap, the highest score wins (`pii/redactor.py`).
- The mapping is a `RedactionMap` written to disk by
  `ingestion/redaction_store.py`. It is never embedded and never sent to
  the LLM.
- The system prompt tells the model to treat placeholders as opaque
  tokens and not to guess the original values (`generation/prompt.py`).

Draft reasoning, from the previous version of this ADR: typed placeholders
keep grammar and roles, so `[PERSON_1] approved [PERSON_2]` stays useful
for retrieval and generation, and the dashboard can show a clean
before/after.

### Consequences

- An answer about an entity names its placeholder, not the value. For
  organisations this costs answers; see
  [0013](0013-redact-organisation-names.md).
- Placeholders are per document: the same person in two documents gets
  two unrelated tokens. Cross-document resolution is on the
  [roadmap](../roadmap.md).
- This is pseudonymisation, not anonymisation: whoever holds the redaction
  maps can reverse it.
