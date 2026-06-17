# ADR 005: Typed placeholders, not blank tokens

## Context

Blank redaction (`John Smith`, `********`) destroys document structure
and breaks downstream RAG quality. A chunk that reads
"`******** approved ********`" is useless as generation context.

## Decision

Replace each PII span with a typed, indexed placeholder:

```
"John Smith's IBAN is BE71 0961 2345 6769"
                  becomes
"[PERSON_1]'s IBAN is [BANK_ACCOUNT_1]"
```

The same source span resolves to the same placeholder within a document
(`John Smith` -> `[PERSON_1]` on every occurrence). Different spans get
different indices, isolated per entity type.

The mapping lives in a `RedactionMap` stored separately
(see `pii/redaction_store.py`). The map is **never embedded**.

## Consequences

- Retrieval still works: `[PERSON_1] approved [PERSON_2]` is useful context.
- Generation prompt explicitly teaches the LLM to treat placeholders as
  opaque tokens (see `generation/prompt.py`).
- Same-span consistency means the dashboard can render a clean before/after.
- Cross-document entity resolution (the same person under different
  spellings) is roadmap.
- See `pii/redactor.py` for the overlap-resolution rule (highest score wins).
