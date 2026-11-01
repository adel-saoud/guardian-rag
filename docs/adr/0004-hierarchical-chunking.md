---
status: accepted
date: 2026-10-31
---

# Hierarchical chunking, parent 1024 / child 256 tokens

## Context and Problem Statement

Small chunks retrieve precisely but lose paragraph context; large chunks
give the generator better context but rank poorly. Redacted text makes
this worse: placeholders such as `[PERSON_1]` carry less meaning than the
names they replace, so the generator needs more surrounding text. What
chunk sizes should retrieval and generation use?

## Considered Options

1. One flat chunk size for both retrieval and generation.
2. Two levels: small children for retrieval, larger parents for
   generation.

## Decision Outcome

Chosen option: 2.

- Child chunks: 256 tokens, 25 tokens of overlap (about 10%), embedded
  for retrieval.
- Parent chunks: 1024 tokens, no overlap, stored in the Qdrant payload,
  not embedded; the generator reads parents.
- Each child stores its `parent_id`, so retrieval hands off to generation
  with one payload lookup.
- `langchain_text_splitters.RecursiveCharacterTextSplitter` with the
  `tiktoken` encoding `cl100k_base` (`ingestion/chunker.py`).

PII detection runs on the whole document before chunking
(`ingestion/pipeline.py`), so chunks only ever hold redacted text.

Draft reasoning, from the previous version of this ADR: children keep
retrieval precise, parents keep paragraph-level context for generation.

### Consequences

- Parent text is duplicated in every sibling child's payload. Acceptable
  for this scale; a larger deployment would move parents to a second
  collection.
- Injection screening and quarantine work per parent, the unit the
  generator reads ([0009](0009-prompt-injection-screening.md)).
- Retrieval numbers for this layout: see
  [0008](0008-cross-encoder-reranking.md).
- Storage layer: [0007](0007-qdrant-over-chroma.md).
- Not measured: other chunk sizes; no eval compares them.
