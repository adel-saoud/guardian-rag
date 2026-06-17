# ADR 004: Hierarchical chunking, parent 1024 / child 256

## Context

Standard RAG chunking forces a tradeoff: small chunks retrieve precisely
but lose paragraph context; large chunks generate better but rank poorly.
Privacy work amplifies the tradeoff because PII-detection precision is
best at sentence granularity.

## Decision

Use a hierarchical split:

- **Child** chunks: 256 tokens, 10% overlap (25 tokens). Embedded for
  retrieval. PII detection runs at this level.
- **Parent** chunks: 1024 tokens, no overlap. Stored in Qdrant payload,
  not embedded. Generation uses these as grounded context.

Each child stores its `parent_id` so retrieval-to-generation handoff is
a single payload lookup.

## Consequences

- Sentence-precise PII detection at the child level.
- Paragraph-level generation context at the parent level.
- Storage cost: parent text is duplicated across siblings
  (~4x). Acceptable for the demo. Production would split into a second
  collection.
- Implementation: `langchain_text_splitters.RecursiveCharacterTextSplitter`
  with `tiktoken` encoding `cl100k_base`.
- See [ADR 007](007-qdrant-over-chroma.md) for the storage layer choice.
