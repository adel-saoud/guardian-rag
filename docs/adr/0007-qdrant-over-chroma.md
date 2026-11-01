---
status: accepted
date: 2026-10-31
---

# Qdrant over Chroma

## Context and Problem Statement

Guardian needs a self-hosted vector store for child embeddings with the
parent text in the payload ([0004](0004-hierarchical-chunking.md)), and a
test suite that runs without Docker or network
([architecture](../architecture.md), test strategy). Which store?

## Considered Options

1. Chroma.
2. Qdrant.

## Decision Outcome

Chosen option: 2, Qdrant.

- Tests use `QdrantClient(location=":memory:")`: a real in-process store,
  no Docker, nothing to clean between tests.
- The local stack runs `qdrant/qdrant` in Docker (`docker-compose.yml`);
  the minimum supported server is 1.17.0 because the pinned
  `qdrant-client` 1.18 supports servers at most one minor version away
  (`service_versions.py`).
- One collection: child vectors with the parent text in the payload, and
  a payload filter that excludes quarantined parents
  ([0009](0009-prompt-injection-screening.md)).

Draft reasoning, from the previous version of this ADR: the deciding
factor was the in-memory client and its test ergonomics.

### Consequences

- The test suite runs against a real vector store in milliseconds,
  without Docker.
- BM25 runs in-process with `rank-bm25`, not inside Qdrant
  (`retrieval/hybrid.py`), so hybrid scoring does not depend on the
  store.
- Point ids are dashed UUIDs (`str(uuid.uuid4())` in the chunker), valid
  both as Qdrant point ids and as `chunk_id`, because Qdrant's local mode
  validates ids strictly.
- The comparison table in the previous version of this ADR is not kept:
  its Chroma cells had no source.
