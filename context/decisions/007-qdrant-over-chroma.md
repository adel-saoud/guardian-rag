# ADR 007: Qdrant over Chroma

## Context

The two open-source vector databases most often considered for a
self-hosted RAG stack are Chroma and Qdrant. Both are MIT/Apache
licensed, both run in Docker, both have a Python client.

## Decision

Use Qdrant.

|                       | Chroma                            | Qdrant                          |
|-----------------------|-----------------------------------|---------------------------------|
| In-process mode       | Yes (DuckDB-backed)               | Yes (`location=":memory:"`)     |
| Production deployment | Pinned to specific node           | Sharded, replicated, snapshots  |
| Hybrid search         | Add-on                            | Built-in payload filtering      |
| Payload size limits   | Limited                           | Unbounded                       |
| Filter pushdown       | Limited                           | First-class                     |
| Test ergonomics       | Good                              | Excellent (in-memory client)    |

The deciding factor is `QdrantClient(location=":memory:")`. It gives the
test suite a real, hermetic vector store with no Docker, no `rm -rf` between
tests, and no network. Chroma's in-process mode requires DuckDB and is
slower to spin up.

## Consequences

- Tests run in milliseconds without Docker.
- Production stack runs Qdrant in Docker via `docker-compose.yml`.
- Single collection design (children with embedded parent payload) keeps
  the storage layer simple. See [[004-hierarchical-chunking]] for the
  parent/child rationale.
- Point IDs use dashed-UUID format. Qdrant's local mode validates these
  strictly; the chunker emits `str(uuid.uuid4())` so the same id is
  valid as a Qdrant point id and as a domain-model `chunk_id`.
