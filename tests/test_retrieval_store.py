"""QdrantStore tests using an in-process Qdrant client."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pytest
from qdrant_client import QdrantClient

from guardian_rag.ingestion.models import ChildChunk, ParentChunk
from guardian_rag.retrieval.embedder import Embedder
from guardian_rag.retrieval.store import QdrantStore


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class ConstantEmbedder:
    """Deterministic embedder: each text maps to a vector mostly determined by
    a tiny lookup. Lets us hand-craft similarity expectations."""

    vectors_by_text: dict[str, list[float]]
    dim: int = 4

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors_by_text.get(t, [0.0] * self.dim) for t in texts]


def _store(embed_map: dict[str, list[float]], dim: int = 4) -> QdrantStore:
    client = QdrantClient(location=":memory:")
    embedder = Embedder(ConstantEmbedder(embed_map, dim=dim), dim=dim)
    return QdrantStore(client=client, embedder=embedder, collection_name="t", recreate=True)


def _make_chunks(
    parent_id: str | None = None,
    document_id: str = "doc-1",
    child_texts: list[str] | None = None,
    parent_text: str = "Parent paragraph context.",
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    pid = parent_id if parent_id is not None else _uuid()
    texts = child_texts if child_texts is not None else ["child a", "child b"]
    parent = ParentChunk(
        chunk_id=pid,
        document_id=document_id,
        text=parent_text,
        order=0,
    )
    children = [
        ChildChunk(
            chunk_id=_uuid(),
            document_id=document_id,
            parent_id=pid,
            text=text,
            order=i,
        )
        for i, text in enumerate(texts)
    ]
    return [parent], children


def test_upsert_and_count():
    store = _store({"child a": [1.0, 0.0, 0.0, 0.0], "child b": [0.0, 1.0, 0.0, 0.0]})
    parents, children = _make_chunks()
    store.upsert(parents, children)
    assert store.count() == 2


def test_search_dense_returns_closest_child():
    store = _store(
        {
            "child a": [1.0, 0.0, 0.0, 0.0],
            "child b": [0.0, 1.0, 0.0, 0.0],
            "query": [0.95, 0.05, 0.0, 0.0],
        }
    )
    parent_id = _uuid()
    parents, children = _make_chunks(parent_id=parent_id)
    store.upsert(parents, children)
    hits = store.search_dense("query", top_k=1)
    assert len(hits) == 1
    assert hits[0].text == "child a"
    assert hits[0].parent_id == parent_id


def test_search_dense_filters_by_document_id():
    store = _store(
        {
            "child a": [1.0, 0.0, 0.0, 0.0],
            "child b": [0.0, 1.0, 0.0, 0.0],
            "query": [1.0, 0.0, 0.0, 0.0],
        }
    )
    parents1, children1 = _make_chunks(document_id="doc-1")
    parents2, children2 = _make_chunks(
        document_id="doc-2",
        child_texts=["child a", "child b"],
    )
    store.upsert(parents1, children1)
    store.upsert(parents2, children2)

    hits = store.search_dense("query", top_k=4, document_id="doc-2")
    assert hits
    assert all(h.document_id == "doc-2" for h in hits)


def test_orphan_child_raises():
    store = _store({"x": [1.0, 0.0, 0.0, 0.0]})
    children = [
        ChildChunk(
            chunk_id=_uuid(),
            document_id="doc-1",
            parent_id=_uuid(),
            text="x",
            order=0,
        )
    ]
    with pytest.raises(ValueError, match="unknown parent"):
        store.upsert([], children)


def test_iter_children_for_bm25_returns_all_texts():
    store = _store(
        {
            "alpha": [1.0, 0.0, 0.0, 0.0],
            "beta": [0.0, 1.0, 0.0, 0.0],
        }
    )
    parents, children = _make_chunks(child_texts=["alpha", "beta"])
    store.upsert(parents, children)
    docs = store.iter_children_for_bm25()
    texts = {text for _id, text in docs}
    assert texts == {"alpha", "beta"}


def test_parents_from_hits_dedupes():
    store = _store(
        {
            "alpha": [1.0, 0.0, 0.0, 0.0],
            "beta": [0.95, 0.0, 0.0, 0.0],
            "query": [1.0, 0.0, 0.0, 0.0],
        }
    )
    parent_id = _uuid()
    parents, children = _make_chunks(parent_id=parent_id, child_texts=["alpha", "beta"])
    store.upsert(parents, children)
    hits = store.search_dense("query", top_k=2)
    parents_out = store.parents_from_hits(hits)
    assert len(parents_out) == 1
    assert parents_out[0].chunk_id == parent_id


def test_empty_upsert_is_a_noop():
    store = _store({})
    store.upsert([], [])
    assert store.count() == 0


def test_search_zero_top_k_returns_empty():
    store = _store({"query": [1.0, 0.0, 0.0, 0.0]})
    assert store.search_dense("query", top_k=0) == []


def test_parents_from_hits_carries_screening_verdict():
    store = _store({"child a": [1.0, 0.0, 0.0, 0.0]})
    parents, children = _make_chunks(child_texts=["child a"])
    stamped = [p.model_copy(update={"screener_id": "rules:v1"}) for p in parents]
    store.upsert(stamped, children)
    hits = store.search_dense("child a", top_k=1)
    loaded = store.parents_from_hits(hits)
    assert loaded[0].screener_id == "rules:v1"
    assert loaded[0].injection_quarantined is False
