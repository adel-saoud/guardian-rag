"""Hybrid retrieval tests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pytest
from qdrant_client import QdrantClient

from guardian_rag.ingestion.models import ChildChunk, ParentChunk
from guardian_rag.retrieval.embedder import Embedder
from guardian_rag.retrieval.hybrid import BM25Index, HybridRetriever
from guardian_rag.retrieval.reranker import Reranker
from guardian_rag.retrieval.store import QdrantStore


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class ConstantEmbedder:
    vectors_by_text: dict[str, list[float]]
    dim: int = 4

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors_by_text.get(t, [0.0] * self.dim) for t in texts]


def _store(embed_map: dict[str, list[float]]) -> QdrantStore:
    client = QdrantClient(location=":memory:")
    embedder = Embedder(ConstantEmbedder(embed_map, dim=4), dim=4)
    return QdrantStore(client=client, embedder=embedder, collection_name="t", recreate=True)


def _seed(store: QdrantStore, child_texts: list[str]) -> str:
    parent_id = _uuid()
    parents = [ParentChunk(chunk_id=parent_id, document_id="doc-1", text="parent text", order=0)]
    children = [
        ChildChunk(
            chunk_id=_uuid(),
            document_id="doc-1",
            parent_id=parent_id,
            text=text,
            order=i,
        )
        for i, text in enumerate(child_texts)
    ]
    store.upsert(parents, children)
    return parent_id


def test_bm25_index_ranks_term_overlap_higher():
    bm25 = BM25Index(
        [
            ("c0", "salary policy review process"),
            ("c1", "annual leave entitlement table"),
            ("c2", "harassment reporting procedure"),
        ]
    )
    hits = bm25.search("salary review", top_k=3)
    assert hits[0][0] == "c0"


def test_bm25_index_returns_empty_when_no_overlap():
    bm25 = BM25Index([("c0", "foo bar"), ("c1", "baz qux")])
    assert bm25.search("zzz", top_k=2) == []


def test_bm25_empty_corpus_returns_empty():
    bm25 = BM25Index()
    assert bm25.search("any query", top_k=2) == []


def test_hybrid_dense_only_when_bm25_empty():
    store = _store(
        {
            "alpha doc": [1.0, 0.0, 0.0, 0.0],
            "beta doc": [0.0, 1.0, 0.0, 0.0],
            "alpha query": [1.0, 0.0, 0.0, 0.0],
        }
    )
    parent_id = _seed(store, ["alpha doc", "beta doc"])
    retriever = HybridRetriever(store)
    result = retriever.retrieve("alpha query", top_k=1)
    assert result.children
    assert result.children[0].text == "alpha doc"
    assert result.parents
    assert result.parents[0].chunk_id == parent_id


def test_hybrid_blends_bm25_into_ranking():
    store = _store(
        {
            "alpha foo": [1.0, 0.0, 0.0, 0.0],
            "beta bar": [0.0, 1.0, 0.0, 0.0],
            "gamma baz": [0.0, 0.0, 1.0, 0.0],
            "delta qux": [0.0, 0.0, 0.0, 1.0],
            # The query vector leans toward `beta` to confuse dense alone.
            "alpha query": [0.0, 1.0, 0.0, 0.0],
        }
    )
    # BM25 IDF needs more than 2 docs to give meaningful scores. Seed
    # four chunks so the term-rare `alpha` still has positive IDF.
    _seed(store, ["alpha foo", "beta bar", "gamma baz", "delta qux"])
    retriever = HybridRetriever(store, bm25_weight=0.9)
    retriever.refresh()
    result = retriever.retrieve("alpha query", top_k=1)
    # With BM25 weighted heavily, the lexical overlap on `alpha` should
    # surface "alpha foo" above the dense-favoured "beta bar".
    assert result.children[0].text == "alpha foo"


def test_hybrid_top_k_zero_returns_empty():
    store = _store({"x": [1.0, 0.0, 0.0, 0.0]})
    _seed(store, ["x"])
    retriever = HybridRetriever(store)
    assert retriever.retrieve("anything", top_k=0).children == []


def test_invalid_weights_rejected():
    store = _store({})
    with pytest.raises(ValueError, match="bm25_weight"):
        HybridRetriever(store, bm25_weight=1.5)


def test_refresh_picks_up_new_documents():
    store = _store({"alpha": [1.0, 0.0, 0.0, 0.0]})
    _seed(store, ["alpha"])
    bm25 = BM25Index()
    retriever = HybridRetriever(store, bm25=bm25)
    assert bm25.size == 0
    retriever.refresh()
    assert bm25.size == 1


@dataclass
class ReverseTextBackend:
    """Fake cross-encoder that prefers texts sorting last alphabetically."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        order = sorted(texts)
        return [float(order.index(t)) for t in texts]


def test_hybrid_reranker_reorders_fused_pool():
    store = _store(
        {
            "alpha doc": [1.0, 0.0, 0.0, 0.0],
            "zeta doc": [0.5, 0.5, 0.0, 0.0],
            "alpha query": [1.0, 0.0, 0.0, 0.0],
        }
    )
    _seed(store, ["alpha doc", "zeta doc"])
    retriever = HybridRetriever(
        store, reranker=Reranker(ReverseTextBackend(), "fake"), rerank_candidates=5
    )
    result = retriever.retrieve("alpha query", top_k=1)
    # Dense alone ranks "alpha doc" first; the reranker overrides it.
    assert [h.text for h in result.children] == ["zeta doc"]
    assert result.children[0].rerank_score is not None
    assert result.children[0].fused_score is not None


def test_invalid_rerank_candidates_rejected():
    with pytest.raises(ValueError, match="rerank_candidates"):
        HybridRetriever(_store({}), rerank_candidates=0)


def test_quarantined_parent_never_retrieved():
    store = _store({"alpha doc": [1.0, 0.0, 0.0, 0.0], "alpha query": [1.0, 0.0, 0.0, 0.0]})
    parent_id = _uuid()
    parent = ParentChunk(
        chunk_id=parent_id,
        document_id="doc-1",
        text="poisoned",
        order=0,
        injection_quarantined=True,
        screener_id="rules:v1",
    )
    child = ChildChunk(
        chunk_id=_uuid(), document_id="doc-1", parent_id=parent_id, text="alpha doc", order=0
    )
    store.upsert([parent], [child])
    retriever = HybridRetriever(store)
    retriever.refresh()
    assert retriever.retrieve("alpha query", top_k=3).children == []
    assert store.iter_children_for_bm25() == []
