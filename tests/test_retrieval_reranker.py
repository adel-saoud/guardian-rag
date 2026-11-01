"""Cross-encoder reranker tests."""

from __future__ import annotations

from dataclasses import dataclass

from guardian_rag.retrieval.models import RetrievalHit
from guardian_rag.retrieval.reranker import Reranker


@dataclass
class LengthBackend:
    """Fake cross-encoder: longer text is more relevant."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(len(t)) for t in texts]


class BrokenBackend:
    def score(self, query: str, texts: list[str]) -> list[float]:
        raise RuntimeError("model unavailable")


class ShortBackend:
    def score(self, query: str, texts: list[str]) -> list[float]:
        return [1.0]


def _hit(chunk_id: str, text: str, score: float) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id="d",
        parent_id="p",
        text=text,
        order=0,
        score=score,
        fused_score=score,
    )


HITS = [_hit("a", "x", 0.9), _hit("b", "xxx", 0.5), _hit("c", "xx", 0.1)]


def test_rerank_orders_by_backend_score_and_truncates():
    reranked = Reranker(LengthBackend(), "fake").rerank("q", HITS, top_k=2)
    assert [h.chunk_id for h in reranked] == ["b", "c"]
    assert reranked[0].rerank_score == 3.0
    assert reranked[0].score == 3.0
    assert reranked[0].fused_score == 0.5


def test_rerank_empty_inputs():
    reranker = Reranker(LengthBackend(), "fake")
    assert reranker.rerank("q", [], top_k=3) == []
    assert reranker.rerank("q", HITS, top_k=0) == []


def test_rerank_failure_falls_back_to_fused_order():
    reranked = Reranker(BrokenBackend(), "fake").rerank("q", HITS, top_k=2)
    assert [h.chunk_id for h in reranked] == ["a", "b"]
    assert reranked[0].rerank_score is None


def test_rerank_score_count_mismatch_falls_back():
    reranked = Reranker(ShortBackend(), "fake").rerank("q", HITS, top_k=3)
    assert [h.chunk_id for h in reranked] == ["a", "b", "c"]
