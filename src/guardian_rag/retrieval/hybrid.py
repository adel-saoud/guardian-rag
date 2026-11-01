"""Hybrid retrieval: dense (Qdrant) + BM25, fused by weighted min-max,
then optionally reranked by a cross-encoder.

Dense retrieval handles paraphrase. BM25 catches rare-term queries that
cosine similarity tends to mis-rank. Fusing them gives both safety nets.

Score fusion uses min-max normalisation within each ranker, then a
weighted sum. Min-max keeps the math simple and is robust to the very
different score ranges Qdrant (cosine, ~0..1) and BM25 (positive,
unbounded) produce.

When a `Reranker` is wired in, fusion only builds the candidate pool
(`rerank_candidates` chunks) and the cross-encoder picks the final top-K.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from guardian_rag.retrieval.models import RetrievalHit, RetrievalResult
from guardian_rag.retrieval.reranker import Reranker
from guardian_rag.retrieval.store import QdrantStore

logger = structlog.get_logger(__name__)


_TOKEN_RE = re.compile(r"\w+")

CANDIDATES_PER_RANKER_MULTIPLIER = 3
"""Each ranker returns this many times the pool size before fusion.

A chunk ranked low by one ranker but high by the other still enters the
fused pool. Raising it trades latency for fused recall.
"""


def _tokenise(text: str) -> list[str]:
    """Lowercase tokenisation by alphanumeric run. Good enough for BM25."""
    return _TOKEN_RE.findall(text.lower())


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    """Scale scores to [0, 1]; all-equal scores map to 1.0 so a lone hit is not zeroed."""
    if not scores:
        return {}
    values = list(scores.values())
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-9:
        return dict.fromkeys(scores, 1.0)
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class BM25Index:
    """In-memory BM25 index over child chunk texts.

    The index is rebuilt by `HybridRetriever.refresh()` after ingestion and
    lives in memory only; a persistent inverted index is on the roadmap.
    """

    def __init__(self, chunks: list[tuple[str, str]] | None = None) -> None:
        self._chunk_ids: list[str] = []
        self._index: BM25Okapi | None = None
        if chunks:
            self.rebuild(chunks)

    def rebuild(self, chunks: list[tuple[str, str]]) -> None:
        """Replace the index with `(chunk_id, text)` pairs. `rank_bm25` has no incremental add."""
        self._chunk_ids = [chunk_id for chunk_id, _ in chunks]
        corpus = [_tokenise(text) for _, text in chunks]
        if not corpus:
            self._index = None
            return
        self._index = BM25Okapi(corpus)

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunk_ids)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return up to `top_k` `(chunk_id, bm25_score)` pairs with a non-zero score."""
        if self._index is None or not self._chunk_ids or top_k <= 0:
            return []
        tokens = _tokenise(query)
        if not tokens:
            return []
        raw_scores: Any = self._index.get_scores(tokens)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        scores: list[float] = [float(s) for s in raw_scores]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        ranked = sorted(zip(self._chunk_ids, scores, strict=True), key=lambda x: x[1], reverse=True)
        # Filter out exactly-zero matches; BM25Okapi returns 0 for docs
        # with no token overlap and we do not want those polluting the
        # fused ranking.
        return [(chunk_id, score) for chunk_id, score in ranked[:top_k] if score > 0.0]


class HybridRetriever:
    """Combine Qdrant dense search with an in-memory BM25 index.

    Parameters
    ----------
    store:
        QdrantStore used for dense retrieval and parent fetch.
    bm25:
        Optional BM25 index. When None or empty, retrieval is dense-only.
    bm25_weight:
        Weight on the normalised BM25 score in [0, 1]. Dense weight is
        `1 - bm25_weight`. Default 0.4, matching the configuration default.
    reranker:
        Optional cross-encoder stage. When None, the fused order is final.
    rerank_candidates:
        Size of the fused pool handed to the reranker. Ignored without a
        reranker. Never smaller than the requested `top_k`.
    """

    def __init__(
        self,
        store: QdrantStore,
        bm25: BM25Index | None = None,
        bm25_weight: float = 0.4,
        reranker: Reranker | None = None,
        rerank_candidates: int = 30,
    ) -> None:
        if not 0.0 <= bm25_weight <= 1.0:
            raise ValueError("bm25_weight must be in [0, 1]")
        if rerank_candidates <= 0:
            raise ValueError("rerank_candidates must be positive")
        self._store = store
        self._bm25 = bm25 if bm25 is not None else BM25Index()
        self._bm25_weight = bm25_weight
        self._dense_weight = 1.0 - bm25_weight
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates

    def refresh(self) -> None:
        """Rebuild the BM25 index from the current Qdrant contents."""
        self._bm25.rebuild(self._store.iter_children_for_bm25())
        logger.info("hybrid.bm25.refreshed", n_chunks=self._bm25.size)

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        document_id: str | None = None,
    ) -> RetrievalResult:
        """Return the top-K fused hits plus their parent context."""
        if top_k <= 0:
            return RetrievalResult(query=query, children=[], parents=[])

        pool_k = max(top_k, self._rerank_candidates) if self._reranker is not None else top_k
        candidate_k = pool_k * CANDIDATES_PER_RANKER_MULTIPLIER
        dense_hits = self._store.search_dense(query, candidate_k, document_id=document_id)
        bm25_pairs = self._bm25.search(query, candidate_k) if self._bm25.size > 0 else []

        dense_scores = {hit.chunk_id: hit.score for hit in dense_hits}
        bm25_scores = dict(bm25_pairs)

        norm_dense = _min_max(dense_scores)
        norm_bm25 = _min_max(bm25_scores)

        fused: dict[str, float] = {}
        for chunk_id in set(norm_dense) | set(norm_bm25):
            fused[chunk_id] = self._dense_weight * norm_dense.get(
                chunk_id, 0.0
            ) + self._bm25_weight * norm_bm25.get(chunk_id, 0.0)

        top_ids = sorted(fused, key=lambda c: fused[c], reverse=True)[:pool_k]

        # Rehydrate hits: prefer the dense object (it has the parent
        # payload). BM25-only hits get fetched individually.
        by_id: dict[str, RetrievalHit] = {hit.chunk_id: hit for hit in dense_hits}
        missing_ids = [c for c in top_ids if c not in by_id]
        if missing_ids:
            by_id.update(self._store.fetch_by_ids(missing_ids))

        final_hits: list[RetrievalHit] = []
        for chunk_id in top_ids:
            base = by_id.get(chunk_id)
            if base is None:
                continue
            final_hits.append(
                base.model_copy(
                    update={
                        "score": fused[chunk_id],
                        "dense_score": dense_scores.get(chunk_id),
                        "bm25_score": bm25_scores.get(chunk_id),
                        "fused_score": fused[chunk_id],
                    }
                )
            )

        if self._reranker is not None:
            final_hits = self._reranker.rerank(query, final_hits, top_k)

        parents = self._store.parents_from_hits(final_hits)
        return RetrievalResult(query=query, children=final_hits, parents=parents)
