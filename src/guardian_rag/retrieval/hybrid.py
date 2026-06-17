"""Hybrid retrieval: dense (Qdrant) + BM25, fused by weighted min-max.

Dense retrieval handles paraphrase. BM25 catches rare-term queries that
cosine similarity tends to mis-rank. Fusing them gives both safety nets.

Score fusion uses min-max normalisation within each ranker, then a
weighted sum. Min-max keeps the math simple and is robust to the very
different score ranges Qdrant (cosine, ~0..1) and BM25 (positive,
unbounded) produce.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from guardian_rag.retrieval.models import RetrievalHit, RetrievalResult
from guardian_rag.retrieval.store import QdrantStore

logger = structlog.get_logger(__name__)


_TOKEN_RE = re.compile(r"\w+")


def _tokenise(text: str) -> list[str]:
    """Lowercase tokenisation by alphanumeric run. Good enough for BM25."""
    return _TOKEN_RE.findall(text.lower())


def _min_max(scores: dict[str, float]) -> dict[str, float]:
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

    The index is rebuilt by `HybridRetriever.refresh()` after ingestion.
    For the portfolio demo this stays in memory; production deployments
    would swap in a persistent inverted index.
    """

    def __init__(self, docs: list[tuple[str, str]] | None = None) -> None:
        self._chunk_ids: list[str] = []
        self._index: BM25Okapi | None = None
        if docs:
            self.rebuild(docs)

    def rebuild(self, docs: list[tuple[str, str]]) -> None:
        self._chunk_ids = [chunk_id for chunk_id, _ in docs]
        corpus = [_tokenise(text) for _, text in docs]
        if not corpus:
            self._index = None
            return
        self._index = BM25Okapi(corpus)

    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
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
    candidate_multiplier:
        How many candidates to pull from each ranker before fusion. Larger
        values improve recall of the fused result at the cost of latency.
    """

    def __init__(
        self,
        store: QdrantStore,
        bm25: BM25Index | None = None,
        bm25_weight: float = 0.4,
        candidate_multiplier: int = 3,
    ) -> None:
        if not 0.0 <= bm25_weight <= 1.0:
            raise ValueError("bm25_weight must be in [0, 1]")
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")
        self._store = store
        self._bm25 = bm25 if bm25 is not None else BM25Index()
        self._bm25_weight = bm25_weight
        self._dense_weight = 1.0 - bm25_weight
        self._candidate_multiplier = candidate_multiplier

    def refresh(self) -> None:
        """Rebuild the BM25 index from the current Qdrant contents."""
        docs = self._store.iter_children_for_bm25()
        self._bm25.rebuild(docs)
        logger.info("hybrid.bm25.refreshed", n_docs=self._bm25.size)

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        document_id: str | None = None,
    ) -> RetrievalResult:
        """Return the top-K fused hits plus their parent context."""
        if top_k <= 0:
            return RetrievalResult(query=query, children=[], parents=[])

        candidate_k = top_k * self._candidate_multiplier
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

        top_ids = sorted(fused, key=lambda c: fused[c], reverse=True)[:top_k]

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
                    }
                )
            )

        parents = self._store.parents_from_hits(final_hits)
        return RetrievalResult(query=query, children=final_hits, parents=parents)
