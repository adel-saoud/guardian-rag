"""Cross-encoder reranking over the hybrid candidate pool.

Hybrid fusion scores each chunk against the query independently (a dot
product, a term-frequency formula). A cross-encoder reads the query and the
chunk together, which ranks better but costs one forward pass per pair. The
standard compromise is retrieve-then-rerank: fuse a wide candidate pool
cheaply, then rerank only that pool.

The backend is a `Protocol` so tests inject a deterministic fake. The
sentence-transformers backend lives in `_real_backends.py`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog

from guardian_rag.retrieval.models import RetrievalHit

logger = structlog.get_logger(__name__)


@runtime_checkable
class RerankerBackend(Protocol):
    """Anything that scores `(query, text)` pairs. Higher means more relevant."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text, in input order."""
        ...


class Reranker:
    """Reorder retrieval hits by cross-encoder relevance.

    Ranking is not a security control, so a backend failure falls back to
    the incoming (fused) order instead of failing the request. The
    injection screener and the post-generation audit are the fail-closed
    stages; the reranker must never be the reason a query errors.
    """

    def __init__(self, backend: RerankerBackend, model_name: str) -> None:
        """`model_name` is only recorded in logs, so failures name the model."""
        self._backend = backend
        self._model_name = model_name

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        """Return the `top_k` hits ordered by rerank score, with `rerank_score` set."""
        if not hits or top_k <= 0:
            return []
        try:
            scores = self._backend.score(query, [hit.text for hit in hits])
        except Exception:
            logger.exception("reranker.failed", model=self._model_name, n_hits=len(hits))
            return hits[:top_k]
        if len(scores) != len(hits):
            logger.error(
                "reranker.score_count_mismatch",
                model=self._model_name,
                n_hits=len(hits),
                n_scores=len(scores),
            )
            return hits[:top_k]
        ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            hit.model_copy(update={"score": score, "rerank_score": score})
            for hit, score in ranked[:top_k]
        ]
