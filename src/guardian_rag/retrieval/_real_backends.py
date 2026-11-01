"""Production embedder backend.

Excluded from pyright and coverage. The `sentence_transformers` import is
performed lazily inside the constructor so the test suite, which substitutes
a deterministic fake, never has to install or download the model weights.
"""

from __future__ import annotations

from typing import Any, cast


class RealEmbedderBackend:
    """Sentence-transformers backend for the production embedder."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Unit-normalised vectors, so Qdrant cosine and dot product rank the same."""
        vectors = cast(
            "list[list[float]] | Any",
            self._model.encode(texts, normalize_embeddings=True),
        )
        return [list(map(float, v)) for v in vectors]


class RealCrossEncoderBackend:
    """Sentence-transformers `CrossEncoder` backend for the reranker.

    `predict` returns raw logits for ms-marco cross-encoders. Only the
    order matters for reranking, so no activation is applied.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """One raw logit per `(query, text)` pair."""
        if not texts:
            return []
        scores = cast(
            "list[float] | Any",
            self._model.predict([(query, text) for text in texts], show_progress_bar=False),
        )
        return [float(s) for s in scores]
