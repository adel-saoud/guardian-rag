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
        vectors = cast(
            "list[list[float]] | Any",
            self._model.encode(texts, normalize_embeddings=True),
        )
        return [list(map(float, v)) for v in vectors]
