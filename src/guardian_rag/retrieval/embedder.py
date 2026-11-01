"""Embedding adapter.

The backend is a `Protocol` so tests can inject a deterministic fake. The
real backend (sentence-transformers) lives in `_real_backends.py` and is
imported lazily, mirroring the pattern used for GLiNER and DeBERTa.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbedderBackend(Protocol):
    """Anything that can turn text into a list of float vectors."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text. Order must be preserved."""
        ...


class Embedder:
    """Wraps an `EmbedderBackend` and exposes a uniform `embed()` interface.

    The wrapper is intentionally thin. Most of the value is the Protocol
    boundary, which lets the rest of the codebase depend on a single
    abstraction instead of sentence-transformers directly.
    """

    def __init__(self, backend: EmbedderBackend, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._backend = backend
        self._dim = dim

    @property
    def dim(self) -> int:
        """Vector size, used to create the Qdrant collection."""
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text, failing loudly if the backend drops any."""
        if not texts:
            return []
        vectors = self._backend.encode(texts)
        if len(vectors) != len(texts):
            raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(texts)} inputs")
        return vectors

    def embed_one(self, text: str) -> list[float]:
        """Embed a single query."""
        return self.embed([text])[0]
