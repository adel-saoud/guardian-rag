"""Embedder tests with a deterministic fake backend."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pytest

from guardian_rag.retrieval.embedder import Embedder


@dataclass
class HashEmbedder:
    """Deterministic backend that hashes each text into a fixed-length vector."""

    dim: int = 8
    calls: list[list[str]] = field(default_factory=list)

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            v = [digest[i % len(digest)] / 255.0 for i in range(self.dim)]
            vectors.append(v)
        return vectors


def test_embed_round_trip():
    backend = HashEmbedder(dim=8)
    emb = Embedder(backend, dim=8)
    vectors = emb.embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == 8 for v in vectors)


def test_embed_empty_returns_empty():
    backend = HashEmbedder()
    emb = Embedder(backend, dim=8)
    assert emb.embed([]) == []
    assert backend.calls == []


def test_embed_one_returns_single_vector():
    emb = Embedder(HashEmbedder(dim=8), dim=8)
    v = emb.embed_one("hello")
    assert len(v) == 8


def test_dim_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        Embedder(HashEmbedder(), dim=0)


def test_dim_property_exposed():
    emb = Embedder(HashEmbedder(dim=16), dim=16)
    assert emb.dim == 16


def test_backend_count_mismatch_raises():
    @dataclass
    class BadBackend:
        def encode(self, texts: list[str]) -> list[list[float]]:
            del texts
            return [[0.0]]

    emb = Embedder(BadBackend(), dim=1)
    with pytest.raises(RuntimeError, match="2 inputs"):
        emb.embed(["a", "b"])
