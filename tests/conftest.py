"""Shared pytest fixtures.

Heavy ML stacks (GLiNER, DeBERTa, sentence-transformers, Qdrant client, Ollama)
are mocked or substituted with deterministic fakes. Nothing in the test suite
downloads weights or makes a network call.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def golden_corpus_dir() -> Path:
    """Resolve the on-disk golden corpus directory."""
    return Path(__file__).resolve().parent.parent / "eval" / "golden"
