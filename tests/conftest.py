"""Shared pytest fixtures.

Heavy ML stacks (GLiNER, DeBERTa, sentence-transformers, Qdrant client, Ollama)
are mocked or substituted with deterministic fakes. Nothing in the test suite
downloads weights or makes a network call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from guardian_rag.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Hermetic Settings with a tmp_path redaction-map directory."""
    return Settings(
        redaction_map_dir=str(tmp_path / "redaction_maps"),
        qdrant_url="http://localhost:6333",
        ollama_base_url="http://localhost:11434",
    )


@pytest.fixture
def golden_corpus_dir() -> Path:
    """Resolve the on-disk golden corpus directory."""
    return Path(__file__).resolve().parent.parent / "eval" / "golden"
