"""Smoke tests for `Settings`."""

from __future__ import annotations

import pytest

from guardian_rag.config import Settings, get_settings


def test_defaults_validate():
    settings = Settings()
    assert settings.gliner_threshold == 0.50
    assert settings.deberta_threshold == 0.85
    assert settings.child_chunk_tokens == 256
    assert settings.parent_chunk_tokens == 1024


def test_env_prefix_applied(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GUARDIAN_OLLAMA_MODEL", "llama3.2:1b")
    settings = get_settings()
    assert settings.ollama_model == "llama3.2:1b"


def test_threshold_bounds_enforced():
    with pytest.raises(ValueError, match="less than or equal"):
        Settings(gliner_threshold=1.5)
    with pytest.raises(ValueError, match="greater than or equal"):
        Settings(deberta_threshold=-0.1)


def test_settings_are_frozen():
    settings = Settings()
    with pytest.raises(ValueError, match=r"frozen"):
        settings.ollama_model = "different"  # type: ignore[misc]


def test_main_module_creates_app():
    """`uvicorn guardian_rag.main:app` needs the module-level `app` to exist."""
    from guardian_rag.main import app

    assert app.title == "Guardian RAG"
