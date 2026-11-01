"""Smoke tests for `Settings`."""

from __future__ import annotations

from pathlib import Path

import pytest

from guardian_rag.config import (
    Settings,
    UnknownSettingError,
    get_settings,
    reject_unknown_env_names,
)


def test_defaults_validate():
    settings = Settings()
    assert settings.gliner_threshold == 0.50
    assert settings.deberta_threshold == 0.85
    assert settings.child_chunk_tokens == 256
    assert settings.parent_chunk_tokens == 1024
    assert settings.ollama_think is False


def test_env_prefix_applied(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GUARDIAN_OLLAMA_MODEL", "qwen3.5:0.8b")
    settings = get_settings()
    assert settings.ollama_model == "qwen3.5:0.8b"


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


def test_misspelled_env_var_is_rejected_with_closest_name(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GUARDIAN_OLAMA_MODEL", "qwen3.5:0.8b")
    with pytest.raises(UnknownSettingError, match="did you mean GUARDIAN_OLLAMA_MODEL"):
        Settings()


def test_removed_env_var_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GUARDIAN_TOP_K", "8")
    with pytest.raises(UnknownSettingError, match="GUARDIAN_TOP_K"):
        get_settings()


def test_unknown_name_in_dotenv_is_rejected(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("GUARDIAN_RERANKER_ENABLE=true\n", encoding="utf-8")
    with pytest.raises(UnknownSettingError, match="did you mean GUARDIAN_RERANKER_ENABLED"):
        Settings(_env_file=env_file)  # type: ignore[call-arg]


def test_unrelated_and_lowercase_names_pass():
    reject_unknown_env_names(
        ["PATH", "OTEL_SERVICE_NAME", "guardian_ollama_model"], ["GUARDIAN_OLLAMA_MODEL"]
    )


def test_unknown_name_without_close_match_says_so():
    with pytest.raises(UnknownSettingError, match="no setting with a close name"):
        reject_unknown_env_names(["GUARDIAN_ZZZ"], ["GUARDIAN_OLLAMA_MODEL"])
