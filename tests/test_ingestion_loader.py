"""Loader tests for plain text, markdown, and unsupported types."""

from __future__ import annotations

from pathlib import Path

import pytest

from guardian_rag.ingestion.loader import load_path


def test_load_plain_text(tmp_path: Path):
    path = tmp_path / "policy.txt"
    path.write_text("Hello world.", encoding="utf-8")
    assert load_path(path) == "Hello world."


def test_load_markdown(tmp_path: Path):
    path = tmp_path / "policy.md"
    path.write_text("# Heading\n\nBody.", encoding="utf-8")
    assert load_path(path) == "# Heading\n\nBody."


def test_unsupported_extension_raises(tmp_path: Path):
    path = tmp_path / "file.docx"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_path(path)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_path(tmp_path / "missing.txt")
