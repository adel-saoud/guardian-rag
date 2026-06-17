"""Loader tests for the golden corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from guardian_rag.eval.loader import load_golden_corpus
from guardian_rag.eval.models import GoldenDocument, GoldenEntity
from guardian_rag.pii.models import PiiEntityType


def _doc(document_id: str) -> GoldenDocument:
    return GoldenDocument(
        document_id=document_id,
        category="hr",
        text="hello world",
        entities=[GoldenEntity(text="hello", entity_type=PiiEntityType.PERSON, start=0, end=5)],
    )


def test_load_round_trips_documents(tmp_path: Path):
    a = _doc("a")
    b = _doc("b")
    (tmp_path / "a.json").write_text(a.model_dump_json(), encoding="utf-8")
    (tmp_path / "b.json").write_text(b.model_dump_json(), encoding="utf-8")
    loaded = load_golden_corpus(tmp_path)
    assert [d.document_id for d in loaded] == ["a", "b"]


def test_missing_root_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_golden_corpus(tmp_path / "nope")


def test_malformed_json_raises(tmp_path: Path):
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        load_golden_corpus(tmp_path)
