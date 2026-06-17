"""Filesystem persistence tests for `RedactionMap`."""

from __future__ import annotations

from pathlib import Path

import pytest

from guardian_rag.ingestion.redaction_store import RedactionMapStore
from guardian_rag.pii.models import PiiEntity, PiiEntityType, PiiStage, RedactionMap


def _map(document_id: str = "doc-1") -> RedactionMap:
    return RedactionMap(
        document_id=document_id,
        placeholders={"[PERSON_1]": "John Smith"},
        entities=[
            PiiEntity(
                entity_type=PiiEntityType.PERSON,
                start=0,
                end=10,
                score=0.9,
                text="John Smith",
                stage=PiiStage.GLINER,
            )
        ],
    )


def test_save_then_load_round_trip(tmp_path: Path):
    store = RedactionMapStore(tmp_path)
    original = _map()
    store.save(original)
    loaded = store.load("doc-1")
    assert loaded == original


def test_save_is_atomic_no_tmp_leftovers(tmp_path: Path):
    store = RedactionMapStore(tmp_path)
    store.save(_map())
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []


def test_exists_and_delete(tmp_path: Path):
    store = RedactionMapStore(tmp_path)
    assert not store.exists("doc-1")
    store.save(_map())
    assert store.exists("doc-1")
    assert store.delete("doc-1") is True
    assert not store.exists("doc-1")
    assert store.delete("doc-1") is False


def test_unsafe_document_id_rejected(tmp_path: Path):
    store = RedactionMapStore(tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        store.save(_map(document_id="../escape"))
    with pytest.raises(ValueError, match="unsafe"):
        store.load("../escape")


def test_load_missing_raises(tmp_path: Path):
    store = RedactionMapStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("doc-1")


def test_root_created_if_missing(tmp_path: Path):
    target = tmp_path / "nested" / "redaction_maps"
    RedactionMapStore(target)
    assert target.exists()
