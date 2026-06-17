"""Hierarchical chunker tests."""

from __future__ import annotations

import pytest

from guardian_rag.ingestion.chunker import HierarchicalChunker


def _big_text(words: int) -> str:
    sentences = [f"This is sentence number {i}." for i in range(words)]
    return " ".join(sentences)


def test_empty_input_yields_no_chunks():
    parents, children = HierarchicalChunker().split("   ", "doc-1")
    assert parents == []
    assert children == []


def test_small_text_emits_single_parent_and_at_least_one_child():
    chunker = HierarchicalChunker(parent_tokens=64, child_tokens=32, child_overlap=4)
    parents, children = chunker.split("A short policy statement about HR rules.", "doc-1")
    assert len(parents) == 1
    assert len(children) >= 1
    assert parents[0].order == 0
    assert all(c.parent_id == parents[0].chunk_id for c in children)
    assert all(c.document_id == "doc-1" for c in children)


def test_large_text_emits_multiple_parents_and_children():
    chunker = HierarchicalChunker(parent_tokens=64, child_tokens=24, child_overlap=4)
    parents, children = chunker.split(_big_text(400), "doc-1")
    assert len(parents) > 1
    assert len(children) >= len(parents)
    parent_ids = {p.chunk_id for p in parents}
    assert all(c.parent_id in parent_ids for c in children)


def test_parent_order_is_monotonic():
    chunker = HierarchicalChunker(parent_tokens=64, child_tokens=24, child_overlap=4)
    parents, _ = chunker.split(_big_text(400), "doc-1")
    assert [p.order for p in parents] == list(range(len(parents)))


def test_child_order_resets_per_parent():
    chunker = HierarchicalChunker(parent_tokens=64, child_tokens=24, child_overlap=4)
    _parents, children = chunker.split(_big_text(400), "doc-1")
    by_parent: dict[str, list[int]] = {}
    for c in children:
        by_parent.setdefault(c.parent_id, []).append(c.order)
    for orders in by_parent.values():
        assert orders == list(range(len(orders)))


def test_chunk_ids_are_unique():
    chunker = HierarchicalChunker(parent_tokens=64, child_tokens=24, child_overlap=4)
    parents, children = chunker.split(_big_text(400), "doc-1")
    ids = [p.chunk_id for p in parents] + [c.chunk_id for c in children]
    assert len(set(ids)) == len(ids)


def test_invalid_sizes_rejected():
    with pytest.raises(ValueError, match="positive"):
        HierarchicalChunker(parent_tokens=0)
    with pytest.raises(ValueError, match="<="):
        HierarchicalChunker(parent_tokens=128, child_tokens=256)
    with pytest.raises(ValueError, match="overlap"):
        HierarchicalChunker(child_tokens=32, child_overlap=32)
