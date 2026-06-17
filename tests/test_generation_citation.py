"""Citation extraction and verification tests."""

from __future__ import annotations

from guardian_rag.generation.citation import (
    extract_citation_indices,
    verify_citations,
)
from guardian_rag.ingestion.models import ParentChunk


def _parents(n: int) -> list[ParentChunk]:
    return [
        ParentChunk(chunk_id=f"p{i + 1}", document_id="doc-1", text=f"text-{i}", order=i)
        for i in range(n)
    ]


def test_extract_indices_in_source_order():
    indices = extract_citation_indices("Per [2] and [1], the rule is X. See [2] again.")
    assert indices == [2, 1, 2]


def test_no_citations_yields_empty_list():
    assert extract_citation_indices("a plain answer") == []


def test_verify_pairs_indices_with_parent_ids():
    parents = _parents(2)
    answer = "Salary policy is reviewed yearly [1]. Bonuses are discretionary [2]."
    result = verify_citations(answer, parents)
    assert result.cited_parent_ids == ["p1", "p2"]
    assert result.has_citations
    assert result.all_citations_grounded


def test_duplicate_citations_dedupe_preserve_first_order():
    parents = _parents(2)
    answer = "[2] [1] [2]"
    result = verify_citations(answer, parents)
    assert result.cited_parent_ids == ["p2", "p1"]


def test_out_of_range_citation_flagged():
    parents = _parents(2)
    answer = "According to [3], the policy says X."
    result = verify_citations(answer, parents)
    assert result.cited_parent_ids == []
    assert result.ungrounded_citations == [3]
    assert not result.all_citations_grounded


def test_no_parents_no_citations():
    result = verify_citations("answer with no citation tokens", [])
    assert not result.has_citations
    assert result.all_citations_grounded
