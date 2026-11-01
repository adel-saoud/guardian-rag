"""Tests for the synthetic in-domain retrieval corpus and its queries."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from guardian_rag.eval.domain_corpus import (
    DomainDocument,
    DomainTopic,
    load_domain_queries,
    load_domain_topics,
    render_domain_corpus,
)
from guardian_rag.ingestion.chunker import HierarchicalChunker
from guardian_rag.pii.redactor import Redactor

RETRIEVAL_DIR = Path(__file__).resolve().parent.parent / "eval" / "retrieval"


def _topic(**overrides: object) -> DomainTopic:
    raw: dict[str, object] = {
        "topic_id": "hr-leave",
        "category": "hr",
        "title": "Leave policy",
        "sections": [
            "Staff in {office} get {weeks} weeks. Ask {person} ({email}).",
            "Requests for {office} go through the portal.",
            "Payroll questions: {phone}.",
        ],
        "variants": [
            {"variant_id": v, "label": v, "slots": {"office": v, "weeks": str(i)}}
            for i, v in enumerate(["lyon", "leeds", "porto", "graz"])
        ],
    }
    raw.update(overrides)
    return DomainTopic.model_validate(raw)


@pytest.fixture(scope="module")
def corpus() -> list[DomainDocument]:
    return render_domain_corpus(load_domain_topics(RETRIEVAL_DIR / "domain"))


def test_render_records_exact_pii_spans():
    documents = render_domain_corpus([_topic()])
    assert [d.document_id for d in documents][:2] == ["hr-leave--lyon", "hr-leave--leeds"]
    for doc in documents:
        assert {e.entity_type.value for e in doc.entities} == {"PERSON", "EMAIL", "PHONE"}
        for entity in doc.entities:
            assert doc.text[entity.start : entity.end] == entity.text


def test_render_is_deterministic_per_seed():
    first = render_domain_corpus([_topic()], seed=0)
    assert first == render_domain_corpus([_topic()], seed=0)
    assert first != render_domain_corpus([_topic()], seed=1)


def test_variant_missing_a_slot_is_rejected():
    variants = [{"variant_id": v, "label": v, "slots": {"office": v}} for v in "abcd"]
    with pytest.raises(ValidationError, match="missing slots"):
        _topic(variants=variants)


def test_identical_siblings_are_rejected():
    variants = [
        {"variant_id": v, "label": v, "slots": {"office": "x", "weeks": "1"}} for v in "abcd"
    ]
    with pytest.raises(ValidationError, match="identical slot values"):
        _topic(variants=variants)


def test_committed_corpus_is_multi_chunk_and_redacts_cleanly(corpus: list[DomainDocument]):
    assert len(corpus) >= 200
    chunker = HierarchicalChunker()
    redactor = Redactor()
    for doc in corpus:
        redacted, _ = redactor.redact(doc.text, doc.entities, doc.document_id)
        assert re.search(r"\[[A-Z_]+_1\]", redacted), doc.document_id
        assert all(e.text not in redacted for e in doc.entities)
        _, children = chunker.split(redacted, doc.document_id)
        assert len(children) >= 2, doc.document_id


def test_queries_name_one_existing_variant_each(corpus: list[DomainDocument]):
    queries = load_domain_queries(RETRIEVAL_DIR / "domain_queries.jsonl")
    document_ids = {d.document_id for d in corpus}
    assert len(queries) >= 100
    assert len({q.query_id for q in queries}) == len(queries)
    assert {q.document_id for q in queries} <= document_ids
    per_topic = Counter(q.document_id.split("--")[0] for q in queries)
    assert set(per_topic.values()) == {3}
    assert len({q.document_id for q in queries}) == len(queries)


def test_queries_carry_no_person_names(corpus: list[DomainDocument]):
    name_tokens = {
        token.lower()
        for doc in corpus
        for entity in doc.entities
        if entity.entity_type.value == "PERSON"
        for token in entity.text.split()
    }
    for query in load_domain_queries(RETRIEVAL_DIR / "domain_queries.jsonl"):
        assert not name_tokens & set(re.findall(r"[a-z]+", query.query.lower())), query.query_id
