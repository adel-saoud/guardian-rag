"""Prompt builder tests."""

from __future__ import annotations

from guardian_rag.generation.prompt import (
    REFUSAL_TEXT,
    SYSTEM_PROMPT,
    build_messages,
)
from guardian_rag.ingestion.models import ParentChunk


def _parents(texts: list[str]) -> list[ParentChunk]:
    return [
        ParentChunk(chunk_id=f"p{i}", document_id="doc-1", text=text, order=i)
        for i, text in enumerate(texts)
    ]


def test_messages_include_system_and_user():
    msgs = build_messages("what is the policy?", _parents(["Policy A says...", "Policy B says..."]))
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"
    assert msgs[0].content == SYSTEM_PROMPT


def test_user_message_numbers_parents_starting_at_one():
    msgs = build_messages("q", _parents(["alpha", "beta"]))
    user = msgs[1].content
    assert "[1] alpha" in user
    assert "[2] beta" in user


def test_empty_parents_produces_no_context_marker():
    msgs = build_messages("q", [])
    assert "(no context available)" in msgs[1].content


def test_refusal_text_referenced_in_system_prompt():
    assert REFUSAL_TEXT in SYSTEM_PROMPT


def test_placeholder_guidance_present():
    assert "[PERSON_1]" in SYSTEM_PROMPT
    assert "opaque" in SYSTEM_PROMPT.lower()
