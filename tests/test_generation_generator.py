"""End-to-end Generator tests with a mocked Ollama transport."""

from __future__ import annotations

from typing import Any

import httpx

from guardian_rag.generation.generator import Generator
from guardian_rag.generation.ollama import OllamaClient
from guardian_rag.generation.prompt import REFUSAL_TEXT
from guardian_rag.ingestion.models import ParentChunk


def _parents(texts: list[str]) -> list[ParentChunk]:
    return [
        ParentChunk(chunk_id=f"p{i}", document_id="doc-1", text=text, order=i)
        for i, text in enumerate(texts)
    ]


def _generator(content: str) -> Generator:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "m",
                "message": {"role": "assistant", "content": content},
                "prompt_eval_count": 1,
                "eval_count": 2,
            },
        )

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://ollama")
    return Generator(ollama=OllamaClient(http=http), model="m")


async def test_generator_returns_refusal_with_no_parents():
    gen = _generator("(unused)")
    result = await gen.generate("q", [])
    assert result.answer == REFUSAL_TEXT
    assert not result.has_citations


async def test_generator_returns_grounded_answer():
    gen = _generator("Salary rule applies [1]. Bonus rule applies [2].")
    result = await gen.generate("policy?", _parents(["a", "b"]))
    assert result.cited_parent_ids == ["p0", "p1"]
    assert result.all_citations_grounded


async def test_generator_flags_ungrounded_citation():
    gen = _generator("Per [9], answer is X.")
    result = await gen.generate("policy?", _parents(["only one"]))
    assert result.ungrounded_citations == [9]
    assert not result.all_citations_grounded


async def test_generator_passes_payload_options():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={"model": "m", "message": {"role": "assistant", "content": "ok"}},
        )

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://ollama")
    gen = Generator(
        ollama=OllamaClient(http=http),
        model="m",
        max_output_tokens=32,
        temperature=0.2,
    )
    await gen.generate("q", _parents(["c"]))
    body = captured["body"]
    assert '"num_predict":32' in body
    assert '"temperature":0.2' in body
