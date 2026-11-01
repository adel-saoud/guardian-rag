"""OllamaClient tests using httpx.MockTransport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from guardian_rag.generation.ollama import ChatMessage, OllamaClient, OllamaError


def _client(handler: Any) -> OllamaClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://ollama")
    return OllamaClient(http=http, think=False, max_retries=2)


async def test_chat_parses_structured_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen3.5:4b",
                "message": {"role": "assistant", "content": "hello back"},
                "prompt_eval_count": 12,
                "eval_count": 7,
                "total_duration": 123456,
            },
        )

    client = _client(handler)
    result = await client.chat(
        model="qwen3.5:4b",
        messages=[ChatMessage(role="user", content="hi")],
    )
    assert result.content == "hello back"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 7
    assert result.total_duration_ns == 123456
    assert result.latency_ms >= 0.0


async def test_chat_sends_options_when_provided():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read()
        return httpx.Response(
            200,
            json={
                "model": "m",
                "message": {"role": "assistant", "content": "x"},
            },
        )

    client = _client(handler)
    await client.chat(
        model="m",
        messages=[ChatMessage(role="user", content="hi")],
        max_output_tokens=64,
        temperature=0.3,
    )
    body = captured["payload"].decode("utf-8")
    assert '"num_predict":64' in body
    assert '"temperature":0.3' in body


@pytest.mark.parametrize("think", [False, True])
async def test_chat_sends_think(think: bool):
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.read())
        return httpx.Response(200, json={"model": "m", "message": {"content": "x"}})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama")
    await OllamaClient(http=http, think=think).chat(
        model="m", messages=[ChatMessage(role="user", content="hi")]
    )
    assert captured["payload"]["think"] is think


async def test_4xx_raises_ollama_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    client = _client(handler)
    with pytest.raises(OllamaError, match="404"):
        await client.chat(
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
        )


async def test_5xx_retried_then_raises():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(500, text="boom")

    client = _client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat(
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
        )
    assert calls["count"] == 2  # both attempts used


async def test_malformed_body_raises_ollama_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "m"})  # no message

    client = _client(handler)
    with pytest.raises(OllamaError, match="missing message"):
        await client.chat(
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
        )


def test_max_retries_must_be_positive():
    http = httpx.AsyncClient()
    with pytest.raises(ValueError, match="max_retries"):
        OllamaClient(http=http, think=False, max_retries=0)
