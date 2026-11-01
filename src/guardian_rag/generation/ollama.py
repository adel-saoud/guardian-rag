"""Async adapter for the local Ollama HTTP API.

Non-streaming `POST /api/chat`, plus `GET /api/version` for the startup
version check. The underlying `httpx.AsyncClient`
is injected so tests can plug in a `MockTransport` and never touch the
network.

Reference: https://docs.ollama.com/api/chat
"""

from __future__ import annotations

import time
from typing import Any, Literal, cast

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = structlog.get_logger(__name__)


Role = Literal["system", "user", "assistant"]
"""Message roles accepted by Ollama's `/api/chat` that this client sends."""


class ChatMessage(BaseModel):
    """A single chat-turn message, the wire shape Ollama expects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    content: str


class OllamaChatResponse(BaseModel):
    """Structured response from a non-streaming `/api/chat` call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    content: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_duration_ns: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)


class OllamaError(RuntimeError):
    """Raised when Ollama returns a non-2xx response or an unparsable body."""


class OllamaClient:
    """Thin async wrapper around the Ollama HTTP API.

    Parameters
    ----------
    http:
        An `httpx.AsyncClient` configured with the desired base URL and
        timeout. Injected so tests can plug in `httpx.MockTransport`.
    think:
        Sent as Ollama's `think` on every chat. Qwen3.5 thinks by default,
        which multiplies output tokens and latency and can push a trace in
        front of the answer, so Guardian sends `false` unless
        `GUARDIAN_OLLAMA_THINK` turns it on.
    max_retries:
        Total attempts on transient transport errors (network, timeout, 5xx).
        Use 1 to disable retries entirely.
    """

    def __init__(self, http: httpx.AsyncClient, *, think: bool, max_retries: int = 3) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self._http = http
        self._think = think
        self._max_retries = max_retries

    async def chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> OllamaChatResponse:
        """Send one chat turn and return a structured response.

        `max_output_tokens` and `temperature` map to Ollama's `num_predict`
        and `temperature` options; None keeps the model default.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "think": self._think,
        }
        options: dict[str, Any] = {}
        if max_output_tokens is not None:
            options["num_predict"] = max_output_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if options:
            payload["options"] = options

        t0 = time.perf_counter()
        response = await self._post_with_retry("/api/chat", payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        body = self._parse_chat_body(response)

        return OllamaChatResponse(
            model=cast("str", body.get("model", model)),
            content=cast("str", body["message"]["content"]),
            prompt_tokens=int(body.get("prompt_eval_count", 0)),
            completion_tokens=int(body.get("eval_count", 0)),
            total_duration_ns=int(body.get("total_duration", 0)),
            latency_ms=latency_ms,
        )

    async def version(self) -> str:
        """Return the server version from `GET /api/version`, for the startup check."""
        response = await self._http.get("/api/version")
        if response.status_code >= 400:
            raise OllamaError(f"Ollama /api/version returned {response.status_code}")
        try:
            return str(cast("dict[str, Any]", response.json())["version"])
        except (ValueError, KeyError) as exc:
            raise OllamaError(f"no version in /api/version: {response.text[:200]}") from exc

    async def _post_with_retry(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        """POST with retries on transport errors and 5xx; a 4xx fails at once as `OllamaError`."""
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential_jitter(initial=0.1, max=2.0),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                response = await self._http.post(path, json=payload)
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Ollama returned {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                if response.status_code >= 400:
                    raise OllamaError(
                        f"Ollama returned {response.status_code}: {response.text[:200]}"
                    )
                return response
        raise OllamaError("retry loop exited without a response")  # pragma: no cover

    @staticmethod
    def _parse_chat_body(response: httpx.Response) -> dict[str, Any]:
        """Decode the body and check it carries `message.content`."""
        try:
            body = cast("dict[str, Any]", response.json())
        except ValueError as exc:
            raise OllamaError(f"Ollama returned non-JSON body: {response.text[:200]}") from exc
        if "message" not in body or "content" not in body["message"]:
            raise OllamaError(f"Ollama response missing message/content: {body}")
        return body
