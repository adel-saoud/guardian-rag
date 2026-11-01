"""Startup version checks for Ollama and Qdrant."""

from __future__ import annotations

import httpx
import pytest

from guardian_rag.generation.ollama import OllamaClient, OllamaError
from guardian_rag.service_versions import (
    MIN_OLLAMA_VERSION,
    MIN_QDRANT_VERSION,
    IncompatibleServiceError,
    require_min_version,
)


@pytest.mark.parametrize("found", ["0.17.5", "0.40.10", "v1.0.0", "0.18.0-rc1"])
def test_ollama_at_or_above_minimum_passes(found: str):
    require_min_version("Ollama", found, MIN_OLLAMA_VERSION)


def test_older_service_names_the_minimum():
    with pytest.raises(IncompatibleServiceError, match=r"Qdrant 1\.16\.3 is older than 1\.17\.0"):
        require_min_version("Qdrant", "1.16.3", MIN_QDRANT_VERSION)


def test_unreadable_version_names_the_minimum():
    with pytest.raises(IncompatibleServiceError, match=r"needs Ollama 0\.17\.5 or newer"):
        require_min_version("Ollama", "dev", MIN_OLLAMA_VERSION)


def test_malformed_minimum_is_a_programming_error():
    with pytest.raises(ValueError, match=r"not major\.minor\.patch"):
        require_min_version("Ollama", "1.0.0", "latest")


def _client(status: int, body: object) -> OllamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/version"
        return httpx.Response(status, json=body)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama")
    return OllamaClient(http=http, think=False)


async def test_ollama_version_reads_the_version_field():
    assert await _client(200, {"version": "0.40.10"}).version() == "0.40.10"


async def test_ollama_version_errors_on_http_failure():
    with pytest.raises(OllamaError, match="404"):
        await _client(404, {"error": "not found"}).version()


async def test_ollama_version_errors_on_missing_field():
    with pytest.raises(OllamaError, match="no version"):
        await _client(200, {"other": 1}).version()
