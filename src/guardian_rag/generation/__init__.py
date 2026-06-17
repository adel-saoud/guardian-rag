"""Grounded generation: Ollama client, prompt, citation verification."""

from __future__ import annotations

from guardian_rag.generation.citation import (
    GroundedAnswer,
    extract_citation_indices,
    verify_citations,
)
from guardian_rag.generation.generator import Generator
from guardian_rag.generation.ollama import (
    ChatMessage,
    OllamaChatResponse,
    OllamaClient,
    OllamaError,
)
from guardian_rag.generation.prompt import REFUSAL_TEXT, SYSTEM_PROMPT, build_messages

__all__ = [
    "REFUSAL_TEXT",
    "SYSTEM_PROMPT",
    "ChatMessage",
    "Generator",
    "GroundedAnswer",
    "OllamaChatResponse",
    "OllamaClient",
    "OllamaError",
    "build_messages",
    "extract_citation_indices",
    "verify_citations",
]
