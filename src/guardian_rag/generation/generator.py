"""End-to-end RAG generation: prompt, call Ollama, verify citations.

This is the only object the API layer touches. It holds the OllamaClient
plus configuration (model id, sampling). The post-generation PII audit
runs over this object's output before the answer reaches the user.
"""

from __future__ import annotations

import structlog

from guardian_rag.generation.citation import GroundedAnswer, verify_citations
from guardian_rag.generation.ollama import OllamaClient
from guardian_rag.generation.prompt import REFUSAL_TEXT, build_messages
from guardian_rag.ingestion.models import ParentChunk

logger = structlog.get_logger(__name__)


class Generator:
    """Run one grounded answer pass over a query and retrieved parents.

    Parameters
    ----------
    ollama:
        The async Ollama client.
    model:
        Ollama model id, e.g. `llama3.2:3b`.
    max_output_tokens:
        Optional cap on the model's output length.
    temperature:
        Optional sampling temperature. Default None means use Ollama's default.
    """

    def __init__(
        self,
        ollama: OllamaClient,
        model: str,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self._ollama = ollama
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self,
        query: str,
        parents: list[ParentChunk],
    ) -> GroundedAnswer:
        """Generate an answer grounded in `parents` and verify its citations.

        When the parent list is empty we short-circuit to the refusal text
        so the LLM is never asked to answer without context.
        """
        if not parents:
            logger.info("generation.refused.no_context")
            return GroundedAnswer(answer=REFUSAL_TEXT)

        messages = build_messages(query, parents)
        response = await self._ollama.chat(
            model=self._model,
            messages=messages,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )
        result = verify_citations(response.content, parents)
        logger.info(
            "generation.complete",
            model=self._model,
            n_parents=len(parents),
            n_citations=len(result.cited_parent_ids),
            n_ungrounded=len(result.ungrounded_citations),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
        )
        return result
