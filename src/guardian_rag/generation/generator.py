"""End-to-end RAG generation: prompt, call Ollama, verify citations.

This is the only generation object the API layer touches. The
post-generation PII audit runs over its output before the answer reaches
the user.
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
        Ollama model id, e.g. `qwen3.5:4b`.
    temperature:
        Sampling temperature; None keeps the model's Ollama default. The
        faithfulness eval sets 0.0 so a rerun gives the same answers.
    """

    def __init__(self, ollama: OllamaClient, model: str, temperature: float | None = None) -> None:
        self._ollama = ollama
        self._model = model
        self._temperature = temperature

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
            model=self._model, messages=messages, temperature=self._temperature
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
