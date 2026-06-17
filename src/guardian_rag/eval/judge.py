"""LLM-as-Judge faithfulness scoring.

The judge compares a candidate answer (produced from the redacted
context) against a reference answer (produced from the unredacted
context). It returns a binary verdict so the report can show a single
percentage.

The judge runs on the same Ollama daemon as the generator. To keep the
prompt cheap, we ask for "yes" or "no" only. We accept any answer that
starts with "yes" (case-insensitive) as a positive verdict; anything else
counts as a failure.

Faithfulness is informational, not gated. The brief calls this out
explicitly: judge variance is real and we do not want a noisy gate.
"""

from __future__ import annotations

import structlog

from guardian_rag.eval.models import FaithfulnessResult
from guardian_rag.generation.ollama import ChatMessage, OllamaClient

logger = structlog.get_logger(__name__)


JUDGE_SYSTEM_PROMPT = (
    "You are a strict eval judge. You compare two answers to the same "
    "question and decide if the candidate is semantically equivalent to "
    "the reference. Treat redacted placeholders like [PERSON_1] as opaque "
    "tokens and accept them as matches for the underlying names. Reply "
    "with only `yes` or `no` on a single line. Nothing else."
)


def _parse_verdict(content: str) -> bool:
    return content.strip().lower().startswith("yes")


class FaithfulnessJudge:
    """Run pairwise yes/no judging against a local Ollama model."""

    def __init__(self, ollama: OllamaClient, model: str) -> None:
        self._ollama = ollama
        self._model = model

    async def judge(
        self,
        *,
        document_id: str,
        question: str,
        reference_answer: str,
        candidate_answer: str,
    ) -> FaithfulnessResult:
        messages = [
            ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    f"Question: {question}\n\n"
                    f"Reference answer: {reference_answer}\n\n"
                    f"Candidate answer: {candidate_answer}\n\n"
                    "Is the candidate semantically equivalent to the reference? Reply yes or no."
                ),
            ),
        ]
        response = await self._ollama.chat(
            model=self._model,
            messages=messages,
            temperature=0.0,
            max_output_tokens=8,
        )
        verdict = _parse_verdict(response.content)
        logger.info(
            "eval.judge.complete",
            document_id=document_id,
            verdict=verdict,
            raw=response.content[:40],
        )
        return FaithfulnessResult(
            document_id=document_id,
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            verdict=verdict,
        )
