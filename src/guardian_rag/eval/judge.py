"""LLM-as-Judge faithfulness scoring.

The judge checks whether a candidate answer states the key fact of a
hand-written reference answer. It is a reference-guided correctness
check, not strict equivalence: an earlier "semantically equivalent"
prompt rejected correct answers that carried extra detail. It returns a
binary verdict so the report can show a single percentage.

The judge runs on the same Ollama daemon as the generator. To keep the
prompt cheap, we ask for "yes" or "no" only. We accept any answer that
starts with "yes" (case-insensitive) as a positive verdict; anything else
counts as a failure. A thinking model (Qwen3.5 thinks by default) can put a
`<think>` trace before the verdict when `think` is not sent as false: a
closed trace is skipped, and an unclosed one (cut by the 8-token limit)
raises `JudgeOutputError` instead of being read as a silent "no".

Faithfulness is informational, not gated: a small local judge is noisy,
and a noisy gate would fail builds at random. `scripts/eval_faithfulness.py`
runs it and reports its agreement with hand labels next to the rates.
"""

from __future__ import annotations

import re

import structlog

from guardian_rag.eval.models import FaithfulnessResult
from guardian_rag.generation.ollama import ChatMessage, OllamaClient

logger = structlog.get_logger(__name__)


JUDGE_SYSTEM_PROMPT = (
    "You are a strict eval judge. You get a question, a reference answer and "
    "a candidate answer. Reply `yes` if the candidate states the key fact of "
    "the reference answer and does not contradict it. Extra detail, citations "
    "like [1] and redacted placeholders like [PERSON_1] do not count against "
    "the candidate. Reply `no` if the key fact is missing, wrong, or the "
    "candidate says it does not know. Reply with only `yes` or `no`."
)


_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)
"""A leading reasoning trace, as Qwen-family models emit it inside the content."""


class JudgeOutputError(ValueError):
    """The judge reply holds no verdict, only the start of a reasoning trace."""


def _parse_verdict(content: str) -> bool:
    """Read any reply starting with "yes" as positive, after a closed `<think>` trace."""
    reply = _THINK_BLOCK.sub("", content, count=1).strip().lower()
    if reply.startswith("<think>"):
        raise JudgeOutputError(
            f"judge reply is an unfinished <think> trace, set GUARDIAN_OLLAMA_THINK=false: "
            f"{content[:80]!r}"
        )
    return reply.startswith("yes")


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
        """Ask the judge model whether `candidate_answer` matches `reference_answer`."""
        messages = [
            ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    f"Question: {question}\n\n"
                    f"Reference answer: {reference_answer}\n\n"
                    f"Candidate answer: {candidate_answer}\n\n"
                    "Does the candidate state the key fact of the reference? Reply yes or no."
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
