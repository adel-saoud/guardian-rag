"""LLM-as-Judge faithfulness tests with mocked Ollama."""

from __future__ import annotations

import httpx

from guardian_rag.eval.judge import FaithfulnessJudge
from guardian_rag.generation.ollama import OllamaClient


def _judge(reply: str) -> FaithfulnessJudge:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "m",
                "message": {"role": "assistant", "content": reply},
            },
        )

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://ollama")
    return FaithfulnessJudge(OllamaClient(http=http), model="m")


async def test_judge_returns_true_on_yes():
    judge = _judge("yes")
    result = await judge.judge(
        document_id="d1",
        question="q",
        reference_answer="ref",
        candidate_answer="cand",
    )
    assert result.verdict is True


async def test_judge_returns_false_on_no():
    judge = _judge("no, they differ on key points")
    result = await judge.judge(
        document_id="d1",
        question="q",
        reference_answer="ref",
        candidate_answer="cand",
    )
    assert result.verdict is False


async def test_judge_normalises_whitespace_and_casing():
    judge = _judge("  YES   \n")
    result = await judge.judge(
        document_id="d1",
        question="q",
        reference_answer="ref",
        candidate_answer="cand",
    )
    assert result.verdict is True
