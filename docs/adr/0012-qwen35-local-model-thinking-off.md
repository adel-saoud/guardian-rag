---
status: accepted
date: 2026-10-31
---

# `qwen3.5:4b` on Ollama for generation and judging, with thinking off

## Context and Problem Statement

Guardian generates answers, and the faithfulness eval judges them
([0011](0011-faithfulness-reported-not-gated.md)), on a local Ollama with
no API key. The previous default, `llama3.2:3b`, was removed from the
local Ollama install. Which local model replaces it, and how is Qwen3.5's
default thinking mode handled?

## Decision Drivers

- Runs on a laptop CPU through Ollama, no API key.
- One model for generator and judge keeps the reproduction to one pull.
- The judge reads a bare `yes` or `no`, so a reasoning trace in the
  content breaks parsing.

## Considered Options

1. `qwen3.5:4b`, thinking left at the model default (on).
2. `qwen3.5:4b`, thinking off through Ollama's `think: false`.
3. `qwen3.5:0.8b` as the judge.

## Decision Outcome

Chosen option: 2.

- `GUARDIAN_OLLAMA_MODEL` and `GUARDIAN_OLLAMA_JUDGE_MODEL` default to
  `qwen3.5:4b` (`config.py`).
- `GUARDIAN_OLLAMA_THINK` (default `false`) is sent as `think` on every
  `/api/chat` call; `OllamaClient` requires it (`generation/ollama.py`).
- The judge skips a closed `<think>` block before its verdict and raises
  `JudgeOutputError` on an unfinished one (`eval/judge.py`).
- Minimum Ollama is 0.17.5, the first release with the small Qwen3.5
  sizes and the fix for Qwen3.5 repeating itself
  (`service_versions.py`).

Draft reasoning: with thinking on, a short prompt costs many times the
output tokens, and an 8-token judge reply cut inside a trace has no
verdict. `qwen3.5:0.8b` was checked by the maintainer as a judge and
found weak (2026-10-24, not recorded in this repo).

### Consequences

- One-off live check on 2026-10-24, Ollama 0.40.10, prompt "Say hi.",
  temperature 0: 10 output tokens and an empty `thinking` field with
  `think: false`, 284 tokens and a 978-character trace with
  `think: true`. Not part of `scripts/reproduce.py`, so not in the
  manifest.
- Manifest records the model digest, Ollama version and `think` value
  (`results/manifest.json`, `models`, roles `generator` and `judge`).
- Faithfulness on this model: see [0011](0011-faithfulness-reported-not-gated.md).
- Generator and judge share a model, a known limit of the faithfulness
  numbers.
- BREAKING for users of `OllamaClient` (new required argument) and for
  deployments that relied on `llama3.2:3b` (CHANGELOG).
