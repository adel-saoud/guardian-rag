---
status: accepted
date: 2026-10-31
---

# Faithfulness as a reported, non-gating metric, checked against hand labels

## Context and Problem Statement

Redaction replaces facts with placeholders before the LLM sees them, so it
can lower answer quality. The PII and retrieval evals do not show that
cost. How should answer quality after redaction be measured, and can a
small local judge be trusted enough to act on the number?

## Decision Drivers

- No paid API key in the default reproduction (`scripts/reproduce.py`).
- A judge's verdicts are only useful if their error rate is known.
- A noisy number must not fail builds at random.

## Considered Options

1. No answer-quality metric.
2. A per-request faithfulness score on every query (the former
   `X-Guardian-Faithfulness` header idea).
3. An offline eval: answer each golden QA pair from the original and the
   redacted document, judge both against a reference, report the gap and
   the judge's agreement with hand labels; never gate on it.
4. Option 3, used as a CI gate.

## Decision Outcome

Chosen option: 3.

- `scripts/eval_faithfulness.py` answers the 30 golden QA pairs
  (`eval/golden/`, one non-PII fact per document) from the original and
  from the redacted document with the same generator at temperature 0,
  and `FaithfulnessJudge` (`eval/judge.py`) answers yes or no against the
  reference.
- The judge also scores 30 hand-labelled cases
  (`eval/faithfulness/judge_labels.jsonl`, reviewed by the maintainer); agreement
  and Cohen's kappa are reported next to the rates.
- Rates are informational: no test, CI job or threshold reads them.

Draft reasoning, from `eval/judge.py` and the CHANGELOG: a small local
judge is noisy, and a noisy gate would fail builds at random. The
per-request score was removed because it would add a judge call to every
query and its value was always empty. Measuring the judge against labels
is what makes the rates readable at all.

### Consequences

- Measured (`results/manifest.json`, suite `faithfulness`, n=30): 93.3%
  judged correct with original context, 80.0% with redacted context.
  All five drops are company-name questions
  ([0013](0013-redact-organisation-names.md)).
- Judge agreement with the labels 86.7%, Cohen's kappa 0.73 (suite
  `faithfulness/judge`); 4 false positives, 0 false negatives
  (`eval/results/faithfulness.json`, `judge_agreement`), so the judge
  leans lenient.
- The first judge prompt ("semantically equivalent") agreed with the
  labels on 18 of 30 cases with `llama3.2:3b` (CHANGELOG); the prompt now
  asks whether the key fact is stated.
- Judge and generator are the same model
  ([0012](0012-qwen35-local-model-thinking-off.md)). Self-preference
  mostly cancels out of the original-versus-redacted gap, not out of the
  absolute rates. `GUARDIAN_OLLAMA_JUDGE_MODEL` allows a different judge.
- The labels are reviewed but only 30 cases; the agreement number has a
  wide uncertainty until the set grows
  ([roadmap](../roadmap.md)).
