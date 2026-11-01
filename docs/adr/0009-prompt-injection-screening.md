---
status: accepted
date: 2026-10-31
---

# Prompt-injection screening: classifier on queries, rules on chunks

## Context and Problem Statement

RAG puts two kinds of untrusted text in front of the LLM: the user query
(direct injection) and the retrieved documents (indirect injection,
Greshake et al. 2023, arXiv:2302.12173). OWASP ranks prompt injection
first in its 2025 LLM Top 10 (LLM01).

The threat model here is narrower than in most RAG systems. The LLM only
sees redacted text and never the redaction map, so an injected instruction
cannot make it reveal original PII. It can still hijack the answer: ignore
the context, invent facts, send the user to a link. Screening protects
answer integrity; PII protection stays with the detector and the audit
([0006](0006-post-generation-audit.md)).

How should queries and documents be screened, where in the pipeline, and
what happens on a hit or an error?

## Decision Drivers

- A false positive at ingestion silently removes a real document from
  every future answer; a refused query can be rephrased.
- The screener must never see raw PII.
- Runs locally, no API key.

## Considered Options

1. No screening; rely on the system prompt.
2. Regex rules only.
3. `protectai/deberta-v3-base-prompt-injection-v2` classifier (Apache-2.0,
   184M parameters, English-only per its model card).
4. `meta-llama/Llama-Prompt-Guard-2-22M` (8 languages, gated under the
   Llama license).
5. An LLM judge per text.
6. Classifier plus rules, configured separately for queries and chunks.

## Decision Outcome

Chosen option: 6.

1. `GUARDIAN_QUERY_INJECTION_SCREENER` (default `classifier+rules`) and
   `GUARDIAN_CHUNK_INJECTION_SCREENER` (default `rules`). Modes:
   `classifier`, `rules`, `classifier+rules`, `off`.
2. The classifier scores long texts in overlapping 1,500-character
   windows because the model truncates at 512 tokens; the max window
   score wins. Threshold `GUARDIAN_INJECTION_THRESHOLD`, default 0.5.
3. Rules: five regex families for instruction-override phrasing.
4. Query screening runs after PII redaction. A flagged query is refused
   before retrieval: HTTP 200, `injection_blocked: true`,
   `X-Guardian-Injection: blocked`, matching how the audit reports a
   redacted answer.
5. Chunk screening runs once per parent chunk at ingestion. A flagged
   parent is stored with `injection_quarantined=true` for audit and
   excluded from dense search and from the BM25 corpus. Quarantine is per
   parent because the generator reads parent text.
6. Each parent stores the `screener_id` that judged it; parents judged by
   another screener, or before v0.2, are re-screened at query time.
7. Fail-closed: a classifier error flags every text it was asked about.
   `off` is explicit and logged as a warning at startup.
8. Spotlighting (Hines et al. 2024, arXiv:2403.14720): context is wrapped
   in `<document>` tags that the system prompt marks as data; a chunk
   cannot close its own tag.

Draft reasoning, from the previous version of this ADR: documents are known
before any query, so screening them once at ingestion is cheaper than on
every query. Rules guard chunks because of the false-positive measurement
below. Prompt Guard 2 and an LLM judge were not adopted: the first is gated
and the second adds a model call per text.

### Consequences

Threshold 0.5, `scripts/eval_injection.py` (`results/manifest.json`,
suites `injection/*`; latencies from `eval/results/injection.json`):

| Set | Screener | Detection | False positives | p50 ms |
|:--|:--|--:|--:|--:|
| Golden corpus, 30 benign redacted docs | rules | n/a | 0 of 30 | <1 |
| same | classifier | n/a | 3 of 30, all support tickets | 27 |
| Own set, 20 indirect injections | rules | 60.0% | | |
| same | classifier | 95.0% | | |
| `deepset/prompt-injections` test, 60 injections / 56 benign | rules | 10.0% | 0.0% | <1 |
| same | classifier | 36.7% | 0.0% | 28 |
| same | classifier+rules | 43.3% | 0.0% | 28 |
| Own set, 40 injections / 20 benign | classifier+rules | 97.5% | 25.0% | 25 |

- No classifier threshold separates the benign tickets from injections:
  at 0.999 the golden corpus still has 1 false positive and detection on
  the own set drops to 75.0% (`eval/results/injection.json`,
  `threshold_sweep`).
- Indirect-injection recall at ingestion is the rules' 60.0%, not the
  classifier's 95.0%. Spotlighting is the second layer; a
  domain-calibrated classifier is on the [roadmap](../roadmap.md).
- The own set was written by the same author as the rules and its benign
  cases are deliberately adversarial: a regression set, not evidence.
- Much of the deepset set is role-play, jailbreak and non-English text,
  which the classifier card says it does not cover.
- Every screener change re-screens retrieved parents at query time until
  the collection is re-ingested.
- Multilingual corpora need option 4 or another classifier.
