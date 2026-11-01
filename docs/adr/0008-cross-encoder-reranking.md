---
status: accepted
date: 2026-10-31
---

# Cross-encoder reranking, built and shipped off

## Context and Problem Statement

Hybrid retrieval (dense MiniLM plus BM25, `retrieval/hybrid.py`) scores
each chunk against the query independently. A cross-encoder reads query
and chunk together, which usually ranks better but costs one forward pass
per pair. The standard pattern is retrieve-then-rerank: fuse a wide pool
cheaply, rerank only that pool
([sentence-transformers docs](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)).
Should Guardian rerank, with which model, over how many candidates, and
on by default?

## Decision Drivers

- CPU only, no API key.
- A default must be backed by a measured gain, not by the pattern's
  reputation.
- Measure on public data and on text that looks like the target domain
  (HR, contracts, support).

## Considered Options

1. No reranker.
2. `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0, 22.7M parameters,
   English).
3. `BAAI/bge-reranker-v2-m3` (568M parameters, multilingual).
4. A hosted reranker API.
5. Reranker on by default, or built but off by default.

## Decision Outcome

Chosen options: 2, built and off by default.

- `Reranker` behind a `RerankerBackend` Protocol, with a lazily loaded
  sentence-transformers `CrossEncoder` (`retrieval/reranker.py`).
- `GUARDIAN_RERANKER_MODEL` defaults to option 2; option 3 is the
  documented swap.
- It scores the top `GUARDIAN_RERANK_CANDIDATES` (default 30) fused child
  chunks, all redacted, so it never sees raw PII.
- A reranker failure falls back to the fused order: ranking is not a
  security control.
- `GUARDIAN_RERANKER_ENABLED` defaults to `false` because of the
  measurements below.

Draft reasoning: a hosted API would break the no-API-key rule; the small
English model is the cheapest way to test whether reranking helps at all
before paying for the 568M multilingual one. The 30-candidate pool and the
English-only default are not backed by a measurement in this repo.

### Consequences

BEIR SciFact test split, 5,183 abstracts, 300 queries, CPU
(`results/manifest.json`, suites `retrieval/scifact/*`):

| Config | nDCG@10 | Recall@10 | MRR@10 | p50 ms | p95 ms |
|:--|--:|--:|--:|--:|--:|
| dense | 0.658 | 0.809 | 0.612 | 71 | 120 |
| hybrid (bm25_weight 0.4) | 0.703 | 0.825 | 0.666 | 69 | 144 |
| hybrid + rerank | 0.711 | 0.858 | 0.671 | 131 | 257 |

Rerank minus hybrid, nDCG@10: +0.009, 95% paired bootstrap CI -0.015 to
+0.033 (`retrieval/scifact`, `rerank_minus_hybrid_ndcg@10`). The interval
contains 0, while p50 latency goes from 69 to 131 ms.

In-domain sanity check: 270 synthetic HR, contract and support documents
in families of six near-duplicates (`guardian_rag.eval.domain_corpus`,
specs in `eval/retrieval/domain/`), 135 queries written by the project
author (`eval/retrieval/domain_queries.jsonl`), CPU (suites
`retrieval/domain/*`):

| Config | nDCG@10 | Recall@10 | MRR@10 | p50 ms | p95 ms |
|:--|--:|--:|--:|--:|--:|
| dense | 0.818 | 0.993 | 0.761 | 11 | 18 |
| hybrid (bm25_weight 0.4) | 0.902 | 1.000 | 0.869 | 13 | 16 |
| hybrid + rerank | 0.888 | 1.000 | 0.852 | 65 | 117 |

Rerank minus hybrid, nDCG@10: -0.014, CI -0.043 to +0.014
(`retrieval/domain`). Recall@10 is 1.000 for both, so the reranker only
reorders documents hybrid already found.

- No gain on either set, so the default stays off. The stage is built,
  tested and one variable away.
- The golden smoke set (30 queries over the 30 redacted golden documents)
  scores 1.0 on every configuration (`retrieval/golden/*`): it shows that
  placeholders do not break the reranker, nothing more.
- The in-domain set is templated and author-written, so it is a sanity
  check, not independent evidence; templated wording likely favours BM25.
- Revisit with labelled real queries, or with option 3.
- SciFact (CC-BY-SA-4.0) is downloaded at run time into the gitignored
  `data/`; only scores are committed.
