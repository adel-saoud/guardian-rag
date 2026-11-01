"""Retrieval-quality metrics at document level.

Guardian retrieves child chunks, but relevance judgements (BEIR qrels, the
golden smoke queries) are per document. `rank_documents` collapses a chunk
ranking into a document ranking by first appearance, which is how BEIR
scores passage-level systems.

Definitions follow the BEIR evaluation (`pytrec_eval` conventions):
nDCG@k with graded gains and log2 discount, Recall@k as the share of
relevant documents found in the top k, MRR@k as the reciprocal rank of the
first relevant document.

These are a few lines of stdlib on purpose. `pytrec_eval` and `ir-measures`
compute the same metrics but add a C extension to the core package, and
`scipy.stats.bootstrap` is only available with the `ml` extra.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from guardian_rag.retrieval.models import RetrievalHit


def rank_documents(hits: Sequence[RetrievalHit]) -> list[str]:
    """Return document ids in order of their best-ranked chunk."""
    seen: set[str] = set()
    ranked: list[str] = []
    for hit in hits:
        if hit.document_id not in seen:
            seen.add(hit.document_id)
            ranked.append(hit.document_id)
    return ranked


def ndcg_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int) -> float:
    """Normalised discounted cumulative gain over the top k documents."""
    dcg = sum(qrels.get(doc_id, 0) / math.log2(rank + 2) for rank, doc_id in enumerate(ranked[:k]))
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int) -> float:
    """Share of relevant documents present in the top k."""
    relevant = {doc_id for doc_id, gain in qrels.items() if gain > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int) -> float:
    """Reciprocal rank of the first relevant document in the top k, else 0."""
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, `q` in [0, 100]. Used for latency p50/p95."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(q / 100 * len(ordered)) - 1)
    return ordered[index]


def paired_bootstrap_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean per-query difference `candidate - baseline` with a bootstrap CI.

    Queries are resampled with replacement and both systems are scored on
    the same resample, which is the standard paired bootstrap for IR
    comparisons. An interval that contains 0 means the query sample cannot
    tell the two systems apart.
    """
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("baseline and candidate must be non-empty and the same length")
    rng = random.Random(seed)  # noqa: S311, statistical resampling, not security
    deltas = [c - b for b, c in zip(baseline, candidate, strict=True)]
    n = len(deltas)
    means = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(n_resamples))
    tail = (1.0 - confidence) / 2
    low = means[int(tail * n_resamples)]
    high = means[min(n_resamples - 1, int((1.0 - tail) * n_resamples))]
    return sum(deltas) / n, low, high
