"""Tests for retrieval ranking metrics and injection detection counts."""

from __future__ import annotations

import math

import pytest

from guardian_rag.eval.injection import count_detections
from guardian_rag.eval.ranking import (
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_ci,
    percentile,
    rank_documents,
    recall_at_k,
)
from guardian_rag.retrieval.models import RetrievalHit


def _hit(chunk_id: str, document_id: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id, document_id=document_id, parent_id="p", text="t", order=0, score=1.0
    )


def test_rank_documents_dedupes_by_first_appearance():
    hits = [_hit("1", "a"), _hit("2", "b"), _hit("3", "a"), _hit("4", "c")]
    assert rank_documents(hits) == ["a", "b", "c"]


def test_ndcg_perfect_and_partial():
    qrels = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], qrels, 10) == pytest.approx(1.0)
    dcg = 1 / math.log2(3)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(["c", "a"], qrels, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["c"], {}, 10) == 0.0


def test_recall_and_mrr():
    qrels = {"a": 1, "b": 1, "z": 0}
    assert recall_at_k(["x", "a", "y"], qrels, 2) == 0.5
    assert recall_at_k(["x"], {}, 2) == 0.0
    assert mrr_at_k(["x", "a"], qrels, 10) == 0.5
    assert mrr_at_k(["x", "a"], qrels, 1) == 0.0


def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile([], 50) == 0.0


def test_count_detections_rates():
    counts = count_detections([True, True, False, False, False], [True, False, False, True, False])
    assert (counts.true_positives, counts.false_negatives) == (1, 1)
    assert (counts.true_negatives, counts.false_positives) == (2, 1)
    assert counts.detection_rate == 0.5
    assert counts.false_positive_rate == pytest.approx(1 / 3)


def test_count_detections_empty_sides_and_mismatch():
    only_benign = count_detections([False], [False])
    assert only_benign.detection_rate is None
    assert only_benign.false_positive_rate == 0.0
    assert count_detections([True], [True]).false_positive_rate is None
    with pytest.raises(ValueError, match="labels"):
        count_detections([True], [])


def test_paired_bootstrap_ci_brackets_mean_and_detects_no_change():
    mean, low, high = paired_bootstrap_ci([0.5] * 50, [0.6] * 50, n_resamples=200)
    assert mean == pytest.approx(0.1)
    assert low == pytest.approx(0.1)
    assert high == pytest.approx(0.1)
    mixed_mean, mixed_low, mixed_high = paired_bootstrap_ci(
        [0.0, 1.0] * 25, [1.0, 0.0] * 25, n_resamples=500
    )
    assert mixed_mean == 0.0
    assert mixed_low < 0.0 < mixed_high


def test_paired_bootstrap_ci_rejects_bad_input():
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_ci([], [])
