"""Tests for EN8.4B IR metrics: NDCG@10, MRR@10, Recall@k.

These metrics are the foundation of the BEIR benchmark evaluation.
We validate against hand-computed examples and edge cases to ensure
correctness before running the full experiment.

References:
    - NDCG: Jarvelin & Kekalainen (2002), "Cumulated gain-based
      evaluation of IR techniques"
    - BEIR: Thakur et al. (2021), "BEIR: A Heterogeneous Benchmark
      for Zero-shot Evaluation of Information Retrieval Models"
"""

import math
import sys
from pathlib import Path

import pytest

# -- Path setup (same pattern as other experiment tests) ---------------
_TEST_DIR = Path(__file__).resolve().parent
_EN8_DIR = _TEST_DIR.parent
_EXPERIMENTS_DIR = _EN8_DIR.parent

sys.path.insert(0, str(_EXPERIMENTS_DIR))
sys.path.insert(0, str(_EN8_DIR))

from en8_4b_beir_benchmarks import (
    ndcg_at_k,
    mrr_at_k,
    recall_at_k,
    compute_metrics_for_query,
)


# =====================================================================
# NDCG@k Tests
# =====================================================================

class TestNDCGAtK:
    """Test Normalized Discounted Cumulative Gain at k."""

    def test_perfect_ranking(self):
        """Perfect ranking should yield NDCG = 1.0."""
        # qrel: doc_a has relevance 2, doc_b has relevance 1
        qrel = {"doc_a": 2, "doc_b": 1}
        # System returns them in perfect order
        ranked = ["doc_a", "doc_b", "doc_c"]
        assert ndcg_at_k(qrel, ranked, k=10) == pytest.approx(1.0)

    def test_reversed_ranking(self):
        """Reversed ranking should yield NDCG < 1.0."""
        qrel = {"doc_a": 2, "doc_b": 1}
        # System returns them in reversed order
        ranked = ["doc_b", "doc_a", "doc_c"]
        # DCG = (2^1 - 1)/log2(2) + (2^2 - 1)/log2(3) = 1/1 + 3/1.585 = 1 + 1.893 = 2.893
        # IDCG = (2^2 - 1)/log2(2) + (2^1 - 1)/log2(3) = 3/1 + 1/1.585 = 3 + 0.631 = 3.631
        # NDCG = 2.893 / 3.631 = 0.7967
        assert ndcg_at_k(qrel, ranked, k=10) == pytest.approx(
            2.892789 / 3.630930, rel=1e-4
        )

    def test_no_relevant_docs_returned(self):
        """If no relevant docs in ranking, NDCG = 0."""
        qrel = {"doc_a": 2, "doc_b": 1}
        ranked = ["doc_x", "doc_y", "doc_z"]
        assert ndcg_at_k(qrel, ranked, k=10) == pytest.approx(0.0)

    def test_empty_qrel(self):
        """No relevant docs in ground truth => NDCG = 0."""
        qrel = {}
        ranked = ["doc_a", "doc_b"]
        assert ndcg_at_k(qrel, ranked, k=10) == pytest.approx(0.0)

    def test_k_truncation(self):
        """Only top-k results should matter."""
        qrel = {"doc_a": 1, "doc_b": 1}
        # doc_b is at position 2, within k=2
        ranked = ["doc_x", "doc_b", "doc_a"]
        ndcg_k2 = ndcg_at_k(qrel, ranked, k=2)
        # Only first 2 considered: doc_x (0), doc_b (1)
        # DCG = 0 + (2^1 - 1)/log2(3) = 0 + 0.631 = 0.631
        # IDCG for k=2 with two rel=1 docs: 1/log2(2) + 1/log2(3) = 1 + 0.631 = 1.631
        # NDCG = 0.631 / 1.631 = 0.387
        assert ndcg_k2 == pytest.approx(0.6309 / 1.6309, rel=1e-3)

    def test_graded_relevance(self):
        """NDCG should handle graded relevance (0, 1, 2, 3)."""
        qrel = {"doc_a": 3, "doc_b": 2, "doc_c": 1, "doc_d": 0}
        # Perfect order
        ranked = ["doc_a", "doc_b", "doc_c", "doc_d"]
        assert ndcg_at_k(qrel, ranked, k=10) == pytest.approx(1.0)

    def test_single_relevant_doc(self):
        """Single relevant doc at position 1 => NDCG = 1.0."""
        qrel = {"doc_a": 1}
        ranked = ["doc_a"]
        assert ndcg_at_k(qrel, ranked, k=10) == pytest.approx(1.0)

    def test_single_relevant_doc_at_position_3(self):
        """Single relevant doc at position 3."""
        qrel = {"doc_a": 1}
        ranked = ["doc_x", "doc_y", "doc_a"]
        # DCG = (2^1 - 1)/log2(4) = 1/2 = 0.5
        # IDCG = (2^1 - 1)/log2(2) = 1/1 = 1.0
        # NDCG = 0.5
        assert ndcg_at_k(qrel, ranked, k=10) == pytest.approx(0.5, rel=1e-4)

    def test_zero_relevance_excluded(self):
        """Docs with relevance 0 in qrel should not contribute to IDCG."""
        qrel = {"doc_a": 1, "doc_b": 0}
        ranked = ["doc_a", "doc_b"]
        # Only doc_a is relevant. IDCG considers only rel>0 docs.
        # DCG = 1/log2(2) = 1.0
        # IDCG = 1/log2(2) = 1.0
        assert ndcg_at_k(qrel, ranked, k=10) == pytest.approx(1.0)


# =====================================================================
# MRR@k Tests
# =====================================================================

class TestMRRAtK:
    """Test Mean Reciprocal Rank at k."""

    def test_first_position(self):
        """Relevant doc at position 1 => RR = 1.0."""
        qrel = {"doc_a": 1}
        ranked = ["doc_a", "doc_b", "doc_c"]
        assert mrr_at_k(qrel, ranked, k=10) == pytest.approx(1.0)

    def test_second_position(self):
        """Relevant doc at position 2 => RR = 0.5."""
        qrel = {"doc_a": 1}
        ranked = ["doc_x", "doc_a", "doc_c"]
        assert mrr_at_k(qrel, ranked, k=10) == pytest.approx(0.5)

    def test_third_position(self):
        """Relevant doc at position 3 => RR = 1/3."""
        qrel = {"doc_a": 1}
        ranked = ["doc_x", "doc_y", "doc_a"]
        assert mrr_at_k(qrel, ranked, k=10) == pytest.approx(1.0 / 3.0)

    def test_no_relevant(self):
        """No relevant doc in ranking => RR = 0."""
        qrel = {"doc_a": 1}
        ranked = ["doc_x", "doc_y", "doc_z"]
        assert mrr_at_k(qrel, ranked, k=10) == pytest.approx(0.0)

    def test_k_truncation(self):
        """Relevant doc beyond k => RR = 0."""
        qrel = {"doc_a": 1}
        ranked = ["doc_x", "doc_y", "doc_a"]
        assert mrr_at_k(qrel, ranked, k=2) == pytest.approx(0.0)

    def test_multiple_relevant_first_matters(self):
        """MRR uses the FIRST relevant doc only."""
        qrel = {"doc_a": 1, "doc_b": 2}
        ranked = ["doc_x", "doc_b", "doc_a"]
        # First relevant is doc_b at position 2
        assert mrr_at_k(qrel, ranked, k=10) == pytest.approx(0.5)

    def test_empty_qrel(self):
        """No ground truth => RR = 0."""
        qrel = {}
        ranked = ["doc_a", "doc_b"]
        assert mrr_at_k(qrel, ranked, k=10) == pytest.approx(0.0)

    def test_zero_relevance_ignored(self):
        """Docs with relevance 0 should not count as relevant."""
        qrel = {"doc_a": 0, "doc_b": 1}
        ranked = ["doc_a", "doc_b"]
        # doc_a has rel=0, not relevant. First relevant is doc_b at pos 2.
        assert mrr_at_k(qrel, ranked, k=10) == pytest.approx(0.5)


# =====================================================================
# Recall@k Tests
# =====================================================================

class TestRecallAtK:
    """Test Recall at k."""

    def test_perfect_recall(self):
        """All relevant docs in top-k."""
        qrel = {"doc_a": 1, "doc_b": 2}
        ranked = ["doc_a", "doc_b", "doc_c"]
        assert recall_at_k(qrel, ranked, k=10) == pytest.approx(1.0)

    def test_partial_recall(self):
        """One of two relevant docs in top-k."""
        qrel = {"doc_a": 1, "doc_b": 2}
        ranked = ["doc_a", "doc_c", "doc_d"]
        assert recall_at_k(qrel, ranked, k=10) == pytest.approx(0.5)

    def test_zero_recall(self):
        """No relevant docs in top-k."""
        qrel = {"doc_a": 1, "doc_b": 2}
        ranked = ["doc_c", "doc_d", "doc_e"]
        assert recall_at_k(qrel, ranked, k=10) == pytest.approx(0.0)

    def test_k_truncation(self):
        """Only top-k results considered."""
        qrel = {"doc_a": 1, "doc_b": 1}
        ranked = ["doc_c", "doc_a", "doc_b"]
        # k=2: only [doc_c, doc_a], so recall = 1/2 = 0.5
        assert recall_at_k(qrel, ranked, k=2) == pytest.approx(0.5)

    def test_empty_qrel(self):
        """No relevant docs => recall = 0 (avoid division by zero)."""
        qrel = {}
        ranked = ["doc_a", "doc_b"]
        assert recall_at_k(qrel, ranked, k=10) == pytest.approx(0.0)

    def test_zero_relevance_excluded(self):
        """Docs with relevance 0 should not count as relevant."""
        qrel = {"doc_a": 0, "doc_b": 1, "doc_c": 2}
        ranked = ["doc_a", "doc_b"]
        # Only doc_b and doc_c are relevant (rel>0). doc_b found => 1/2
        assert recall_at_k(qrel, ranked, k=10) == pytest.approx(0.5)

    def test_recall_at_100(self):
        """Recall@100 with more results than relevant docs."""
        qrel = {"doc_a": 1}
        ranked = [f"doc_{i}" for i in range(100)]
        ranked[50] = "doc_a"
        assert recall_at_k(qrel, ranked, k=100) == pytest.approx(1.0)


# =====================================================================
# Integration: compute_metrics_for_query
# =====================================================================

class TestComputeMetricsForQuery:
    """Test the combined metric computation for a single query."""

    def test_perfect_system(self):
        """Perfect ranking produces all-1.0 metrics."""
        qrel = {"doc_a": 2, "doc_b": 1}
        ranked = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
        metrics = compute_metrics_for_query(qrel, ranked)
        assert metrics["ndcg@10"] == pytest.approx(1.0)
        assert metrics["mrr@10"] == pytest.approx(1.0)
        assert metrics["recall@1"] == pytest.approx(0.5)
        assert metrics["recall@5"] == pytest.approx(1.0)
        assert metrics["recall@10"] == pytest.approx(1.0)

    def test_empty_ranking(self):
        """Empty ranking produces all-zero metrics."""
        qrel = {"doc_a": 1}
        ranked = []
        metrics = compute_metrics_for_query(qrel, ranked)
        assert metrics["ndcg@10"] == pytest.approx(0.0)
        assert metrics["mrr@10"] == pytest.approx(0.0)
        assert metrics["recall@1"] == pytest.approx(0.0)

    def test_returns_all_expected_keys(self):
        """Verify all expected metric keys are present."""
        qrel = {"doc_a": 1}
        ranked = ["doc_a"]
        metrics = compute_metrics_for_query(qrel, ranked)
        expected_keys = {
            "ndcg@10", "mrr@10",
            "recall@1", "recall@5", "recall@10", "recall@100",
        }
        assert expected_keys == set(metrics.keys())
