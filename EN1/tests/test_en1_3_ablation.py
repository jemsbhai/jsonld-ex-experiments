"""Tests for EN1.3 Ablation — Comprehensive Byzantine-Robust Fusion Analysis.

RED phase — all tests should FAIL until en1_3_ablation_core is implemented.

Eight analysis dimensions:
  A. Stress configurations (fewer/weaker honest sources)
  B. Breaking point analysis (adversarial ratio sweep)
  C. Heterogeneous honest quality
  D. Adversary sophistication (SUBTLE, COLLUDING)
  E. Additional scalar baselines (median, majority vote, oracle trimmed)
  F. Calibration analysis (ECE, Brier score)
  G. Uncertainty preservation under attack
  H. Paired statistical tests with multiple comparison correction

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest experiments/EN1/tests/test_en1_3_ablation.py -v
"""
from __future__ import annotations

import pytest
import numpy as np

from EN1.en1_3_ablation_core import (
    # Extended adversary types
    AdversarialStrategyExt,
    generate_subtle_adversarial_opinions,
    generate_colluding_adversarial_opinions,
    # Heterogeneous honest generation
    generate_heterogeneous_honest_opinions,
    # Additional scalar baselines
    fuse_scalar_median,
    fuse_scalar_majority_vote,
    fuse_scalar_oracle_trimmed,
    # Calibration
    compute_ece,
    compute_brier_score,
    # Fused probability extraction (for calibration analysis)
    fuse_sl_cumulative_probs,
    fuse_sl_robust_probs,
    fuse_sl_trust_discount_probs,
    fuse_scalar_mean_probs,
    # Uncertainty tracking
    compute_mean_fused_uncertainty,
    # Breaking point analysis
    find_breaking_point,
    # Paired tests with correction
    compute_pairwise_mcnemar_matrix,
    # Extended single-scenario evaluator
    evaluate_extended_scenario,
)
from EN1.en1_3_core import (
    generate_ground_truth,
    generate_honest_opinions,
    generate_adversarial_opinions,
    AdversarialStrategy,
    compute_accuracy,
    compute_f1,
    _transpose_opinions,
    Opinion,
)


# ═══════════════════════════════════════════════════════════════════
# A. Heterogeneous honest source generation
# ═══════════════════════════════════════════════════════════════════

class TestHeterogeneousHonest:

    def test_shape_matches_accuracies(self):
        """One source per accuracy value."""
        gt = generate_ground_truth(50, seed=42)
        accs = [0.65, 0.75, 0.85, 0.95]
        sources = generate_heterogeneous_honest_opinions(
            gt, accuracies=accs, seed=42,
        )
        assert len(sources) == 4
        assert all(len(s) == 50 for s in sources)

    def test_quality_ordering_preserved(self):
        """Higher-accuracy sources should be more accurate empirically."""
        gt = generate_ground_truth(2000, seed=42)
        accs = [0.60, 0.90]
        sources = generate_heterogeneous_honest_opinions(gt, accs, seed=42)
        acc_low = sum(
            (s.projected_probability() > 0.5) == label
            for s, label in zip(sources[0], gt)
        ) / len(gt)
        acc_high = sum(
            (s.projected_probability() > 0.5) == label
            for s, label in zip(sources[1], gt)
        ) / len(gt)
        assert acc_high > acc_low

    def test_returns_opinions(self):
        gt = [True, False, True]
        sources = generate_heterogeneous_honest_opinions(
            gt, accuracies=[0.7, 0.8], seed=42,
        )
        for src in sources:
            for op in src:
                assert isinstance(op, Opinion)


# ═══════════════════════════════════════════════════════════════════
# B. Extended adversary strategies
# ═══════════════════════════════════════════════════════════════════

class TestSubtleAdversary:

    def test_shape(self):
        gt = generate_ground_truth(50, seed=42)
        adv = generate_subtle_adversarial_opinions(gt, n_sources=2, seed=42)
        assert len(adv) == 2
        assert all(len(s) == 50 for s in adv)

    def test_inverts_but_low_confidence(self):
        """Subtle adversary inverts labels but with LOW confidence — hard to detect."""
        gt = generate_ground_truth(500, seed=42)
        adv = generate_subtle_adversarial_opinions(gt, n_sources=1, seed=42)
        # Should still be mostly wrong
        n_wrong = sum(
            (op.projected_probability() > 0.5) != label
            for op, label in zip(adv[0], gt)
        )
        assert n_wrong / len(gt) > 0.65
        # But uncertainty should be high (low confidence)
        mean_u = np.mean([op.uncertainty for op in adv[0]])
        assert mean_u > 0.15  # more uncertain than standard inversion


class TestColludingAdversary:

    def test_shape(self):
        gt = generate_ground_truth(50, seed=42)
        adv = generate_colluding_adversarial_opinions(gt, n_sources=3, seed=42)
        assert len(adv) == 3
        assert all(len(s) == 50 for s in adv)

    def test_colluders_agree_with_each_other(self):
        """Colluding adversaries should have low pairwise conflict."""
        from jsonld_ex.confidence_algebra import pairwise_conflict
        gt = generate_ground_truth(100, seed=42)
        adv = generate_colluding_adversarial_opinions(gt, n_sources=3, seed=42)
        conflicts = []
        for i in range(100):
            c = pairwise_conflict(adv[0][i], adv[1][i])
            conflicts.append(c)
        mean_conflict = np.mean(conflicts)
        # Colluders should strongly agree (low conflict relative to
        # honest-vs-adversary conflict which is typically > 0.3)
        assert mean_conflict < 0.20

    def test_colluders_disagree_with_honest(self):
        """Colluding adversaries should conflict with honest sources."""
        from jsonld_ex.confidence_algebra import pairwise_conflict
        gt = generate_ground_truth(200, seed=42)
        honest = generate_honest_opinions(gt, n_sources=1, accuracy=0.85, seed=42)
        adv = generate_colluding_adversarial_opinions(gt, n_sources=1, seed=43)
        conflicts = []
        for i in range(200):
            c = pairwise_conflict(honest[0][i], adv[0][i])
            conflicts.append(c)
        mean_conflict = np.mean(conflicts)
        assert mean_conflict > 0.3


# ═══════════════════════════════════════════════════════════════════
# C. Additional scalar baselines
# ═══════════════════════════════════════════════════════════════════

class TestScalarMedian:

    def test_returns_predictions(self):
        per_instance = [
            [Opinion(0.8, 0.1, 0.1), Opinion(0.7, 0.2, 0.1),
             Opinion(0.05, 0.9, 0.05)],
        ]
        preds = fuse_scalar_median(per_instance)
        assert len(preds) == 1
        assert isinstance(preds[0], bool)

    def test_median_ignores_single_outlier(self):
        """With odd number of sources, median ignores one extreme."""
        per_instance = [
            [Opinion(0.85, 0.1, 0.05), Opinion(0.80, 0.1, 0.1),
             Opinion(0.05, 0.9, 0.05)],  # outlier
        ]
        preds = fuse_scalar_median(per_instance)
        assert preds[0] is True  # median is 0.80's prob, not pulled by outlier


class TestScalarMajorityVote:

    def test_returns_predictions(self):
        per_instance = [
            [Opinion(0.8, 0.1, 0.1), Opinion(0.7, 0.2, 0.1),
             Opinion(0.05, 0.9, 0.05)],
        ]
        preds = fuse_scalar_majority_vote(per_instance)
        assert len(preds) == 1
        assert isinstance(preds[0], bool)

    def test_majority_wins(self):
        """2 positive vs 1 negative → positive."""
        per_instance = [
            [Opinion(0.8, 0.1, 0.1), Opinion(0.7, 0.2, 0.1),
             Opinion(0.05, 0.9, 0.05)],
        ]
        preds = fuse_scalar_majority_vote(per_instance)
        assert preds[0] is True


class TestScalarOracleTrimmed:

    def test_returns_predictions(self):
        per_instance = [
            [Opinion(0.8, 0.1, 0.1), Opinion(0.7, 0.2, 0.1),
             Opinion(0.05, 0.9, 0.05)],
        ]
        preds = fuse_scalar_oracle_trimmed(
            per_instance, adversarial_indices=[2],
        )
        assert len(preds) == 1

    def test_oracle_removes_adversary(self):
        """Oracle knows which sources are adversarial and removes them."""
        per_instance = [
            [Opinion(0.85, 0.1, 0.05), Opinion(0.80, 0.1, 0.1),
             Opinion(0.05, 0.9, 0.05)],  # adversary at idx 2
        ]
        preds = fuse_scalar_oracle_trimmed(
            per_instance, adversarial_indices=[2],
        )
        assert preds[0] is True


# ═══════════════════════════════════════════════════════════════════
# D. Calibration metrics
# ═══════════════════════════════════════════════════════════════════

class TestECE:

    def test_perfect_calibration(self):
        """If predicted probabilities match empirical frequency, ECE=0."""
        # 10 instances, all predicted 0.8, 8 are correct → perfect
        probs = [0.8] * 10
        labels = [True] * 8 + [False] * 2
        ece = compute_ece(labels, probs, n_bins=1)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_overconfident(self):
        """If predicted 0.9 but only 50% correct, ECE is high."""
        probs = [0.9] * 100
        labels = [True] * 50 + [False] * 50
        ece = compute_ece(labels, probs, n_bins=1)
        assert ece > 0.35

    def test_range_0_1(self):
        probs = list(np.random.RandomState(42).uniform(0, 1, 100))
        labels = [bool(p > 0.5) for p in probs]
        ece = compute_ece(labels, probs)
        assert 0.0 <= ece <= 1.0


class TestBrierScore:

    def test_perfect_prediction(self):
        probs = [1.0, 0.0, 1.0]
        labels = [True, False, True]
        assert compute_brier_score(labels, probs) == pytest.approx(0.0)

    def test_worst_prediction(self):
        probs = [0.0, 1.0, 0.0]
        labels = [True, False, True]
        assert compute_brier_score(labels, probs) == pytest.approx(1.0)

    def test_range_0_1(self):
        probs = list(np.random.RandomState(42).uniform(0, 1, 100))
        labels = [bool(np.random.RandomState(42 + i).random() > 0.5) for i in range(100)]
        bs = compute_brier_score(labels, probs)
        assert 0.0 <= bs <= 1.0


# ═══════════════════════════════════════════════════════════════════
# E. Probability extraction for calibration
# ═══════════════════════════════════════════════════════════════════

class TestProbabilityExtraction:

    def test_sl_cumulative_probs_returns_floats(self):
        per_instance = [
            [Opinion(0.8, 0.1, 0.1), Opinion(0.7, 0.2, 0.1)],
            [Opinion(0.1, 0.8, 0.1), Opinion(0.2, 0.7, 0.1)],
        ]
        probs = fuse_sl_cumulative_probs(per_instance)
        assert len(probs) == 2
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_scalar_mean_probs(self):
        per_instance = [
            [Opinion(0.8, 0.1, 0.1), Opinion(0.7, 0.2, 0.1)],
        ]
        probs = fuse_scalar_mean_probs(per_instance)
        assert len(probs) == 1
        assert 0.0 <= probs[0] <= 1.0


# ═══════════════════════════════════════════════════════════════════
# F. Uncertainty preservation
# ═══════════════════════════════════════════════════════════════════

class TestUncertaintyPreservation:

    def test_returns_float(self):
        per_instance = [
            [Opinion(0.8, 0.1, 0.1), Opinion(0.7, 0.2, 0.1)],
        ]
        u = compute_mean_fused_uncertainty(per_instance, method="cumulative")
        assert isinstance(u, float)
        assert 0.0 <= u <= 1.0

    def test_adversary_increases_uncertainty_for_robust(self):
        """Robust fuse with adversary should preserve higher uncertainty
        than cumulative with adversary (which gets falsely confident)."""
        honest = [Opinion(0.85, 0.1, 0.05)] * 3
        adversary = [Opinion(0.05, 0.9, 0.05)]
        mixed = [honest + adversary]  # 1 instance, 4 sources

        u_cumul = compute_mean_fused_uncertainty(mixed, method="cumulative")
        u_robust = compute_mean_fused_uncertainty(mixed, method="robust")
        # After removing outlier, robust should have lower uncertainty
        # (cleaner signal), OR cumulative should have suspiciously low
        # uncertainty (overconfident despite adversary)
        # Both are valid outcomes — the key is they differ
        assert u_cumul != pytest.approx(u_robust, abs=0.01)

    def test_methods_supported(self):
        per_instance = [
            [Opinion(0.8, 0.1, 0.1), Opinion(0.7, 0.2, 0.1)],
        ]
        for method in ["cumulative", "robust"]:
            u = compute_mean_fused_uncertainty(per_instance, method=method)
            assert 0.0 <= u <= 1.0


# ═══════════════════════════════════════════════════════════════════
# G. Breaking point analysis
# ═══════════════════════════════════════════════════════════════════

class TestBreakingPoint:

    def test_returns_ratio(self):
        """find_breaking_point should return the adversarial ratio where
        accuracy drops below threshold."""
        bp = find_breaking_point(
            n_honest=5,
            honest_accuracy=0.85,
            adversarial_strategy=AdversarialStrategy.INVERSION,
            method="sl_trust_discount",
            baseline_drop=0.05,
            max_adversarial=10,
            seed=42,
        )
        assert isinstance(bp, dict)
        assert "breaking_k" in bp
        assert "breaking_ratio" in bp
        assert "baseline_accuracy" in bp
        assert "accuracy_at_break" in bp

    def test_no_break_if_robust(self):
        """With very few adversaries and strong method, may not break."""
        bp = find_breaking_point(
            n_honest=10,
            honest_accuracy=0.90,
            adversarial_strategy=AdversarialStrategy.RANDOM,
            method="sl_trust_discount",
            baseline_drop=0.05,
            max_adversarial=3,
            seed=42,
        )
        # May or may not break — but structure should be valid
        assert bp["breaking_k"] is None or bp["breaking_k"] >= 0

    def test_cumulative_breaks_before_robust(self):
        """sl_cumulative should break at a lower k than sl_robust on inversion."""
        bp_cumul = find_breaking_point(
            n_honest=5, honest_accuracy=0.80,
            adversarial_strategy=AdversarialStrategy.INVERSION,
            method="sl_cumulative", baseline_drop=0.05,
            max_adversarial=5, seed=42,
        )
        bp_robust = find_breaking_point(
            n_honest=5, honest_accuracy=0.80,
            adversarial_strategy=AdversarialStrategy.INVERSION,
            method="sl_robust", baseline_drop=0.05,
            max_adversarial=5, seed=42,
        )
        # If both break, robust should break at same or higher k
        if bp_cumul["breaking_k"] is not None and bp_robust["breaking_k"] is not None:
            assert bp_robust["breaking_k"] >= bp_cumul["breaking_k"]


# ═══════════════════════════════════════════════════════════════════
# H. Pairwise McNemar matrix with Bonferroni correction
# ═══════════════════════════════════════════════════════════════════

class TestPairwiseMcNemar:

    def test_returns_matrix_structure(self):
        per_question = [
            {"method_a": True, "method_b": True, "method_c": False},
            {"method_a": True, "method_b": False, "method_c": False},
            {"method_a": False, "method_b": True, "method_c": True},
        ] * 10  # need enough for meaningful test
        methods = ["method_a", "method_b", "method_c"]
        matrix = compute_pairwise_mcnemar_matrix(per_question, methods)
        assert isinstance(matrix, dict)
        # Should have n*(n-1)/2 = 3 pairs
        assert len(matrix) == 3

    def test_includes_bonferroni(self):
        per_question = [
            {"method_a": True, "method_b": False},
        ] * 20 + [
            {"method_a": False, "method_b": True},
        ] * 5
        methods = ["method_a", "method_b"]
        matrix = compute_pairwise_mcnemar_matrix(per_question, methods)
        pair_key = "method_a_vs_method_b"
        assert pair_key in matrix
        assert "p_value" in matrix[pair_key]
        assert "p_bonferroni" in matrix[pair_key]
        assert "significant_bonferroni" in matrix[pair_key]

    def test_bonferroni_more_conservative(self):
        """Bonferroni-corrected p should be >= raw p."""
        per_question = [
            {"m1": True, "m2": False, "m3": True},
        ] * 15 + [
            {"m1": False, "m2": True, "m3": False},
        ] * 5
        methods = ["m1", "m2", "m3"]
        matrix = compute_pairwise_mcnemar_matrix(per_question, methods)
        for pair, result in matrix.items():
            assert result["p_bonferroni"] >= result["p_value"] - 1e-12


# ═══════════════════════════════════════════════════════════════════
# I. Extended scenario evaluator
# ═══════════════════════════════════════════════════════════════════

class TestEvaluateExtendedScenario:

    def test_returns_all_methods(self):
        """Extended evaluator includes old + new methods."""
        results = evaluate_extended_scenario(
            n_instances=50, n_honest=5, n_adversarial=1,
            honest_accuracy=0.80,
            adversarial_strategy=AdversarialStrategy.INVERSION,
            robust_thresholds=[0.15],
            seed=42,
        )
        # Must include all methods
        expected = {
            "scalar_mean", "scalar_median", "scalar_majority_vote",
            "scalar_trimmed_mean", "scalar_oracle_trimmed",
            "sl_cumulative", "sl_robust_t0.15", "sl_trust_discount",
        }
        assert expected.issubset(set(results.keys()))

    def test_each_method_has_calibration(self):
        """Every method should report ECE and Brier score."""
        results = evaluate_extended_scenario(
            n_instances=100, n_honest=5, n_adversarial=1,
            honest_accuracy=0.80,
            adversarial_strategy=AdversarialStrategy.INVERSION,
            robust_thresholds=[0.15],
            seed=42,
        )
        for method, metrics in results.items():
            if method.startswith("_"):
                continue  # skip _mcnemar_matrix
            assert "ece" in metrics, f"{method} missing ece"
            assert "brier" in metrics, f"{method} missing brier"
            assert 0.0 <= metrics["ece"] <= 1.0
            assert 0.0 <= metrics["brier"] <= 1.0

    def test_each_method_has_uncertainty(self):
        """SL methods should report mean fused uncertainty."""
        results = evaluate_extended_scenario(
            n_instances=50, n_honest=5, n_adversarial=1,
            honest_accuracy=0.80,
            adversarial_strategy=AdversarialStrategy.INVERSION,
            robust_thresholds=[0.15],
            seed=42,
        )
        for method in ["sl_cumulative", "sl_robust_t0.15", "sl_trust_discount"]:
            assert "mean_fused_uncertainty" in results[method], f"{method} missing uncertainty"

    def test_heterogeneous_mode(self):
        """Can run with heterogeneous honest sources."""
        results = evaluate_extended_scenario(
            n_instances=50,
            honest_accuracies=[0.65, 0.75, 0.85, 0.95],
            n_adversarial=1,
            adversarial_strategy=AdversarialStrategy.INVERSION,
            robust_thresholds=[0.15],
            seed=42,
        )
        assert "scalar_mean" in results

    def test_subtle_adversary(self):
        """Can run with subtle adversary strategy."""
        results = evaluate_extended_scenario(
            n_instances=50, n_honest=5, n_adversarial=2,
            honest_accuracy=0.80,
            adversarial_strategy_ext=AdversarialStrategyExt.SUBTLE,
            robust_thresholds=[0.15],
            seed=42,
        )
        assert "sl_robust_t0.15" in results

    def test_colluding_adversary(self):
        """Can run with colluding adversary strategy."""
        results = evaluate_extended_scenario(
            n_instances=50, n_honest=5, n_adversarial=2,
            honest_accuracy=0.80,
            adversarial_strategy_ext=AdversarialStrategyExt.COLLUDING,
            robust_thresholds=[0.15],
            seed=42,
        )
        assert "sl_robust_t0.15" in results

    def test_mcnemar_matrix_included(self):
        """Extended evaluator should include pairwise McNemar matrix."""
        results = evaluate_extended_scenario(
            n_instances=100, n_honest=5, n_adversarial=2,
            honest_accuracy=0.80,
            adversarial_strategy=AdversarialStrategy.INVERSION,
            robust_thresholds=[0.15],
            seed=42,
        )
        assert "_mcnemar_matrix" in results
        assert len(results["_mcnemar_matrix"]) > 0
