"""Tests for EN1.3 — Byzantine-Robust Fusion.

RED phase — all tests should FAIL until en1_3_core is implemented.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest experiments/EN1/tests/test_en1_3.py -v
"""
from __future__ import annotations

import pytest
import numpy as np

from EN1.en1_3_core import (
    # Data generation
    generate_ground_truth,
    generate_honest_opinions,
    generate_adversarial_opinions,
    AdversarialStrategy,
    # Fusion methods
    fuse_scalar_mean,
    fuse_scalar_trimmed_mean,
    fuse_sl_cumulative,
    fuse_sl_robust,
    fuse_sl_trust_discount,
    # Evaluation
    compute_accuracy,
    compute_f1,
    compute_detection_rate,
    # Full pipeline
    evaluate_single_scenario,
    FUSION_METHODS,
)
from jsonld_ex.confidence_algebra import Opinion


# ═══════════════════════════════════════════════════════════════════
# 1. Data generation
# ═══════════════════════════════════════════════════════════════════

class TestGenerateGroundTruth:

    def test_returns_binary_array(self):
        gt = generate_ground_truth(n=100, positive_rate=0.6, seed=42)
        assert len(gt) == 100
        assert set(gt).issubset({True, False})

    def test_positive_rate_approximate(self):
        gt = generate_ground_truth(n=10000, positive_rate=0.7, seed=42)
        rate = sum(gt) / len(gt)
        assert abs(rate - 0.7) < 0.03  # within 3% for 10K samples

    def test_deterministic_with_seed(self):
        gt1 = generate_ground_truth(n=50, positive_rate=0.5, seed=99)
        gt2 = generate_ground_truth(n=50, positive_rate=0.5, seed=99)
        assert gt1 == gt2


class TestGenerateHonestOpinions:

    def test_shape(self):
        gt = [True, False, True, True, False]
        opinions = generate_honest_opinions(
            gt, n_sources=3, accuracy=0.85, seed=42,
        )
        assert len(opinions) == 3  # 3 sources
        assert all(len(src) == 5 for src in opinions)  # 5 instances each

    def test_returns_opinions(self):
        gt = [True, False]
        opinions = generate_honest_opinions(gt, n_sources=2, accuracy=0.85, seed=42)
        for src in opinions:
            for op in src:
                assert isinstance(op, Opinion)
                assert pytest.approx(op.belief + op.disbelief + op.uncertainty, abs=1e-9) == 1.0

    def test_honest_sources_mostly_correct(self):
        """Honest sources with acc=0.85 should get >75% right on 500 instances."""
        gt = generate_ground_truth(n=500, positive_rate=0.6, seed=42)
        opinions = generate_honest_opinions(gt, n_sources=5, accuracy=0.85, seed=42)
        for src in opinions:
            n_correct = sum(
                (op.projected_probability() > 0.5) == label
                for op, label in zip(src, gt)
            )
            assert n_correct / len(gt) > 0.75


class TestGenerateAdversarialOpinions:

    def test_shape(self):
        gt = [True, False, True]
        adv = generate_adversarial_opinions(
            gt, n_sources=2, strategy=AdversarialStrategy.INVERSION, seed=42,
        )
        assert len(adv) == 2
        assert all(len(src) == 3 for src in adv)

    def test_inversion_mostly_wrong(self):
        """Inversion strategy should get most labels wrong."""
        gt = generate_ground_truth(n=500, positive_rate=0.6, seed=42)
        adv = generate_adversarial_opinions(
            gt, n_sources=1, strategy=AdversarialStrategy.INVERSION, seed=42,
        )
        n_wrong = sum(
            (op.projected_probability() > 0.5) != label
            for op, label in zip(adv[0], gt)
        )
        assert n_wrong / len(gt) > 0.75

    def test_random_near_chance(self):
        """Random strategy should be near 50% accuracy."""
        gt = generate_ground_truth(n=1000, positive_rate=0.5, seed=42)
        adv = generate_adversarial_opinions(
            gt, n_sources=1, strategy=AdversarialStrategy.RANDOM, seed=42,
        )
        n_correct = sum(
            (op.projected_probability() > 0.5) == label
            for op, label in zip(adv[0], gt)
        )
        rate = n_correct / len(gt)
        assert 0.35 < rate < 0.65

    def test_targeted_flips_only_positives(self):
        """Targeted strategy should flip positive→negative but leave negatives alone."""
        gt = [True, True, False, False, True]
        adv = generate_adversarial_opinions(
            gt, n_sources=1, strategy=AdversarialStrategy.TARGETED, seed=42,
        )
        # For negatives (indices 2,3), adversary should agree with truth
        for idx in [2, 3]:
            assert adv[0][idx].projected_probability() < 0.5
        # For positives (indices 0,1,4), adversary should disagree
        for idx in [0, 1, 4]:
            assert adv[0][idx].projected_probability() < 0.5


# ═══════════════════════════════════════════════════════════════════
# 2. Fusion methods
# ═══════════════════════════════════════════════════════════════════

class TestFuseScalarMean:

    def test_unanimous_positive(self):
        ops = [Opinion(0.9, 0.05, 0.05), Opinion(0.85, 0.1, 0.05)]
        predictions = fuse_scalar_mean([ops])  # 1 instance, 2 sources
        assert len(predictions) == 1
        assert predictions[0] is True  # high belief → positive

    def test_unanimous_negative(self):
        ops = [Opinion(0.05, 0.9, 0.05), Opinion(0.1, 0.85, 0.05)]
        predictions = fuse_scalar_mean([ops])
        assert predictions[0] is False

    def test_returns_list_of_bools(self):
        # 3 instances, 2 sources per instance
        per_instance = [
            [Opinion(0.8, 0.1, 0.1), Opinion(0.7, 0.2, 0.1)],
            [Opinion(0.1, 0.8, 0.1), Opinion(0.2, 0.7, 0.1)],
            [Opinion(0.5, 0.3, 0.2), Opinion(0.4, 0.4, 0.2)],
        ]
        predictions = fuse_scalar_mean(per_instance)
        assert len(predictions) == 3
        assert all(isinstance(p, bool) for p in predictions)


class TestFuseScalarTrimmedMean:

    def test_trims_outlier(self):
        """With k=1, one extreme outlier should be trimmed."""
        # 1 instance, 3 sources. Two honest (positive), one adversarial (strong negative)
        per_instance = [
            [Opinion(0.85, 0.1, 0.05), Opinion(0.8, 0.1, 0.1),
             Opinion(0.05, 0.9, 0.05)],
        ]
        # Without trimming, adversary might flip decision
        pred_trimmed = fuse_scalar_trimmed_mean(per_instance, k=1)
        assert pred_trimmed[0] is True  # trimmed mean should keep positive


class TestFuseSLCumulative:

    def test_returns_predictions(self):
        per_instance = [
            [Opinion(0.8, 0.1, 0.1), Opinion(0.7, 0.2, 0.1)],
        ]
        preds = fuse_sl_cumulative(per_instance)
        assert len(preds) == 1
        assert isinstance(preds[0], bool)


class TestFuseSLRobust:

    def test_returns_predictions_and_removed(self):
        per_instance = [
            [Opinion(0.8, 0.1, 0.1), Opinion(0.7, 0.2, 0.1),
             Opinion(0.05, 0.9, 0.05)],  # outlier
        ]
        preds, per_instance_removed = fuse_sl_robust(
            per_instance, threshold=0.15,
        )
        assert len(preds) == 1
        assert isinstance(preds[0], bool)
        assert len(per_instance_removed) == 1
        # Outlier at index 2 should be removed
        assert 2 in per_instance_removed[0]

    def test_no_removals_when_cohesive(self):
        per_instance = [
            [Opinion(0.8, 0.1, 0.1), Opinion(0.75, 0.15, 0.1)],
        ]
        _, removed = fuse_sl_robust(per_instance, threshold=0.15)
        assert removed[0] == []


class TestFuseSLTrustDiscount:

    def test_returns_predictions(self):
        per_instance = [
            [Opinion(0.8, 0.1, 0.1), Opinion(0.05, 0.9, 0.05)],
        ]
        trust = [Opinion(0.9, 0.05, 0.05), Opinion(0.2, 0.3, 0.5)]
        preds = fuse_sl_trust_discount(per_instance, trust)
        assert len(preds) == 1
        assert isinstance(preds[0], bool)

    def test_low_trust_downweights_adversary(self):
        """With low trust on adversary, honest source should dominate."""
        per_instance = [
            [Opinion(0.85, 0.1, 0.05), Opinion(0.05, 0.9, 0.05)],
        ]
        trust = [
            Opinion(0.9, 0.05, 0.05),   # high trust on honest
            Opinion(0.1, 0.4, 0.5),      # low trust on adversary
        ]
        preds = fuse_sl_trust_discount(per_instance, trust)
        assert preds[0] is True  # honest source dominates


# ═══════════════════════════════════════════════════════════════════
# 3. Evaluation metrics
# ═══════════════════════════════════════════════════════════════════

class TestMetrics:

    def test_accuracy(self):
        gt = [True, True, False, False, True]
        pred = [True, False, False, True, True]
        assert compute_accuracy(gt, pred) == pytest.approx(0.6)

    def test_f1_perfect(self):
        gt = [True, True, False, False]
        pred = [True, True, False, False]
        assert compute_f1(gt, pred) == pytest.approx(1.0)

    def test_f1_all_wrong(self):
        gt = [True, True, True]
        pred = [False, False, False]
        assert compute_f1(gt, pred) == pytest.approx(0.0)

    def test_detection_rate_all_detected(self):
        removed_indices = [{10, 11}]  # single instance, removed indices 10, 11
        adversarial_indices = [10, 11]
        rate = compute_detection_rate(removed_indices, adversarial_indices, n_sources=12)
        assert rate["recall"] == pytest.approx(1.0)

    def test_detection_rate_none_detected(self):
        removed_indices = [set()]
        adversarial_indices = [10, 11]
        rate = compute_detection_rate(removed_indices, adversarial_indices, n_sources=12)
        assert rate["recall"] == pytest.approx(0.0)

    def test_detection_rate_false_positive(self):
        """If honest sources are removed, precision drops."""
        removed_indices = [{0, 10}]  # 0 is honest, 10 is adversarial
        adversarial_indices = [10]
        rate = compute_detection_rate(removed_indices, adversarial_indices, n_sources=11)
        assert rate["recall"] == pytest.approx(1.0)
        assert rate["precision"] == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════
# 4. Full pipeline
# ═══════════════════════════════════════════════════════════════════

class TestEvaluateSingleScenario:

    def test_returns_all_methods(self):
        results = evaluate_single_scenario(
            n_instances=50,
            n_honest=5,
            n_adversarial=1,
            honest_accuracy=0.85,
            adversarial_strategy=AdversarialStrategy.INVERSION,
            robust_thresholds=[0.10, 0.20],
            seed=42,
        )
        assert isinstance(results, dict)
        assert "scalar_mean" in results
        assert "scalar_trimmed_mean" in results
        assert "sl_cumulative" in results
        assert "sl_robust_t0.10" in results
        assert "sl_robust_t0.20" in results
        assert "sl_trust_discount" in results

    def test_each_method_has_metrics(self):
        results = evaluate_single_scenario(
            n_instances=50, n_honest=5, n_adversarial=1,
            honest_accuracy=0.85,
            adversarial_strategy=AdversarialStrategy.INVERSION,
            robust_thresholds=[0.15],
            seed=42,
        )
        for method, metrics in results.items():
            assert "accuracy" in metrics
            assert "f1" in metrics
            assert 0.0 <= metrics["accuracy"] <= 1.0
            assert 0.0 <= metrics["f1"] <= 1.0

    def test_robust_beats_cumulative_with_adversary(self):
        """With adversarial injection, robust should beat or match cumulative."""
        results = evaluate_single_scenario(
            n_instances=500, n_honest=10, n_adversarial=3,
            honest_accuracy=0.85,
            adversarial_strategy=AdversarialStrategy.INVERSION,
            robust_thresholds=[0.15],
            seed=42,
        )
        # Robust should be at least as good as cumulative (usually better)
        assert results["sl_robust_t0.15"]["accuracy"] >= results["sl_cumulative"]["accuracy"] - 0.02

    def test_no_adversary_baseline(self):
        """With 0 adversaries, all methods should perform similarly."""
        results = evaluate_single_scenario(
            n_instances=200, n_honest=10, n_adversarial=0,
            honest_accuracy=0.85,
            adversarial_strategy=AdversarialStrategy.RANDOM,
            robust_thresholds=[0.15],
            seed=42,
        )
        # All methods should be above 80% with 10 honest sources at 0.85 accuracy
        for method, metrics in results.items():
            assert metrics["accuracy"] > 0.80, f"{method} accuracy too low: {metrics['accuracy']}"
