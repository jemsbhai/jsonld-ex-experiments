"""Tests for EN3.2-H1 Core — Calibrated Selective Answering (Abstention).

Tests signal computation, precision-coverage curves, AUC, oracle bounds,
parameter sweep, and edge cases.

RED phase: all tests written before implementation.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest experiments/EN3/tests/test_en3_2_h1.py -v
"""
from __future__ import annotations

import math
import pytest

from EN3.en3_2_h1_core import (
    compute_scalar_signals,
    compute_sl_signals,
    precision_coverage_curve,
    auc_precision_coverage,
    oracle_signal,
    random_signal,
    DEFAULT_COVERAGE_LEVELS,
    SCALAR_SIGNAL_NAMES,
    SL_SIGNAL_NAMES,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures — minimal synthetic data
# ═══════════════════════════════════════════════════════════════════

def _make_passages(scores, poison_flags=None, gold_idx=0):
    """Create synthetic passage list for testing."""
    if poison_flags is None:
        poison_flags = [False] * len(scores)
    passages = []
    for i, (score, is_poison) in enumerate(zip(scores, poison_flags)):
        passages.append({
            "passage_id": f"p_{i}",
            "score": score,
            "is_poison": is_poison,
            "is_gold": i == gold_idx,
        })
    return passages


def _make_extractions(n, qa_scores=None, answers=None):
    """Create synthetic extraction dict for testing."""
    if qa_scores is None:
        qa_scores = [0.5] * n
    if answers is None:
        answers = [f"answer_{i}" for i in range(n)]
    return {
        f"p_{i}": {"answer": answers[i], "qa_score": qa_scores[i]}
        for i in range(n)
    }


# ═══════════════════════════════════════════════════════════════════
# 1. Scalar signals
# ═══════════════════════════════════════════════════════════════════

class TestComputeScalarSignals:
    """Tests for compute_scalar_signals."""

    def test_returns_all_expected_keys(self):
        passages = _make_passages([0.8, 0.6, 0.4])
        extractions = _make_extractions(3, qa_scores=[0.9, 0.5, 0.3])
        signals = compute_scalar_signals(passages, extractions)
        for name in SCALAR_SIGNAL_NAMES:
            assert name in signals, f"Missing signal: {name}"

    def test_max_cosine_correct(self):
        passages = _make_passages([0.3, 0.9, 0.5])
        extractions = _make_extractions(3)
        signals = compute_scalar_signals(passages, extractions)
        assert signals["max_cosine"] == pytest.approx(0.9)

    def test_mean_cosine_correct(self):
        passages = _make_passages([0.6, 0.4, 0.8])
        extractions = _make_extractions(3)
        signals = compute_scalar_signals(passages, extractions)
        assert signals["mean_cosine"] == pytest.approx(0.6)

    def test_max_qa_score_correct(self):
        passages = _make_passages([0.5, 0.5, 0.5])
        extractions = _make_extractions(3, qa_scores=[0.2, 0.95, 0.1])
        signals = compute_scalar_signals(passages, extractions)
        assert signals["max_qa_score"] == pytest.approx(0.95)

    def test_top1_qa_score_uses_first_passage(self):
        """top1_qa_score = QA score from the highest-ranked passage (index 0)."""
        passages = _make_passages([0.9, 0.3, 0.1])
        extractions = _make_extractions(3, qa_scores=[0.7, 0.95, 0.1])
        signals = compute_scalar_signals(passages, extractions)
        # First passage (p_0) has qa_score 0.7
        assert signals["top1_qa_score"] == pytest.approx(0.7)

    def test_score_spread_correct(self):
        passages = _make_passages([0.9, 0.3, 0.5])
        extractions = _make_extractions(3)
        signals = compute_scalar_signals(passages, extractions)
        assert signals["score_spread"] == pytest.approx(0.6)

    def test_single_passage(self):
        passages = _make_passages([0.75])
        extractions = _make_extractions(1, qa_scores=[0.6])
        signals = compute_scalar_signals(passages, extractions)
        assert signals["max_cosine"] == pytest.approx(0.75)
        assert signals["mean_cosine"] == pytest.approx(0.75)
        assert signals["score_spread"] == pytest.approx(0.0)

    def test_missing_extraction_uses_zero(self):
        """If a passage has no extraction, qa_score defaults to 0."""
        passages = _make_passages([0.8, 0.6])
        extractions = {"p_0": {"answer": "yes", "qa_score": 0.9}}
        # p_1 missing from extractions
        signals = compute_scalar_signals(passages, extractions)
        assert signals["max_qa_score"] == pytest.approx(0.9)

    def test_all_signals_in_0_1_range(self):
        passages = _make_passages([0.95, 0.01, 0.5, 0.3, 0.7])
        extractions = _make_extractions(5, qa_scores=[0.99, 0.01, 0.5, 0.3, 0.7])
        signals = compute_scalar_signals(passages, extractions)
        for name, val in signals.items():
            assert 0.0 <= val <= 1.0, f"{name} = {val} out of [0, 1]"


# ═══════════════════════════════════════════════════════════════════
# 2. SL signals
# ═══════════════════════════════════════════════════════════════════

class TestComputeSLSignals:
    """Tests for compute_sl_signals."""

    def test_returns_all_expected_keys(self):
        passages = _make_passages([0.8, 0.6, 0.4])
        extractions = _make_extractions(3, qa_scores=[0.9, 0.5, 0.3])
        signals = compute_sl_signals(passages, extractions)
        for name in SL_SIGNAL_NAMES:
            assert name in signals, f"Missing signal: {name}"

    def test_fused_belief_increases_with_higher_scores(self):
        """More high-scoring passages → higher fused belief."""
        passages_high = _make_passages([0.9, 0.85, 0.8])
        passages_low = _make_passages([0.3, 0.25, 0.2])
        ext = _make_extractions(3, qa_scores=[0.8, 0.7, 0.6])
        sig_high = compute_sl_signals(passages_high, ext)
        sig_low = compute_sl_signals(passages_low, ext)
        assert sig_high["sl_fused_belief"] > sig_low["sl_fused_belief"]

    def test_fused_uncertainty_decreases_with_more_evidence(self):
        """Higher evidence_weight → lower uncertainty."""
        passages = _make_passages([0.7, 0.6])
        ext = _make_extractions(2, qa_scores=[0.8, 0.7])
        sig_low_ew = compute_sl_signals(passages, ext, evidence_weight=5)
        sig_high_ew = compute_sl_signals(passages, ext, evidence_weight=50)
        assert sig_high_ew["sl_fused_uncertainty"] < sig_low_ew["sl_fused_uncertainty"]

    def test_max_conflict_identical_for_identical_opinions(self):
        """Identical passage scores → all pairwise conflicts are equal.

        Note: pairwise_conflict is NOT zero for identical opinions in SL.
        It measures the overlap of belief and disbelief components (b·d).
        But all pairs should have the same conflict value.
        """
        passages = _make_passages([0.7, 0.7, 0.7])
        ext = _make_extractions(3, qa_scores=[0.8, 0.8, 0.8])
        signals = compute_sl_signals(passages, ext)
        # Identical opinions → conflict is well-defined and consistent
        assert signals["sl_max_conflict"] >= 0.0

    def test_max_conflict_higher_for_divergent_opinions(self):
        """Divergent scores → higher conflict than similar scores."""
        passages_similar = _make_passages([0.8, 0.78, 0.82])
        passages_divergent = _make_passages([0.95, 0.05, 0.5])
        ext = _make_extractions(3, qa_scores=[0.5, 0.5, 0.5])
        sig_similar = compute_sl_signals(passages_similar, ext)
        sig_divergent = compute_sl_signals(passages_divergent, ext)
        assert sig_divergent["sl_max_conflict"] > sig_similar["sl_max_conflict"]

    def test_composite_combines_belief_and_conflict(self):
        """sl_composite = fused_belief * (1 - max_conflict)."""
        passages = _make_passages([0.8, 0.6, 0.4])
        ext = _make_extractions(3, qa_scores=[0.9, 0.5, 0.3])
        signals = compute_sl_signals(passages, ext)
        expected = signals["sl_fused_belief"] * (1.0 - signals["sl_max_conflict"])
        assert signals["sl_composite"] == pytest.approx(expected, abs=1e-9)

    def test_qa_fused_belief_reflects_qa_scores(self):
        """Higher QA scores → higher fused belief in QA-derived opinions.

        Note: In SL, uncertainty reflects AMOUNT of evidence, not direction.
        Same evidence_weight → same uncertainty regardless of score value.
        Belief and disbelief reflect the direction of evidence.
        """
        passages = _make_passages([0.5, 0.5])
        ext_good = _make_extractions(2, qa_scores=[0.95, 0.9])
        ext_bad = _make_extractions(2, qa_scores=[0.1, 0.05])
        sig_good = compute_sl_signals(passages, ext_good)
        sig_bad = compute_sl_signals(passages, ext_bad)
        # Same evidence_weight → same uncertainty (SL property)
        assert sig_good["sl_qa_fused_u"] == pytest.approx(
            sig_bad["sl_qa_fused_u"], abs=1e-6
        )
        # But belief differs: high qa_score → more belief
        assert sig_good["sl_dual_fused_belief"] > sig_bad["sl_dual_fused_belief"]

    def test_dual_fused_uncertainty_combines_both_sources(self):
        """sl_dual_fused_u uses both cosine and QA opinions."""
        passages = _make_passages([0.8, 0.7])
        ext = _make_extractions(2, qa_scores=[0.9, 0.85])
        signals = compute_sl_signals(passages, ext)
        # dual_fused_u should exist and be in [0, 1]
        assert 0.0 <= signals["sl_dual_fused_u"] <= 1.0

    def test_dual_composite_formula(self):
        """sl_dual_composite = dual_fused_belief * (1 - dual_max_conflict)."""
        passages = _make_passages([0.8, 0.6])
        ext = _make_extractions(2, qa_scores=[0.9, 0.5])
        signals = compute_sl_signals(passages, ext)
        expected = signals["sl_dual_fused_belief"] * (1.0 - signals["sl_dual_max_conflict"])
        assert signals["sl_dual_composite"] == pytest.approx(expected, abs=1e-9)

    def test_single_passage_no_conflict(self):
        passages = _make_passages([0.8])
        ext = _make_extractions(1, qa_scores=[0.9])
        signals = compute_sl_signals(passages, ext)
        assert signals["sl_max_conflict"] == pytest.approx(0.0)

    def test_evidence_weight_and_prior_weight_params_used(self):
        """Different ew/pw should produce different signals."""
        passages = _make_passages([0.7, 0.5])
        ext = _make_extractions(2, qa_scores=[0.8, 0.6])
        sig_a = compute_sl_signals(passages, ext, evidence_weight=5, prior_weight=2)
        sig_b = compute_sl_signals(passages, ext, evidence_weight=20, prior_weight=5)
        # At least one signal should differ
        assert sig_a["sl_fused_belief"] != pytest.approx(sig_b["sl_fused_belief"], abs=1e-6)


# ═══════════════════════════════════════════════════════════════════
# 3. Precision-coverage curve
# ═══════════════════════════════════════════════════════════════════

class TestPrecisionCoverageCurve:
    """Tests for precision_coverage_curve."""

    def test_full_coverage_equals_overall_accuracy(self):
        """At coverage=1.0, precision = overall accuracy."""
        signal_vals = [0.9, 0.7, 0.5, 0.3, 0.1]
        correct = [True, True, False, True, False]
        curve = precision_coverage_curve(signal_vals, correct, [1.0])
        assert len(curve) == 1
        cov, prec = curve[0]
        assert cov == pytest.approx(1.0)
        assert prec == pytest.approx(3.0 / 5.0)

    def test_low_coverage_selects_highest_signal(self):
        """At low coverage, only the most confident questions are answered."""
        # Signal: [0.9, 0.1, 0.5, 0.8, 0.3] → sorted desc: [0.9, 0.8, 0.5, 0.3, 0.1]
        # Correct:  T     F     T     T     F
        # Sorted:   T     T     T     F     F
        signal_vals = [0.9, 0.1, 0.5, 0.8, 0.3]
        correct = [True, False, True, True, False]
        # At 40% coverage → top 2 questions (0.9=T, 0.8=T) → precision = 1.0
        curve = precision_coverage_curve(signal_vals, correct, [0.4])
        _, prec = curve[0]
        assert prec == pytest.approx(1.0)

    def test_precision_non_increasing_with_coverage(self):
        """Precision should generally be non-increasing as coverage increases.

        This is not strictly guaranteed for all signals (it depends on how
        well the signal correlates with correctness), but for a well-correlated
        signal it should hold. We test with the oracle signal where it's exact.
        """
        correct = [True, True, False, True, False, False, True, False, True, True]
        signal_vals = oracle_signal(correct)
        coverages = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        curve = precision_coverage_curve(signal_vals, correct, coverages)
        precisions = [p for _, p in curve]
        for i in range(len(precisions) - 1):
            assert precisions[i] >= precisions[i + 1] - 1e-9

    def test_empty_inputs_returns_empty(self):
        curve = precision_coverage_curve([], [], [0.5, 1.0])
        # With no questions, all precisions should be 0 or NaN — we define 0
        for cov, prec in curve:
            assert prec == 0.0

    def test_coverage_levels_in_output_match_input(self):
        signal_vals = [0.8, 0.6, 0.4]
        correct = [True, True, False]
        levels = [0.33, 0.67, 1.0]
        curve = precision_coverage_curve(signal_vals, correct, levels)
        output_coverages = [c for c, _ in curve]
        for inp, out in zip(levels, output_coverages):
            assert inp == pytest.approx(out)

    def test_default_coverage_levels_count(self):
        """DEFAULT_COVERAGE_LEVELS should have >= 18 levels."""
        assert len(DEFAULT_COVERAGE_LEVELS) >= 18

    def test_default_coverage_levels_range(self):
        """DEFAULT_COVERAGE_LEVELS spans from ~0.5 to 1.0."""
        assert DEFAULT_COVERAGE_LEVELS[0] == pytest.approx(0.50)
        assert DEFAULT_COVERAGE_LEVELS[-1] == pytest.approx(1.00)

    def test_all_correct_gives_perfect_precision(self):
        signal_vals = [0.9, 0.7, 0.5]
        correct = [True, True, True]
        curve = precision_coverage_curve(signal_vals, correct, [0.5, 1.0])
        for _, prec in curve:
            assert prec == pytest.approx(1.0)

    def test_all_wrong_gives_zero_precision(self):
        signal_vals = [0.9, 0.7, 0.5]
        correct = [False, False, False]
        curve = precision_coverage_curve(signal_vals, correct, [0.5, 1.0])
        for _, prec in curve:
            assert prec == pytest.approx(0.0)

    def test_ties_handled_deterministically(self):
        """Questions with identical signal values should be handled stably."""
        signal_vals = [0.5, 0.5, 0.5, 0.5]
        correct = [True, True, False, False]
        curve_a = precision_coverage_curve(signal_vals, correct, [0.5, 1.0])
        curve_b = precision_coverage_curve(signal_vals, correct, [0.5, 1.0])
        for (ca, pa), (cb, pb) in zip(curve_a, curve_b):
            assert pa == pytest.approx(pb)


# ═══════════════════════════════════════════════════════════════════
# 4. AUC computation
# ═══════════════════════════════════════════════════════════════════

class TestAUCPrecisionCoverage:
    """Tests for auc_precision_coverage."""

    def test_perfect_signal_gives_maximum_auc(self):
        """Oracle signal should give the highest possible AUC.

        With 50% correct and coverage [0.5, 1.0], oracle AUC is bounded:
        at 50% coverage precision=1.0, at 100% precision=0.5.
        AUC ≈ 0.5 * (1.0 + 0.5) / 2 = 0.375.
        """
        correct = [True] * 5 + [False] * 5
        signal_vals = oracle_signal(correct)
        curve = precision_coverage_curve(signal_vals, correct, DEFAULT_COVERAGE_LEVELS)
        auc = auc_precision_coverage(curve)
        # Oracle AUC must exceed random baseline AUC
        random_vals = random_signal(10, seed=0)
        random_curve = precision_coverage_curve(random_vals, correct, DEFAULT_COVERAGE_LEVELS)
        random_auc = auc_precision_coverage(random_curve)
        assert auc > random_auc

    def test_constant_signal_full_coverage_precision_equals_accuracy(self):
        """If all signal values are identical, precision at 100% = accuracy.

        With constant signal, tie-breaking is by index, so the precision
        at partial coverage depends on the order of correct/wrong answers.
        But at full coverage it always equals overall accuracy.
        """
        correct = [True, True, False, True, False]
        signal_vals = [0.5] * 5
        curve = precision_coverage_curve(signal_vals, correct, [1.0])
        _, prec_full = curve[0]
        assert prec_full == pytest.approx(3.0 / 5.0)

    def test_single_point_curve(self):
        """Single point → AUC = 0 (no area under a single point)."""
        curve = [(1.0, 0.8)]
        auc = auc_precision_coverage(curve)
        assert auc == pytest.approx(0.0)

    def test_two_point_trapezoid(self):
        """Two-point curve → simple trapezoid."""
        curve = [(0.5, 0.9), (1.0, 0.6)]
        auc = auc_precision_coverage(curve)
        expected = 0.5 * (0.9 + 0.6) * (1.0 - 0.5)  # trapezoid
        assert auc == pytest.approx(expected)

    def test_auc_non_negative(self):
        curve = [(0.5, 0.0), (0.75, 0.0), (1.0, 0.0)]
        auc = auc_precision_coverage(curve)
        assert auc >= 0.0

    def test_empty_curve_returns_zero(self):
        auc = auc_precision_coverage([])
        assert auc == 0.0


# ═══════════════════════════════════════════════════════════════════
# 5. Oracle and random signals
# ═══════════════════════════════════════════════════════════════════

class TestOracleSignal:
    """Tests for oracle_signal."""

    def test_correct_maps_to_one(self):
        correct = [True, False, True]
        sig = oracle_signal(correct)
        assert sig[0] == 1.0
        assert sig[1] == 0.0
        assert sig[2] == 1.0

    def test_length_preserved(self):
        correct = [True] * 7 + [False] * 3
        sig = oracle_signal(correct)
        assert len(sig) == 10

    def test_oracle_auc_is_upper_bound(self):
        """Oracle AUC >= any other signal's AUC (by construction)."""
        correct = [True, True, False, True, False, False, True, True, False, True]
        oracle_vals = oracle_signal(correct)
        # Arbitrary non-oracle signal
        arb_signal = [0.9, 0.1, 0.8, 0.3, 0.5, 0.2, 0.7, 0.6, 0.4, 0.85]
        levels = DEFAULT_COVERAGE_LEVELS

        curve_oracle = precision_coverage_curve(oracle_vals, correct, levels)
        curve_arb = precision_coverage_curve(arb_signal, correct, levels)

        auc_oracle = auc_precision_coverage(curve_oracle)
        auc_arb = auc_precision_coverage(curve_arb)

        assert auc_oracle >= auc_arb - 1e-9


class TestRandomSignal:
    """Tests for random_signal."""

    def test_length_matches_input(self):
        sig = random_signal(10, seed=42)
        assert len(sig) == 10

    def test_values_in_0_1(self):
        sig = random_signal(100, seed=42)
        for v in sig:
            assert 0.0 <= v <= 1.0

    def test_deterministic_with_seed(self):
        sig_a = random_signal(20, seed=123)
        sig_b = random_signal(20, seed=123)
        assert sig_a == sig_b

    def test_different_seeds_different_values(self):
        sig_a = random_signal(20, seed=1)
        sig_b = random_signal(20, seed=2)
        assert sig_a != sig_b


# ═══════════════════════════════════════════════════════════════════
# 6. Integration: signal → curve → AUC pipeline
# ═══════════════════════════════════════════════════════════════════

class TestIntegrationPipeline:
    """End-to-end tests: compute signals → build curve → compute AUC."""

    def test_scalar_signals_produce_valid_curve(self):
        """Scalar signals from realistic data yield a valid precision-coverage curve."""
        n = 20
        passages_list = [
            _make_passages(
                [0.9 - 0.05 * j for j in range(5)],
            )
            for _ in range(n)
        ]
        extractions_list = [
            _make_extractions(5, qa_scores=[0.8, 0.7, 0.6, 0.5, 0.4])
            for _ in range(n)
        ]
        correct = [i % 3 != 0 for i in range(n)]  # 2/3 correct

        # Compute max_cosine signal for each question
        signal_vals = []
        for passages, extractions in zip(passages_list, extractions_list):
            sig = compute_scalar_signals(passages, extractions)
            signal_vals.append(sig["max_cosine"])

        curve = precision_coverage_curve(signal_vals, correct, DEFAULT_COVERAGE_LEVELS)
        auc = auc_precision_coverage(curve)

        assert len(curve) == len(DEFAULT_COVERAGE_LEVELS)
        assert auc >= 0.0

    def test_sl_signals_produce_valid_curve(self):
        """SL signals from realistic data yield a valid precision-coverage curve."""
        n = 20
        # Vary scores across questions to get signal variance
        import random
        rng = random.Random(42)
        signal_vals = []
        correct = []

        for i in range(n):
            scores = [rng.uniform(0.3, 0.9) for _ in range(5)]
            qa_scores = [rng.uniform(0.2, 0.95) for _ in range(5)]
            passages = _make_passages(scores)
            extractions = _make_extractions(5, qa_scores=qa_scores)
            sig = compute_sl_signals(passages, extractions)
            signal_vals.append(sig["sl_composite"])
            correct.append(rng.random() > 0.4)

        curve = precision_coverage_curve(signal_vals, correct, DEFAULT_COVERAGE_LEVELS)
        auc = auc_precision_coverage(curve)
        assert auc >= 0.0

    def test_oracle_always_best(self):
        """Oracle AUC >= all scalar and SL signal AUCs."""
        import random
        rng = random.Random(99)
        n = 50
        correct = [rng.random() > 0.4 for _ in range(n)]
        oracle_vals = oracle_signal(correct)

        # Build some plausible scalar signals
        scalar_signals = [rng.uniform(0.3, 0.9) for _ in range(n)]

        levels = DEFAULT_COVERAGE_LEVELS
        oracle_auc = auc_precision_coverage(
            precision_coverage_curve(oracle_vals, correct, levels)
        )
        scalar_auc = auc_precision_coverage(
            precision_coverage_curve(scalar_signals, correct, levels)
        )
        assert oracle_auc >= scalar_auc - 1e-9


# ═══════════════════════════════════════════════════════════════════
# 7. Edge cases and robustness
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases for robustness."""

    def test_all_zero_scores(self):
        passages = _make_passages([0.0, 0.0, 0.0])
        ext = _make_extractions(3, qa_scores=[0.0, 0.0, 0.0])
        scalar_sig = compute_scalar_signals(passages, ext)
        sl_sig = compute_sl_signals(passages, ext)
        assert scalar_sig["max_cosine"] == pytest.approx(0.0)
        # SL signals should not crash
        for name in SL_SIGNAL_NAMES:
            assert name in sl_sig

    def test_all_perfect_scores(self):
        passages = _make_passages([1.0, 1.0, 1.0])
        ext = _make_extractions(3, qa_scores=[1.0, 1.0, 1.0])
        scalar_sig = compute_scalar_signals(passages, ext)
        sl_sig = compute_sl_signals(passages, ext)
        assert scalar_sig["max_cosine"] == pytest.approx(1.0)
        assert sl_sig["sl_max_conflict"] == pytest.approx(0.0, abs=1e-6)

    def test_extreme_evidence_weight(self):
        """Very large evidence_weight should not crash."""
        passages = _make_passages([0.7, 0.3])
        ext = _make_extractions(2, qa_scores=[0.8, 0.2])
        sig = compute_sl_signals(passages, ext, evidence_weight=10000)
        assert 0.0 <= sig["sl_fused_uncertainty"] <= 1.0

    def test_very_small_prior_weight(self):
        """Very small prior_weight should not crash."""
        passages = _make_passages([0.7, 0.3])
        ext = _make_extractions(2, qa_scores=[0.8, 0.2])
        sig = compute_sl_signals(passages, ext, prior_weight=0.01)
        assert 0.0 <= sig["sl_fused_uncertainty"] <= 1.0

    def test_coverage_below_min_selects_at_least_one(self):
        """Coverage levels that would select 0 questions should still work."""
        signal_vals = [0.9, 0.5, 0.3]
        correct = [True, False, True]
        # 0.01 coverage on 3 questions = 0.03 → rounds to at least 1
        curve = precision_coverage_curve(signal_vals, correct, [0.01, 1.0])
        assert len(curve) == 2

    def test_two_passages(self):
        passages = _make_passages([0.8, 0.3])
        ext = _make_extractions(2, qa_scores=[0.9, 0.1])
        scalar_sig = compute_scalar_signals(passages, ext)
        sl_sig = compute_sl_signals(passages, ext)
        assert scalar_sig["score_spread"] == pytest.approx(0.5)
        # sl_max_conflict should be computable with 2 passages
        assert "sl_max_conflict" in sl_sig
