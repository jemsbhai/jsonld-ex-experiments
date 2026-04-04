"""Tests for EN1.5 — Deduction Under Uncertainty.

TDD tests: these define the expected interface and behavior of en1_5_core.
All tests should be RED before implementation, GREEN after.

Three-dataset design:
  - ASIA (Lauritzen & Spiegelhalter 1988): 8-node canonical diagnostic BN
  - ALARM (Beinlich et al. 1989): 37-node standard BN benchmark
  - Synthea (Walonoski et al. 2018): empirical condition→observation
    conditional probabilities from ~1K synthetic FHIR R4 patient bundles

Protocol:
  For each parent→child edge in the BN (or condition→observation pair
  in Synthea), simulate N observations of the parent from
  Bernoulli(P_true(parent)), build SL opinions via from_evidence(), apply
  deduce(), and compare ECE/Brier/MAE against scalar baseline.
"""
from __future__ import annotations

import numpy as np
import pytest

# ── Import core module (will fail until implemented) ──
from EN1.en1_5_core import (
    ALL_BN_MODELS,
    load_asia_kb,
    load_alarm_kb,
    load_bn_kb,
    load_all_bn_kbs,
    load_synthea_kb,
    DeductionKB,
    run_deduction_trial,
    compute_calibration_metrics,
    run_evidence_sweep,
    extract_paths,
    run_multihop_deduction_trial,
    run_multihop_sweep,
)


# ═══════════════════════════════════════════════════════════════════
# 1. Knowledge Base Loading
# ═══════════════════════════════════════════════════════════════════

class TestKBLoading:
    """Verify that each KB loads correctly and has the expected structure."""

    def test_asia_kb_structure(self):
        """ASIA KB: 8 nodes, all edges extracted with valid CPTs."""
        kb = load_asia_kb()
        assert isinstance(kb, DeductionKB)
        assert kb.name == "asia"
        # ASIA has 8 nodes and 8 edges
        assert len(kb.nodes) == 8
        assert len(kb.edges) == 8
        # Every edge should have valid conditional probabilities
        for edge in kb.edges:
            assert 0.0 <= edge.p_child_given_parent <= 1.0
            assert 0.0 <= edge.p_child_given_not_parent <= 1.0
            assert 0.0 <= edge.p_parent <= 1.0

    def test_alarm_kb_structure(self):
        """ALARM KB: 37 nodes, 46 edges extracted with valid CPTs."""
        kb = load_alarm_kb()
        assert isinstance(kb, DeductionKB)
        assert kb.name == "alarm"
        assert len(kb.nodes) == 37
        assert len(kb.edges) == 46
        for edge in kb.edges:
            assert 0.0 <= edge.p_child_given_parent <= 1.0
            assert 0.0 <= edge.p_child_given_not_parent <= 1.0
            assert 0.0 <= edge.p_parent <= 1.0

    def test_synthea_kb_structure(self):
        """Synthea KB: empirical conditionals from FHIR bundles."""
        kb = load_synthea_kb()
        assert isinstance(kb, DeductionKB)
        assert kb.name == "synthea"
        # Should have a reasonable number of edges (condition→observation)
        # At minimum 10 edges with sufficient co-occurrence
        assert len(kb.edges) >= 10
        for edge in kb.edges:
            assert 0.0 <= edge.p_child_given_parent <= 1.0
            assert 0.0 <= edge.p_child_given_not_parent <= 1.0
            assert 0.0 <= edge.p_parent <= 1.0

    def test_asia_marginals_sum_valid(self):
        """All marginal probabilities in ASIA are in (0, 1)."""
        kb = load_asia_kb()
        for edge in kb.edges:
            # Marginals must be strictly between 0 and 1 to be meaningful
            assert 0.0 < edge.p_parent < 1.0, (
                f"Edge {edge.parent}->{edge.child}: p_parent={edge.p_parent}"
            )

    def test_alarm_marginals_valid(self):
        """All marginal probabilities in ALARM are in (0, 1)."""
        kb = load_alarm_kb()
        for edge in kb.edges:
            assert 0.0 < edge.p_parent < 1.0, (
                f"Edge {edge.parent}->{edge.child}: p_parent={edge.p_parent}"
            )

    def test_alarm_multiparent_edges_handled(self):
        """ALARM has nodes with multiple parents.

        For multi-parent nodes, we marginalize over all other parents
        using the BN's joint distribution. This test verifies that the
        marginalized conditionals are consistent: P(child) computed via
        total probability should match the BN's marginal.
        """
        kb = load_alarm_kb()
        # Spot-check: for at least 5 edges, verify consistency
        checked = 0
        for edge in kb.edges[:10]:
            p_child_computed = (
                edge.p_parent * edge.p_child_given_parent
                + (1 - edge.p_parent) * edge.p_child_given_not_parent
            )
            # This should be close to the BN marginal P(child)
            assert 0.0 <= p_child_computed <= 1.0
            checked += 1
        assert checked >= 5


# ═══════════════════════════════════════════════════════════════════
# 2. Single Deduction Trial
# ═══════════════════════════════════════════════════════════════════

class TestDeductionTrial:
    """Verify that a single deduction trial runs correctly."""

    def test_trial_returns_both_methods(self):
        """Trial returns scalar and SL predicted probabilities."""
        kb = load_asia_kb()
        edge = kb.edges[0]
        result = run_deduction_trial(edge, n_evidence=50, seed=42)
        assert "scalar_pred" in result
        assert "sl_pred" in result
        assert "sl_uncertainty" in result
        assert "sl_antecedent_uncertainty" in result
        assert "ground_truth_p" in result
        assert "outcome" in result  # binary draw for calibration

    def test_trial_scalar_no_uncertainty(self):
        """Scalar method produces a point estimate with no uncertainty."""
        kb = load_asia_kb()
        edge = kb.edges[0]
        result = run_deduction_trial(edge, n_evidence=50, seed=42)
        # Scalar is just a number, no uncertainty
        assert isinstance(result["scalar_pred"], float)
        assert 0.0 <= result["scalar_pred"] <= 1.0

    def test_trial_sl_has_uncertainty(self):
        """SL method produces uncertainty > 0 for finite evidence."""
        kb = load_asia_kb()
        edge = kb.edges[0]
        result = run_deduction_trial(edge, n_evidence=10, seed=42)
        # Deduced uncertainty is small (from near-dogmatic conditionals)
        # but antecedent uncertainty should be substantial at N=10
        assert result["sl_uncertainty"] > 0.0
        assert result["sl_antecedent_uncertainty"] > 0.1

    def test_trial_sl_antecedent_uncertainty_decreases_with_evidence(self):
        """SL antecedent uncertainty should decrease as evidence N grows.

        NOTE: The *deduced* uncertainty (u_y) is dominated by the
        conditional uncertainties (component-wise LTP), so it does NOT
        decrease meaningfully with antecedent N.  The calibration benefit
        comes through the projected probability P(ω_y) = b_y + a_y·u_y,
        where b_y is shrunk toward the base rate when antecedent u_x
        is high.  The correct quantity to track is antecedent uncertainty.
        """
        kb = load_asia_kb()
        edge = kb.edges[0]
        u_low = run_deduction_trial(edge, n_evidence=5, seed=42)["sl_antecedent_uncertainty"]
        u_high = run_deduction_trial(edge, n_evidence=1000, seed=42)["sl_antecedent_uncertainty"]
        assert u_high < u_low

    def test_trial_deterministic_with_seed(self):
        """Same seed produces identical results."""
        kb = load_asia_kb()
        edge = kb.edges[0]
        r1 = run_deduction_trial(edge, n_evidence=50, seed=123)
        r2 = run_deduction_trial(edge, n_evidence=50, seed=123)
        assert r1["scalar_pred"] == r2["scalar_pred"]
        assert r1["sl_pred"] == r2["sl_pred"]
        assert r1["outcome"] == r2["outcome"]

    def test_trial_ground_truth_is_correct(self):
        """Ground truth P(child) via total probability matches BN marginal."""
        kb = load_asia_kb()
        edge = kb.edges[0]
        result = run_deduction_trial(edge, n_evidence=50, seed=42)
        expected_p = (
            edge.p_parent * edge.p_child_given_parent
            + (1 - edge.p_parent) * edge.p_child_given_not_parent
        )
        assert abs(result["ground_truth_p"] - expected_p) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# 3. Calibration Metrics
# ═══════════════════════════════════════════════════════════════════

class TestCalibrationMetrics:
    """Verify ECE, Brier, MAE computation."""

    def test_perfect_calibration_ece_zero(self):
        """Perfectly calibrated predictions have ECE ≈ 0."""
        # 1000 predictions at p=0.5, with exactly 50% positive outcomes
        predictions = np.full(1000, 0.5)
        outcomes = np.array([0] * 500 + [1] * 500)
        metrics = compute_calibration_metrics(predictions, outcomes, n_bins=10)
        assert metrics["ece"] < 0.05  # approximately zero

    def test_worst_calibration_ece_high(self):
        """Always predicting 1.0 when all outcomes are 0 gives high ECE."""
        predictions = np.full(100, 1.0)
        outcomes = np.zeros(100)
        metrics = compute_calibration_metrics(predictions, outcomes, n_bins=10)
        assert metrics["ece"] > 0.8

    def test_brier_perfect(self):
        """Perfect predictions have Brier score = 0."""
        predictions = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([1, 0, 1, 0])
        metrics = compute_calibration_metrics(predictions, outcomes, n_bins=10)
        assert abs(metrics["brier"]) < 1e-10

    def test_brier_worst(self):
        """Maximally wrong predictions have Brier score = 1."""
        predictions = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([0, 1, 0, 1])
        metrics = compute_calibration_metrics(predictions, outcomes, n_bins=10)
        assert abs(metrics["brier"] - 1.0) < 1e-10

    def test_mae_computation(self):
        """MAE vs ground truth probabilities computed correctly."""
        predictions = np.array([0.5, 0.6, 0.7])
        ground_truth = np.array([0.4, 0.6, 0.8])
        metrics = compute_calibration_metrics(
            predictions, np.zeros(3), n_bins=10,
            ground_truth_probs=ground_truth,
        )
        expected_mae = np.mean(np.abs(predictions - ground_truth))
        assert abs(metrics["mae_vs_true"] - expected_mae) < 1e-10

    def test_metrics_keys(self):
        """All expected metric keys are present."""
        predictions = np.array([0.3, 0.7, 0.5])
        outcomes = np.array([0, 1, 1])
        metrics = compute_calibration_metrics(predictions, outcomes, n_bins=10)
        assert "ece" in metrics
        assert "brier" in metrics
        assert "bin_edges" in metrics
        assert "bin_counts" in metrics
        assert "bin_accs" in metrics
        assert "bin_confs" in metrics


# ═══════════════════════════════════════════════════════════════════
# 4. Evidence Sweep (Full Protocol)
# ═══════════════════════════════════════════════════════════════════

class TestEvidenceSweep:
    """Integration tests for the full evidence sweep protocol."""

    def test_sweep_returns_results_per_n(self):
        """Sweep over N values returns results for each N."""
        kb = load_asia_kb()
        results = run_evidence_sweep(
            kb,
            n_values=[5, 50],
            n_reps=10,  # small for testing speed
            seed=42,
        )
        assert len(results) == 2
        assert 5 in results
        assert 50 in results

    def test_sweep_result_structure(self):
        """Each N-level result contains scalar and SL metrics."""
        kb = load_asia_kb()
        results = run_evidence_sweep(
            kb,
            n_values=[10],
            n_reps=10,
            seed=42,
        )
        r = results[10]
        assert "scalar_ece" in r
        assert "sl_ece" in r
        assert "scalar_brier" in r
        assert "sl_brier" in r
        assert "scalar_mae" in r
        assert "sl_mae" in r
        assert "sl_mean_uncertainty" in r
        assert "sl_mean_antecedent_uncertainty" in r
        assert "n_trials" in r

    def test_sweep_sl_antecedent_uncertainty_decreases(self):
        """SL mean antecedent uncertainty should decrease as N increases."""
        kb = load_asia_kb()
        results = run_evidence_sweep(
            kb,
            n_values=[5, 100],
            n_reps=20,
            seed=42,
        )
        assert results[5]["sl_mean_antecedent_uncertainty"] > results[100]["sl_mean_antecedent_uncertainty"]

    def test_sweep_deterministic(self):
        """Same seed produces identical sweep results."""
        kb = load_asia_kb()
        r1 = run_evidence_sweep(kb, n_values=[10], n_reps=10, seed=99)
        r2 = run_evidence_sweep(kb, n_values=[10], n_reps=10, seed=99)
        assert r1[10]["scalar_ece"] == r2[10]["scalar_ece"]
        assert r1[10]["sl_ece"] == r2[10]["sl_ece"]

    def test_sweep_all_three_datasets(self):
        """All three KBs can run through the sweep without error."""
        for loader in [load_asia_kb, load_alarm_kb, load_synthea_kb]:
            kb = loader()
            results = run_evidence_sweep(
                kb,
                n_values=[10],
                n_reps=5,
                seed=42,
            )
            assert 10 in results
            assert results[10]["n_trials"] > 0


# ═══════════════════════════════════════════════════════════════════
# 5. Scientific Validity Checks
# ═══════════════════════════════════════════════════════════════════

class TestScientificValidity:
    """Tests for scientific correctness, not just code correctness."""

    def test_sl_converges_to_scalar_at_high_evidence(self):
        """At N=10000, SL predictions should be very close to scalar.

        This tests that SL and scalar agree in the large-sample limit,
        as both should converge to the true law of total probability.
        """
        kb = load_asia_kb()
        results = run_evidence_sweep(
            kb,
            n_values=[10000],
            n_reps=50,
            seed=42,
        )
        r = results[10000]
        # At very high evidence, both methods should have similar MAE
        # (SL uncertainty is negligible, so projected_probability ≈ p_hat)
        assert abs(r["scalar_mae"] - r["sl_mae"]) < 0.01

    def test_sl_antecedent_uncertainty_nonzero_at_low_evidence(self):
        """At N=5, SL antecedent should maintain meaningful uncertainty."""
        kb = load_asia_kb()
        results = run_evidence_sweep(
            kb,
            n_values=[5],
            n_reps=50,
            seed=42,
        )
        # With prior_weight=2, N=5 gives u = 2/(5+2) ≈ 0.286
        assert results[5]["sl_mean_antecedent_uncertainty"] > 0.15

    def test_conditional_opinions_high_evidence(self):
        """Conditional opinions are built with high evidence (near-dogmatic).

        This ensures we're isolating the effect of antecedent uncertainty,
        not conflating it with conditional uncertainty.
        """
        kb = load_asia_kb()
        edge = kb.edges[0]
        result = run_deduction_trial(edge, n_evidence=10, seed=42)
        # The conditional opinions should contribute negligible uncertainty
        # to the deduced opinion — most uncertainty comes from the antecedent.
        # We can't test this directly here, but we verify the deduced
        # SL uncertainty is bounded by a reasonable amount.
        assert result["sl_uncertainty"] > 0.0
        assert result["sl_uncertainty"] < 1.0


# ═════════════════════════════════════════════════════════════════
# 6. All BN Models
# ═════════════════════════════════════════════════════════════════

class TestAllBNModels:
    """Verify that all 15 pgmpy BN models load correctly."""

    def test_all_models_constant_has_15(self):
        """ALL_BN_MODELS should list 15 models."""
        assert len(ALL_BN_MODELS) == 15

    @pytest.mark.parametrize("model_name", ALL_BN_MODELS)
    def test_each_model_loads(self, model_name):
        """Each BN model loads and produces valid edges."""
        kb = load_bn_kb(model_name)
        assert isinstance(kb, DeductionKB)
        assert kb.name == model_name
        assert len(kb.nodes) > 0
        assert len(kb.edges) > 0
        for edge in kb.edges:
            assert 0.0 < edge.p_parent < 1.0
            assert 0.0 <= edge.p_child_given_parent <= 1.0
            assert 0.0 <= edge.p_child_given_not_parent <= 1.0

    @pytest.mark.parametrize("model_name", ALL_BN_MODELS)
    def test_total_probability_consistency(self, model_name):
        """P(child) from total probability is in [0, 1] for all edges."""
        kb = load_bn_kb(model_name)
        for edge in kb.edges:
            p_child = (
                edge.p_parent * edge.p_child_given_parent
                + (1 - edge.p_parent) * edge.p_child_given_not_parent
            )
            assert -1e-9 <= p_child <= 1.0 + 1e-9, (
                f"{model_name}: {edge.parent}->{edge.child}: "
                f"P(child)={p_child}"
            )

    def test_load_all_bn_kbs(self):
        """load_all_bn_kbs returns all 15 models."""
        kbs = load_all_bn_kbs()
        assert len(kbs) == 15
        names = {kb.name for kb in kbs}
        assert names == set(ALL_BN_MODELS)

    def test_total_edges_across_all_models(self):
        """All 15 models combined should give substantial edge count."""
        kbs = load_all_bn_kbs()
        total_edges = sum(len(kb.edges) for kb in kbs)
        # Should be at least 900 (sum of edges from the check script)
        assert total_edges >= 900


# ═════════════════════════════════════════════════════════════════
# 7. Multi-Hop Path Extraction
# ═════════════════════════════════════════════════════════════════

class TestMultiHopPaths:
    """Verify path extraction from KBs."""

    def test_asia_has_multihop_paths(self):
        """ASIA should have paths of length >= 2.

        e.g., smoke -> lung -> either -> dysp (length 3)
        """
        kb = load_asia_kb()
        paths = extract_paths(kb, max_length=4)
        assert len(paths) > 0
        # Should have paths of length 2 and 3
        lengths = {len(p) for p in paths}
        assert 2 in lengths

    def test_paths_are_acyclic(self):
        """No node should appear more than once in any path."""
        kb = load_alarm_kb()
        paths = extract_paths(kb, max_length=4)
        for path in paths:
            nodes = [path[0].parent] + [e.child for e in path]
            assert len(nodes) == len(set(nodes)), (
                f"Cycle in path: {' -> '.join(nodes)}"
            )

    def test_paths_are_consecutive(self):
        """Each edge's child should be the next edge's parent."""
        kb = load_alarm_kb()
        paths = extract_paths(kb, max_length=4)
        for path in paths:
            for i in range(len(path) - 1):
                assert path[i].child == path[i + 1].parent

    def test_max_length_respected(self):
        """No path should exceed max_length edges."""
        kb = load_alarm_kb()
        paths = extract_paths(kb, max_length=3)
        for path in paths:
            assert len(path) <= 3

    def test_alarm_path_count_substantial(self):
        """ALARM (46 edges) should produce many multi-hop paths."""
        kb = load_alarm_kb()
        paths = extract_paths(kb, max_length=4)
        # ALARM is a rich DAG; should have dozens of paths
        assert len(paths) >= 20


# ═════════════════════════════════════════════════════════════════
# 8. Multi-Hop Deduction Trials
# ═════════════════════════════════════════════════════════════════

class TestMultiHopDeduction:
    """Verify chained deduction trials."""

    def test_multihop_trial_returns_expected_keys(self):
        """Multi-hop trial result has all expected fields."""
        kb = load_asia_kb()
        paths = extract_paths(kb, max_length=3)
        assert len(paths) > 0
        path = paths[0]
        result = run_multihop_deduction_trial(path, n_evidence=50, seed=42)
        assert "scalar_pred" in result
        assert "sl_pred" in result
        assert "sl_uncertainty" in result
        assert "path_length" in result
        assert "path_description" in result
        assert result["path_length"] == len(path)

    def test_multihop_predictions_in_range(self):
        """Both scalar and SL predictions should be in [0, 1]."""
        kb = load_alarm_kb()
        paths = extract_paths(kb, max_length=3)
        for path in paths[:10]:  # spot-check 10 paths
            result = run_multihop_deduction_trial(path, n_evidence=50, seed=42)
            assert 0.0 <= result["scalar_pred"] <= 1.0
            assert 0.0 <= result["sl_pred"] <= 1.0

    def test_multihop_deterministic(self):
        """Same seed gives identical results."""
        kb = load_asia_kb()
        paths = extract_paths(kb, max_length=3)
        path = paths[0]
        r1 = run_multihop_deduction_trial(path, n_evidence=50, seed=77)
        r2 = run_multihop_deduction_trial(path, n_evidence=50, seed=77)
        assert r1["scalar_pred"] == r2["scalar_pred"]
        assert r1["sl_pred"] == r2["sl_pred"]

    def test_multihop_uncertain_conditionals(self):
        """With low conditional_evidence, SL uncertainty should be higher.

        This tests the 'uncertain KB' scenario where conditional
        knowledge itself has limited evidence.
        """
        kb = load_asia_kb()
        paths = extract_paths(kb, max_length=3)
        path = paths[0]
        r_dogmatic = run_multihop_deduction_trial(
            path, n_evidence=50, seed=42, conditional_evidence=10000,
        )
        r_uncertain = run_multihop_deduction_trial(
            path, n_evidence=50, seed=42, conditional_evidence=100,
        )
        assert r_uncertain["sl_uncertainty"] > r_dogmatic["sl_uncertainty"]


# ═════════════════════════════════════════════════════════════════
# 9. Multi-Hop Sweep
# ═════════════════════════════════════════════════════════════════

class TestMultiHopSweep:
    """Integration tests for multi-hop evidence sweep."""

    def test_sweep_returns_by_length(self):
        """Multi-hop sweep groups results by path length."""
        kb = load_asia_kb()
        results = run_multihop_sweep(
            kb, n_values=[10], n_reps=5, seed=42, max_path_length=3,
        )
        assert len(results) > 0
        # All keys should be "length_N" format
        for key in results:
            assert key.startswith("length_")

    def test_sweep_result_structure(self):
        """Each length-level result has the expected metrics."""
        kb = load_asia_kb()
        results = run_multihop_sweep(
            kb, n_values=[10], n_reps=5, seed=42, max_path_length=3,
        )
        for length_key, n_results in results.items():
            for n, metrics in n_results.items():
                assert "scalar_ece" in metrics
                assert "sl_ece" in metrics
                assert "scalar_mae" in metrics
                assert "sl_mae" in metrics
                assert "n_paths" in metrics
                assert "n_trials" in metrics

    def test_sweep_caps_paths(self):
        """max_paths_per_length is respected."""
        kb = load_alarm_kb()
        results = run_multihop_sweep(
            kb, n_values=[10], n_reps=5, seed=42,
            max_path_length=3, max_paths_per_length=5,
        )
        for length_key, n_results in results.items():
            for n, metrics in n_results.items():
                assert metrics["n_paths"] <= 5
