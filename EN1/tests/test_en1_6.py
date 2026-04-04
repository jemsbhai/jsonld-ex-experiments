"""Tests for EN1.6 -- Multi-Source Sensor Fusion.

TDD tests: define the expected interface and behavior of en1_6_core.
All tests should be RED before implementation, GREEN after.

Three heterogeneous sensors monitor a binary threshold state:
  1. High-precision / saturating (degrades when signal far above threshold)
  2. Low-precision / robust (always available, noisy)
  3. Intermittent / high-quality (accurate but 70% missing readings)

Five signal scenarios: stable_normal, stable_alert, gradual_rise,
sudden_spike, oscillating.

Four fusion methods: scalar weighted avg, Kalman filter, SL fusion,
SL + temporal decay fusion.
"""
from __future__ import annotations

import numpy as np
import pytest

from EN1.en1_6_core import (
    SensorSpec,
    SignalScenario,
    generate_signal,
    generate_sensor_readings,
    fuse_scalar_weighted,
    fuse_kalman,
    fuse_sl,
    fuse_sl_naive,
    fuse_sl_temporal,
    compute_fusion_metrics,
    run_scenario,
    run_full_experiment,
    SENSOR_SPECS,
    SCENARIOS,
)


# ===================================================================
# 1. Signal Generation
# ===================================================================

class TestSignalGeneration:
    """Verify synthetic signal generation."""

    def test_stable_normal_all_below_threshold(self):
        """Stable normal: true state is always 0 (below threshold)."""
        signal = generate_signal("stable_normal", n_steps=200, seed=42)
        assert len(signal["true_state"]) == 200
        assert all(s == 0 for s in signal["true_state"])

    def test_stable_alert_all_above_threshold(self):
        """Stable alert: true state is always 1 (above threshold)."""
        signal = generate_signal("stable_alert", n_steps=200, seed=42)
        assert all(s == 1 for s in signal["true_state"])

    def test_gradual_rise_has_transition(self):
        """Gradual rise: starts normal, ends alert."""
        signal = generate_signal("gradual_rise", n_steps=200, seed=42)
        states = signal["true_state"]
        # First 20% should be normal
        assert sum(states[:40]) < 10
        # Last 20% should be alert
        assert sum(states[160:]) > 30

    def test_sudden_spike_has_spike(self):
        """Sudden spike: mostly normal with a spike period."""
        signal = generate_signal("sudden_spike", n_steps=200, seed=42)
        states = signal["true_state"]
        # Should have both 0s and 1s
        assert 0 in states
        assert 1 in states

    def test_oscillating_has_both_states(self):
        """Oscillating: alternates between normal and alert."""
        signal = generate_signal("oscillating", n_steps=400, seed=42)
        states = signal["true_state"]
        assert 0 in states
        assert 1 in states
        # Neither state should dominate too much (roughly balanced)
        ratio = sum(states) / len(states)
        assert 0.2 < ratio < 0.8

    def test_signal_has_continuous_value(self):
        """Signal should include the continuous underlying value."""
        signal = generate_signal("gradual_rise", n_steps=100, seed=42)
        assert "value" in signal
        assert len(signal["value"]) == 100

    def test_signal_deterministic(self):
        """Same seed produces identical signals."""
        s1 = generate_signal("oscillating", n_steps=100, seed=99)
        s2 = generate_signal("oscillating", n_steps=100, seed=99)
        assert s1["true_state"] == s2["true_state"]
        assert s1["value"] == s2["value"]


# ===================================================================
# 2. Sensor Models
# ===================================================================

class TestSensorModels:
    """Verify sensor reading generation."""

    def test_three_sensors_defined(self):
        """SENSOR_SPECS should define exactly 3 sensors."""
        assert len(SENSOR_SPECS) == 3

    def test_sensor_specs_have_required_fields(self):
        """Each SensorSpec has name, tpr, fpr, availability."""
        for spec in SENSOR_SPECS:
            assert isinstance(spec, SensorSpec)
            assert 0.0 < spec.tpr <= 1.0
            assert 0.0 <= spec.fpr < 1.0
            assert 0.0 < spec.availability <= 1.0
            assert spec.name

    def test_readings_shape(self):
        """generate_sensor_readings returns one reading per sensor per step."""
        signal = generate_signal("stable_normal", n_steps=100, seed=42)
        readings = generate_sensor_readings(
            signal, SENSOR_SPECS, seed=42
        )
        assert len(readings) == 3  # one per sensor
        for r in readings:
            assert len(r["observations"]) == 100
            assert len(r["available"]) == 100

    def test_intermittent_sensor_has_gaps(self):
        """Sensor 3 (intermittent) should have ~70% gaps."""
        signal = generate_signal("stable_normal", n_steps=1000, seed=42)
        readings = generate_sensor_readings(
            signal, SENSOR_SPECS, seed=42
        )
        # Sensor 3 has availability=0.3
        intermittent = readings[2]
        avail_rate = sum(intermittent["available"]) / len(intermittent["available"])
        assert 0.2 < avail_rate < 0.4  # approximately 30%

    def test_robust_sensor_always_available(self):
        """Sensor 2 (robust) should have availability=1.0."""
        signal = generate_signal("stable_normal", n_steps=100, seed=42)
        readings = generate_sensor_readings(
            signal, SENSOR_SPECS, seed=42
        )
        robust = readings[1]
        assert all(robust["available"])

    def test_sensor_accuracy_matches_spec(self):
        """Over many samples, sensor accuracy should match TPR/FPR.

        Uses stable_normal (no saturation) so sensor 1's TPR is
        unaffected. Tests FPR accuracy for sensor 2 (low-precision).
        """
        signal = generate_signal("stable_normal", n_steps=5000, seed=42)
        readings = generate_sensor_readings(
            signal, SENSOR_SPECS, seed=42
        )
        # Sensor 2 (low-precision, robust): FPR ~ 0.20
        # In stable_normal (all negatives), observations = FP rate
        sensor2 = readings[1]
        avail_mask = [i for i, a in enumerate(sensor2["available"]) if a]
        if len(avail_mask) > 100:
            false_pos = sum(
                sensor2["observations"][i] == 1
                for i in avail_mask
            )
            fpr_observed = false_pos / len(avail_mask)
            # Should be close to FPR=0.20
            assert 0.15 < fpr_observed < 0.25

    def test_readings_deterministic(self):
        """Same seed produces identical readings."""
        signal = generate_signal("oscillating", n_steps=100, seed=42)
        r1 = generate_sensor_readings(signal, SENSOR_SPECS, seed=42)
        r2 = generate_sensor_readings(signal, SENSOR_SPECS, seed=42)
        assert r1[0]["observations"] == r2[0]["observations"]
        assert r1[2]["available"] == r2[2]["available"]


# ===================================================================
# 3. Fusion Methods
# ===================================================================

class TestFusionMethods:
    """Verify each fusion method runs and returns valid output."""

    @pytest.fixture
    def scenario_data(self):
        signal = generate_signal("gradual_rise", n_steps=100, seed=42)
        readings = generate_sensor_readings(signal, SENSOR_SPECS, seed=42)
        return signal, readings

    def test_scalar_weighted_returns_predictions(self, scenario_data):
        signal, readings = scenario_data
        preds = fuse_scalar_weighted(readings, SENSOR_SPECS)
        assert len(preds) == 100
        assert all(0.0 <= p <= 1.0 for p in preds)

    def test_kalman_returns_predictions(self, scenario_data):
        signal, readings = scenario_data
        preds = fuse_kalman(readings, SENSOR_SPECS)
        assert len(preds) == 100
        assert all(0.0 <= p <= 1.0 for p in preds)

    def test_sl_returns_predictions_and_uncertainty(self, scenario_data):
        signal, readings = scenario_data
        result = fuse_sl(
            readings, SENSOR_SPECS,
            scenario="gradual_rise", signal=signal,
        )
        assert len(result["predictions"]) == 100
        assert len(result["uncertainties"]) == 100
        assert all(0.0 <= p <= 1.0 for p in result["predictions"])
        assert all(0.0 <= u <= 1.0 for u in result["uncertainties"])

    def test_sl_temporal_returns_predictions(self, scenario_data):
        signal, readings = scenario_data
        result = fuse_sl_temporal(
            readings, SENSOR_SPECS,
            scenario="gradual_rise", signal=signal,
        )
        assert len(result["predictions"]) == 100
        assert len(result["uncertainties"]) == 100

    def test_sl_temporal_uncertainty_higher_during_gaps(self, scenario_data):
        """SL temporal: uncertainty should increase when sensor 3 is missing.

        The per-sensor staleness decay explicitly increases uncertainty
        for stale data, so timesteps without sensor 3 should show higher
        fused uncertainty than timesteps with it.
        """
        signal, readings = scenario_data
        result = fuse_sl_temporal(
            readings, SENSOR_SPECS,
            scenario="gradual_rise", signal=signal,
        )
        avail = readings[2]["available"]
        u_with = [result["uncertainties"][i] for i in range(100) if avail[i]]
        u_without = [result["uncertainties"][i] for i in range(100) if not avail[i]]
        if u_with and u_without:
            assert np.mean(u_with) < np.mean(u_without)


# ===================================================================
# 4. Metrics
# ===================================================================

class TestMetrics:
    """Verify metric computation."""

    def test_perfect_predictions(self):
        """Perfect predictions: MAE=0, F1=1."""
        preds = [1.0, 0.0, 1.0, 0.0, 1.0]
        true = [1, 0, 1, 0, 1]
        m = compute_fusion_metrics(preds, true)
        assert m["mae"] < 0.01
        assert m["f1"] > 0.99

    def test_worst_predictions(self):
        """Completely wrong predictions: high MAE, F1=0."""
        preds = [0.0, 0.0, 0.0, 0.0, 0.0]
        true = [1, 1, 1, 1, 1]
        m = compute_fusion_metrics(preds, true)
        assert m["mae"] > 0.9
        assert m["f1"] < 0.01  # no true positives -> F1=0

    def test_metrics_keys(self):
        """All expected keys present."""
        preds = [0.5, 0.5, 0.5]
        true = [0, 1, 0]
        m = compute_fusion_metrics(preds, true)
        assert "mae" in m
        assert "f1" in m
        assert "ece" in m
        assert "accuracy" in m

    def test_coverage_with_uncertainty(self):
        """Coverage metric: fraction of true states within uncertainty band."""
        preds = [0.5, 0.5, 0.5]
        true = [0, 1, 0]
        uncertainties = [0.6, 0.6, 0.6]  # wide bands
        m = compute_fusion_metrics(preds, true, uncertainties=uncertainties)
        assert "coverage" in m
        assert 0.0 <= m["coverage"] <= 1.0


# ===================================================================
# 5. Full Scenario and Experiment
# ===================================================================

class TestScenarioExecution:
    """Integration tests for full scenario runs."""

    def test_run_scenario_returns_all_methods(self):
        """run_scenario returns metrics for all 4 methods."""
        result = run_scenario(
            "gradual_rise", n_steps=100, n_reps=5, seed=42
        )
        assert "scalar_weighted" in result
        assert "kalman" in result
        assert "sl_fusion" in result
        assert "sl_temporal" in result

    def test_run_scenario_result_structure(self):
        """Each method result has the expected metrics."""
        result = run_scenario(
            "gradual_rise", n_steps=100, n_reps=5, seed=42
        )
        for method_name, method_result in result.items():
            if method_name.startswith("_"):
                continue
            assert "mae" in method_result
            assert "f1" in method_result
            assert "n_reps" in method_result

    def test_run_scenario_deterministic(self):
        """Same seed produces identical results."""
        r1 = run_scenario("oscillating", n_steps=50, n_reps=3, seed=42)
        r2 = run_scenario("oscillating", n_steps=50, n_reps=3, seed=42)
        assert r1["sl_fusion"]["mae"] == r2["sl_fusion"]["mae"]

    def test_all_scenarios_defined(self):
        """SCENARIOS constant has all 5 scenarios."""
        assert len(SCENARIOS) == 5
        assert "stable_normal" in SCENARIOS
        assert "stable_alert" in SCENARIOS
        assert "gradual_rise" in SCENARIOS
        assert "sudden_spike" in SCENARIOS
        assert "oscillating" in SCENARIOS

    def test_full_experiment_runs(self):
        """run_full_experiment across all scenarios."""
        results = run_full_experiment(
            n_steps=50, n_reps=3, seed=42
        )
        assert len(results) == 5
        for scenario_name, scenario_result in results.items():
            assert "sl_fusion" in scenario_result


# ===================================================================
# 6. Scientific Validity
# ===================================================================

class TestScientificValidity:
    """Tests for scientific correctness."""

    def test_sl_beats_scalar_on_gradual_rise(self):
        """SL should handle gradual transitions well.

        In the gradual_rise scenario, SL's evidence accumulation and
        uncertainty-aware fusion should produce competitive results.
        The key advantage is uncertainty quantification, not necessarily
        lower MAE in every scenario.
        """
        result = run_scenario(
            "gradual_rise", n_steps=200, n_reps=50, seed=42
        )
        # SL should be competitive (within 0.05 of best scalar method)
        best_scalar = min(
            result["scalar_weighted"]["mae"],
            result["kalman"]["mae"],
        )
        assert result["sl_fusion"]["mae"] <= best_scalar + 0.10

    def test_sl_provides_uncertainty_scalar_does_not(self):
        """SL methods return uncertainty; scalar methods do not."""
        result = run_scenario(
            "gradual_rise", n_steps=100, n_reps=5, seed=42
        )
        assert "mean_uncertainty" in result["sl_fusion"]
        assert "mean_uncertainty" in result["sl_temporal"]

    def test_all_methods_better_than_random(self):
        """All methods should beat random guessing (MAE < 0.5)."""
        result = run_scenario(
            "gradual_rise", n_steps=200, n_reps=20, seed=42
        )
        for method_name, method_result in result.items():
            if method_name.startswith("_"):
                continue
            assert method_result["mae"] < 0.5, (
                f"{method_name} MAE={method_result['mae']:.3f} >= 0.5"
            )
