"""Tests for EN3.4 Phase A0 — Calibration Analysis.

RED phase: these tests define the contract for en3_4_calibration.py.
All should FAIL before implementation.

Tests cover:
  - ECE computation from (score, is_correct) pairs
  - Reliability diagram bin data
  - Temperature scaling fit
  - Per-model uncertainty derivation
  - Integration with Opinion.from_confidence()
"""
from __future__ import annotations

import math
import pytest
import numpy as np

# -- Path setup (same pattern as other EN3 tests) --
import sys
from pathlib import Path

_en3_dir = Path(__file__).resolve().parent.parent
_experiments_root = _en3_dir.parent
for p in [str(_en3_dir), str(_experiments_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# =====================================================================
# Import the module under test (will fail until implemented)
# =====================================================================

from EN3.en3_4_calibration import (
    compute_ece,
    reliability_diagram_bins,
    fit_temperature,
    derive_model_uncertainty,
    CalibrationReport,
)


# =====================================================================
# 1. ECE Computation
# =====================================================================


class TestComputeECE:
    """Tests for Expected Calibration Error calculation."""

    def test_perfect_calibration_has_zero_ece(self):
        """A perfectly calibrated model has ECE = 0."""
        # 100 predictions: score = 0.9, accuracy = 0.9
        # All in the [0.85, 0.95) bin → accuracy matches confidence
        scores = [0.9] * 100
        correct = [True] * 90 + [False] * 10
        ece = compute_ece(scores, correct, n_bins=10)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_maximally_overconfident_model(self):
        """Model predicts 1.0 confidence but is always wrong → ECE = 1.0."""
        scores = [0.95] * 100
        correct = [False] * 100
        ece = compute_ece(scores, correct, n_bins=10)
        assert ece == pytest.approx(0.95, abs=0.05)

    def test_maximally_underconfident_model(self):
        """Model predicts 0.05 confidence but is always right → high ECE."""
        scores = [0.05] * 100
        correct = [True] * 100
        ece = compute_ece(scores, correct, n_bins=10)
        assert ece == pytest.approx(0.95, abs=0.05)

    def test_known_ece_value(self):
        """Hand-computed ECE for a simple case.

        Two bins populated:
          Bin [0.2, 0.3): 50 predictions, mean_conf=0.25, accuracy=0.50
            contribution = (50/100) * |0.50 - 0.25| = 0.125
          Bin [0.7, 0.8): 50 predictions, mean_conf=0.75, accuracy=0.80
            contribution = (50/100) * |0.80 - 0.75| = 0.025
          ECE = 0.125 + 0.025 = 0.150
        """
        scores = [0.25] * 50 + [0.75] * 50
        correct = [True] * 25 + [False] * 25 + [True] * 40 + [False] * 10
        ece = compute_ece(scores, correct, n_bins=10)
        assert ece == pytest.approx(0.150, abs=0.01)

    def test_empty_input_raises(self):
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError):
            compute_ece([], [], n_bins=10)

    def test_mismatched_lengths_raises(self):
        """Mismatched scores/correct lengths should raise ValueError."""
        with pytest.raises(ValueError):
            compute_ece([0.5, 0.6], [True], n_bins=10)

    def test_ece_bounded_zero_one(self):
        """ECE must always be in [0, 1]."""
        rng = np.random.RandomState(42)
        scores = rng.uniform(0, 1, 500).tolist()
        correct = (rng.uniform(0, 1, 500) < 0.5).tolist()
        ece = compute_ece(scores, correct, n_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_n_bins_parameter(self):
        """Different bin counts should produce (potentially) different ECE."""
        rng = np.random.RandomState(123)
        scores = rng.uniform(0, 1, 200).tolist()
        correct = (rng.uniform(0, 1, 200) < np.array(scores)).tolist()
        ece_10 = compute_ece(scores, correct, n_bins=10)
        ece_20 = compute_ece(scores, correct, n_bins=20)
        # Both valid, just potentially different granularity
        assert 0.0 <= ece_10 <= 1.0
        assert 0.0 <= ece_20 <= 1.0


# =====================================================================
# 2. Reliability Diagram Bins
# =====================================================================


class TestReliabilityDiagramBins:
    """Tests for reliability diagram data generation."""

    def test_returns_correct_number_of_bins(self):
        """Should return exactly n_bins entries."""
        scores = [0.1, 0.5, 0.9]
        correct = [True, False, True]
        bins = reliability_diagram_bins(scores, correct, n_bins=10)
        assert len(bins) == 10

    def test_bin_structure(self):
        """Each bin should have: bin_lower, bin_upper, count, mean_confidence,
        accuracy, abs_diff."""
        scores = [0.25] * 20 + [0.75] * 20
        correct = [True] * 10 + [False] * 10 + [True] * 15 + [False] * 5
        bins = reliability_diagram_bins(scores, correct, n_bins=10)

        for b in bins:
            assert "bin_lower" in b
            assert "bin_upper" in b
            assert "count" in b
            assert "mean_confidence" in b
            assert "accuracy" in b
            assert "abs_diff" in b

    def test_populated_bins_have_valid_accuracy(self):
        """Bins with count > 0 should have accuracy in [0, 1]."""
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        correct = [False, True, False, True, True]
        bins = reliability_diagram_bins(scores, correct, n_bins=10)

        for b in bins:
            if b["count"] > 0:
                assert 0.0 <= b["accuracy"] <= 1.0
                assert 0.0 <= b["mean_confidence"] <= 1.0

    def test_empty_bins_have_zero_count(self):
        """Bins with no predictions should have count=0."""
        # All predictions in [0.9, 1.0) bin
        scores = [0.95] * 10
        correct = [True] * 10
        bins = reliability_diagram_bins(scores, correct, n_bins=10)
        empty_bins = [b for b in bins if b["count"] == 0]
        assert len(empty_bins) >= 8  # at least 8 of 10 bins should be empty

    def test_total_count_matches_input(self):
        """Sum of all bin counts should equal number of predictions."""
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        correct = [False, True, False, True, True]
        bins = reliability_diagram_bins(scores, correct, n_bins=10)
        total = sum(b["count"] for b in bins)
        assert total == len(scores)


# =====================================================================
# 3. Temperature Scaling
# =====================================================================


class TestFitTemperature:
    """Tests for temperature scaling calibration."""

    def test_perfectly_calibrated_returns_t_near_one(self):
        """A perfectly calibrated model should have T ≈ 1.0."""
        rng = np.random.RandomState(42)
        # Generate calibrated predictions: accuracy ≈ confidence
        n = 1000
        scores = rng.uniform(0.1, 0.9, n)
        correct = rng.uniform(0, 1, n) < scores
        t = fit_temperature(scores.tolist(), correct.tolist())
        assert 0.5 < t < 2.0  # should be near 1.0

    def test_overconfident_model_returns_t_greater_one(self):
        """An overconfident model needs T > 1 to soften scores."""
        # Predict 0.9 but only 50% correct → overconfident
        scores = [0.9] * 100
        correct = [True] * 50 + [False] * 50
        t = fit_temperature(scores, correct)
        assert t > 1.0

    def test_temperature_is_positive(self):
        """Temperature must always be positive."""
        rng = np.random.RandomState(99)
        scores = rng.uniform(0, 1, 200).tolist()
        correct = (rng.uniform(0, 1, 200) < 0.5).tolist()
        t = fit_temperature(scores, correct)
        assert t > 0.0

    def test_calibrated_scores_have_lower_ece(self):
        """After temperature scaling, ECE should decrease (or stay same)."""
        # Overconfident model
        rng = np.random.RandomState(77)
        n = 500
        true_p = rng.uniform(0.2, 0.8, n)
        # Inflate scores: model is overconfident
        scores = np.clip(true_p + 0.2, 0.01, 0.99)
        correct = rng.uniform(0, 1, n) < true_p

        t = fit_temperature(scores.tolist(), correct.tolist())

        # Apply temperature scaling
        logits = np.log(scores / (1 - scores))
        calibrated = 1 / (1 + np.exp(-logits / t))

        ece_before = compute_ece(scores.tolist(), correct.tolist(), n_bins=10)
        ece_after = compute_ece(calibrated.tolist(), correct.tolist(), n_bins=10)
        assert ece_after <= ece_before + 0.01  # allow tiny float noise


# =====================================================================
# 4. Per-Model Uncertainty Derivation
# =====================================================================


class TestDeriveModelUncertainty:
    """Tests for ECE → SL uncertainty parameter conversion."""

    def test_low_ece_gives_low_uncertainty(self):
        """Well-calibrated model (low ECE) → low uncertainty."""
        u = derive_model_uncertainty(ece=0.02)
        assert 0.02 <= u <= 0.10

    def test_high_ece_gives_high_uncertainty(self):
        """Poorly calibrated model (high ECE) → high uncertainty."""
        u = derive_model_uncertainty(ece=0.40)
        assert u >= 0.30

    def test_floor_enforced(self):
        """Even zero ECE should produce uncertainty ≥ 0.02."""
        u = derive_model_uncertainty(ece=0.0)
        assert u >= 0.02

    def test_ceiling_enforced(self):
        """Even extreme ECE should not exceed 0.50."""
        u = derive_model_uncertainty(ece=1.0)
        assert u <= 0.50

    def test_monotonicity(self):
        """Higher ECE → higher or equal uncertainty."""
        u_low = derive_model_uncertainty(ece=0.05)
        u_mid = derive_model_uncertainty(ece=0.20)
        u_high = derive_model_uncertainty(ece=0.45)
        assert u_low <= u_mid <= u_high

    def test_integrates_with_opinion(self):
        """Derived uncertainty must work with Opinion.from_confidence()."""
        from jsonld_ex.confidence_algebra import Opinion

        u = derive_model_uncertainty(ece=0.15)
        op = Opinion.from_confidence(0.85, uncertainty=u)
        assert op.belief >= 0
        assert op.disbelief >= 0
        assert op.uncertainty >= 0
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


# =====================================================================
# 5. CalibrationReport Dataclass
# =====================================================================


class TestCalibrationReport:
    """Tests for the CalibrationReport container."""

    def test_construction(self):
        """CalibrationReport should hold all required fields."""
        report = CalibrationReport(
            model_name="test_model",
            ece=0.12,
            ece_post_tempscale=0.05,
            temperature=1.8,
            derived_uncertainty=0.14,
            n_predictions=500,
            reliability_bins=[],
        )
        assert report.model_name == "test_model"
        assert report.ece == 0.12
        assert report.temperature == 1.8
        assert report.derived_uncertainty == 0.14

    def test_to_dict(self):
        """Should serialize to a JSON-compatible dict."""
        report = CalibrationReport(
            model_name="test_model",
            ece=0.12,
            ece_post_tempscale=0.05,
            temperature=1.8,
            derived_uncertainty=0.14,
            n_predictions=500,
            reliability_bins=[{"bin_lower": 0.0, "bin_upper": 0.1, "count": 10}],
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["model_name"] == "test_model"
        assert d["ece"] == 0.12
        assert isinstance(d["reliability_bins"], list)

    def test_specialist_should_have_lower_uncertainty_than_generalist(self):
        """Sanity check: if specialist ECE < generalist ECE,
        derived uncertainties should respect that ordering."""
        u_specialist = derive_model_uncertainty(ece=0.08)
        u_generalist = derive_model_uncertainty(ece=0.22)
        assert u_specialist < u_generalist
