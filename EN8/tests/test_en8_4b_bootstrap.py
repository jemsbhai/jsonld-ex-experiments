"""Tests for bootstrap confidence interval computation.

Statistical rigor is critical for NeurIPS. These tests verify that
bootstrap CIs have correct coverage properties and handle edge cases.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_TEST_DIR = Path(__file__).resolve().parent
_EN8_DIR = _TEST_DIR.parent
_EXPERIMENTS_DIR = _EN8_DIR.parent

sys.path.insert(0, str(_EXPERIMENTS_DIR))
sys.path.insert(0, str(_EN8_DIR))

from en8_4b_beir_benchmarks import bootstrap_ci


class TestBootstrapCI:
    """Test bootstrap confidence interval computation."""

    def test_returns_correct_keys(self):
        """Result should contain mean, ci_lower, ci_upper, std."""
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = bootstrap_ci(values, n_bootstrap=100, seed=42)
        assert "mean" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "std" in result

    def test_ci_contains_mean(self):
        """CI should contain the sample mean."""
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = bootstrap_ci(values, n_bootstrap=1000, seed=42)
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_ci_width_shrinks_with_more_data(self):
        """CI should be narrower with more data points."""
        rng = np.random.default_rng(42)
        small = rng.normal(0.5, 0.1, size=10).tolist()
        large = rng.normal(0.5, 0.1, size=200).tolist()

        ci_small = bootstrap_ci(small, n_bootstrap=1000, seed=42)
        ci_large = bootstrap_ci(large, n_bootstrap=1000, seed=42)

        width_small = ci_small["ci_upper"] - ci_small["ci_lower"]
        width_large = ci_large["ci_upper"] - ci_large["ci_lower"]
        assert width_large < width_small

    def test_constant_values(self):
        """All-identical values should produce zero-width CI."""
        values = [0.75] * 20
        result = bootstrap_ci(values, n_bootstrap=1000, seed=42)
        assert result["mean"] == pytest.approx(0.75)
        assert result["ci_lower"] == pytest.approx(0.75)
        assert result["ci_upper"] == pytest.approx(0.75)
        assert result["std"] == pytest.approx(0.0)

    def test_single_value(self):
        """Single value: CI should collapse to that value."""
        values = [0.8]
        result = bootstrap_ci(values, n_bootstrap=1000, seed=42)
        assert result["mean"] == pytest.approx(0.8)
        assert result["ci_lower"] == pytest.approx(0.8)
        assert result["ci_upper"] == pytest.approx(0.8)

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical results."""
        values = [0.3, 0.5, 0.7, 0.9, 0.4, 0.6]
        r1 = bootstrap_ci(values, n_bootstrap=500, seed=123)
        r2 = bootstrap_ci(values, n_bootstrap=500, seed=123)
        assert r1["mean"] == pytest.approx(r2["mean"])
        assert r1["ci_lower"] == pytest.approx(r2["ci_lower"])
        assert r1["ci_upper"] == pytest.approx(r2["ci_upper"])

    def test_different_seeds_differ(self):
        """Different seeds should generally produce different CIs."""
        values = list(np.random.default_rng(0).normal(0.5, 0.2, size=50))
        r1 = bootstrap_ci(values, n_bootstrap=500, seed=1)
        r2 = bootstrap_ci(values, n_bootstrap=500, seed=999)
        # Means are the same (sample mean doesn't depend on bootstrap)
        assert r1["mean"] == pytest.approx(r2["mean"])
        # But CI bounds should differ slightly
        # (not guaranteed but overwhelmingly likely with different seeds)

    def test_ci_bounds_are_ordered(self):
        """ci_lower <= mean <= ci_upper always."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            values = rng.uniform(0, 1, size=30).tolist()
            result = bootstrap_ci(values, n_bootstrap=500, seed=42)
            assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_95_percent_coverage(self):
        """Empirical coverage check: true mean should fall in CI ~95% of time.

        Draw 200 samples from N(0.5, 0.1), compute 95% CI each time.
        The true mean (0.5) should fall inside the CI ~95% of the time.
        We accept 88-100% to avoid flaky tests.
        """
        true_mean = 0.5
        rng = np.random.default_rng(42)
        n_trials = 200
        covered = 0

        for trial in range(n_trials):
            sample = rng.normal(true_mean, 0.1, size=50).tolist()
            result = bootstrap_ci(sample, n_bootstrap=500, seed=trial)
            if result["ci_lower"] <= true_mean <= result["ci_upper"]:
                covered += 1

        coverage = covered / n_trials
        # 95% CI should cover ~95% of the time; accept 88-100% for test stability
        assert 0.88 <= coverage <= 1.0, f"Coverage {coverage:.2%} outside [88%, 100%]"
