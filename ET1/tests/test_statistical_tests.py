"""Tests for the statistical tests module.

TDD: Written BEFORE statistical_tests.py implementation.
Tests use analytically known distributions and hand-computed values
to verify correctness.
"""

import pytest
import math


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    """Bootstrap CIs for metric differences between conditions."""

    def test_ci_contains_true_difference(self):
        """When two samples clearly differ, CI should not contain 0."""
        from src.statistical_tests import bootstrap_ci_difference

        # Group A: high values, Group B: low values
        a = [0.9, 0.85, 0.88, 0.92, 0.87] * 20
        b = [0.3, 0.35, 0.32, 0.28, 0.31] * 20

        lo, hi, mean_diff = bootstrap_ci_difference(
            a, b, metric_fn=lambda x: sum(x) / len(x),
            n_resamples=5000, ci=0.95, seed=42,
        )
        # A is clearly higher than B
        assert lo > 0, f"CI lower bound {lo} should be > 0"
        assert hi > lo
        assert mean_diff > 0.4

    def test_ci_contains_zero_for_identical(self):
        """When samples are identical, CI should contain 0."""
        from src.statistical_tests import bootstrap_ci_difference

        a = [0.5, 0.6, 0.55, 0.52, 0.58] * 20
        b = [0.5, 0.6, 0.55, 0.52, 0.58] * 20

        lo, hi, mean_diff = bootstrap_ci_difference(
            a, b, metric_fn=lambda x: sum(x) / len(x),
            n_resamples=5000, ci=0.95, seed=42,
        )
        assert lo <= 0 <= hi, f"CI [{lo}, {hi}] should contain 0"

    def test_ci_width_decreases_with_samples(self):
        """More data → narrower CI."""
        from src.statistical_tests import bootstrap_ci_difference

        small_a = [0.7, 0.8, 0.75]
        small_b = [0.3, 0.4, 0.35]
        big_a = small_a * 30
        big_b = small_b * 30

        metric = lambda x: sum(x) / len(x)
        lo_s, hi_s, _ = bootstrap_ci_difference(small_a, small_b, metric, 5000, 0.95, 42)
        lo_b, hi_b, _ = bootstrap_ci_difference(big_a, big_b, metric, 5000, 0.95, 42)

        width_small = hi_s - lo_s
        width_big = hi_b - lo_b
        assert width_big < width_small

    def test_deterministic_with_seed(self):
        from src.statistical_tests import bootstrap_ci_difference

        a = [0.7, 0.8, 0.6, 0.75]
        b = [0.3, 0.4, 0.5, 0.35]
        metric = lambda x: sum(x) / len(x)

        r1 = bootstrap_ci_difference(a, b, metric, 1000, 0.95, seed=42)
        r2 = bootstrap_ci_difference(a, b, metric, 1000, 0.95, seed=42)
        assert r1 == r2

    def test_99_ci_wider_than_95(self):
        from src.statistical_tests import bootstrap_ci_difference

        a = [0.7, 0.8, 0.6, 0.75, 0.72] * 10
        b = [0.3, 0.4, 0.5, 0.35, 0.38] * 10
        metric = lambda x: sum(x) / len(x)

        lo95, hi95, _ = bootstrap_ci_difference(a, b, metric, 5000, 0.95, 42)
        lo99, hi99, _ = bootstrap_ci_difference(a, b, metric, 5000, 0.99, 42)

        assert (hi99 - lo99) >= (hi95 - lo95) - 1e-9


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

class TestMcNemar:
    """McNemar's test for paired binary outcomes (hallucination rates)."""

    def test_significant_difference(self):
        """When models clearly differ in errors, p should be small."""
        from src.statistical_tests import mcnemar_test

        # Model A correct, Model B wrong on 30 items; opposite on 5
        a_correct = [1] * 30 + [0] * 5 + [1] * 65
        b_correct = [0] * 30 + [1] * 5 + [1] * 65

        stat, p_value = mcnemar_test(a_correct, b_correct)
        assert p_value < 0.05

    def test_no_difference(self):
        """Identical models → p should be large."""
        from src.statistical_tests import mcnemar_test

        same = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        stat, p_value = mcnemar_test(same, same)
        assert p_value >= 0.99 or math.isnan(p_value)  # no discordant pairs

    def test_symmetric_inputs(self):
        """McNemar should be symmetric in the models being compared."""
        from src.statistical_tests import mcnemar_test

        a = [1, 0, 1, 0, 1, 1, 0, 0]
        b = [0, 1, 1, 0, 0, 1, 1, 0]

        _, p_ab = mcnemar_test(a, b)
        _, p_ba = mcnemar_test(b, a)
        assert abs(p_ab - p_ba) < 1e-9

    def test_returns_statistic_and_pvalue(self):
        from src.statistical_tests import mcnemar_test

        a = [1, 0, 1, 0]
        b = [0, 1, 1, 0]
        stat, p = mcnemar_test(a, b)
        assert isinstance(stat, float)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Holm-Šídák correction
# ---------------------------------------------------------------------------

class TestHolmSidak:
    """Multiple comparison correction per protocol §9.2."""

    def test_single_pvalue_unchanged(self):
        from src.statistical_tests import holm_sidak_correction

        adjusted = holm_sidak_correction([0.03])
        assert len(adjusted) == 1
        assert abs(adjusted[0] - 0.03) < 1e-6

    def test_corrected_pvalues_geq_raw(self):
        """Adjusted p-values must be >= raw p-values."""
        from src.statistical_tests import holm_sidak_correction

        raw = [0.01, 0.03, 0.05, 0.10, 0.20]
        adjusted = holm_sidak_correction(raw)
        for r, a in zip(raw, adjusted):
            assert a >= r - 1e-9, f"Adjusted {a} < raw {r}"

    def test_corrected_order_preserved(self):
        """Adjusted p-values should maintain monotonicity."""
        from src.statistical_tests import holm_sidak_correction

        raw = [0.001, 0.01, 0.05, 0.10]
        adjusted = holm_sidak_correction(raw)
        for i in range(len(adjusted) - 1):
            assert adjusted[i] <= adjusted[i + 1] + 1e-9

    def test_capped_at_one(self):
        from src.statistical_tests import holm_sidak_correction

        raw = [0.80, 0.90, 0.95]
        adjusted = holm_sidak_correction(raw)
        for a in adjusted:
            assert a <= 1.0

    def test_known_correction(self):
        """Hand-verified: 3 p-values through Holm-Šídák."""
        from src.statistical_tests import holm_sidak_correction

        raw = [0.01, 0.04, 0.05]
        adjusted = holm_sidak_correction(raw)

        # Sorted raw: [0.01, 0.04, 0.05] (already sorted)
        # Step 1: k=3 comparisons remaining. p_adj = 1-(1-0.01)^3 ≈ 0.0297
        # Step 2: k=2 remaining. p_adj = 1-(1-0.04)^2 ≈ 0.0784
        # Step 3: k=1 remaining. p_adj = 0.05
        # Enforce monotonicity: [0.0297, 0.0784, 0.0784]
        # (Step 3 adjusted < Step 2 adjusted, so take max)
        assert adjusted[0] < 0.04  # Should be ~0.03
        assert adjusted[0] < adjusted[1]


# ---------------------------------------------------------------------------
# Cohen's d and h
# ---------------------------------------------------------------------------

class TestEffectSizes:
    """Effect sizes per protocol §9.3."""

    def test_cohens_d_zero_for_identical(self):
        from src.statistical_tests import cohens_d

        a = [0.5, 0.6, 0.55, 0.52]
        d = cohens_d(a, a)
        assert abs(d) < 1e-9

    def test_cohens_d_large_effect(self):
        from src.statistical_tests import cohens_d

        a = [10.0, 11.0, 10.5, 10.8]
        b = [5.0, 5.5, 5.2, 5.3]
        d = cohens_d(a, b)
        assert d > 0.8  # Large effect by convention

    def test_cohens_d_sign(self):
        """Positive d means group A has higher mean."""
        from src.statistical_tests import cohens_d

        a = [10.0, 11.0]
        b = [5.0, 6.0]
        d = cohens_d(a, b)
        assert d > 0

    def test_cohens_h_zero_for_equal(self):
        from src.statistical_tests import cohens_h

        h = cohens_h(0.5, 0.5)
        assert abs(h) < 1e-9

    def test_cohens_h_large_for_extreme(self):
        from src.statistical_tests import cohens_h

        h = cohens_h(0.95, 0.05)
        assert abs(h) > 0.8

    def test_cohens_h_bounds(self):
        """h should be in [-π, π] by definition."""
        from src.statistical_tests import cohens_h

        h = cohens_h(0.01, 0.99)
        assert -math.pi <= h <= math.pi
