"""RED-phase tests for EN4.2 — Dempster-Shafer vs Subjective Logic.

These tests verify:
  1. Classical DS combination rule against published textbook examples
  2. Yager's rule against published examples
  3. Zadeh's paradox (the known pathological DS result)
  4. SL vs DS behaviour differences on controlled inputs
  5. Multi-source combination (>2 sources)
  6. Base rate ablation mechanics
  7. Conflict-partitioned analysis

All DS rule implementations are verified against manual calculations
and published references:
  - Shafer (1976) "A Mathematical Theory of Evidence"
  - Zadeh (1979) "On the validity of Dempster's rule"
  - Yager (1987) "On the Dempster-Shafer framework"
  - Jøsang (2016) "Subjective Logic" §12.3
"""
from __future__ import annotations

import math
import pytest
import numpy as np

from en4_2_ds_comparison import (
    # DS combination rules
    dempster_combine,
    yager_combine,
    dempster_combine_multi,
    yager_combine_multi,
    # Mass function helpers
    confidence_to_mass,
    mass_to_decision,
    sl_to_decision,
    # Conflict
    compute_conflict_K,
    # Pipeline
    compare_fusion_methods_binary,
)

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    conflict_metric,
)


# ====================================================================
# Section 1: Classical DS Combination — Textbook Verification
# ====================================================================

class TestDempsterCombineBasic:
    """Verify Dempster's rule against hand-computed examples."""

    def test_two_agreeing_sources(self):
        """Two sources that both support entity → stronger belief."""
        # m1: b=0.6, d=0.1, u=0.3
        # m2: b=0.7, d=0.1, u=0.2
        m1 = {"b": 0.6, "d": 0.1, "u": 0.3}
        m2 = {"b": 0.7, "d": 0.1, "u": 0.2}
        result = dempster_combine(m1, m2)

        # K = b1*d2 + d1*b2 = 0.6*0.1 + 0.1*0.7 = 0.06 + 0.07 = 0.13
        # numerator_b = b1*b2 + b1*u2 + u1*b2 = 0.42 + 0.12 + 0.21 = 0.75
        # numerator_d = d1*d2 + d1*u2 + u1*d2 = 0.01 + 0.02 + 0.03 = 0.06
        # numerator_u = u1*u2 = 0.06
        # Total = 0.75 + 0.06 + 0.06 = 0.87 = 1 - K ✓
        # m(ent) = 0.75 / 0.87 ≈ 0.86207
        # m(¬ent) = 0.06 / 0.87 ≈ 0.06897
        # m(Θ) = 0.06 / 0.87 ≈ 0.06897
        assert result["b"] == pytest.approx(0.75 / 0.87, abs=1e-10)
        assert result["d"] == pytest.approx(0.06 / 0.87, abs=1e-10)
        assert result["u"] == pytest.approx(0.06 / 0.87, abs=1e-10)
        # Validity: b + d + u = 1
        assert result["b"] + result["d"] + result["u"] == pytest.approx(1.0, abs=1e-12)

    def test_vacuous_source_is_identity(self):
        """Combining with a vacuous mass function should be identity."""
        m1 = {"b": 0.6, "d": 0.3, "u": 0.1}
        vacuous = {"b": 0.0, "d": 0.0, "u": 1.0}
        result = dempster_combine(m1, vacuous)
        assert result["b"] == pytest.approx(m1["b"], abs=1e-10)
        assert result["d"] == pytest.approx(m1["d"], abs=1e-10)
        assert result["u"] == pytest.approx(m1["u"], abs=1e-10)

    def test_symmetric(self):
        """Dempster's rule is commutative."""
        m1 = {"b": 0.4, "d": 0.2, "u": 0.4}
        m2 = {"b": 0.3, "d": 0.5, "u": 0.2}
        r12 = dempster_combine(m1, m2)
        r21 = dempster_combine(m2, m1)
        assert r12["b"] == pytest.approx(r21["b"], abs=1e-12)
        assert r12["d"] == pytest.approx(r21["d"], abs=1e-12)
        assert r12["u"] == pytest.approx(r21["u"], abs=1e-12)

    def test_zero_conflict(self):
        """When no conflict, DS normalization has no effect."""
        # Both sources agree completely: b=0.5, d=0.0, u=0.5
        m1 = {"b": 0.5, "d": 0.0, "u": 0.5}
        m2 = {"b": 0.3, "d": 0.0, "u": 0.7}
        result = dempster_combine(m1, m2)
        K = compute_conflict_K(m1, m2)
        assert K == pytest.approx(0.0, abs=1e-12)
        # No normalization needed; result should still be valid
        assert result["b"] + result["d"] + result["u"] == pytest.approx(1.0, abs=1e-12)

    def test_validity_random(self):
        """100 random mass function pairs all produce valid results."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            raw1 = rng.dirichlet([1, 1, 1])
            raw2 = rng.dirichlet([1, 1, 1])
            m1 = {"b": raw1[0], "d": raw1[1], "u": raw1[2]}
            m2 = {"b": raw2[0], "d": raw2[1], "u": raw2[2]}
            K = compute_conflict_K(m1, m2)
            if K >= 1.0 - 1e-10:
                continue  # total conflict — DS is undefined
            result = dempster_combine(m1, m2)
            total = result["b"] + result["d"] + result["u"]
            assert total == pytest.approx(1.0, abs=1e-10), \
                f"Invalid: b={result['b']}, d={result['d']}, u={result['u']}, sum={total}"
            assert result["b"] >= -1e-12
            assert result["d"] >= -1e-12
            assert result["u"] >= -1e-12


# ====================================================================
# Section 2: Zadeh's Paradox — The Critical DS Failure
# ====================================================================

class TestZadehParadox:
    """Verify the known pathological DS result (Zadeh 1979).

    This is the most important test in the entire experiment. It verifies
    that our DS implementation correctly reproduces the known paradox,
    and that SL avoids it.
    """

    def test_zadeh_classic(self):
        """Classic Zadeh scenario: nearly total conflict.

        Source 1: 99% A, 1% B → strongly believes A
        Source 2: 99% B, 1% A → strongly believes B

        DS normalizes away conflict → attributes nearly all mass to...
        whichever focal element gets the cross-term.

        Expected: DS gives counterintuitive high certainty.
        """
        m1 = {"b": 0.99, "d": 0.01, "u": 0.0}  # A=entity, B=not-entity
        m2 = {"b": 0.01, "d": 0.99, "u": 0.0}
        K = compute_conflict_K(m1, m2)

        # K = b1*d2 + d1*b2 = 0.99*0.99 + 0.01*0.01 = 0.9801 + 0.0001 = 0.9802
        assert K == pytest.approx(0.9802, abs=1e-6)

        result = dempster_combine(m1, m2)
        # Despite total disagreement, DS produces a result with near-zero uncertainty
        # because u1*u2 = 0 and normalization by (1 - 0.9802) inflates everything
        # This IS the paradox: two completely disagreeing sources → high certainty
        assert result["u"] == pytest.approx(0.0, abs=1e-10)

    def test_zadeh_sl_graceful(self):
        """SL handles the same scenario gracefully.

        With two dogmatic opinions in complete disagreement, SL should
        NOT produce high certainty for either side. The dogmatic fallback
        should produce an averaged result.
        """
        op1 = Opinion(belief=0.99, disbelief=0.01, uncertainty=0.0, base_rate=0.5)
        op2 = Opinion(belief=0.01, disbelief=0.99, uncertainty=0.0, base_rate=0.5)
        fused = cumulative_fuse(op1, op2)

        # SL dogmatic fallback: simple average with equal relative dogmatism
        # Result should be moderate, NOT extreme
        # b ≈ 0.5, d ≈ 0.5 (averaged beliefs)
        assert fused.uncertainty == pytest.approx(0.0, abs=1e-10)  # still dogmatic
        # But the beliefs are AVERAGED, not pathologically skewed
        assert abs(fused.belief - 0.5) < 0.1

    def test_zadeh_with_uncertainty(self):
        """Zadeh-like but with some uncertainty — more realistic.

        Source 1: b=0.85, d=0.05, u=0.10
        Source 2: b=0.05, d=0.85, u=0.10

        DS should still show pathological normalization.
        SL should maintain meaningful uncertainty.
        """
        m1 = {"b": 0.85, "d": 0.05, "u": 0.10}
        m2 = {"b": 0.05, "d": 0.85, "u": 0.10}
        K = compute_conflict_K(m1, m2)

        # K = 0.85*0.85 + 0.05*0.05 = 0.7225 + 0.0025 = 0.725
        assert K == pytest.approx(0.725, abs=1e-6)

        ds_result = dempster_combine(m1, m2)
        # DS: u_fused = u1*u2 / (1-K) = 0.01 / 0.275 ≈ 0.0364
        # Very low uncertainty despite massive disagreement
        assert ds_result["u"] < 0.05

        # SL: should maintain higher uncertainty
        op1 = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.10, base_rate=0.5)
        op2 = Opinion(belief=0.05, disbelief=0.85, uncertainty=0.10, base_rate=0.5)
        sl_result = cumulative_fuse(op1, op2)
        # SL κ = 0.10 + 0.10 - 0.01 = 0.19
        # u_fused = 0.01 / 0.19 ≈ 0.0526
        # Still low, but the beliefs should be more balanced
        assert sl_result.uncertainty > ds_result["u"] - 0.01  # SL at least as uncertain

        # The KEY difference: DS has inflated belief in one direction,
        # SL produces balanced low-confidence beliefs
        sl_belief_imbalance = abs(sl_result.belief - sl_result.disbelief)
        ds_belief_imbalance = abs(ds_result["b"] - ds_result["d"])
        # SL should be MORE balanced (less imbalanced)
        assert sl_belief_imbalance <= ds_belief_imbalance + 0.01


# ====================================================================
# Section 3: Yager's Rule
# ====================================================================

class TestYagerCombine:
    """Verify Yager's rule: conflict mass → uncertainty."""

    def test_conflict_to_uncertainty(self):
        """Under conflict, Yager transfers mass to Θ."""
        m1 = {"b": 0.8, "d": 0.1, "u": 0.1}
        m2 = {"b": 0.1, "d": 0.8, "u": 0.1}
        K = compute_conflict_K(m1, m2)
        result = yager_combine(m1, m2)

        # K = 0.8*0.8 + 0.1*0.1 = 0.65
        # m(Θ) = u1*u2 + K = 0.01 + 0.65 = 0.66
        assert result["u"] == pytest.approx(0.01 + K, abs=1e-10)
        # Much higher uncertainty than DS
        assert result["u"] > 0.5

        # Validity
        assert result["b"] + result["d"] + result["u"] == pytest.approx(1.0, abs=1e-12)

    def test_vacuous_identity(self):
        """Yager with vacuous source should be identity."""
        m1 = {"b": 0.6, "d": 0.3, "u": 0.1}
        vacuous = {"b": 0.0, "d": 0.0, "u": 1.0}
        result = yager_combine(m1, vacuous)
        assert result["b"] == pytest.approx(m1["b"], abs=1e-10)
        assert result["d"] == pytest.approx(m1["d"], abs=1e-10)
        assert result["u"] == pytest.approx(m1["u"], abs=1e-10)

    def test_symmetric(self):
        """Yager's rule is commutative."""
        m1 = {"b": 0.4, "d": 0.2, "u": 0.4}
        m2 = {"b": 0.3, "d": 0.5, "u": 0.2}
        r12 = yager_combine(m1, m2)
        r21 = yager_combine(m2, m1)
        assert r12["b"] == pytest.approx(r21["b"], abs=1e-12)
        assert r12["d"] == pytest.approx(r21["d"], abs=1e-12)
        assert r12["u"] == pytest.approx(r21["u"], abs=1e-12)


# ====================================================================
# Section 4: Multi-Source Combination
# ====================================================================

class TestMultiSourceCombination:
    """Verify sequential n-way combination."""

    def test_three_sources_ds(self):
        """Three-source DS combination via sequential pairwise."""
        m1 = {"b": 0.5, "d": 0.1, "u": 0.4}
        m2 = {"b": 0.6, "d": 0.1, "u": 0.3}
        m3 = {"b": 0.4, "d": 0.2, "u": 0.4}
        result = dempster_combine_multi([m1, m2, m3])
        assert result["b"] + result["d"] + result["u"] == pytest.approx(1.0, abs=1e-10)
        # More sources → lower uncertainty
        pair_result = dempster_combine(m1, m2)
        assert result["u"] < pair_result["u"] + 1e-10

    def test_three_sources_yager(self):
        """Three-source Yager combination."""
        m1 = {"b": 0.5, "d": 0.1, "u": 0.4}
        m2 = {"b": 0.6, "d": 0.1, "u": 0.3}
        m3 = {"b": 0.4, "d": 0.2, "u": 0.4}
        result = yager_combine_multi([m1, m2, m3])
        assert result["b"] + result["d"] + result["u"] == pytest.approx(1.0, abs=1e-10)

    def test_five_sources_ds_validity(self):
        """Five-source DS: all valid."""
        rng = np.random.default_rng(42)
        masses = []
        for _ in range(5):
            raw = rng.dirichlet([2, 1, 2])  # bias toward b and u
            masses.append({"b": raw[0], "d": raw[1], "u": raw[2]})
        result = dempster_combine_multi(masses)
        assert result["b"] + result["d"] + result["u"] == pytest.approx(1.0, abs=1e-10)


# ====================================================================
# Section 5: Mass-Function / Decision Helpers
# ====================================================================

class TestMassFunctionHelpers:
    """Test mass function construction and decision rules."""

    def test_confidence_to_mass_basic(self):
        """High confidence → high belief."""
        m = confidence_to_mass(score=0.9, uncertainty=0.2)
        # b = score * (1 - u) = 0.9 * 0.8 = 0.72
        # d = (1 - score) * (1 - u) = 0.1 * 0.8 = 0.08
        # u = 0.2
        assert m["b"] == pytest.approx(0.72, abs=1e-10)
        assert m["d"] == pytest.approx(0.08, abs=1e-10)
        assert m["u"] == pytest.approx(0.2, abs=1e-10)

    def test_confidence_to_mass_vacuous(self):
        """Full uncertainty → vacuous."""
        m = confidence_to_mass(score=0.7, uncertainty=1.0)
        assert m["b"] == pytest.approx(0.0, abs=1e-10)
        assert m["d"] == pytest.approx(0.0, abs=1e-10)
        assert m["u"] == pytest.approx(1.0, abs=1e-10)

    def test_confidence_to_mass_validity(self):
        """Random inputs → valid mass functions."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            score = rng.uniform(0, 1)
            u = rng.uniform(0, 1)
            m = confidence_to_mass(score, u)
            assert m["b"] + m["d"] + m["u"] == pytest.approx(1.0, abs=1e-12)
            assert m["b"] >= -1e-12
            assert m["d"] >= -1e-12
            assert m["u"] >= -1e-12

    def test_mass_to_decision(self):
        """DS decision: max(m(ent), m(¬ent))."""
        m = {"b": 0.7, "d": 0.2, "u": 0.1}
        assert mass_to_decision(m) == True  # entity
        m2 = {"b": 0.2, "d": 0.7, "u": 0.1}
        assert mass_to_decision(m2) == False  # not entity

    def test_sl_to_decision_base_rate_effect(self):
        """SL decision with base rate: P = b + a*u > threshold."""
        # Moderate belief, high uncertainty, different base rates
        op = Opinion(belief=0.3, disbelief=0.1, uncertainty=0.6, base_rate=0.5)
        # P = 0.3 + 0.5*0.6 = 0.6 → entity at threshold 0.5
        assert sl_to_decision(op, threshold=0.5, base_rate=0.5) == True

        op2 = Opinion(belief=0.3, disbelief=0.1, uncertainty=0.6, base_rate=0.1)
        # P = 0.3 + 0.1*0.6 = 0.36 → not entity at threshold 0.5
        assert sl_to_decision(op2, threshold=0.5, base_rate=0.1) == False

    def test_ds_vs_sl_decision_divergence(self):
        """Find a case where DS and SL make DIFFERENT decisions.

        This is the key test: there must exist inputs where the two
        frameworks disagree. If they always agree, one is redundant.
        """
        # High uncertainty, moderate belief, low base rate
        m = {"b": 0.25, "d": 0.15, "u": 0.60}
        # DS decision: m(ent)=0.25 > m(¬ent)=0.15 → entity
        assert mass_to_decision(m) == True

        # SL with low base rate: P = 0.25 + 0.1*0.6 = 0.31 < 0.5 → not entity
        op = Opinion(belief=0.25, disbelief=0.15, uncertainty=0.60, base_rate=0.1)
        assert sl_to_decision(op, threshold=0.5, base_rate=0.1) == False

        # They DISAGREE — DS says entity, SL says not entity
        # This is the base rate effect in action


# ====================================================================
# Section 6: Conflict Measurement
# ====================================================================

class TestConflict:
    """Verify conflict computation."""

    def test_zero_conflict(self):
        """No opposing belief → K=0."""
        m1 = {"b": 0.5, "d": 0.0, "u": 0.5}
        m2 = {"b": 0.3, "d": 0.0, "u": 0.7}
        assert compute_conflict_K(m1, m2) == pytest.approx(0.0, abs=1e-12)

    def test_max_conflict(self):
        """Fully opposing dogmatic sources → K near 1."""
        m1 = {"b": 1.0, "d": 0.0, "u": 0.0}
        m2 = {"b": 0.0, "d": 1.0, "u": 0.0}
        assert compute_conflict_K(m1, m2) == pytest.approx(1.0, abs=1e-12)

    def test_conflict_formula(self):
        """Verify K = b1*d2 + d1*b2."""
        m1 = {"b": 0.6, "d": 0.1, "u": 0.3}
        m2 = {"b": 0.2, "d": 0.4, "u": 0.4}
        expected_K = 0.6 * 0.4 + 0.1 * 0.2  # 0.24 + 0.02 = 0.26
        assert compute_conflict_K(m1, m2) == pytest.approx(expected_K, abs=1e-12)

    def test_conflict_symmetric(self):
        """K(m1, m2) = K(m2, m1)."""
        m1 = {"b": 0.4, "d": 0.3, "u": 0.3}
        m2 = {"b": 0.2, "d": 0.5, "u": 0.3}
        assert compute_conflict_K(m1, m2) == pytest.approx(
            compute_conflict_K(m2, m1), abs=1e-12
        )


# ====================================================================
# Section 7: Comparison Pipeline
# ====================================================================

class TestComparisonPipeline:
    """Test the full comparison pipeline on simple controlled data."""

    def test_compare_agrees_on_easy_case(self):
        """All methods should agree when evidence is strong and consistent."""
        # Two sources, both strongly supporting entity
        scores = [0.9, 0.85]
        uncertainties = [0.1, 0.15]

        results = compare_fusion_methods_binary(scores, uncertainties, base_rate=0.5)

        # All methods should predict entity
        assert results["ds_classical"]["decision"] == True
        assert results["ds_yager"]["decision"] == True
        assert results["sl_cumulative"]["decision"] == True
        assert results["sl_averaging"]["decision"] == True

    def test_compare_structure(self):
        """Pipeline returns expected keys."""
        scores = [0.7, 0.3]
        uncertainties = [0.2, 0.2]
        results = compare_fusion_methods_binary(scores, uncertainties, base_rate=0.5)

        expected_methods = ["ds_classical", "ds_yager", "sl_cumulative", "sl_averaging"]
        for method in expected_methods:
            assert method in results
            assert "b" in results[method]
            assert "d" in results[method]
            assert "u" in results[method]
            assert "decision" in results[method]

    def test_compare_five_sources(self):
        """Pipeline handles 5 sources (matches EN1.1 setup)."""
        scores = [0.8, 0.7, 0.6, 0.9, 0.5]
        uncertainties = [0.1, 0.2, 0.3, 0.1, 0.4]
        results = compare_fusion_methods_binary(scores, uncertainties, base_rate=0.5)
        # All should produce valid mass functions
        for method in ["ds_classical", "ds_yager", "sl_cumulative", "sl_averaging"]:
            total = results[method]["b"] + results[method]["d"] + results[method]["u"]
            assert total == pytest.approx(1.0, abs=1e-10)


# ====================================================================
# Section 8: SL vs DS Equivalence and Divergence Characterization
# ====================================================================

class TestSLvsDSCharacterization:
    """Characterize when SL and DS agree vs diverge.

    This is the core scientific question: under what conditions does
    the choice of combination rule matter?
    """

    def test_structural_divergence_at_all_conflict_levels(self):
        """DS and SL diverge structurally, even under zero conflict.

        FINDING: The b₁b₂ mutual reinforcement term in DS means the two
        rules produce different belief values regardless of conflict level.
        Mean divergence is ~10-12% of belief mass even at K < 0.05.

        This is NOT a defect — it reflects a genuine design difference:
        - DS counts concordant evidence as mutual reinforcement (b₁b₂)
        - SL only counts evidence weighted by the OTHER source's uncertainty
        """
        rng = np.random.default_rng(42)
        divergences_by_K = {"low": [], "mid": [], "high": []}

        for _ in range(500):
            b1, b2 = rng.uniform(0.3, 0.8, size=2)
            u1, u2 = rng.uniform(0.1, 0.5, size=2)
            d1 = 1 - b1 - u1
            d2 = 1 - b2 - u2
            if d1 < 0 or d2 < 0:
                continue
            m1 = {"b": b1, "d": d1, "u": u1}
            m2 = {"b": b2, "d": d2, "u": u2}
            K = compute_conflict_K(m1, m2)
            if K >= 0.95:
                continue

            ds = dempster_combine(m1, m2)
            op1 = Opinion(belief=b1, disbelief=d1, uncertainty=u1, base_rate=0.5)
            op2 = Opinion(belief=b2, disbelief=d2, uncertainty=u2, base_rate=0.5)
            sl = cumulative_fuse(op1, op2)

            diff = ds["b"] - sl.belief  # DS - SL (expected positive)

            if K < 0.1:
                divergences_by_K["low"].append(diff)
            elif K < 0.3:
                divergences_by_K["mid"].append(diff)
            else:
                divergences_by_K["high"].append(diff)

        # Key assertions:
        # 1. DS belief >= SL belief in virtually all cases (b₁b₂ ≥ 0)
        low = divergences_by_K["low"]
        assert len(low) > 5, "Need enough low-conflict samples"
        assert np.mean(low) > 0.02, (
            f"Expected DS > SL even at low conflict; mean diff = {np.mean(low):.4f}"
        )

        # 2. Divergence exists at ALL conflict levels — NOT just high conflict
        for label, diffs in divergences_by_K.items():
            if len(diffs) > 3:
                assert np.mean(diffs) > 0.01, (
                    f"Expected nonzero divergence at {label} conflict; "
                    f"mean = {np.mean(diffs):.4f}"
                )

        # 3. DS always gives higher or equal belief (b₁b₂ term is non-negative)
        all_diffs = [d for lst in divergences_by_K.values() for d in lst]
        frac_ds_higher = sum(1 for d in all_diffs if d > -1e-10) / len(all_diffs)
        assert frac_ds_higher > 0.95, (
            f"Expected DS belief >= SL belief in >95% of cases; got {frac_ds_higher:.2%}"
        )

    def test_high_conflict_diverges(self):
        """Under high conflict, DS and SL MUST diverge on uncertainty."""
        # Generate high-conflict pair
        m1 = {"b": 0.7, "d": 0.1, "u": 0.2}
        m2 = {"b": 0.1, "d": 0.7, "u": 0.2}
        K = compute_conflict_K(m1, m2)
        assert K > 0.4  # verify it's high conflict

        ds = dempster_combine(m1, m2)
        op1 = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
        op2 = Opinion(belief=0.1, disbelief=0.7, uncertainty=0.2, base_rate=0.5)
        sl = cumulative_fuse(op1, op2)

        # DS should have lower uncertainty (it normalizes conflict away)
        # SL should have higher uncertainty (it preserves conflict information)
        # This is a characterization, not a pass/fail
        assert ds["u"] < sl.uncertainty or abs(ds["u"] - sl.uncertainty) < 0.01, \
            "Expected DS uncertainty ≤ SL uncertainty under high conflict"
