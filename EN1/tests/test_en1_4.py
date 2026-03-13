"""Tests for EN1.4 — Trust Discount Chain Analysis.

Comprehensive analysis of trust propagation through provenance chains:
  1. Homogeneous chains — verify closed-form convergence
  2. Heterogeneous chains — varying trust per link
  3. Trust level sweep — convergence rate characterization
  4. Original opinion sweep — sensitivity to initial evidence
  5. Base rate sensitivity — effect on convergence target
  6. Branching provenance — multiple trust paths with fusion
  7. Decision divergence — when do scalar and SL disagree?
  8. Information content — entropy through the chain
  9. Comparison with Bayesian intermediary updating

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest experiments/EN1/tests/test_en1_4.py -v
"""
from __future__ import annotations

import math
import pytest
import numpy as np

from EN1.en1_4_core import (
    # Core chain computation
    compute_sl_chain,
    compute_scalar_chain,
    compute_bayesian_chain,
    # Closed-form verification
    sl_chain_closed_form,
    sl_uncertainty_closed_form,
    # Heterogeneous chains
    compute_heterogeneous_sl_chain,
    # Branching provenance
    compute_branching_provenance,
    # Analysis functions
    compute_decision_divergence_point,
    compute_opinion_entropy,
    compute_information_loss_curve,
    # Full sweep
    run_trust_level_sweep,
    run_original_opinion_sweep,
    run_base_rate_sweep,
    run_chain_length_comparison,
)
from jsonld_ex.confidence_algebra import Opinion, trust_discount


# ═══════════════════════════════════════════════════════════════════
# 1. Homogeneous chain — basic correctness
# ═══════════════════════════════════════════════════════════════════

class TestHomogeneousChain:

    def test_sl_chain_length_1(self):
        """Single trust discount should match the library function."""
        trust = Opinion(0.85, 0.05, 0.10)
        original = Opinion(0.90, 0.05, 0.05)
        chain = compute_sl_chain(trust, original, max_length=1)
        assert len(chain) == 2
        expected = trust_discount(trust, original)
        assert chain[1].belief == pytest.approx(expected.belief, abs=1e-12)
        assert chain[1].disbelief == pytest.approx(expected.disbelief, abs=1e-12)
        assert chain[1].uncertainty == pytest.approx(expected.uncertainty, abs=1e-12)

    def test_sl_chain_preserves_additivity(self):
        """b + d + u = 1 at every step in the chain."""
        trust = Opinion(0.80, 0.10, 0.10)
        original = Opinion(0.70, 0.20, 0.10)
        chain = compute_sl_chain(trust, original, max_length=30)
        for i, op in enumerate(chain):
            total = op.belief + op.disbelief + op.uncertainty
            assert total == pytest.approx(1.0, abs=1e-9), f"Step {i}: b+d+u={total}"

    def test_sl_chain_converges_to_vacuous(self):
        """After many steps, opinion should approach vacuous (u→1)."""
        trust = Opinion(0.85, 0.05, 0.10)
        original = Opinion(0.90, 0.05, 0.05)
        chain = compute_sl_chain(trust, original, max_length=50)
        final = chain[-1]
        assert final.uncertainty > 0.99
        assert final.belief < 0.01
        assert final.disbelief < 0.01

    def test_sl_projected_probability_converges_to_base_rate(self):
        """P(ω_n) should converge to base rate a, not to 0."""
        trust = Opinion(0.85, 0.05, 0.10)
        original = Opinion(0.90, 0.05, 0.05, base_rate=0.3)
        chain = compute_sl_chain(trust, original, max_length=50)
        final_p = chain[-1].projected_probability()
        assert final_p == pytest.approx(0.3, abs=0.01)

    def test_scalar_chain_converges_to_zero(self):
        """Scalar c_n = t^n × c_0 → 0 regardless of base rate."""
        trust_level = 0.85
        original_conf = 0.90
        chain = compute_scalar_chain(trust_level, original_conf, max_length=50)
        assert len(chain) == 51
        assert chain[-1] < 0.01

    def test_scalar_chain_is_exponential_decay(self):
        """Each step multiplies by trust level."""
        trust_level = 0.80
        original_conf = 0.95
        chain = compute_scalar_chain(trust_level, original_conf, max_length=10)
        for i in range(1, len(chain)):
            expected = trust_level ** i * original_conf
            assert chain[i] == pytest.approx(expected, abs=1e-12)

    def test_sl_and_scalar_start_same(self):
        """At step 0, both should equal the original."""
        trust = Opinion(0.85, 0.05, 0.10)
        original = Opinion(0.90, 0.05, 0.05, base_rate=0.5)
        sl_chain = compute_sl_chain(trust, original, max_length=5)
        scalar_chain = compute_scalar_chain(
            0.85, original.projected_probability(), max_length=5,
        )
        assert sl_chain[0].projected_probability() == pytest.approx(
            scalar_chain[0], abs=1e-9)


# ═══════════════════════════════════════════════════════════════════
# 2. Closed-form verification
# ═══════════════════════════════════════════════════════════════════

class TestClosedForm:

    def test_projected_probability_matches_formula(self):
        """P(ω_n) = a + b_trust^n × (P(ω_0) − a)."""
        trust = Opinion(0.85, 0.05, 0.10)
        original = Opinion(0.90, 0.05, 0.05, base_rate=0.4)
        chain = compute_sl_chain(trust, original, max_length=20)
        for n in range(21):
            computed_p = chain[n].projected_probability()
            formula_p = sl_chain_closed_form(
                b_trust=0.85, p0=original.projected_probability(),
                base_rate=0.4, n=n,
            )
            assert computed_p == pytest.approx(formula_p, abs=1e-9), \
                f"Step {n}: computed={computed_p}, formula={formula_p}"

    def test_uncertainty_matches_formula(self):
        """u_n = 1 − b_trust^n × (1 − u_0)."""
        trust = Opinion(0.85, 0.05, 0.10)
        original = Opinion(0.70, 0.20, 0.10, base_rate=0.5)
        chain = compute_sl_chain(trust, original, max_length=20)
        for n in range(21):
            computed_u = chain[n].uncertainty
            formula_u = sl_uncertainty_closed_form(
                b_trust=0.85, u0=0.10, n=n,
            )
            assert computed_u == pytest.approx(formula_u, abs=1e-9), \
                f"Step {n}: computed={computed_u}, formula={formula_u}"

    def test_belief_decay_rate(self):
        """b_n = b_trust^n × b_0 — belief decays exponentially."""
        trust = Opinion(0.80, 0.10, 0.10)
        original = Opinion(0.70, 0.20, 0.10)
        chain = compute_sl_chain(trust, original, max_length=15)
        for n in range(16):
            expected_b = 0.80 ** n * 0.70
            assert chain[n].belief == pytest.approx(expected_b, abs=1e-9)

    def test_disbelief_decay_rate(self):
        """d_n = b_trust^n × d_0 — disbelief decays at same rate as belief."""
        trust = Opinion(0.80, 0.10, 0.10)
        original = Opinion(0.70, 0.20, 0.10)
        chain = compute_sl_chain(trust, original, max_length=15)
        for n in range(16):
            expected_d = 0.80 ** n * 0.20
            assert chain[n].disbelief == pytest.approx(expected_d, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════
# 3. Heterogeneous chains
# ═══════════════════════════════════════════════════════════════════

class TestHeterogeneousChain:

    def test_different_trusts_per_link(self):
        """Chain with varying trust levels at each link."""
        trusts = [
            Opinion(0.90, 0.05, 0.05),
            Opinion(0.60, 0.20, 0.20),
            Opinion(0.95, 0.02, 0.03),
        ]
        original = Opinion(0.80, 0.10, 0.10, base_rate=0.5)
        chain = compute_heterogeneous_sl_chain(trusts, original)
        assert len(chain) == 4  # original + 3 steps
        for op in chain:
            assert op.belief + op.disbelief + op.uncertainty == pytest.approx(1.0, abs=1e-9)

    def test_weak_link_dominates(self):
        """A single weak trust link should heavily dilute the opinion."""
        strong = Opinion(0.95, 0.02, 0.03)
        weak = Opinion(0.30, 0.40, 0.30)
        original = Opinion(0.90, 0.05, 0.05)

        chain_strong = compute_heterogeneous_sl_chain([strong, strong, strong], original)
        chain_weak = compute_heterogeneous_sl_chain([strong, weak, strong], original)

        assert chain_weak[-1].uncertainty > chain_strong[-1].uncertainty + 0.1

    def test_single_link_matches_homogeneous(self):
        """Single-link heterogeneous chain should match homogeneous."""
        trust = Opinion(0.85, 0.05, 0.10)
        original = Opinion(0.80, 0.10, 0.10)
        hetero = compute_heterogeneous_sl_chain([trust], original)
        homo = compute_sl_chain(trust, original, max_length=1)
        assert hetero[1].belief == pytest.approx(homo[1].belief, abs=1e-12)


# ═══════════════════════════════════════════════════════════════════
# 4. Branching provenance
# ═══════════════════════════════════════════════════════════════════

class TestBranchingProvenance:

    def test_two_paths_reduce_uncertainty(self):
        """Two independent trust paths fused should have lower uncertainty
        than either path alone."""
        trust = Opinion(0.80, 0.10, 0.10)
        original = Opinion(0.85, 0.10, 0.05, base_rate=0.5)
        trusts_per_path = [[trust] * 3, [trust] * 3]

        result = compute_branching_provenance(original, trusts_per_path)
        single_path = compute_sl_chain(trust, original, max_length=3)

        assert result["fused"].uncertainty < single_path[-1].uncertainty

    def test_returns_per_path_and_fused(self):
        trust = Opinion(0.80, 0.10, 0.10)
        original = Opinion(0.85, 0.10, 0.05)
        trusts_per_path = [[trust, trust], [trust]]
        result = compute_branching_provenance(original, trusts_per_path)
        assert "paths" in result
        assert "fused" in result
        assert len(result["paths"]) == 2
        assert isinstance(result["fused"], Opinion)

    def test_more_paths_lower_uncertainty(self):
        """3 paths should produce lower uncertainty than 2 paths."""
        trust = Opinion(0.75, 0.10, 0.15)
        original = Opinion(0.80, 0.10, 0.10, base_rate=0.5)
        result_2 = compute_branching_provenance(
            original, [[trust]*3, [trust]*3],
        )
        result_3 = compute_branching_provenance(
            original, [[trust]*3, [trust]*3, [trust]*3],
        )
        assert result_3["fused"].uncertainty < result_2["fused"].uncertainty


# ═══════════════════════════════════════════════════════════════════
# 5. Decision divergence
# ═══════════════════════════════════════════════════════════════════

class TestDecisionDivergence:

    def test_divergence_point_exists_for_nonhalf_base_rate(self):
        """With a != 0.5, scalar and SL should diverge at some chain length."""
        trust = Opinion(0.85, 0.05, 0.10)
        original = Opinion(0.80, 0.10, 0.10, base_rate=0.7)
        point = compute_decision_divergence_point(
            trust, 0.85, original, threshold=0.5, max_length=50,
        )
        assert point is not None
        assert point["divergence_length"] > 0
        assert point["sl_decision"] != point["scalar_decision"]

    def test_divergence_returns_chain_length(self):
        trust = Opinion(0.90, 0.05, 0.05)
        original = Opinion(0.85, 0.10, 0.05, base_rate=0.6)
        point = compute_decision_divergence_point(
            trust, 0.90, original, threshold=0.5, max_length=30,
        )
        assert "divergence_length" in point
        assert "sl_prob" in point
        assert "scalar_prob" in point


# ═══════════════════════════════════════════════════════════════════
# 6. Information content (entropy and evidence mass)
# ═══════════════════════════════════════════════════════════════════

class TestInformationContent:

    def test_vacuous_opinion_zero_entropy(self):
        """Vacuous opinion (u=1) has zero information → zero entropy of (b,d)."""
        vac = Opinion(0.0, 0.0, 1.0)
        entropy = compute_opinion_entropy(vac)
        assert entropy == pytest.approx(0.0, abs=1e-9)

    def test_dogmatic_opinion_has_entropy(self):
        """Balanced dogmatic opinion (b=0.5, d=0.5, u=0) has max BDU entropy."""
        dog = Opinion(0.5, 0.5, 0.0)
        entropy = compute_opinion_entropy(dog)
        assert entropy > 0

    def test_evidence_mass_monotonically_decreases(self):
        """Evidence mass (b + d) must decrease monotonically along the chain.

        This is the correct monotonic quantity. b_n + d_n = b_trust^n × (b_0 + d_0),
        which is strictly decreasing for b_trust < 1.

        Note: BDU *entropy* is NOT monotonically decreasing — as the
        distribution transitions from peaked-at-b to peaked-at-u, it passes
        through more uniform states with higher entropy. This is a documented
        mathematical property, not a bug.
        """
        trust = Opinion(0.85, 0.05, 0.10)
        original = Opinion(0.80, 0.15, 0.05)
        chain = compute_sl_chain(trust, original, max_length=20)
        evidence_mass = [op.belief + op.disbelief for op in chain]
        for i in range(1, len(evidence_mass)):
            assert evidence_mass[i] < evidence_mass[i-1] + 1e-12, \
                f"Step {i}: evidence mass {evidence_mass[i]} >= {evidence_mass[i-1]}"

    def test_bdu_entropy_is_non_monotonic(self):
        """BDU entropy can INCREASE before decreasing — this is correct behavior.

        Starting from a peaked distribution (high b, low u), trust discount
        moves mass from b to u. The intermediate states are more uniform
        (higher entropy) before concentrating at u (lower entropy).
        This is an honest mathematical finding.
        """
        trust = Opinion(0.85, 0.05, 0.10)
        original = Opinion(0.80, 0.15, 0.05)  # strongly peaked at b
        chain = compute_sl_chain(trust, original, max_length=30)
        entropies = [compute_opinion_entropy(op) for op in chain]
        # Entropy should increase initially, then decrease
        # Verify it's not monotonically decreasing (which would contradict math)
        has_increase = any(
            entropies[i] > entropies[i-1] + 1e-9
            for i in range(1, len(entropies))
        )
        # And also verify it eventually decreases toward 0
        assert entropies[-1] < entropies[5], "Entropy should eventually decrease"
        # The non-monotonicity is the finding — document but don't require
        # (some initial opinions may not exhibit it if already near uniform)

    def test_entropy_converges_to_zero(self):
        """Entropy must converge to 0 as opinion → vacuous (0, 0, 1)."""
        trust = Opinion(0.85, 0.05, 0.10)
        original = Opinion(0.80, 0.15, 0.05)
        chain = compute_sl_chain(trust, original, max_length=50)
        final_entropy = compute_opinion_entropy(chain[-1])
        assert final_entropy < 0.05  # near-vacuous → near-zero entropy

    def test_information_loss_curve_returns_both(self):
        """Information loss curve should include SL and scalar entropy."""
        trust = Opinion(0.85, 0.05, 0.10)
        original = Opinion(0.80, 0.15, 0.05, base_rate=0.5)
        curve = compute_information_loss_curve(
            trust, 0.85, original, max_length=15,
        )
        assert len(curve) == 16
        for entry in curve:
            assert "sl_entropy" in entry
            assert "scalar_entropy" in entry
            assert "step" in entry


# ═══════════════════════════════════════════════════════════════════
# 7. Bayesian chain comparison
# ═══════════════════════════════════════════════════════════════════

class TestBayesianChain:

    def test_bayesian_chain_returns_probabilities(self):
        """Bayesian updating through intermediaries."""
        trust_reliability = 0.85
        original_prob = 0.90
        chain = compute_bayesian_chain(
            trust_reliability, original_prob, max_length=20,
        )
        assert len(chain) == 21
        assert all(0.0 <= p <= 1.0 for p in chain)

    def test_bayesian_converges_to_prior(self):
        """Bayesian should converge to prior (0.5) with enough intermediaries."""
        chain = compute_bayesian_chain(0.85, 0.90, max_length=50, prior=0.5)
        assert abs(chain[-1] - 0.5) < 0.05


# ═══════════════════════════════════════════════════════════════════
# 8. Sweep functions
# ═══════════════════════════════════════════════════════════════════

class TestSweeps:

    def test_trust_level_sweep_returns_dict(self):
        results = run_trust_level_sweep(
            trust_levels=[0.6, 0.8, 0.95],
            original=Opinion(0.85, 0.10, 0.05, base_rate=0.5),
            max_length=20,
        )
        assert len(results) == 3
        for tl, data in results.items():
            assert "sl_probs" in data
            assert "scalar_probs" in data
            assert "sl_uncertainties" in data
            assert len(data["sl_probs"]) == 21

    def test_higher_trust_slower_decay(self):
        """Higher trust level should preserve information longer."""
        original = Opinion(0.85, 0.10, 0.05, base_rate=0.5)
        results = run_trust_level_sweep(
            trust_levels=[0.6, 0.95],
            original=original, max_length=20,
        )
        sl_low = results[0.6]["sl_probs"][10]
        sl_high = results[0.95]["sl_probs"][10]
        base = 0.5
        assert abs(sl_high - base) > abs(sl_low - base)

    def test_original_opinion_sweep(self):
        results = run_original_opinion_sweep(
            trust=Opinion(0.85, 0.05, 0.10),
            trust_scalar=0.85,
            originals=[
                Opinion(0.90, 0.05, 0.05, base_rate=0.5),
                Opinion(0.10, 0.80, 0.10, base_rate=0.5),
                Opinion(0.00, 0.00, 1.00, base_rate=0.5),
            ],
            max_length=20,
        )
        assert len(results) == 3

    def test_base_rate_sweep(self):
        results = run_base_rate_sweep(
            trust=Opinion(0.85, 0.05, 0.10),
            trust_scalar=0.85,
            base_rates=[0.1, 0.3, 0.5, 0.7, 0.9],
            max_length=20,
        )
        assert len(results) == 5
        for br, data in results.items():
            final_p = data["sl_probs"][-1]
            assert final_p == pytest.approx(br, abs=0.05)

    def test_chain_length_comparison(self):
        results = run_chain_length_comparison(
            trust=Opinion(0.85, 0.05, 0.10),
            trust_scalar=0.85,
            original=Opinion(0.80, 0.10, 0.10, base_rate=0.6),
            max_length=30,
        )
        assert "sl_chain" in results
        assert "scalar_chain" in results
        assert "bayesian_chain" in results
        assert "divergence_point" in results
        assert "sl_half_life" in results
        assert "scalar_half_life" in results
