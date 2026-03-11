#!/usr/bin/env python
"""EN7.1 — Property-Based Verification of Confidence Algebra.

NeurIPS 2026 D&B, Suite EN7 (Formal Algebra Properties), Experiment 1.

Verifies 11 mathematically formal properties of the jsonld-ex confidence
algebra using the Hypothesis property-based testing framework with 10,000
random opinions per property.  Establishes mathematical credibility of the
SL opinion algebra by demonstrating that all operators preserve validity
invariants, satisfy expected algebraic identities, and degrade gracefully
where identities are known not to hold.

Properties verified:
    P1  — Opinion validity invariant (b+d+u=1, all in [0,1]) after EVERY operator
    P2  — Cumulative fusion commutativity
    P3  — Cumulative fusion associativity (equal base rates)
    P4  — Cumulative fusion non-associativity documentation (different base rates)
    P5  — Vacuous opinion as fusion identity element
    P6  — Trust discount with full trust preserves opinion
    P7  — Trust discount with vacuous trust yields vacuous opinion
    P8  — Deduction factorization (component-wise law of total probability)
    P9  — Decay monotonicity (uncertainty non-decreasing with elapsed time)
    P10 — Decay convergence to vacuous opinion as elapsed -> infinity
    P11 — Projected probability consistency (P(w) = b + a*u)

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN7/en7_1_property_verification.py

Output:
    experiments/EN7/results/en7_1_results.json

References:
    Josang, A. (2016). Subjective Logic: A Formalism for Reasoning Under
    Uncertainty. Springer.  ISBN 978-3-319-42337-1.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

# ── Path setup ─────────────────────────────────────────────────────
# Ensure repo root is on sys.path so experiments.infra and jsonld_ex
# are both importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "packages" / "python" / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

# ── Imports: jsonld-ex ─────────────────────────────────────────────
from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    trust_discount,
    deduce,
    conflict_metric,
    pairwise_conflict,
    robust_fuse,
)
from jsonld_ex.confidence_decay import (
    decay_opinion,
    exponential_decay,
    linear_decay,
    step_decay,
)

# ── Imports: experiment infrastructure ─────────────────────────────
from experiments.infra.config import set_global_seed, get_global_seed
from experiments.infra.results import ExperimentResult
from experiments.infra.env_log import log_environment

# ── Imports: Hypothesis ────────────────────────────────────────────
from hypothesis import given, settings, HealthCheck, Phase
from hypothesis import strategies as st

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

GLOBAL_SEED = 42
N_EXAMPLES = 10_000
FLOAT_TOL = 1e-9          # Tolerance for floating-point equality
VALIDITY_TOL = 1e-9        # Tolerance for b+d+u=1 check
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Hypothesis settings shared by all property tests
PROP_SETTINGS = settings(
    max_examples=N_EXAMPLES,
    derandomize=True,       # Deterministic: seed derived from test name
    database=None,          # No persistent database (clean runs)
    deadline=None,          # No per-example timeout
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
    ],
    phases=[Phase.explicit, Phase.generate],  # Skip shrinking for speed
)

# ═══════════════════════════════════════════════════════════════════
# HYPOTHESIS STRATEGIES
# ═══════════════════════════════════════════════════════════════════


@st.composite
def opinion_strategy(draw: st.DrawFn, base_rate: float | None = None) -> Opinion:
    """Generate a valid SL opinion uniformly on the 2-simplex.

    Uses the sorted-uniform method for uniform sampling on the simplex:
        1. Draw U1, U2 ~ Uniform(0, 1)
        2. Sort to get U_(1) <= U_(2)
        3. Set b = U_(1), d = U_(2) - U_(1), u = 1 - U_(2)

    This produces a uniform distribution over all valid (b, d, u) triples
    satisfying b + d + u = 1, b >= 0, d >= 0, u >= 0.

    Args:
        base_rate: If provided, fix the base rate to this value.
                   Otherwise, draw uniformly from [0, 1].
    """
    u1 = draw(st.floats(min_value=0.0, max_value=1.0,
                         allow_nan=False, allow_infinity=False))
    u2 = draw(st.floats(min_value=0.0, max_value=1.0,
                         allow_nan=False, allow_infinity=False))
    lo, hi = min(u1, u2), max(u1, u2)
    b = lo
    d = hi - lo
    u = 1.0 - hi

    if base_rate is not None:
        a = base_rate
    else:
        a = draw(st.floats(min_value=0.0, max_value=1.0,
                            allow_nan=False, allow_infinity=False))

    return Opinion(belief=b, disbelief=d, uncertainty=u, base_rate=a)


@st.composite
def non_dogmatic_opinion_strategy(draw: st.DrawFn, base_rate: float | None = None) -> Opinion:
    """Generate a valid opinion with u > 0 (non-dogmatic).

    Required for properties where dogmatic opinions cause 0/0
    indeterminate forms (e.g., cumulative fusion).
    """
    u1 = draw(st.floats(min_value=0.0, max_value=1.0,
                         allow_nan=False, allow_infinity=False))
    # Ensure u2 < 1.0 so that u = 1 - u2 > 0
    u2 = draw(st.floats(min_value=0.0, max_value=1.0 - 1e-12,
                         allow_nan=False, allow_infinity=False))
    lo, hi = min(u1, u2), max(u1, u2)
    b = lo
    d = hi - lo
    u = 1.0 - hi

    # If u ended up as 0 due to float, nudge it
    if u <= 0.0:
        u = 1e-12
        total = b + d + u
        b /= total
        d /= total
        u /= total

    if base_rate is not None:
        a = base_rate
    else:
        a = draw(st.floats(min_value=0.0, max_value=1.0,
                            allow_nan=False, allow_infinity=False))

    return Opinion(belief=b, disbelief=d, uncertainty=u, base_rate=a)


def positive_float_strategy(
    min_value: float = 1e-6,
    max_value: float = 1e6,
) -> st.SearchStrategy[float]:
    """Generate a positive float for elapsed time, half-life, etc."""
    return st.floats(min_value=min_value, max_value=max_value,
                     allow_nan=False, allow_infinity=False)


# ═══════════════════════════════════════════════════════════════════
# VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════


def is_valid_opinion(op: Opinion, tol: float = VALIDITY_TOL) -> bool:
    """Check whether an Opinion satisfies all validity constraints."""
    if not (0.0 - tol <= op.belief <= 1.0 + tol):
        return False
    if not (0.0 - tol <= op.disbelief <= 1.0 + tol):
        return False
    if not (0.0 - tol <= op.uncertainty <= 1.0 + tol):
        return False
    if not (0.0 - tol <= op.base_rate <= 1.0 + tol):
        return False
    total = op.belief + op.disbelief + op.uncertainty
    if abs(total - 1.0) > tol:
        return False
    return True


def opinion_close(a: Opinion, b: Opinion, tol: float = FLOAT_TOL) -> bool:
    """Check whether two opinions are close in all components."""
    return (
        abs(a.belief - b.belief) < tol
        and abs(a.disbelief - b.disbelief) < tol
        and abs(a.uncertainty - b.uncertainty) < tol
        and abs(a.base_rate - b.base_rate) < tol
    )


def opinion_bdu_close(a: Opinion, b: Opinion, tol: float = FLOAT_TOL) -> bool:
    """Check whether two opinions are close in b, d, u (ignoring base_rate)."""
    return (
        abs(a.belief - b.belief) < tol
        and abs(a.disbelief - b.disbelief) < tol
        and abs(a.uncertainty - b.uncertainty) < tol
    )


def opinion_max_deviation(a: Opinion, b: Opinion) -> float:
    """Maximum absolute deviation between two opinions across all components."""
    return max(
        abs(a.belief - b.belief),
        abs(a.disbelief - b.disbelief),
        abs(a.uncertainty - b.uncertainty),
        abs(a.base_rate - b.base_rate),
    )


def opinion_bdu_max_deviation(a: Opinion, b: Opinion) -> float:
    """Maximum absolute deviation in b, d, u only."""
    return max(
        abs(a.belief - b.belief),
        abs(a.disbelief - b.disbelief),
        abs(a.uncertainty - b.uncertainty),
    )


# ═══════════════════════════════════════════════════════════════════
# PROPERTY RESULT CONTAINER
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PropertyResult:
    """Result of verifying a single algebraic property."""

    property_id: str
    property_name: str
    description: str
    expected_outcome: str       # "PASS" or "COUNTEREXAMPLES"
    actual_outcome: str         # "PASS", "FAIL", "COUNTEREXAMPLES_FOUND"
    n_examples_tested: int
    wall_time_seconds: float
    max_deviation: float | None = None
    mean_deviation: float | None = None
    std_deviation: float | None = None
    counterexamples: list[dict[str, Any]] | None = None
    sub_properties: dict[str, Any] | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════
# PROPERTY RUNNERS
# ═══════════════════════════════════════════════════════════════════


def run_property(
    property_id: str,
    property_name: str,
    description: str,
    expected_outcome: str,
    test_fn: Callable[[], dict[str, Any]],
) -> PropertyResult:
    """Execute a property test function and capture results.

    Args:
        property_id:      e.g. "P1"
        property_name:    Human-readable name
        description:      Mathematical description
        expected_outcome: "PASS" or "COUNTEREXAMPLES"
        test_fn:          Callable that runs the property and returns
                          a dict with keys: n_tested, max_deviation,
                          mean_deviation, std_deviation, counterexamples,
                          sub_properties, notes, outcome.
    """
    print(f"  Running {property_id}: {property_name}...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        result_data = test_fn()
        elapsed = time.perf_counter() - t0
        outcome = result_data.get("outcome", "PASS")
        print(f"{outcome} ({elapsed:.2f}s)")
        return PropertyResult(
            property_id=property_id,
            property_name=property_name,
            description=description,
            expected_outcome=expected_outcome,
            actual_outcome=outcome,
            n_examples_tested=result_data.get("n_tested", N_EXAMPLES),
            wall_time_seconds=round(elapsed, 4),
            max_deviation=result_data.get("max_deviation"),
            mean_deviation=result_data.get("mean_deviation"),
            std_deviation=result_data.get("std_deviation"),
            counterexamples=result_data.get("counterexamples"),
            sub_properties=result_data.get("sub_properties"),
            notes=result_data.get("notes"),
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"FAIL ({elapsed:.2f}s) — {e}")
        return PropertyResult(
            property_id=property_id,
            property_name=property_name,
            description=description,
            expected_outcome=expected_outcome,
            actual_outcome=f"FAIL: {type(e).__name__}: {e}",
            n_examples_tested=0,
            wall_time_seconds=round(elapsed, 4),
            notes=str(e),
        )


# ═══════════════════════════════════════════════════════════════════
# P1: OPINION VALIDITY INVARIANT
# ═══════════════════════════════════════════════════════════════════


def test_p1_validity_invariant() -> dict[str, Any]:
    """Verify that EVERY operator preserves the opinion validity invariant.

    For every operator in the algebra, the output must satisfy:
        b + d + u = 1,  b >= 0,  d >= 0,  u >= 0,  a in [0,1]

    Operators tested:
        - cumulative_fuse (2-ary and 3-ary)
        - averaging_fuse (2-ary and 3-ary)
        - trust_discount
        - deduce
        - decay_opinion (exponential, linear, step)
        - robust_fuse
        - Opinion.from_confidence
        - Opinion.from_evidence
        - conflict_metric (output in [0, 1])
        - pairwise_conflict (output in [0, 1])
    """
    sub_results: dict[str, str] = {}
    n_tested_total = 0

    # ── Helper: run a @given test and record pass/fail ──
    def run_sub(name: str, test_func: Callable) -> None:
        nonlocal n_tested_total
        try:
            test_func()
            sub_results[name] = "PASS"
        except Exception as e:
            sub_results[name] = f"FAIL: {e}"

    # --- cumulative_fuse (2-ary) ---
    @given(a=opinion_strategy(), b=opinion_strategy())
    @PROP_SETTINGS
    def _cf2(a: Opinion, b: Opinion) -> None:
        result = cumulative_fuse(a, b)
        assert is_valid_opinion(result), f"Invalid: {result}"

    run_sub("cumulative_fuse_2ary", _cf2)
    n_tested_total += N_EXAMPLES

    # --- cumulative_fuse (3-ary) ---
    @given(a=opinion_strategy(), b=opinion_strategy(), c=opinion_strategy())
    @PROP_SETTINGS
    def _cf3(a: Opinion, b: Opinion, c: Opinion) -> None:
        result = cumulative_fuse(a, b, c)
        assert is_valid_opinion(result), f"Invalid: {result}"

    run_sub("cumulative_fuse_3ary", _cf3)
    n_tested_total += N_EXAMPLES

    # --- averaging_fuse (2-ary) ---
    @given(a=opinion_strategy(), b=opinion_strategy())
    @PROP_SETTINGS
    def _af2(a: Opinion, b: Opinion) -> None:
        result = averaging_fuse(a, b)
        assert is_valid_opinion(result), f"Invalid: {result}"

    run_sub("averaging_fuse_2ary", _af2)
    n_tested_total += N_EXAMPLES

    # --- averaging_fuse (3-ary) ---
    @given(a=opinion_strategy(), b=opinion_strategy(), c=opinion_strategy())
    @PROP_SETTINGS
    def _af3(a: Opinion, b: Opinion, c: Opinion) -> None:
        result = averaging_fuse(a, b, c)
        assert is_valid_opinion(result), f"Invalid: {result}"

    run_sub("averaging_fuse_3ary", _af3)
    n_tested_total += N_EXAMPLES

    # --- trust_discount ---
    @given(trust=opinion_strategy(), opinion=opinion_strategy())
    @PROP_SETTINGS
    def _td(trust: Opinion, opinion: Opinion) -> None:
        result = trust_discount(trust, opinion)
        assert is_valid_opinion(result), f"Invalid: {result}"

    run_sub("trust_discount", _td)
    n_tested_total += N_EXAMPLES

    # --- deduce ---
    @given(
        ox=opinion_strategy(),
        oy_x=opinion_strategy(),
        oy_nx=opinion_strategy(),
    )
    @PROP_SETTINGS
    def _ded(ox: Opinion, oy_x: Opinion, oy_nx: Opinion) -> None:
        result = deduce(ox, oy_x, oy_nx)
        assert is_valid_opinion(result), f"Invalid: {result}"

    run_sub("deduce", _ded)
    n_tested_total += N_EXAMPLES

    # --- decay_opinion (exponential) ---
    @given(
        op=opinion_strategy(),
        elapsed=positive_float_strategy(),
        half_life=positive_float_strategy(),
    )
    @PROP_SETTINGS
    def _dec_exp(op: Opinion, elapsed: float, half_life: float) -> None:
        result = decay_opinion(op, elapsed, half_life, decay_fn=exponential_decay)
        assert is_valid_opinion(result), f"Invalid: {result}"

    run_sub("decay_exponential", _dec_exp)
    n_tested_total += N_EXAMPLES

    # --- decay_opinion (linear) ---
    @given(
        op=opinion_strategy(),
        elapsed=positive_float_strategy(),
        half_life=positive_float_strategy(),
    )
    @PROP_SETTINGS
    def _dec_lin(op: Opinion, elapsed: float, half_life: float) -> None:
        result = decay_opinion(op, elapsed, half_life, decay_fn=linear_decay)
        assert is_valid_opinion(result), f"Invalid: {result}"

    run_sub("decay_linear", _dec_lin)
    n_tested_total += N_EXAMPLES

    # --- decay_opinion (step) ---
    @given(
        op=opinion_strategy(),
        elapsed=positive_float_strategy(),
        half_life=positive_float_strategy(),
    )
    @PROP_SETTINGS
    def _dec_step(op: Opinion, elapsed: float, half_life: float) -> None:
        result = decay_opinion(op, elapsed, half_life, decay_fn=step_decay)
        assert is_valid_opinion(result), f"Invalid: {result}"

    run_sub("decay_step", _dec_step)
    n_tested_total += N_EXAMPLES

    # --- robust_fuse ---
    @given(
        a=opinion_strategy(),
        b=opinion_strategy(),
        c=opinion_strategy(),
    )
    @PROP_SETTINGS
    def _rf(a: Opinion, b: Opinion, c: Opinion) -> None:
        result, _ = robust_fuse([a, b, c])
        assert is_valid_opinion(result), f"Invalid: {result}"

    run_sub("robust_fuse", _rf)
    n_tested_total += N_EXAMPLES

    # --- from_confidence ---
    @given(
        c=st.floats(min_value=0.0, max_value=1.0,
                     allow_nan=False, allow_infinity=False),
        u=st.floats(min_value=0.0, max_value=1.0,
                     allow_nan=False, allow_infinity=False),
        a=st.floats(min_value=0.0, max_value=1.0,
                     allow_nan=False, allow_infinity=False),
    )
    @PROP_SETTINGS
    def _fc(c: float, u: float, a: float) -> None:
        result = Opinion.from_confidence(c, uncertainty=u, base_rate=a)
        assert is_valid_opinion(result), f"Invalid: {result}"

    run_sub("from_confidence", _fc)
    n_tested_total += N_EXAMPLES

    # --- from_evidence ---
    @given(
        pos=st.integers(min_value=0, max_value=10000),
        neg=st.integers(min_value=0, max_value=10000),
        pw=st.floats(min_value=0.01, max_value=100.0,
                      allow_nan=False, allow_infinity=False),
        a=st.floats(min_value=0.0, max_value=1.0,
                     allow_nan=False, allow_infinity=False),
    )
    @PROP_SETTINGS
    def _fe(pos: int, neg: int, pw: float, a: float) -> None:
        result = Opinion.from_evidence(pos, neg, prior_weight=pw, base_rate=a)
        assert is_valid_opinion(result), f"Invalid: {result}"

    run_sub("from_evidence", _fe)
    n_tested_total += N_EXAMPLES

    # --- conflict_metric (output range [0, 1]) ---
    @given(op=opinion_strategy())
    @PROP_SETTINGS
    def _cm(op: Opinion) -> None:
        result = conflict_metric(op)
        assert 0.0 <= result <= 1.0 + FLOAT_TOL, f"Out of range: {result}"

    run_sub("conflict_metric_range", _cm)
    n_tested_total += N_EXAMPLES

    # --- pairwise_conflict (output range [0, 1]) ---
    @given(a=opinion_strategy(), b=opinion_strategy())
    @PROP_SETTINGS
    def _pc(a: Opinion, b: Opinion) -> None:
        result = pairwise_conflict(a, b)
        assert 0.0 - FLOAT_TOL <= result <= 1.0 + FLOAT_TOL, f"Out of range: {result}"

    run_sub("pairwise_conflict_range", _pc)
    n_tested_total += N_EXAMPLES

    # ── Aggregate ──
    all_passed = all(v == "PASS" for v in sub_results.values())
    return {
        "outcome": "PASS" if all_passed else "FAIL",
        "n_tested": n_tested_total,
        "sub_properties": sub_results,
        "notes": f"Tested {len(sub_results)} sub-operators, {n_tested_total} total examples",
    }


# ═══════════════════════════════════════════════════════════════════
# P2: FUSION COMMUTATIVITY
# ═══════════════════════════════════════════════════════════════════


def test_p2_fusion_commutativity() -> dict[str, Any]:
    """Verify cumulative_fuse(a, b) == cumulative_fuse(b, a).

    Also verifies averaging_fuse commutativity as a sub-property.
    Reports maximum floating-point deviation observed.
    """
    deviations_cf: list[float] = []
    deviations_af: list[float] = []

    @given(a=opinion_strategy(), b=opinion_strategy())
    @PROP_SETTINGS
    def _cf_comm(a: Opinion, b: Opinion) -> None:
        ab = cumulative_fuse(a, b)
        ba = cumulative_fuse(b, a)
        dev = opinion_max_deviation(ab, ba)
        deviations_cf.append(dev)
        assert dev < FLOAT_TOL, (
            f"Commutativity violated: max_dev={dev}, "
            f"a={a}, b={b}, ab={ab}, ba={ba}"
        )

    @given(a=opinion_strategy(), b=opinion_strategy())
    @PROP_SETTINGS
    def _af_comm(a: Opinion, b: Opinion) -> None:
        ab = averaging_fuse(a, b)
        ba = averaging_fuse(b, a)
        dev = opinion_max_deviation(ab, ba)
        deviations_af.append(dev)
        assert dev < FLOAT_TOL, (
            f"Commutativity violated: max_dev={dev}, "
            f"a={a}, b={b}, ab={ab}, ba={ba}"
        )

    sub_results: dict[str, str] = {}
    try:
        _cf_comm()
        sub_results["cumulative_fuse"] = "PASS"
    except Exception as e:
        sub_results["cumulative_fuse"] = f"FAIL: {e}"

    try:
        _af_comm()
        sub_results["averaging_fuse"] = "PASS"
    except Exception as e:
        sub_results["averaging_fuse"] = f"FAIL: {e}"

    import numpy as np
    all_devs = deviations_cf + deviations_af
    all_passed = all(v == "PASS" for v in sub_results.values())

    return {
        "outcome": "PASS" if all_passed else "FAIL",
        "n_tested": len(deviations_cf) + len(deviations_af),
        "max_deviation": float(np.max(all_devs)) if all_devs else 0.0,
        "mean_deviation": float(np.mean(all_devs)) if all_devs else 0.0,
        "std_deviation": float(np.std(all_devs)) if all_devs else 0.0,
        "sub_properties": sub_results,
        "notes": (
            f"Cumulative fusion: max_dev={max(deviations_cf) if deviations_cf else 0:.2e}, "
            f"Averaging fusion: max_dev={max(deviations_af) if deviations_af else 0:.2e}"
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# P3: FUSION ASSOCIATIVITY (EQUAL BASE RATES)
# ═══════════════════════════════════════════════════════════════════


def _make_associativity_test(
    br: float,
    deviations: list[float],
    bdu_deviations: list[float],
    use_non_dogmatic: bool = True,
) -> Callable:
    """Factory: create a @given-decorated associativity test for a fixed base rate.

    Hypothesis rejects @given on functions with default arguments, so we
    use a factory that closes over the base rate value instead.

    Args:
        br: Fixed base rate for all three opinions.
        deviations: Shared list to accumulate max deviations.
        bdu_deviations: Shared list to accumulate BDU deviations.
        use_non_dogmatic: If True, use non_dogmatic_opinion_strategy to
            avoid the dogmatic fallback branch (which is known to break
            associativity).  Josang's associativity proof (S12.3) requires
            at least one non-dogmatic opinion in each pair.
    """
    strat = non_dogmatic_opinion_strategy if use_non_dogmatic else opinion_strategy

    @given(
        a=strat(base_rate=br),
        b=strat(base_rate=br),
        c=strat(base_rate=br),
    )
    @PROP_SETTINGS
    def _assoc(a: Opinion, b: Opinion, c: Opinion) -> None:
        left = cumulative_fuse(cumulative_fuse(a, b), c)
        right = cumulative_fuse(a, cumulative_fuse(b, c))
        dev = opinion_max_deviation(left, right)
        bdu_dev = opinion_bdu_max_deviation(left, right)
        deviations.append(dev)
        bdu_deviations.append(bdu_dev)
        assert dev < FLOAT_TOL, (
            f"Associativity violated at base_rate={br}: max_dev={dev}, "
            f"a={a}, b={b}, c={c}, left={left}, right={right}"
        )

    return _assoc


def test_p3_fusion_associativity_equal_base_rates() -> dict[str, Any]:
    """Verify cumulative fusion associativity when all base rates are equal.

    Josang (2016, S12.3) proves associativity for the standard kappa-based
    formula, which requires at least one non-dogmatic opinion (u > 0) in
    each fused pair.  When both opinions are dogmatic (u = 0), the formula
    has a 0/0 indeterminate form and falls back to simple averaging with
    equal relative dogmatism (gamma_A = gamma_B = 0.5).  Simple averaging
    is NOT associative: avg(avg(a,b),c) != avg(a,avg(b,c)) in general.

    We verify:
      (a) Non-dogmatic case: associativity holds (PASS expected).
      (b) Dogmatic case: associativity does NOT hold (documented as finding).

    We test across 7 fixed base rates for robustness.
    """
    deviations: list[float] = []
    bdu_deviations: list[float] = []
    dogmatic_deviations: list[float] = []
    dogmatic_counterexamples: list[dict[str, Any]] = []

    fixed_base_rates = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    sub_results: dict[str, str] = {}

    # ── (a) Non-dogmatic: associativity should hold ──
    for br in fixed_base_rates:
        test_fn = _make_associativity_test(
            br, deviations, bdu_deviations, use_non_dogmatic=True,
        )
        try:
            test_fn()
            sub_results[f"non_dogmatic_br={br}"] = "PASS"
        except Exception as e:
            sub_results[f"non_dogmatic_br={br}"] = f"FAIL: {e}"

    # ── (b) Dogmatic: document non-associativity ──
    # Use explicit canonical examples rather than random generation to
    # clearly demonstrate the mathematical phenomenon.
    dogmatic_test_cases = [
        # (a, b, c) triples — all dogmatic, same base_rate
        (Opinion(0.7, 0.3, 0.0, 0.5), Opinion(0.6, 0.4, 0.0, 0.5), Opinion(1.0, 0.0, 0.0, 0.5)),
        (Opinion(0.3, 0.7, 0.0, 0.5), Opinion(0.8, 0.2, 0.0, 0.5), Opinion(0.5, 0.5, 0.0, 0.5)),
        (Opinion(0.9, 0.1, 0.0, 0.5), Opinion(0.1, 0.9, 0.0, 0.5), Opinion(0.9, 0.1, 0.0, 0.5)),
        (Opinion(1.0, 0.0, 0.0, 0.5), Opinion(0.0, 1.0, 0.0, 0.5), Opinion(0.5, 0.5, 0.0, 0.5)),
    ]
    n_dogmatic_violations = 0
    for a_op, b_op, c_op in dogmatic_test_cases:
        left = cumulative_fuse(cumulative_fuse(a_op, b_op), c_op)
        right = cumulative_fuse(a_op, cumulative_fuse(b_op, c_op))
        dev = opinion_max_deviation(left, right)
        dogmatic_deviations.append(dev)
        if dev > FLOAT_TOL:
            n_dogmatic_violations += 1
            if len(dogmatic_counterexamples) < 5:
                dogmatic_counterexamples.append({
                    "a": {"b": a_op.belief, "d": a_op.disbelief, "u": a_op.uncertainty},
                    "b": {"b": b_op.belief, "d": b_op.disbelief, "u": b_op.uncertainty},
                    "c": {"b": c_op.belief, "d": c_op.disbelief, "u": c_op.uncertainty},
                    "left": {"b": left.belief, "d": left.disbelief, "u": left.uncertainty},
                    "right": {"b": right.belief, "d": right.disbelief, "u": right.uncertainty},
                    "max_deviation": dev,
                })

    sub_results["dogmatic_non_associativity_documented"] = (
        f"CONFIRMED: {n_dogmatic_violations}/{len(dogmatic_test_cases)} "
        f"dogmatic triples violate associativity"
    )

    import numpy as np
    non_dogmatic_passed = all(
        v == "PASS"
        for k, v in sub_results.items()
        if k.startswith("non_dogmatic_br=")
    )

    return {
        "outcome": "PASS" if non_dogmatic_passed else "FAIL",
        "n_tested": len(deviations) + len(dogmatic_test_cases),
        "max_deviation": float(np.max(deviations)) if deviations else 0.0,
        "mean_deviation": float(np.mean(deviations)) if deviations else 0.0,
        "std_deviation": float(np.std(deviations)) if deviations else 0.0,
        "counterexamples": dogmatic_counterexamples if dogmatic_counterexamples else None,
        "sub_properties": sub_results,
        "notes": (
            f"Tested {len(fixed_base_rates)} fixed base rates with non-dogmatic opinions, "
            f"{len(deviations)} total non-dogmatic examples. "
            f"BDU max_dev={float(np.max(bdu_deviations)) if bdu_deviations else 0:.2e}. "
            f"FINDING: Associativity holds for non-dogmatic opinions (standard kappa formula) "
            f"but fails for dogmatic opinions (equal-weight averaging fallback). "
            f"Dogmatic violations: {n_dogmatic_violations}/{len(dogmatic_test_cases)}, "
            f"max_dev={max(dogmatic_deviations) if dogmatic_deviations else 0:.4f}. "
            f"This is mathematically inherent: simple averaging is not associative."
        ),
    }




# ═══════════════════════════════════════════════════════════════════
# P4: FUSION NON-ASSOCIATIVITY (DIFFERENT BASE RATES)
# ═══════════════════════════════════════════════════════════════════


def test_p4_fusion_non_associativity() -> dict[str, Any]:
    """Document non-associativity of cumulative fusion with different base rates.

    We generate opinion triples with distinct base rates and measure
    the deviation between the two association orderings.  We collect
    counterexamples where the deviation exceeds floating-point noise.

    We separately track:
      1. Base-rate non-associativity (from the averaging of base rates)
      2. BDU non-associativity (from the dogmatic fallback branch)

    KEY FINDING: Non-associativity has two sources:
      (a) Base-rate averaging: always present when base rates differ.
          This is inherent to the pairwise base-rate averaging convention
          a_fused = (a_A + a_B) / 2.
      (b) Dogmatic fallback: when both opinions in a pair have u=0, the
          formula switches from the standard kappa-based formula to simple
          averaging.  This creates a different composition path for b/d/u
          depending on the association order.  These cases are rare with
          uniform sampling but represent a genuine mathematical
          discontinuity in the operator.
    """
    # Dogmatic threshold: opinions with u below this are "effectively dogmatic"
    DOGMATIC_THRESHOLD = 1e-300

    counterexamples: list[dict[str, Any]] = []
    full_deviations: list[float] = []
    bdu_deviations: list[float] = []
    base_rate_deviations: list[float] = []

    # Dogmatic path tracking
    n_involves_dogmatic_pair = 0       # At least one pair is both-dogmatic
    n_bdu_violations_dogmatic = 0      # BDU violation AND dogmatic pair
    n_bdu_violations_nondogmatic = 0   # BDU violation WITHOUT dogmatic pair

    @given(
        a=opinion_strategy(),
        b=opinion_strategy(),
        c=opinion_strategy(),
    )
    @PROP_SETTINGS
    def _non_assoc(a: Opinion, b: Opinion, c: Opinion) -> None:
        nonlocal n_involves_dogmatic_pair
        nonlocal n_bdu_violations_dogmatic
        nonlocal n_bdu_violations_nondogmatic

        left = cumulative_fuse(cumulative_fuse(a, b), c)
        right = cumulative_fuse(a, cumulative_fuse(b, c))

        full_dev = opinion_max_deviation(left, right)
        bdu_dev = opinion_bdu_max_deviation(left, right)
        br_dev = abs(left.base_rate - right.base_rate)

        full_deviations.append(full_dev)
        bdu_deviations.append(bdu_dev)
        base_rate_deviations.append(br_dev)

        # Detect dogmatic pairs: does any pair of inputs both have u ~ 0?
        # This determines whether the dogmatic branch is triggered.
        has_dogmatic_pair = (
            (a.uncertainty < DOGMATIC_THRESHOLD and b.uncertainty < DOGMATIC_THRESHOLD)
            or (b.uncertainty < DOGMATIC_THRESHOLD and c.uncertainty < DOGMATIC_THRESHOLD)
            or (a.uncertainty < DOGMATIC_THRESHOLD and c.uncertainty < DOGMATIC_THRESHOLD)
            # Also check intermediate: fuse(a,b) dogmatic AND c dogmatic
            or (cumulative_fuse(a, b).uncertainty < DOGMATIC_THRESHOLD
                and c.uncertainty < DOGMATIC_THRESHOLD)
            or (a.uncertainty < DOGMATIC_THRESHOLD
                and cumulative_fuse(b, c).uncertainty < DOGMATIC_THRESHOLD)
        )

        if has_dogmatic_pair:
            n_involves_dogmatic_pair += 1

        is_bdu_violation = bdu_dev > FLOAT_TOL

        if is_bdu_violation:
            if has_dogmatic_pair:
                n_bdu_violations_dogmatic += 1
            else:
                n_bdu_violations_nondogmatic += 1

        # Collect notable counterexamples (limit to 10 full + 10 BDU)
        if full_dev > 1e-6 and len(counterexamples) < 10:
            counterexamples.append({
                "a": {"b": a.belief, "d": a.disbelief, "u": a.uncertainty, "a": a.base_rate},
                "b": {"b": b.belief, "d": b.disbelief, "u": b.uncertainty, "a": b.base_rate},
                "c": {"b": c.belief, "d": c.disbelief, "u": c.uncertainty, "a": c.base_rate},
                "left_base_rate": left.base_rate,
                "right_base_rate": right.base_rate,
                "full_deviation": full_dev,
                "bdu_deviation": bdu_dev,
                "base_rate_deviation": br_dev,
                "involves_dogmatic_pair": has_dogmatic_pair,
            })
        # Also collect BDU-specific counterexamples
        elif is_bdu_violation and len(counterexamples) < 20:
            counterexamples.append({
                "type": "bdu_violation",
                "a": {"b": a.belief, "d": a.disbelief, "u": a.uncertainty, "a": a.base_rate},
                "b": {"b": b.belief, "d": b.disbelief, "u": b.uncertainty, "a": b.base_rate},
                "c": {"b": c.belief, "d": c.disbelief, "u": c.uncertainty, "a": c.base_rate},
                "left_bdu": {"b": left.belief, "d": left.disbelief, "u": left.uncertainty},
                "right_bdu": {"b": right.belief, "d": right.disbelief, "u": right.uncertainty},
                "bdu_deviation": bdu_dev,
                "involves_dogmatic_pair": has_dogmatic_pair,
            })

    _non_assoc()

    import numpy as np
    n_violations = sum(1 for d in full_deviations if d > FLOAT_TOL)
    n_bdu_violations = sum(1 for d in bdu_deviations if d > FLOAT_TOL)

    # Characterize whether ALL bdu violations are from dogmatic path
    bdu_all_from_dogmatic = (
        n_bdu_violations > 0
        and n_bdu_violations_nondogmatic == 0
    )

    return {
        "outcome": "COUNTEREXAMPLES_FOUND",
        "n_tested": len(full_deviations),
        "max_deviation": float(np.max(full_deviations)) if full_deviations else 0.0,
        "mean_deviation": float(np.mean(full_deviations)) if full_deviations else 0.0,
        "std_deviation": float(np.std(full_deviations)) if full_deviations else 0.0,
        "counterexamples": counterexamples,
        "sub_properties": {
            "n_full_violations": n_violations,
            "n_bdu_violations": n_bdu_violations,
            "n_bdu_violations_from_dogmatic_path": n_bdu_violations_dogmatic,
            "n_bdu_violations_from_nondogmatic_path": n_bdu_violations_nondogmatic,
            "n_involves_dogmatic_pair": n_involves_dogmatic_pair,
            "bdu_violations_all_from_dogmatic_fallback": bdu_all_from_dogmatic,
            "max_bdu_deviation": float(np.max(bdu_deviations)) if bdu_deviations else 0.0,
            "mean_bdu_deviation": float(np.mean(bdu_deviations)) if bdu_deviations else 0.0,
            "max_base_rate_deviation": float(np.max(base_rate_deviations)) if base_rate_deviations else 0.0,
            "mean_base_rate_deviation": float(np.mean(base_rate_deviations)) if base_rate_deviations else 0.0,
        },
        "notes": (
            f"Full associativity violations: {n_violations}/{len(full_deviations)}. "
            f"BDU-only violations: {n_bdu_violations}/{len(bdu_deviations)}. "
            f"Of BDU violations: {n_bdu_violations_dogmatic} from dogmatic fallback, "
            f"{n_bdu_violations_nondogmatic} from non-dogmatic path. "
            f"BDU violations ALL from dogmatic fallback: "
            f"{'CONFIRMED' if bdu_all_from_dogmatic else 'NOT CONFIRMED'}. "
            f"FINDING: Non-associativity has two sources: "
            f"(1) base-rate averaging (always, when base rates differ), "
            f"(2) dogmatic fallback branch discontinuity "
            f"(rare, when both opinions in a pair have u=0)."
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# P5: VACUOUS IDENTITY
# ═══════════════════════════════════════════════════════════════════


def test_p5_vacuous_identity() -> dict[str, Any]:
    """Verify cumulative_fuse(a, vacuous) == a when base rates match.

    The vacuous opinion v = (0, 0, 1, a) is the identity element for
    cumulative fusion: fusing with total ignorance should not change
    the opinion.

    We also separately verify that b/d/u are preserved even when
    base rates differ (since the fusion formula doesn't use base_rate
    for computing b/d/u).
    """
    deviations_match: list[float] = []
    deviations_bdu_mismatch: list[float] = []

    # --- Matching base rates: full identity ---
    @given(a=opinion_strategy())
    @PROP_SETTINGS
    def _vac_match(a: Opinion) -> None:
        vacuous = Opinion(0.0, 0.0, 1.0, base_rate=a.base_rate)
        result = cumulative_fuse(a, vacuous)
        dev = opinion_max_deviation(a, result)
        deviations_match.append(dev)
        assert dev < FLOAT_TOL, (
            f"Vacuous identity violated: dev={dev}, a={a}, result={result}"
        )

    # --- Mismatched base rates: b/d/u should still be preserved ---
    @given(a=opinion_strategy(), vac_br=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ))
    @PROP_SETTINGS
    def _vac_mismatch(a: Opinion, vac_br: float) -> None:
        vacuous = Opinion(0.0, 0.0, 1.0, base_rate=vac_br)
        result = cumulative_fuse(a, vacuous)
        bdu_dev = opinion_bdu_max_deviation(a, result)
        deviations_bdu_mismatch.append(bdu_dev)
        assert bdu_dev < FLOAT_TOL, (
            f"BDU not preserved with vacuous (different base_rate): "
            f"dev={bdu_dev}, a={a}, result={result}"
        )

    sub_results: dict[str, str] = {}
    try:
        _vac_match()
        sub_results["matching_base_rate_full_identity"] = "PASS"
    except Exception as e:
        sub_results["matching_base_rate_full_identity"] = f"FAIL: {e}"

    try:
        _vac_mismatch()
        sub_results["mismatched_base_rate_bdu_preserved"] = "PASS"
    except Exception as e:
        sub_results["mismatched_base_rate_bdu_preserved"] = f"FAIL: {e}"

    import numpy as np
    all_devs = deviations_match + deviations_bdu_mismatch
    all_passed = all(v == "PASS" for v in sub_results.values())

    return {
        "outcome": "PASS" if all_passed else "FAIL",
        "n_tested": len(deviations_match) + len(deviations_bdu_mismatch),
        "max_deviation": float(np.max(all_devs)) if all_devs else 0.0,
        "mean_deviation": float(np.mean(all_devs)) if all_devs else 0.0,
        "std_deviation": float(np.std(all_devs)) if all_devs else 0.0,
        "sub_properties": sub_results,
        "notes": (
            f"Match: max_dev={max(deviations_match) if deviations_match else 0:.2e}. "
            f"Mismatch BDU: max_dev={max(deviations_bdu_mismatch) if deviations_bdu_mismatch else 0:.2e}."
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# P6: TRUST DISCOUNT WITH FULL TRUST = IDENTITY
# ═══════════════════════════════════════════════════════════════════


def test_p6_trust_discount_full_trust() -> dict[str, Any]:
    """Verify trust_discount(full_trust, opinion) == opinion.

    Full trust: (b=1, d=0, u=0).  Discounting by full trust should
    preserve the original opinion unchanged.

    Mathematical verification:
        b_result = 1 * b_x = b_x
        d_result = 1 * d_x = d_x
        u_result = 0 + 0 + 1 * u_x = u_x
    """
    deviations: list[float] = []

    @given(
        op=opinion_strategy(),
        trust_br=st.floats(min_value=0.0, max_value=1.0,
                            allow_nan=False, allow_infinity=False),
    )
    @PROP_SETTINGS
    def _full_trust(op: Opinion, trust_br: float) -> None:
        full_trust = Opinion(1.0, 0.0, 0.0, base_rate=trust_br)
        result = trust_discount(full_trust, op)
        # Base rate of result should be opinion's base_rate (per formula)
        dev = opinion_max_deviation(op, result)
        deviations.append(dev)
        assert dev < FLOAT_TOL, (
            f"Full trust identity violated: dev={dev}, op={op}, result={result}"
        )

    try:
        _full_trust()
        outcome = "PASS"
    except Exception as e:
        outcome = f"FAIL: {e}"

    import numpy as np
    return {
        "outcome": outcome,
        "n_tested": len(deviations),
        "max_deviation": float(np.max(deviations)) if deviations else 0.0,
        "mean_deviation": float(np.mean(deviations)) if deviations else 0.0,
        "std_deviation": float(np.std(deviations)) if deviations else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════
# P7: TRUST DISCOUNT WITH VACUOUS TRUST = VACUOUS
# ═══════════════════════════════════════════════════════════════════


def test_p7_trust_discount_vacuous_trust() -> dict[str, Any]:
    """Verify trust_discount(vacuous_trust, opinion) == vacuous.

    Vacuous trust: (b=0, d=0, u=1).  Discounting through total
    ignorance about trustworthiness should yield total ignorance
    about the proposition.

    Mathematical verification:
        b_result = 0 * b_x = 0
        d_result = 0 * d_x = 0
        u_result = 0 + 1 + 0 * u_x = 1
    """
    deviations: list[float] = []

    @given(
        op=opinion_strategy(),
        trust_br=st.floats(min_value=0.0, max_value=1.0,
                            allow_nan=False, allow_infinity=False),
    )
    @PROP_SETTINGS
    def _vac_trust(op: Opinion, trust_br: float) -> None:
        vacuous_trust = Opinion(0.0, 0.0, 1.0, base_rate=trust_br)
        result = trust_discount(vacuous_trust, op)

        # Result should be vacuous about the proposition
        expected = Opinion(0.0, 0.0, 1.0, base_rate=op.base_rate)
        dev = opinion_max_deviation(expected, result)
        deviations.append(dev)
        assert dev < FLOAT_TOL, (
            f"Vacuous trust identity violated: dev={dev}, "
            f"op={op}, result={result}, expected={expected}"
        )

    try:
        _vac_trust()
        outcome = "PASS"
    except Exception as e:
        outcome = f"FAIL: {e}"

    import numpy as np
    return {
        "outcome": outcome,
        "n_tested": len(deviations),
        "max_deviation": float(np.max(deviations)) if deviations else 0.0,
        "mean_deviation": float(np.mean(deviations)) if deviations else 0.0,
        "std_deviation": float(np.std(deviations)) if deviations else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════
# P8: DEDUCTION FACTORIZATION (COMPONENT-WISE LTP)
# ═══════════════════════════════════════════════════════════════════


def test_p8_deduction_factorization() -> dict[str, Any]:
    """Verify the component-wise law of total probability for deduction.

    For each component c in {b, d, u}, the deduction operator satisfies:

        c_Y = P(X) * c_{Y|X} + (1 - P(X)) * c_{Y|~X}

    where P(X) = b_X + a_X * u_X is the projected probability of X.

    This factorization was identified as a mathematical finding in the
    jsonld-ex project: SL deduction factorizes through P(x), meaning
    the deduced components depend on the parent ONLY through its
    projected probability, not through its individual b/d/u values.
    """
    b_deviations: list[float] = []
    d_deviations: list[float] = []
    u_deviations: list[float] = []

    @given(
        ox=opinion_strategy(),
        oy_x=opinion_strategy(),
        oy_nx=opinion_strategy(),
    )
    @PROP_SETTINGS
    def _factorization(ox: Opinion, oy_x: Opinion, oy_nx: Opinion) -> None:
        result = deduce(ox, oy_x, oy_nx)
        p_x = ox.projected_probability()
        p_nx = 1.0 - p_x

        # Component-wise LTP
        expected_b = p_x * oy_x.belief + p_nx * oy_nx.belief
        expected_d = p_x * oy_x.disbelief + p_nx * oy_nx.disbelief
        expected_u = p_x * oy_x.uncertainty + p_nx * oy_nx.uncertainty

        dev_b = abs(result.belief - expected_b)
        dev_d = abs(result.disbelief - expected_d)
        dev_u = abs(result.uncertainty - expected_u)

        b_deviations.append(dev_b)
        d_deviations.append(dev_d)
        u_deviations.append(dev_u)

        max_dev = max(dev_b, dev_d, dev_u)
        assert max_dev < FLOAT_TOL, (
            f"Factorization violated: max_dev={max_dev}, "
            f"ox={ox}, oy_x={oy_x}, oy_nx={oy_nx}, result={result}, "
            f"expected_b={expected_b}, expected_d={expected_d}, expected_u={expected_u}"
        )

    try:
        _factorization()
        outcome = "PASS"
    except Exception as e:
        outcome = f"FAIL: {e}"

    import numpy as np
    all_devs = b_deviations + d_deviations + u_deviations

    return {
        "outcome": outcome,
        "n_tested": len(b_deviations),
        "max_deviation": float(np.max(all_devs)) if all_devs else 0.0,
        "mean_deviation": float(np.mean(all_devs)) if all_devs else 0.0,
        "std_deviation": float(np.std(all_devs)) if all_devs else 0.0,
        "sub_properties": {
            "belief_max_dev": float(np.max(b_deviations)) if b_deviations else 0.0,
            "disbelief_max_dev": float(np.max(d_deviations)) if d_deviations else 0.0,
            "uncertainty_max_dev": float(np.max(u_deviations)) if u_deviations else 0.0,
        },
        "notes": (
            "Component-wise LTP: c_Y = P(X)*c_{Y|X} + (1-P(X))*c_{Y|~X} "
            "verified for c in {b, d, u}. This confirms the deduction "
            "factorization finding: SL deduction depends on the parent "
            "only through its projected probability P(X)."
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# P9: DECAY MONOTONICITY
# ═══════════════════════════════════════════════════════════════════


def test_p9_decay_monotonicity() -> dict[str, Any]:
    """Verify that uncertainty is monotonically non-decreasing with elapsed time.

    For any opinion and half_life, if t1 < t2 then:
        decay(op, t1).uncertainty <= decay(op, t2).uncertainty

    Tested for exponential_decay and linear_decay.
    (step_decay has a discontinuity at t=half_life but is trivially
    monotonic within each region.)
    """
    deviations_exp: list[float] = []
    deviations_lin: list[float] = []

    @given(
        op=opinion_strategy(),
        t1=positive_float_strategy(min_value=0.0, max_value=1e4),
        delta=positive_float_strategy(min_value=1e-6, max_value=1e4),
        half_life=positive_float_strategy(min_value=1e-3, max_value=1e4),
    )
    @PROP_SETTINGS
    def _mono_exp(op: Opinion, t1: float, delta: float, half_life: float) -> None:
        t2 = t1 + delta
        d1 = decay_opinion(op, t1, half_life, exponential_decay)
        d2 = decay_opinion(op, t2, half_life, exponential_decay)
        violation = d1.uncertainty - d2.uncertainty  # Should be <= 0
        deviations_exp.append(max(0.0, violation))
        assert violation <= FLOAT_TOL, (
            f"Monotonicity violated (exponential): u(t1={t1})={d1.uncertainty} > "
            f"u(t2={t2})={d2.uncertainty}, diff={violation}"
        )

    @given(
        op=opinion_strategy(),
        t1=positive_float_strategy(min_value=0.0, max_value=1e4),
        delta=positive_float_strategy(min_value=1e-6, max_value=1e4),
        half_life=positive_float_strategy(min_value=1e-3, max_value=1e4),
    )
    @PROP_SETTINGS
    def _mono_lin(op: Opinion, t1: float, delta: float, half_life: float) -> None:
        t2 = t1 + delta
        d1 = decay_opinion(op, t1, half_life, linear_decay)
        d2 = decay_opinion(op, t2, half_life, linear_decay)
        violation = d1.uncertainty - d2.uncertainty  # Should be <= 0
        deviations_lin.append(max(0.0, violation))
        assert violation <= FLOAT_TOL, (
            f"Monotonicity violated (linear): u(t1={t1})={d1.uncertainty} > "
            f"u(t2={t2})={d2.uncertainty}, diff={violation}"
        )

    sub_results: dict[str, str] = {}
    try:
        _mono_exp()
        sub_results["exponential_decay"] = "PASS"
    except Exception as e:
        sub_results["exponential_decay"] = f"FAIL: {e}"

    try:
        _mono_lin()
        sub_results["linear_decay"] = "PASS"
    except Exception as e:
        sub_results["linear_decay"] = f"FAIL: {e}"

    import numpy as np
    all_devs = deviations_exp + deviations_lin
    all_passed = all(v == "PASS" for v in sub_results.values())

    return {
        "outcome": "PASS" if all_passed else "FAIL",
        "n_tested": len(deviations_exp) + len(deviations_lin),
        "max_deviation": float(np.max(all_devs)) if all_devs else 0.0,
        "mean_deviation": float(np.mean(all_devs)) if all_devs else 0.0,
        "std_deviation": float(np.std(all_devs)) if all_devs else 0.0,
        "sub_properties": sub_results,
    }


# ═══════════════════════════════════════════════════════════════════
# P10: DECAY CONVERGENCE TO VACUOUS
# ═══════════════════════════════════════════════════════════════════


def test_p10_decay_to_vacuous() -> dict[str, Any]:
    """Verify that as elapsed time -> infinity, the opinion -> vacuous.

    Exponential decay: as t -> inf, lambda -> 0, so b -> 0, d -> 0, u -> 1.
    Linear decay: at t >= 2*half_life, lambda = 0, so b = 0, d = 0, u = 1.

    We test with very large t values and verify convergence.
    """
    deviations_exp: list[float] = []
    deviations_lin: list[float] = []

    @given(op=opinion_strategy())
    @PROP_SETTINGS
    def _vac_exp(op: Opinion) -> None:
        # Very large elapsed time (100 half-lives should give lambda ~ 2^-100 ~ 0)
        half_life = 100.0
        elapsed = 100.0 * half_life  # 100 half-lives
        result = decay_opinion(op, elapsed, half_life, exponential_decay)
        vacuous = Opinion(0.0, 0.0, 1.0, base_rate=op.base_rate)
        dev = opinion_max_deviation(vacuous, result)
        deviations_exp.append(dev)
        # With 100 half-lives, lambda = 2^-100 ~ 7.9e-31
        # Belief and disbelief should be essentially 0
        assert dev < 1e-10, (
            f"Exponential decay did not converge to vacuous: dev={dev}, "
            f"result={result}"
        )

    @given(op=opinion_strategy())
    @PROP_SETTINGS
    def _vac_lin(op: Opinion) -> None:
        half_life = 100.0
        elapsed = 2.0 * half_life  # Exactly at full decay
        result = decay_opinion(op, elapsed, half_life, linear_decay)
        vacuous = Opinion(0.0, 0.0, 1.0, base_rate=op.base_rate)
        dev = opinion_max_deviation(vacuous, result)
        deviations_lin.append(dev)
        assert dev < FLOAT_TOL, (
            f"Linear decay did not reach vacuous at 2*half_life: dev={dev}, "
            f"result={result}"
        )

    sub_results: dict[str, str] = {}
    try:
        _vac_exp()
        sub_results["exponential_to_vacuous"] = "PASS"
    except Exception as e:
        sub_results["exponential_to_vacuous"] = f"FAIL: {e}"

    try:
        _vac_lin()
        sub_results["linear_to_vacuous"] = "PASS"
    except Exception as e:
        sub_results["linear_to_vacuous"] = f"FAIL: {e}"

    import numpy as np
    all_devs = deviations_exp + deviations_lin
    all_passed = all(v == "PASS" for v in sub_results.values())

    return {
        "outcome": "PASS" if all_passed else "FAIL",
        "n_tested": len(deviations_exp) + len(deviations_lin),
        "max_deviation": float(np.max(all_devs)) if all_devs else 0.0,
        "mean_deviation": float(np.mean(all_devs)) if all_devs else 0.0,
        "std_deviation": float(np.std(all_devs)) if all_devs else 0.0,
        "sub_properties": sub_results,
    }


# ═══════════════════════════════════════════════════════════════════
# P11: PROJECTED PROBABILITY CONSISTENCY
# ═══════════════════════════════════════════════════════════════════


def test_p11_projected_probability_consistency() -> dict[str, Any]:
    """Verify P(omega) = b + a * u for all valid opinions.

    This is definitionally true, but we verify the implementation
    matches the mathematical definition across a wide range of inputs.
    We also verify the result is in [0, 1].
    """
    deviations: list[float] = []

    @given(op=opinion_strategy())
    @PROP_SETTINGS
    def _pp(op: Opinion) -> None:
        pp = op.projected_probability()
        expected = op.belief + op.base_rate * op.uncertainty
        dev = abs(pp - expected)
        deviations.append(dev)
        assert dev < FLOAT_TOL, (
            f"P(w) != b + a*u: P={pp}, expected={expected}, dev={dev}, op={op}"
        )
        # Range check
        assert -FLOAT_TOL <= pp <= 1.0 + FLOAT_TOL, (
            f"P(w) out of [0,1]: P={pp}, op={op}"
        )

    try:
        _pp()
        outcome = "PASS"
    except Exception as e:
        outcome = f"FAIL: {e}"

    import numpy as np
    return {
        "outcome": outcome,
        "n_tested": len(deviations),
        "max_deviation": float(np.max(deviations)) if deviations else 0.0,
        "mean_deviation": float(np.mean(deviations)) if deviations else 0.0,
        "std_deviation": float(np.std(deviations)) if deviations else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    """Run all 11 property verifications and save results."""
    print("=" * 70)
    print("EN7.1 — Property-Based Verification of Confidence Algebra")
    print("=" * 70)

    # ── Reproducibility setup ──
    set_global_seed(GLOBAL_SEED)
    env = log_environment()
    print(f"  Python:      {env['python_version']}")
    print(f"  Platform:    {env['platform']}")
    print(f"  jsonld-ex:   {env.get('jsonld_ex_version', 'unknown')}")
    print(f"  NumPy:       {env.get('numpy_version', 'unknown')}")
    print(f"  Seed:        {GLOBAL_SEED}")
    print(f"  Examples/property: {N_EXAMPLES}")
    print(f"  Float tolerance:   {FLOAT_TOL}")
    print("-" * 70)

    t_start = time.perf_counter()
    results: list[PropertyResult] = []

    # ── P1: Validity Invariant ──
    results.append(run_property(
        "P1", "Opinion Validity Invariant",
        "For all operators: output satisfies b+d+u=1 and all components in [0,1]",
        "PASS", test_p1_validity_invariant,
    ))

    # ── P2: Fusion Commutativity ──
    results.append(run_property(
        "P2", "Fusion Commutativity",
        "cumulative_fuse(a,b) == cumulative_fuse(b,a) for all opinions a,b",
        "PASS", test_p2_fusion_commutativity,
    ))

    # ── P3: Fusion Associativity (Equal Base Rates) ──
    results.append(run_property(
        "P3", "Fusion Associativity (Equal Base Rates)",
        "(a fuse b) fuse c == a fuse (b fuse c) when all base_rates equal",
        "PASS", test_p3_fusion_associativity_equal_base_rates,
    ))

    # ── P4: Fusion Non-Associativity (Different Base Rates) ──
    results.append(run_property(
        "P4", "Fusion Non-Associativity (Different Base Rates)",
        "Document counterexamples where associativity fails; characterize source",
        "COUNTEREXAMPLES", test_p4_fusion_non_associativity,
    ))

    # ── P5: Vacuous Identity ──
    results.append(run_property(
        "P5", "Vacuous Opinion as Fusion Identity",
        "cumulative_fuse(a, vacuous) == a when base_rates match",
        "PASS", test_p5_vacuous_identity,
    ))

    # ── P6: Trust Discount Full Trust ──
    results.append(run_property(
        "P6", "Trust Discount with Full Trust = Identity",
        "trust_discount((1,0,0,a), opinion) == opinion for all opinions",
        "PASS", test_p6_trust_discount_full_trust,
    ))

    # ── P7: Trust Discount Vacuous Trust ──
    results.append(run_property(
        "P7", "Trust Discount with Vacuous Trust = Vacuous",
        "trust_discount((0,0,1,a), opinion) == (0,0,1,opinion.base_rate)",
        "PASS", test_p7_trust_discount_vacuous_trust,
    ))

    # ── P8: Deduction Factorization ──
    results.append(run_property(
        "P8", "Deduction Factorization (Component-wise LTP)",
        "c_Y = P(X)*c_{Y|X} + (1-P(X))*c_{Y|~X} for c in {b,d,u}",
        "PASS", test_p8_deduction_factorization,
    ))

    # ── P9: Decay Monotonicity ──
    results.append(run_property(
        "P9", "Decay Monotonicity",
        "t1 < t2 => uncertainty(decay(op,t1)) <= uncertainty(decay(op,t2))",
        "PASS", test_p9_decay_monotonicity,
    ))

    # ── P10: Decay to Vacuous ──
    results.append(run_property(
        "P10", "Decay Convergence to Vacuous",
        "As elapsed -> infinity, opinion -> vacuous for exp and linear decay",
        "PASS", test_p10_decay_to_vacuous,
    ))

    # ── P11: Projected Probability Consistency ──
    results.append(run_property(
        "P11", "Projected Probability Consistency",
        "P(omega) = b + a*u for all valid opinions, result in [0,1]",
        "PASS", test_p11_projected_probability_consistency,
    ))

    total_time = time.perf_counter() - t_start

    # ── Summary ──
    print("-" * 70)
    n_pass = sum(1 for r in results if r.actual_outcome == "PASS")
    n_counter = sum(1 for r in results if r.actual_outcome == "COUNTEREXAMPLES_FOUND")
    n_fail = sum(1 for r in results if r.actual_outcome.startswith("FAIL"))
    total_examples = sum(r.n_examples_tested for r in results)

    print(f"\n  SUMMARY: {n_pass} PASS, {n_counter} COUNTEREXAMPLES (expected), {n_fail} FAIL")
    print(f"  Total examples tested: {total_examples:,}")
    print(f"  Total wall time: {total_time:.1f}s")

    if n_fail > 0:
        print("\n  FAILURES:")
        for r in results:
            if r.actual_outcome.startswith("FAIL"):
                print(f"    {r.property_id}: {r.actual_outcome}")

    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "en7_1_results.json"

    experiment_result = ExperimentResult(
        experiment_id="EN7.1",
        parameters={
            "global_seed": GLOBAL_SEED,
            "n_examples_per_property": N_EXAMPLES,
            "float_tolerance": FLOAT_TOL,
            "validity_tolerance": VALIDITY_TOL,
            "hypothesis_derandomize": True,
            "hypothesis_phases": ["explicit", "generate"],
        },
        metrics={
            "n_properties_tested": len(results),
            "n_pass": n_pass,
            "n_counterexamples_expected": n_counter,
            "n_fail": n_fail,
            "total_examples_tested": total_examples,
            "total_wall_time_seconds": round(total_time, 4),
            "all_expected_outcomes_met": (n_fail == 0),
        },
        raw_data={
            "property_results": [r.to_dict() for r in results],
        },
        environment=env,
        notes=(
            "EN7.1: Property-based verification of jsonld-ex confidence algebra "
            f"using Hypothesis framework. {len(results)} properties verified "
            f"with {N_EXAMPLES} random opinions per property. "
            f"All expected outcomes {'MET' if n_fail == 0 else 'NOT MET'}."
        ),
    )
    experiment_result.save_json(str(output_path))
    print(f"\n  Results saved to: {output_path}")

    # ── Timestamped archive ──
    ts = experiment_result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    archive_path = RESULTS_DIR / f"en7_1_results_{ts}.json"
    experiment_result.save_json(str(archive_path))
    print(f"  Archived to:      {archive_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()