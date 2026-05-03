"""EN4.2 — Dempster-Shafer vs Subjective Logic: Empirical Comparison.

Implements classical Dempster's Rule, Yager's Rule, and comparison
pipeline against jsonld-ex's Subjective Logic fusion operators.

References:
    Dempster (1967) "Upper and lower probabilities induced by a
        multivalued mapping"
    Shafer (1976) "A Mathematical Theory of Evidence"
    Zadeh (1979) "On the validity of Dempster's rule of combination"
    Yager (1987) "On the Dempster-Shafer framework and new combination
        rules"
    Jøsang (2016) "Subjective Logic" §12.3

All combination rules operate on binary-frame mass functions:
    m = {b, d, u}  where  b + d + u = 1
    b = m(entity),  d = m(¬entity),  u = m(Θ)  (ignorance)
"""
from __future__ import annotations

from typing import Any

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    conflict_metric,
)


# ====================================================================
# Mass Function Helpers
# ====================================================================

def confidence_to_mass(score: float, uncertainty: float) -> dict[str, float]:
    """Convert a scalar confidence score to a binary mass function.

    Uses the same construction as ``Opinion.from_confidence``:
        b = score * (1 - u)
        d = (1 - score) * (1 - u)
        u = uncertainty

    Args:
        score: Model confidence in [0, 1].
        uncertainty: Epistemic uncertainty in [0, 1].

    Returns:
        Dict with keys 'b', 'd', 'u' summing to 1.
    """
    evidence_weight = 1.0 - uncertainty
    b = score * evidence_weight
    d = (1.0 - score) * evidence_weight
    return {"b": b, "d": d, "u": uncertainty}


def mass_to_decision(m: dict[str, float]) -> bool:
    """DS decision rule: predict entity if m(entity) > m(¬entity).

    This is the standard DS decision without any prior adjustment.
    Ties broken in favour of entity (b >= d).
    """
    return m["b"] >= m["d"]


def sl_to_decision(
    op: Opinion,
    threshold: float = 0.5,
    base_rate: float | None = None,
) -> bool:
    """SL decision rule: predict entity if P = b + a*u > threshold.

    Args:
        op: Subjective Logic opinion.
        threshold: Decision threshold on projected probability.
        base_rate: Override for the opinion's base rate. If None, uses
            op.a.

    Returns:
        True if projected probability exceeds threshold.
    """
    a = base_rate if base_rate is not None else op.base_rate
    projected = op.belief + a * op.uncertainty
    return projected > threshold


# ====================================================================
# Conflict Measurement
# ====================================================================

def compute_conflict_K(m1: dict[str, float], m2: dict[str, float]) -> float:
    """Compute DS conflict mass between two binary mass functions.

    K = b₁d₂ + d₁b₂

    This is the total mass assigned to the empty set under the
    conjunctive rule — the amount of evidence that is in direct
    opposition.

    Returns:
        Conflict mass K ∈ [0, 1].
    """
    return m1["b"] * m2["d"] + m1["d"] * m2["b"]


# ====================================================================
# Classical Dempster's Rule (Dempster 1967, Shafer 1976)
# ====================================================================

def dempster_combine(
    m1: dict[str, float],
    m2: dict[str, float],
) -> dict[str, float]:
    """Combine two binary mass functions using Dempster's rule.

    For binary frame {entity, ¬entity}:
        K       = b₁d₂ + d₁b₂
        m(ent)  = (b₁b₂ + b₁u₂ + u₁b₂) / (1 - K)
        m(¬ent) = (d₁d₂ + d₁u₂ + u₁d₂) / (1 - K)
        m(Θ)    = u₁u₂ / (1 - K)

    Args:
        m1, m2: Mass functions with keys 'b', 'd', 'u'.

    Returns:
        Combined mass function.

    Raises:
        ValueError: If K >= 1 (total conflict — rule is undefined).
    """
    b1, d1, u1 = m1["b"], m1["d"], m1["u"]
    b2, d2, u2 = m2["b"], m2["d"], m2["u"]

    K = b1 * d2 + d1 * b2

    if K >= 1.0 - 1e-15:
        raise ValueError(
            f"Total conflict (K={K:.6f}): Dempster's rule is undefined."
        )

    norm = 1.0 - K

    b_out = (b1 * b2 + b1 * u2 + u1 * b2) / norm
    d_out = (d1 * d2 + d1 * u2 + u1 * d2) / norm
    u_out = (u1 * u2) / norm

    return {"b": b_out, "d": d_out, "u": u_out}


def dempster_combine_multi(
    masses: list[dict[str, float]],
) -> dict[str, float]:
    """Sequential pairwise Dempster combination for n sources.

    Note: Dempster's rule is associative, so sequential pairwise
    application produces the same result regardless of order.

    Args:
        masses: List of mass functions (≥2).

    Returns:
        Combined mass function.
    """
    if len(masses) < 2:
        raise ValueError("Need at least 2 mass functions to combine.")

    result = masses[0].copy()
    for m in masses[1:]:
        result = dempster_combine(result, m)
    return result


# ====================================================================
# Yager's Rule (Yager 1987)
# ====================================================================

def yager_combine(
    m1: dict[str, float],
    m2: dict[str, float],
) -> dict[str, float]:
    """Combine two binary mass functions using Yager's rule.

    Unlike Dempster, conflict mass is transferred to the universal
    set (ignorance) rather than being normalized away:

        m(ent)  = b₁b₂ + b₁u₂ + u₁b₂
        m(¬ent) = d₁d₂ + d₁u₂ + u₁d₂
        m(Θ)    = u₁u₂ + K

    where K = b₁d₂ + d₁b₂ is the conflict mass.

    Args:
        m1, m2: Mass functions with keys 'b', 'd', 'u'.

    Returns:
        Combined mass function.
    """
    b1, d1, u1 = m1["b"], m1["d"], m1["u"]
    b2, d2, u2 = m2["b"], m2["d"], m2["u"]

    K = b1 * d2 + d1 * b2

    b_out = b1 * b2 + b1 * u2 + u1 * b2
    d_out = d1 * d2 + d1 * u2 + u1 * d2
    u_out = u1 * u2 + K

    return {"b": b_out, "d": d_out, "u": u_out}


def yager_combine_multi(
    masses: list[dict[str, float]],
) -> dict[str, float]:
    """Sequential pairwise Yager combination for n sources.

    Note: Yager's rule is NOT generally associative, so order may
    matter. We use left-to-right sequential application consistent
    with the standard convention.

    Args:
        masses: List of mass functions (≥2).

    Returns:
        Combined mass function.
    """
    if len(masses) < 2:
        raise ValueError("Need at least 2 mass functions to combine.")

    result = masses[0].copy()
    for m in masses[1:]:
        result = yager_combine(result, m)
    return result


# ====================================================================
# Comparison Pipeline
# ====================================================================

def compare_fusion_methods_binary(
    scores: list[float],
    uncertainties: list[float],
    base_rate: float = 0.5,
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Compare all four fusion methods on a single entity position.

    Given n source confidence scores and uncertainties, constructs
    mass functions / SL opinions and applies each combination rule.

    Args:
        scores: Per-source confidence scores.
        uncertainties: Per-source epistemic uncertainties.
        base_rate: SL base rate parameter.
        threshold: SL decision threshold on projected probability.

    Returns:
        Dict keyed by method name, each containing:
            b, d, u: fused mass/opinion components
            decision: bool (entity or not)
    """
    if len(scores) != len(uncertainties):
        raise ValueError("scores and uncertainties must have same length.")
    if len(scores) < 2:
        raise ValueError("Need at least 2 sources.")

    # Build mass functions and SL opinions
    masses = [confidence_to_mass(s, u) for s, u in zip(scores, uncertainties)]
    opinions = [
        Opinion(belief=m["b"], disbelief=m["d"], uncertainty=m["u"], base_rate=base_rate) for m in masses
    ]

    # --- Classical Dempster ---
    ds_fused = dempster_combine_multi(masses)
    ds_decision = mass_to_decision(ds_fused)

    # --- Yager ---
    yager_fused = yager_combine_multi(masses)
    yager_decision = mass_to_decision(yager_fused)

    # --- SL Cumulative ---
    sl_cum = opinions[0]
    for op in opinions[1:]:
        sl_cum = cumulative_fuse(sl_cum, op)
    sl_cum_decision = sl_to_decision(sl_cum, threshold=threshold, base_rate=base_rate)

    # --- SL Averaging ---
    sl_avg = opinions[0]
    for op in opinions[1:]:
        sl_avg = averaging_fuse(sl_avg, op)
    sl_avg_decision = sl_to_decision(sl_avg, threshold=threshold, base_rate=base_rate)

    return {
        "ds_classical": {
            "b": ds_fused["b"],
            "d": ds_fused["d"],
            "u": ds_fused["u"],
            "decision": ds_decision,
        },
        "ds_yager": {
            "b": yager_fused["b"],
            "d": yager_fused["d"],
            "u": yager_fused["u"],
            "decision": yager_decision,
        },
        "sl_cumulative": {
            "b": sl_cum.belief,
            "d": sl_cum.disbelief,
            "u": sl_cum.uncertainty,
            "decision": sl_cum_decision,
        },
        "sl_averaging": {
            "b": sl_avg.belief,
            "d": sl_avg.disbelief,
            "u": sl_avg.uncertainty,
            "decision": sl_avg_decision,
        },
    }
