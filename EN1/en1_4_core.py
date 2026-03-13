"""EN1.4 Core — Trust Discount Chain Analysis.

Pure computation module (no I/O, no API calls).

Comprehensive analysis of trust propagation through provenance chains,
demonstrating the fundamental difference between SL trust discount
(converges to base rate) and scalar trust multiplication (converges to 0).

Mathematical foundation:
  SL trust discount (Jøsang 2016 §14.3):
    b_n = b_trust^n × b_0
    d_n = b_trust^n × d_0
    u_n = 1 − b_trust^n × (1 − u_0)

  Therefore:
    P(ω_n) = b_n + a × u_n
           = b_trust^n × b_0 + a × (1 − b_trust^n × (1 − u_0))
           = a + b_trust^n × (b_0 + a × u_0 − a)
           = a + b_trust^n × (P(ω_0) − a)

  This converges to a (the base rate) as n → ∞.

  Scalar: c_n = t^n × c_0 → 0 as n → ∞.

  The difference: SL preserves the prior (base rate) in the limit,
  while scalar trust discount destroys all information.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from jsonld_ex.confidence_algebra import (
    Opinion,
    trust_discount,
    cumulative_fuse,
)


# ═══════════════════════════════════════════════════════════════════
# 1. Core chain computation
# ═══════════════════════════════════════════════════════════════════

def compute_sl_chain(
    trust: Opinion,
    original: Opinion,
    max_length: int = 20,
) -> List[Opinion]:
    """Compute SL trust discount chain: apply trust_discount iteratively.

    Returns: List of opinions [ω_0, ω_1, ..., ω_n] where ω_0 = original
    and ω_{i+1} = trust_discount(trust, ω_i).
    """
    chain = [original]
    current = original
    for _ in range(max_length):
        current = trust_discount(trust, current)
        chain.append(current)
    return chain


def compute_scalar_chain(
    trust_level: float,
    original_confidence: float,
    max_length: int = 20,
) -> List[float]:
    """Compute scalar trust chain: c_n = t^n × c_0.

    Returns: List of confidence values [c_0, c_1, ..., c_n].
    """
    chain = [original_confidence]
    for n in range(1, max_length + 1):
        chain.append(trust_level ** n * original_confidence)
    return chain


def compute_bayesian_chain(
    trust_reliability: float,
    original_prob: float,
    max_length: int = 20,
    prior: float = 0.5,
) -> List[float]:
    """Compute Bayesian intermediary chain.

    Each intermediary is a noisy channel with reliability r:
      P(report=1|true=1) = r
      P(report=0|true=0) = r

    After n intermediaries:
      P(x|reports) converges to prior via iterated Bayesian updating.

    This models the "telephone game" where each intermediary may
    flip the message with probability (1-r).
    """
    chain = [original_prob]
    current = original_prob
    for _ in range(max_length):
        # Bayesian update: intermediary reports with reliability trust_reliability
        # P(x=1|report=1) = r*p / (r*p + (1-r)*(1-p))
        # But the intermediary reports the previous posterior, so:
        # Effective: p_new = r * p_old + (1-r) * (1 - p_old)
        #          = (2r - 1) * p_old + (1 - r)
        # This is a linear map that converges to 0.5 when r < 1
        current = trust_reliability * current + (1 - trust_reliability) * (1 - current)
        chain.append(current)
    return chain


# ═══════════════════════════════════════════════════════════════════
# 2. Closed-form verification
# ═══════════════════════════════════════════════════════════════════

def sl_chain_closed_form(
    b_trust: float,
    p0: float,
    base_rate: float,
    n: int,
) -> float:
    """Closed-form projected probability at step n.

    P(ω_n) = a + b_trust^n × (P(ω_0) − a)
    """
    return base_rate + (b_trust ** n) * (p0 - base_rate)


def sl_uncertainty_closed_form(
    b_trust: float,
    u0: float,
    n: int,
) -> float:
    """Closed-form uncertainty at step n.

    u_n = 1 − b_trust^n × (1 − u_0)
    """
    return 1.0 - (b_trust ** n) * (1.0 - u0)


# ═══════════════════════════════════════════════════════════════════
# 3. Heterogeneous chains
# ═══════════════════════════════════════════════════════════════════

def compute_heterogeneous_sl_chain(
    trusts: List[Opinion],
    original: Opinion,
) -> List[Opinion]:
    """Compute SL chain with different trust opinions at each link.

    Returns: [ω_0, ω_1, ..., ω_n] where ω_{i+1} = trust_discount(trusts[i], ω_i).
    """
    chain = [original]
    current = original
    for t in trusts:
        current = trust_discount(t, current)
        chain.append(current)
    return chain


# ═══════════════════════════════════════════════════════════════════
# 4. Branching provenance
# ═══════════════════════════════════════════════════════════════════

def compute_branching_provenance(
    original: Opinion,
    trusts_per_path: List[List[Opinion]],
) -> Dict[str, Any]:
    """Compute branching provenance: multiple trust paths fused.

    Each path independently propagates the original opinion through
    its trust chain, then all path endpoints are fused via cumulative_fuse.

    This models the real-world scenario where information reaches you
    through multiple independent channels.

    Returns: {paths: [endpoint_opinions], fused: Opinion}
    """
    path_endpoints = []
    for trusts in trusts_per_path:
        chain = compute_heterogeneous_sl_chain(trusts, original)
        path_endpoints.append(chain[-1])

    fused = cumulative_fuse(*path_endpoints)

    return {
        "paths": path_endpoints,
        "fused": fused,
    }


# ═══════════════════════════════════════════════════════════════════
# 5. Decision divergence
# ═══════════════════════════════════════════════════════════════════

def compute_decision_divergence_point(
    trust: Opinion,
    trust_scalar: float,
    original: Opinion,
    threshold: float = 0.5,
    max_length: int = 50,
) -> Optional[Dict[str, Any]]:
    """Find the chain length where SL and scalar make different binary decisions.

    Binary decision: positive if P(ω) > threshold (SL) or c > threshold (scalar).

    Returns: {divergence_length, sl_prob, scalar_prob, sl_decision, scalar_decision}
    or None if no divergence within max_length.
    """
    sl_chain = compute_sl_chain(trust, original, max_length=max_length)
    scalar_chain = compute_scalar_chain(
        trust_scalar, original.projected_probability(), max_length=max_length,
    )

    for n in range(max_length + 1):
        sl_p = sl_chain[n].projected_probability()
        sc_p = scalar_chain[n]
        sl_decision = sl_p > threshold
        sc_decision = sc_p > threshold

        if sl_decision != sc_decision:
            return {
                "divergence_length": n,
                "sl_prob": sl_p,
                "scalar_prob": sc_p,
                "sl_decision": sl_decision,
                "scalar_decision": sc_decision,
            }

    return None


# ═══════════════════════════════════════════════════════════════════
# 6. Information content (entropy)
# ═══════════════════════════════════════════════════════════════════

def compute_opinion_entropy(opinion: Opinion) -> float:
    """Compute Shannon entropy of the (b, d, u) distribution.

    H(b, d, u) = −Σ p_i × log2(p_i) for p_i ∈ {b, d, u}

    This measures the information content of the opinion triple.
    A vacuous opinion (0, 0, 1) has zero entropy (all mass in u).
    A balanced dogmatic opinion (0.5, 0.5, 0) has 1 bit.
    """
    components = [opinion.belief, opinion.disbelief, opinion.uncertainty]
    entropy = 0.0
    for p in components:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_information_loss_curve(
    trust: Opinion,
    trust_scalar: float,
    original: Opinion,
    max_length: int = 20,
) -> List[Dict[str, Any]]:
    """Compute entropy along both SL and scalar chains.

    For scalar, we compute entropy of the implied (c, 1-c) distribution.

    Returns: List of {step, sl_entropy, scalar_entropy, sl_prob, scalar_prob, sl_uncertainty}
    """
    sl_chain = compute_sl_chain(trust, original, max_length=max_length)
    scalar_chain = compute_scalar_chain(
        trust_scalar, original.projected_probability(), max_length=max_length,
    )

    curve = []
    for n in range(max_length + 1):
        sl_op = sl_chain[n]
        sl_ent = compute_opinion_entropy(sl_op)

        sc_p = scalar_chain[n]
        # Scalar entropy: H(c, 1-c) — binary entropy
        if 0 < sc_p < 1:
            sc_ent = -sc_p * math.log2(sc_p) - (1 - sc_p) * math.log2(1 - sc_p)
        else:
            sc_ent = 0.0

        curve.append({
            "step": n,
            "sl_entropy": sl_ent,
            "scalar_entropy": sc_ent,
            "sl_prob": sl_op.projected_probability(),
            "scalar_prob": sc_p,
            "sl_uncertainty": sl_op.uncertainty,
        })

    return curve


# ═══════════════════════════════════════════════════════════════════
# 7. Sweep functions
# ═══════════════════════════════════════════════════════════════════

def run_trust_level_sweep(
    trust_levels: List[float],
    original: Opinion,
    max_length: int = 20,
) -> Dict[float, Dict[str, Any]]:
    """Sweep trust levels, compute chains for each.

    Returns: {trust_level: {sl_probs, scalar_probs, sl_uncertainties, sl_beliefs}}
    """
    results = {}
    for tl in trust_levels:
        trust = Opinion(tl, (1.0 - tl) * 0.5, (1.0 - tl) * 0.5)
        sl_chain = compute_sl_chain(trust, original, max_length=max_length)
        scalar_chain = compute_scalar_chain(
            tl, original.projected_probability(), max_length=max_length,
        )
        results[tl] = {
            "sl_probs": [op.projected_probability() for op in sl_chain],
            "scalar_probs": scalar_chain,
            "sl_uncertainties": [op.uncertainty for op in sl_chain],
            "sl_beliefs": [op.belief for op in sl_chain],
        }
    return results


def run_original_opinion_sweep(
    trust: Opinion,
    trust_scalar: float,
    originals: List[Opinion],
    max_length: int = 20,
) -> List[Dict[str, Any]]:
    """Sweep original opinions, compute chains for each.

    Returns: List of {original_p, base_rate, sl_probs, scalar_probs, divergence}
    """
    results = []
    for orig in originals:
        sl_chain = compute_sl_chain(trust, orig, max_length=max_length)
        scalar_chain = compute_scalar_chain(
            trust_scalar, orig.projected_probability(), max_length=max_length,
        )
        div = compute_decision_divergence_point(
            trust, trust_scalar, orig, max_length=max_length,
        )
        results.append({
            "original_p": orig.projected_probability(),
            "base_rate": orig.base_rate,
            "sl_probs": [op.projected_probability() for op in sl_chain],
            "scalar_probs": scalar_chain,
            "divergence": div,
        })
    return results


def run_base_rate_sweep(
    trust: Opinion,
    trust_scalar: float,
    base_rates: List[float],
    max_length: int = 20,
) -> Dict[float, Dict[str, Any]]:
    """Sweep base rates, showing SL converges to each while scalar always → 0.

    For each base rate, constructs an original opinion with the same
    projected probability (0.85) but different base rate.
    """
    results = {}
    for br in base_rates:
        # Construct opinion with P(ω) = 0.85 and base_rate = br
        # P(ω) = b + a*u = 0.85
        # Choose: b=0.80, u=0.05/(br if br > 0 else 1), but let's use
        # a fixed (b, d, u) and vary only the base rate
        orig = Opinion(0.80, 0.10, 0.10, base_rate=br)
        sl_chain = compute_sl_chain(trust, orig, max_length=max_length)
        scalar_chain = compute_scalar_chain(
            trust_scalar, orig.projected_probability(), max_length=max_length,
        )
        results[br] = {
            "sl_probs": [op.projected_probability() for op in sl_chain],
            "scalar_probs": scalar_chain,
            "sl_uncertainties": [op.uncertainty for op in sl_chain],
            "original_p": orig.projected_probability(),
        }
    return results


def run_chain_length_comparison(
    trust: Opinion,
    trust_scalar: float,
    original: Opinion,
    max_length: int = 30,
) -> Dict[str, Any]:
    """Comprehensive comparison of SL, scalar, and Bayesian chains.

    Returns: {sl_chain, scalar_chain, bayesian_chain, divergence_point,
              sl_half_life, scalar_half_life, bayesian_half_life}
    """
    sl_chain_ops = compute_sl_chain(trust, original, max_length=max_length)
    sl_probs = [op.projected_probability() for op in sl_chain_ops]
    sl_uncertainties = [op.uncertainty for op in sl_chain_ops]

    scalar_chain = compute_scalar_chain(
        trust_scalar, original.projected_probability(), max_length=max_length,
    )

    bayesian_chain = compute_bayesian_chain(
        trust_scalar, original.projected_probability(),
        max_length=max_length, prior=original.base_rate,
    )

    # Divergence point
    div = compute_decision_divergence_point(
        trust, trust_scalar, original, max_length=max_length,
    )

    # Half-life: steps until P(ω) is halfway between start and limit
    p0 = original.projected_probability()
    base = original.base_rate

    # SL half-life: steps until P(ω_n) = (p0 + base) / 2
    sl_target = (p0 + base) / 2
    sl_half = None
    for n, p in enumerate(sl_probs):
        if abs(p - base) <= abs(p0 - base) / 2:
            sl_half = n
            break

    # Scalar half-life: steps until c_n = c_0 / 2
    sc_target = p0 / 2
    sc_half = None
    for n, c in enumerate(scalar_chain):
        if c <= sc_target:
            sc_half = n
            break

    # Bayesian half-life: steps until P = (p0 + prior) / 2
    bay_target = (p0 + base) / 2
    bay_half = None
    for n, p in enumerate(bayesian_chain):
        if abs(p - base) <= abs(p0 - base) / 2:
            bay_half = n
            break

    return {
        "sl_chain": sl_probs,
        "sl_uncertainties": sl_uncertainties,
        "scalar_chain": scalar_chain,
        "bayesian_chain": bayesian_chain,
        "divergence_point": div,
        "sl_half_life": sl_half,
        "scalar_half_life": sc_half,
        "bayesian_half_life": bay_half,
        "sl_limit": base,
        "scalar_limit": 0.0,
        "bayesian_limit": base,
    }
