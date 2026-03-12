"""EN3.2-H1 Core — Calibrated Selective Answering (Abstention).

Computes abstention signals (scalar and SL) for RAG questions, builds
precision-coverage curves, and evaluates AUC. Supports parameter sweeps
over evidence_weight and prior_weight.

This module contains NO API calls or I/O — it is pure computation,
fully testable without network access.

Signals computed:
  Scalar group:
    max_cosine    — max cosine similarity among retrieved passages
    mean_cosine   — mean cosine similarity
    max_qa_score  — max QA extraction confidence
    top1_qa_score — QA confidence from highest-ranked passage
    score_spread  — max - min cosine similarity

  SL group (from cosine → Opinion.from_evidence):
    sl_fused_belief       — belief from cumulative_fuse of passage opinions
    sl_fused_uncertainty   — uncertainty from fused opinion
    sl_max_conflict       — max pairwise_conflict between passage opinions
    sl_composite          — fused_belief × (1 - max_conflict)
    sl_qa_fused_u         — uncertainty from fusing QA-derived opinions

  Hybrid SL group (fuse both cosine and QA opinions):
    sl_dual_fused_belief     — belief from fusing both signal types
    sl_dual_fused_u          — uncertainty from fusing both signal types
    sl_dual_max_conflict     — max conflict across all opinions
    sl_dual_composite        — dual_fused_belief × (1 - dual_max_conflict)
"""
from __future__ import annotations

import random as _random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    pairwise_conflict,
)


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

# 21 levels from 0.50 to 1.00 in 2.5pp steps
DEFAULT_COVERAGE_LEVELS: List[float] = [
    round(0.50 + i * 0.025, 3) for i in range(21)
]

SCALAR_SIGNAL_NAMES: List[str] = [
    "max_cosine",
    "mean_cosine",
    "max_qa_score",
    "top1_qa_score",
    "score_spread",
]

SL_SIGNAL_NAMES: List[str] = [
    "sl_fused_belief",
    "sl_fused_uncertainty",
    "sl_max_conflict",
    "sl_composite",
    "sl_qa_fused_u",
    "sl_dual_fused_belief",
    "sl_dual_fused_u",
    "sl_dual_max_conflict",
    "sl_dual_composite",
]


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _score_to_opinion(
    score: float,
    evidence_weight: float,
    prior_weight: float,
) -> Opinion:
    """Convert a scalar score ∈ [0, 1] to an SL opinion via evidence mapping.

    Positive evidence = score × evidence_weight
    Negative evidence = (1 - score) × evidence_weight
    """
    s = max(0.0, min(1.0, score))
    pos = s * evidence_weight
    neg = (1.0 - s) * evidence_weight
    return Opinion.from_evidence(pos, neg, prior_weight=prior_weight)


def _max_pairwise_conflict(opinions: List[Opinion]) -> float:
    """Compute maximum pairwise conflict among a list of opinions."""
    n = len(opinions)
    if n < 2:
        return 0.0
    max_c = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            c = pairwise_conflict(opinions[i], opinions[j])
            if c > max_c:
                max_c = c
    return max_c


def _fuse_opinions(opinions: List[Opinion]) -> Opinion:
    """Fuse a list of opinions (handles single-opinion case)."""
    if len(opinions) == 1:
        return opinions[0]
    return cumulative_fuse(*opinions)


# ═══════════════════════════════════════════════════════════════════
# 1. Scalar signal computation
# ═══════════════════════════════════════════════════════════════════

def compute_scalar_signals(
    passages: List[Dict[str, Any]],
    extractions: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Compute all scalar abstention signals for one question.

    Args:
        passages:    List of retrieved passages, each with 'passage_id'
                     and 'score' (cosine similarity). Ordered by rank.
        extractions: Dict mapping passage_id → {answer, qa_score}.

    Returns:
        Dict with keys from SCALAR_SIGNAL_NAMES.
    """
    cosine_scores = [p["score"] for p in passages]
    qa_scores = [
        extractions.get(p["passage_id"], {}).get("qa_score", 0.0)
        for p in passages
    ]

    max_cos = max(cosine_scores)
    min_cos = min(cosine_scores)
    mean_cos = sum(cosine_scores) / len(cosine_scores)

    return {
        "max_cosine": max_cos,
        "mean_cosine": mean_cos,
        "max_qa_score": max(qa_scores),
        "top1_qa_score": qa_scores[0],  # highest-ranked passage
        "score_spread": max_cos - min_cos,
    }


# ═══════════════════════════════════════════════════════════════════
# 2. SL signal computation
# ═══════════════════════════════════════════════════════════════════

def compute_sl_signals(
    passages: List[Dict[str, Any]],
    extractions: Dict[str, Dict[str, Any]],
    evidence_weight: float = 10.0,
    prior_weight: float = 2.0,
) -> Dict[str, float]:
    """Compute all SL abstention signals for one question.

    Constructs SL opinions from cosine scores and QA scores, fuses them,
    and computes conflict metrics.

    Args:
        passages:        List of retrieved passages with 'passage_id', 'score'.
        extractions:     Dict mapping passage_id → {answer, qa_score}.
        evidence_weight: Total evidence weight for Opinion.from_evidence.
        prior_weight:    Non-informative prior weight (W in Jøsang's notation).

    Returns:
        Dict with keys from SL_SIGNAL_NAMES.
    """
    # --- Cosine-derived opinions ---
    cos_opinions: List[Opinion] = []
    for p in passages:
        cos_opinions.append(
            _score_to_opinion(p["score"], evidence_weight, prior_weight)
        )

    # --- QA-derived opinions ---
    qa_opinions: List[Opinion] = []
    for p in passages:
        pid = p["passage_id"]
        qa_score = extractions.get(pid, {}).get("qa_score", 0.0)
        qa_opinions.append(
            _score_to_opinion(qa_score, evidence_weight, prior_weight)
        )

    # --- Cosine group signals ---
    cos_fused = _fuse_opinions(cos_opinions)
    cos_max_conflict = _max_pairwise_conflict(cos_opinions)

    # --- QA group signals ---
    qa_fused = _fuse_opinions(qa_opinions)

    # --- Dual (hybrid) group: fuse ALL opinions together ---
    all_opinions = cos_opinions + qa_opinions
    dual_fused = _fuse_opinions(all_opinions)
    dual_max_conflict = _max_pairwise_conflict(all_opinions)

    return {
        # Cosine-based SL
        "sl_fused_belief": cos_fused.belief,
        "sl_fused_uncertainty": cos_fused.uncertainty,
        "sl_max_conflict": cos_max_conflict,
        "sl_composite": cos_fused.belief * (1.0 - cos_max_conflict),
        "sl_qa_fused_u": qa_fused.uncertainty,
        # Dual (hybrid)
        "sl_dual_fused_belief": dual_fused.belief,
        "sl_dual_fused_u": dual_fused.uncertainty,
        "sl_dual_max_conflict": dual_max_conflict,
        "sl_dual_composite": dual_fused.belief * (1.0 - dual_max_conflict),
    }


# ═══════════════════════════════════════════════════════════════════
# 3. Precision-coverage evaluation
# ═══════════════════════════════════════════════════════════════════

def precision_coverage_curve(
    signal_values: List[float],
    correct: List[bool],
    coverage_levels: Optional[List[float]] = None,
) -> List[Tuple[float, float]]:
    """Compute precision at each coverage level.

    Ranks questions by signal value (descending — higher = more confident),
    then at each coverage level k, computes precision on the top-k%
    questions.

    Args:
        signal_values:   One value per question (higher = more confident).
        correct:         Whether LLM answered each question correctly.
        coverage_levels: Fractions in (0, 1]. Defaults to DEFAULT_COVERAGE_LEVELS.

    Returns:
        List of (coverage, precision) tuples, one per coverage level.
    """
    if coverage_levels is None:
        coverage_levels = DEFAULT_COVERAGE_LEVELS

    n = len(signal_values)
    if n == 0:
        return [(cov, 0.0) for cov in coverage_levels]

    # Sort by signal descending; break ties by original index (stable)
    indexed = list(enumerate(zip(signal_values, correct)))
    indexed.sort(key=lambda x: (-x[1][0], x[0]))
    sorted_correct = [c for _, (_, c) in indexed]

    # Prefix sum of correct answers
    prefix_correct = [0] * (n + 1)
    for i in range(n):
        prefix_correct[i + 1] = prefix_correct[i] + (1 if sorted_correct[i] else 0)

    result: List[Tuple[float, float]] = []
    for cov in coverage_levels:
        k = max(1, round(cov * n))
        k = min(k, n)
        num_correct = prefix_correct[k]
        precision = num_correct / k
        result.append((cov, precision))

    return result


def auc_precision_coverage(
    curve: List[Tuple[float, float]],
) -> float:
    """Compute area under the precision-coverage curve via trapezoidal rule.

    Args:
        curve: List of (coverage, precision) tuples, sorted by coverage.

    Returns:
        AUC value (non-negative float).
    """
    if len(curve) < 2:
        return 0.0

    auc = 0.0
    for i in range(len(curve) - 1):
        c0, p0 = curve[i]
        c1, p1 = curve[i + 1]
        auc += 0.5 * (p0 + p1) * (c1 - c0)
    return auc


# ═══════════════════════════════════════════════════════════════════
# 4. Oracle and random baselines
# ═══════════════════════════════════════════════════════════════════

def oracle_signal(correct: List[bool]) -> List[float]:
    """Oracle signal: 1.0 if correct, 0.0 if wrong.

    Upper bound on any signal's precision-coverage AUC.
    """
    return [1.0 if c else 0.0 for c in correct]


def random_signal(n: int, seed: int = 42) -> List[float]:
    """Random signal: uniform [0, 1]. Lower bound baseline."""
    rng = _random.Random(seed)
    return [rng.random() for _ in range(n)]
