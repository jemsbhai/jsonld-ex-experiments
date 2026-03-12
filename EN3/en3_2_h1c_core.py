"""EN3.2-H1c Core — Multi-Extractor RAG Fusion.

Fuses answers from multiple independent QA extractors using scalar
baselines and SL algebra. Mirrors EN1.1's proven paradigm (multiple
independent sources of comparable quality) in the RAG setting.

This module contains NO API calls or I/O — it is pure computation.

Answer selection strategies:
  Scalar:
    majority           — most common answer across all model×passage pairs
    weighted_qa        — answer with highest sum of qa_scores
    model_agreement    — answer supported by most distinct models
    model_agree_x_qa   — n_models × sum(qa_score)

  SL:
    sl_fusion          — cumulative_fuse opinions within groups, select by
                         projected probability
    sl_trust_discount  — trust_discount per model before fusion
    sl_conflict_aware  — detect inter-group conflict, prefer high-belief
                         low-conflict answer
"""
from __future__ import annotations

import re
import string
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    pairwise_conflict,
    trust_discount,
)


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

ANSWER_SIMILARITY_THRESHOLD = 0.6

SCALAR_STRATEGY_NAMES: List[str] = [
    "majority",
    "weighted_qa",
    "model_agreement",
    "model_agree_x_qa",
]

SL_STRATEGY_NAMES: List[str] = [
    "sl_fusion",
    "sl_trust_discount",
    "sl_conflict_aware",
]


# ═══════════════════════════════════════════════════════════════════
# Text normalization and matching
# ═══════════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    """Normalize answer text for comparison."""
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def _fuzzy_match(a: str, b: str, threshold: float = ANSWER_SIMILARITY_THRESHOLD) -> bool:
    na, nb = a.strip().lower(), b.strip().lower()
    if na == nb:
        return True
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _score_to_opinion(
    score: float, evidence_weight: float, prior_weight: float,
) -> Opinion:
    """Convert a scalar score to an SL opinion via evidence mapping."""
    s = max(0.0, min(1.0, score))
    pos = s * evidence_weight
    neg = (1.0 - s) * evidence_weight
    return Opinion.from_evidence(pos, neg, prior_weight=prior_weight)


def _fuse_opinions(opinions: List[Opinion]) -> Opinion:
    """Fuse a list of opinions (handles single-opinion case)."""
    if len(opinions) == 1:
        return opinions[0]
    return cumulative_fuse(*opinions)


# ═══════════════════════════════════════════════════════════════════
# Answer grouping
# ═══════════════════════════════════════════════════════════════════

def group_answers_fuzzy(
    multi_model_extractions: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group extracted answers across all models by fuzzy matching.

    Args:
        multi_model_extractions: {model_name: {pid: {answer, qa_score}}}

    Returns:
        {canonical_answer: [{model, passage_id, answer, qa_score}, ...]}
    """
    # Flatten all extractions into a list
    flat: List[Dict[str, Any]] = []
    for model_name, passages in multi_model_extractions.items():
        for pid, info in passages.items():
            answer = info.get("answer", "").strip()
            if not answer:
                continue
            flat.append({
                "model": model_name,
                "passage_id": pid,
                "answer": answer,
                "qa_score": info.get("qa_score", 0.0),
            })

    # Group by fuzzy matching
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for entry in flat:
        ans = entry["answer"]
        placed = False
        for canonical in groups:
            if _fuzzy_match(ans, canonical):
                groups[canonical].append(entry)
                placed = True
                break
        if not placed:
            groups[ans] = [entry]

    return groups


# ═══════════════════════════════════════════════════════════════════
# Scalar strategies
# ═══════════════════════════════════════════════════════════════════

def fuse_answer_groups_scalar(
    groups: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, float]]:
    """Compute scalar metrics for each answer group.

    Returns: {canonical: {count, qa_sum, n_models, model_agree_x_qa}}
    """
    result = {}
    for canonical, members in groups.items():
        qa_sum = sum(m["qa_score"] for m in members)
        distinct_models = len(set(m["model"] for m in members))
        result[canonical] = {
            "count": len(members),
            "qa_sum": qa_sum,
            "n_models": distinct_models,
            "model_agree_x_qa": distinct_models * qa_sum,
        }
    return result


def select_answer_majority(
    groups: Dict[str, List[Dict[str, Any]]],
) -> str:
    """Select answer with most model×passage votes."""
    return max(groups.keys(), key=lambda k: len(groups[k]))


def select_answer_weighted_qa(
    groups: Dict[str, List[Dict[str, Any]]],
) -> str:
    """Select answer with highest sum of qa_scores."""
    return max(groups.keys(), key=lambda k: sum(m["qa_score"] for m in groups[k]))


def select_answer_model_agreement(
    groups: Dict[str, List[Dict[str, Any]]],
) -> str:
    """Select answer supported by most distinct models.

    Ties broken by total count, then qa_sum.
    """
    def _key(canonical):
        members = groups[canonical]
        n_models = len(set(m["model"] for m in members))
        qa_sum = sum(m["qa_score"] for m in members)
        return (n_models, len(members), qa_sum)

    return max(groups.keys(), key=_key)


def select_answer_model_agree_x_qa(
    groups: Dict[str, List[Dict[str, Any]]],
) -> str:
    """Select answer by n_distinct_models × sum(qa_score)."""
    def _score(canonical):
        members = groups[canonical]
        n_models = len(set(m["model"] for m in members))
        qa_sum = sum(m["qa_score"] for m in members)
        return n_models * qa_sum

    return max(groups.keys(), key=_score)


# ═══════════════════════════════════════════════════════════════════
# SL strategies
# ═══════════════════════════════════════════════════════════════════

def fuse_answer_groups_sl(
    groups: Dict[str, List[Dict[str, Any]]],
    evidence_weight: float = 10.0,
    prior_weight: float = 2.0,
) -> Dict[str, Opinion]:
    """Fuse opinions within each answer group using cumulative_fuse.

    Each member contributes an opinion derived from its qa_score.

    Returns: {canonical_answer: fused_opinion}
    """
    result = {}
    for canonical, members in groups.items():
        opinions = [
            _score_to_opinion(m["qa_score"], evidence_weight, prior_weight)
            for m in members
        ]
        result[canonical] = _fuse_opinions(opinions)
    return result


def select_answer_sl_fusion(
    groups: Dict[str, List[Dict[str, Any]]],
    evidence_weight: float = 10.0,
    prior_weight: float = 2.0,
) -> str:
    """Select answer with highest projected probability after SL fusion.

    Each member's qa_score becomes an opinion. Opinions within a group
    are fused via cumulative_fuse. The group with the highest projected
    probability wins. This naturally weights both evidence strength
    (qa_score) and evidence quantity (number of members).
    """
    fused = fuse_answer_groups_sl(groups, evidence_weight, prior_weight)
    return max(fused.keys(), key=lambda k: fused[k].projected_probability())


def select_answer_sl_trust_discount(
    groups: Dict[str, List[Dict[str, Any]]],
    model_trust: Optional[Dict[str, float]] = None,
    evidence_weight: float = 10.0,
    prior_weight: float = 2.0,
) -> str:
    """Select answer after applying trust discount per model.

    Each member's opinion is first discounted by the model's trust level
    before fusion within the group. This downweights unreliable models.

    Args:
        groups: Answer groups from group_answers_fuzzy.
        model_trust: {model_name: trust_level} in [0, 1].
                     If None, all models get equal trust (0.8).
    """
    default_trust = 0.8

    fused_opinions: Dict[str, Opinion] = {}
    for canonical, members in groups.items():
        discounted_opinions = []
        for m in members:
            # Base opinion from qa_score
            base_op = _score_to_opinion(m["qa_score"], evidence_weight, prior_weight)

            # Trust level for this model
            if model_trust is not None:
                t = model_trust.get(m["model"], default_trust)
            else:
                t = default_trust

            # Construct trust opinion
            trust_op = Opinion.from_confidence(t, uncertainty=0.1)

            # Apply trust discount
            discounted = trust_discount(trust_op, base_op)
            discounted_opinions.append(discounted)

        fused_opinions[canonical] = _fuse_opinions(discounted_opinions)

    return max(fused_opinions.keys(),
               key=lambda k: fused_opinions[k].projected_probability())


def select_answer_sl_conflict_aware(
    groups: Dict[str, List[Dict[str, Any]]],
    evidence_weight: float = 10.0,
    prior_weight: float = 2.0,
) -> str:
    """Select answer using conflict-aware SL fusion.

    Fuses within groups, then penalizes groups that have high conflict
    with other groups. Score = projected_probability × (1 - max_conflict_with_others).
    """
    fused = fuse_answer_groups_sl(groups, evidence_weight, prior_weight)
    canonicals = list(fused.keys())

    if len(canonicals) == 1:
        return canonicals[0]

    # Compute max conflict each group has with any other group
    scores: Dict[str, float] = {}
    for i, c_i in enumerate(canonicals):
        max_conf = 0.0
        for j, c_j in enumerate(canonicals):
            if i != j:
                conf = pairwise_conflict(fused[c_i], fused[c_j])
                if conf > max_conf:
                    max_conf = conf
        pp = fused[c_i].projected_probability()
        scores[c_i] = pp * (1.0 - max_conf)

    return max(scores.keys(), key=scores.get)


# ═══════════════════════════════════════════════════════════════════
# Abstention
# ═══════════════════════════════════════════════════════════════════

def compute_abstention_signal(
    groups: Dict[str, List[Dict[str, Any]]],
    evidence_weight: float = 10.0,
    prior_weight: float = 2.0,
) -> float:
    """Compute abstention signal from SL fusion.

    Higher value = less confident = should abstain.
    Combines fused uncertainty of the best group with inter-group conflict.

    Returns: value in [0, 1].
    """
    fused = fuse_answer_groups_sl(groups, evidence_weight, prior_weight)
    canonicals = list(fused.keys())

    if len(canonicals) == 0:
        return 1.0

    # Best group by projected probability
    best = max(canonicals, key=lambda k: fused[k].projected_probability())
    best_u = fused[best].uncertainty

    # Max inter-group conflict
    max_conflict = 0.0
    if len(canonicals) >= 2:
        for i in range(len(canonicals)):
            for j in range(i + 1, len(canonicals)):
                conf = pairwise_conflict(fused[canonicals[i]], fused[canonicals[j]])
                if conf > max_conflict:
                    max_conflict = conf

    # Combine: high uncertainty OR high conflict → should abstain
    # Use max to capture either source of doubt
    signal = max(best_u, max_conflict)
    return min(1.0, max(0.0, signal))
