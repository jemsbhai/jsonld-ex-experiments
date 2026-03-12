"""EN3.2-H1c v2 Core — Per-Passage Multi-Model Fusion.

Two-level architecture mirroring EN1.1's proven paradigm:
  Level 1: Per-passage — fuse 4 models' extractions for one passage
  Level 2: Cross-passage — rank passages by fused confidence × cosine

This module contains NO API calls or I/O — pure computation.

Strategies:
  Scalar:
    scalar_majority         — per-passage majority vote, rank by agreement × cosine
    scalar_qa_weighted      — per-passage qa-weighted vote, rank by qa_frac × cosine
    scalar_majority_x_qa    — per-passage majority, rank by agreement × mean_qa × cosine
    single_roberta          — roberta only, rank by qa_score × cosine
    single_best_model       — best single model per passage by qa_score

  SL:
    sl_fusion              — cumulative_fuse within passage, rank by P(ω) × cosine
    sl_trust_discount      — trust_discount per model before fusion
    sl_3strong             — exclude weakest model, fuse 3 strong models
    sl_conflict_weighted   — penalize passages with high intra-passage conflict

Each strategy produces a per-passage (answer, confidence) pair at Level 1,
then ranks passages at Level 2 to select the final answer.
"""
from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
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

STRATEGY_NAMES: List[str] = [
    "scalar_majority",
    "scalar_qa_weighted",
    "scalar_majority_x_qa",
    "single_roberta",
    "single_best_model",
    "sl_fusion",
    "sl_trust_discount",
    "sl_3strong",
    "sl_conflict_weighted",
]

_DEFAULT_TRUST = 0.8
_WEAKEST_MODEL = "bert_tiny"

# ═══════════════════════════════════════════════════════════════════
# Text normalization
# ═══════════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def _fuzzy_match(a: str, b: str, threshold: float = ANSWER_SIMILARITY_THRESHOLD) -> bool:
    na, nb = a.strip().lower(), b.strip().lower()
    if na == nb: return True
    if not na or not nb: return False
    if na in nb or nb in na: return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def _group_within_passage(
    extractions: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Tuple[str, float]]]:
    """Group a passage's model extractions by fuzzy-matched answer.

    Returns: {canonical_answer: [(model_name, qa_score), ...]}
    """
    groups: Dict[str, List[Tuple[str, float]]] = {}
    for model_name, info in extractions.items():
        answer = info.get("answer", "").strip()
        if not answer:
            continue
        qa_score = info.get("qa_score", 0.0)
        placed = False
        for canonical in groups:
            if _fuzzy_match(answer, canonical):
                groups[canonical].append((model_name, qa_score))
                placed = True
                break
        if not placed:
            groups[answer] = [(model_name, qa_score)]
    return groups


# ═══════════════════════════════════════════════════════════════════
# Internal SL helpers
# ═══════════════════════════════════════════════════════════════════

def _score_to_opinion(score: float, ew: float = 10.0, pw: float = 2.0) -> Opinion:
    s = max(0.0, min(1.0, score))
    return Opinion.from_evidence(s * ew, (1.0 - s) * ew, prior_weight=pw)


def _fuse_opinions(opinions: List[Opinion]) -> Opinion:
    if len(opinions) == 1:
        return opinions[0]
    return cumulative_fuse(*opinions)


# ═══════════════════════════════════════════════════════════════════
# Level 1: Per-passage fusion
# ═══════════════════════════════════════════════════════════════════

def fuse_passage_scalar_majority(
    extractions: Dict[str, Dict[str, Any]],
) -> Tuple[str, float]:
    """Majority vote across models for one passage.

    Returns: (winning_answer, agreement_fraction)
    """
    groups = _group_within_passage(extractions)
    if not groups:
        return ("", 0.0)

    n_total = sum(len(members) for members in groups.values())
    best_canonical = max(groups.keys(), key=lambda k: len(groups[k]))
    agreement = len(groups[best_canonical]) / n_total if n_total > 0 else 0.0
    return (best_canonical, agreement)


def fuse_passage_scalar_qa_weighted(
    extractions: Dict[str, Dict[str, Any]],
) -> Tuple[str, float]:
    """QA-score-weighted vote across models for one passage.

    Returns: (winning_answer, normalized_qa_fraction)
    """
    groups = _group_within_passage(extractions)
    if not groups:
        return ("", 0.0)

    # Sum qa_scores per group
    group_qa: Dict[str, float] = {}
    total_qa = 0.0
    for canonical, members in groups.items():
        qa_sum = sum(qa for _, qa in members)
        group_qa[canonical] = qa_sum
        total_qa += qa_sum

    best_canonical = max(group_qa.keys(), key=group_qa.get)
    confidence = group_qa[best_canonical] / total_qa if total_qa > 0 else 0.0
    return (best_canonical, confidence)


def fuse_passage_sl_fusion(
    extractions: Dict[str, Dict[str, Any]],
    evidence_weight: float = 10.0,
    prior_weight: float = 2.0,
) -> Tuple[str, Opinion]:
    """SL cumulative fusion within one passage.

    Groups answers by fuzzy match. For each group, fuses members'
    opinions via cumulative_fuse. Selects group with highest
    projected probability.

    Returns: (winning_answer, fused_opinion_of_winner)
    """
    groups = _group_within_passage(extractions)
    if not groups:
        return ("", Opinion(0.0, 0.0, 1.0))

    # Fuse opinions within each group
    group_fused: Dict[str, Opinion] = {}
    for canonical, members in groups.items():
        opinions = [_score_to_opinion(qa, evidence_weight, prior_weight)
                    for _, qa in members]
        group_fused[canonical] = _fuse_opinions(opinions)

    best = max(group_fused.keys(),
               key=lambda k: group_fused[k].projected_probability())
    return (best, group_fused[best])


def fuse_passage_sl_trust_discount(
    extractions: Dict[str, Dict[str, Any]],
    model_trust: Optional[Dict[str, float]] = None,
    evidence_weight: float = 10.0,
    prior_weight: float = 2.0,
) -> Tuple[str, Opinion]:
    """SL fusion with per-model trust discount within one passage.

    Each model's opinion is first discounted by its trust level
    before fusion within the answer group.

    Returns: (winning_answer, fused_opinion_of_winner)
    """
    groups = _group_within_passage(extractions)
    if not groups:
        return ("", Opinion(0.0, 0.0, 1.0))

    group_fused: Dict[str, Opinion] = {}
    for canonical, members in groups.items():
        discounted = []
        for model_name, qa_score in members:
            base_op = _score_to_opinion(qa_score, evidence_weight, prior_weight)
            if model_trust is not None:
                t = model_trust.get(model_name, _DEFAULT_TRUST)
            else:
                t = _DEFAULT_TRUST
            trust_op = Opinion.from_confidence(t, uncertainty=0.1)
            discounted.append(trust_discount(trust_op, base_op))
        group_fused[canonical] = _fuse_opinions(discounted)

    best = max(group_fused.keys(),
               key=lambda k: group_fused[k].projected_probability())
    return (best, group_fused[best])


# ═══════════════════════════════════════════════════════════════════
# Level 2: Cross-passage ranking
# ═══════════════════════════════════════════════════════════════════

def rank_passages_by_confidence(
    passage_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank passages by confidence score (descending)."""
    return sorted(passage_results, key=lambda p: -p.get("confidence", 0.0))


def rank_passages_by_confidence_x_cosine(
    passage_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank passages by confidence × cosine (descending)."""
    return sorted(
        passage_results,
        key=lambda p: -(p.get("confidence", 0.0) * p.get("cosine", 0.0)),
    )


def select_answer_from_ranking(
    ranked: List[Dict[str, Any]],
) -> str:
    """Take the answer from the top-ranked passage."""
    if not ranked:
        return ""
    return ranked[0].get("answer", "")


# ═══════════════════════════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════════════════════════

def evaluate_all_strategies(
    question_passages: List[Dict[str, Any]],
    model_trust: Optional[Dict[str, float]] = None,
    evidence_weight: float = 10.0,
    prior_weight: float = 2.0,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all strategies on one question.

    Args:
        question_passages: List of passage dicts, each with:
            cosine: float
            passage_id: str
            extractions: {model: {answer, qa_score}}
        model_trust: Optional per-model trust for SL trust discount.

    Returns: {strategy_name: {answer: str, confidence: float}}
    """
    results: Dict[str, Dict[str, Any]] = {}

    # ── scalar_majority ──
    passage_results_majority = []
    for pd in question_passages:
        answer, agreement = fuse_passage_scalar_majority(pd["extractions"])
        passage_results_majority.append({
            "passage_id": pd["passage_id"], "answer": answer,
            "confidence": agreement, "cosine": pd["cosine"],
        })
    ranked = rank_passages_by_confidence_x_cosine(passage_results_majority)
    results["scalar_majority"] = {
        "answer": select_answer_from_ranking(ranked),
        "confidence": ranked[0]["confidence"] if ranked else 0.0,
    }

    # ── scalar_qa_weighted ──
    passage_results_qa = []
    for pd in question_passages:
        answer, qa_frac = fuse_passage_scalar_qa_weighted(pd["extractions"])
        passage_results_qa.append({
            "passage_id": pd["passage_id"], "answer": answer,
            "confidence": qa_frac, "cosine": pd["cosine"],
        })
    ranked = rank_passages_by_confidence_x_cosine(passage_results_qa)
    results["scalar_qa_weighted"] = {
        "answer": select_answer_from_ranking(ranked),
        "confidence": ranked[0]["confidence"] if ranked else 0.0,
    }

    # ── scalar_majority_x_qa ──
    passage_results_mqa = []
    for pd in question_passages:
        groups = _group_within_passage(pd["extractions"])
        if not groups:
            passage_results_mqa.append({
                "passage_id": pd["passage_id"], "answer": "",
                "confidence": 0.0, "cosine": pd["cosine"],
            })
            continue
        n_total = sum(len(m) for m in groups.values())
        best_canonical = max(groups.keys(), key=lambda k: len(groups[k]))
        agreement = len(groups[best_canonical]) / n_total
        mean_qa = (sum(qa for _, qa in groups[best_canonical])
                   / len(groups[best_canonical]))
        passage_results_mqa.append({
            "passage_id": pd["passage_id"], "answer": best_canonical,
            "confidence": agreement * mean_qa, "cosine": pd["cosine"],
        })
    ranked = rank_passages_by_confidence_x_cosine(passage_results_mqa)
    results["scalar_majority_x_qa"] = {
        "answer": select_answer_from_ranking(ranked),
        "confidence": ranked[0]["confidence"] if ranked else 0.0,
    }

    # ── single_roberta ──
    passage_results_rob = []
    for pd in question_passages:
        rob = pd["extractions"].get("roberta", {})
        if rob:
            passage_results_rob.append({
                "passage_id": pd["passage_id"],
                "answer": rob.get("answer", ""),
                "confidence": rob.get("qa_score", 0.0),
                "cosine": pd["cosine"],
            })
    ranked = rank_passages_by_confidence_x_cosine(passage_results_rob)
    results["single_roberta"] = {
        "answer": select_answer_from_ranking(ranked),
        "confidence": ranked[0]["confidence"] if ranked else 0.0,
    }

    # ── single_best_model ──
    # For each passage, pick the extraction with highest qa_score
    passage_results_best = []
    for pd in question_passages:
        best_ans, best_qa = "", 0.0
        for model_name, info in pd["extractions"].items():
            if info.get("qa_score", 0.0) > best_qa:
                best_qa = info["qa_score"]
                best_ans = info.get("answer", "")
        passage_results_best.append({
            "passage_id": pd["passage_id"], "answer": best_ans,
            "confidence": best_qa, "cosine": pd["cosine"],
        })
    ranked = rank_passages_by_confidence_x_cosine(passage_results_best)
    results["single_best_model"] = {
        "answer": select_answer_from_ranking(ranked),
        "confidence": ranked[0]["confidence"] if ranked else 0.0,
    }

    # ── sl_fusion ──
    passage_results_sl = []
    for pd in question_passages:
        answer, opinion = fuse_passage_sl_fusion(
            pd["extractions"], evidence_weight, prior_weight,
        )
        passage_results_sl.append({
            "passage_id": pd["passage_id"], "answer": answer,
            "confidence": opinion.projected_probability(),
            "cosine": pd["cosine"],
        })
    ranked = rank_passages_by_confidence_x_cosine(passage_results_sl)
    results["sl_fusion"] = {
        "answer": select_answer_from_ranking(ranked),
        "confidence": ranked[0]["confidence"] if ranked else 0.0,
    }

    # ── sl_trust_discount ──
    passage_results_td = []
    for pd in question_passages:
        answer, opinion = fuse_passage_sl_trust_discount(
            pd["extractions"], model_trust, evidence_weight, prior_weight,
        )
        passage_results_td.append({
            "passage_id": pd["passage_id"], "answer": answer,
            "confidence": opinion.projected_probability(),
            "cosine": pd["cosine"],
        })
    ranked = rank_passages_by_confidence_x_cosine(passage_results_td)
    results["sl_trust_discount"] = {
        "answer": select_answer_from_ranking(ranked),
        "confidence": ranked[0]["confidence"] if ranked else 0.0,
    }

    # ── sl_3strong ──
    # Exclude weakest model before fusion
    passage_results_3s = []
    for pd in question_passages:
        strong_ext = {
            k: v for k, v in pd["extractions"].items()
            if k != _WEAKEST_MODEL
        }
        if not strong_ext:
            strong_ext = pd["extractions"]  # fallback
        answer, opinion = fuse_passage_sl_fusion(
            strong_ext, evidence_weight, prior_weight,
        )
        passage_results_3s.append({
            "passage_id": pd["passage_id"], "answer": answer,
            "confidence": opinion.projected_probability(),
            "cosine": pd["cosine"],
        })
    ranked = rank_passages_by_confidence_x_cosine(passage_results_3s)
    results["sl_3strong"] = {
        "answer": select_answer_from_ranking(ranked),
        "confidence": ranked[0]["confidence"] if ranked else 0.0,
    }

    # ── sl_conflict_weighted ──
    # Penalize passages where intra-passage model conflict is high
    passage_results_cw = []
    for pd in question_passages:
        answer, opinion = fuse_passage_sl_fusion(
            pd["extractions"], evidence_weight, prior_weight,
        )
        # Compute intra-passage conflict
        groups = _group_within_passage(pd["extractions"])
        if len(groups) >= 2:
            group_opinions = {}
            for canonical, members in groups.items():
                opinions = [_score_to_opinion(qa, evidence_weight, prior_weight)
                            for _, qa in members]
                group_opinions[canonical] = _fuse_opinions(opinions)
            max_conf = 0.0
            canons = list(group_opinions.keys())
            for i in range(len(canons)):
                for j in range(i + 1, len(canons)):
                    c = pairwise_conflict(group_opinions[canons[i]],
                                          group_opinions[canons[j]])
                    if c > max_conf:
                        max_conf = c
            confidence = opinion.projected_probability() * (1.0 - max_conf)
        else:
            confidence = opinion.projected_probability()

        passage_results_cw.append({
            "passage_id": pd["passage_id"], "answer": answer,
            "confidence": confidence, "cosine": pd["cosine"],
        })
    ranked = rank_passages_by_confidence_x_cosine(passage_results_cw)
    results["sl_conflict_weighted"] = {
        "answer": select_answer_from_ranking(ranked),
        "confidence": ranked[0]["confidence"] if ranked else 0.0,
    }

    return results
