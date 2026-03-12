"""EN3.2-H1b Core — Poison Passage Detection (Binary Classification).

Computes signals for detecting whether a question's retrieved passage set
contains poison (adversarial) passages. Evaluates with AUROC and
precision-at-recall.

This module contains NO API calls or I/O — it is pure computation.

Signals computed:
  Scalar detection signals:
    score_variance       — variance of cosine scores across passages
    score_range          — max - min cosine score
    answer_disagreement  — (n_unique_answers - 1) / n_passages (0 = all agree)
    qa_score_variance    — variance of QA extraction scores
    top_bottom_gap       — mean(top-3 scores) - mean(bottom-3 scores)

  SL detection signals:
    sl_max_conflict           — max pairwise SL conflict
    sl_mean_conflict          — mean pairwise SL conflict
    sl_fused_uncertainty      — uncertainty from cumulative_fuse of all opinions
    sl_fused_disbelief        — disbelief from fused opinion
    sl_belief_disbelief_spread — max(belief_i) - min(belief_i) across opinions
    sl_conflict_answer_weighted — max conflict between passages with different answers
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    pairwise_conflict,
)


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

SCALAR_DETECTION_SIGNALS: List[str] = [
    "score_variance",
    "score_range",
    "answer_disagreement",
    "qa_score_variance",
    "top_bottom_gap",
]

SL_DETECTION_SIGNALS: List[str] = [
    "sl_max_conflict",
    "sl_mean_conflict",
    "sl_fused_uncertainty",
    "sl_fused_disbelief",
    "sl_belief_disbelief_spread",
    "sl_conflict_answer_weighted",
]


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _variance(vals: List[float]) -> float:
    """Population variance."""
    if len(vals) == 0:
        return 0.0
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def _score_to_opinion(
    score: float, evidence_weight: float, prior_weight: float,
) -> Opinion:
    """Convert a scalar score to an SL opinion via evidence mapping."""
    s = max(0.0, min(1.0, score))
    pos = s * evidence_weight
    neg = (1.0 - s) * evidence_weight
    return Opinion.from_evidence(pos, neg, prior_weight=prior_weight)


# ═══════════════════════════════════════════════════════════════════
# Signal computation
# ═══════════════════════════════════════════════════════════════════

def compute_poison_detection_signals(
    passages: List[Dict[str, Any]],
    extractions: Dict[str, Dict[str, Any]],
    evidence_weight: float = 10.0,
    prior_weight: float = 2.0,
) -> Dict[str, float]:
    """Compute all poison detection signals for one question's retrieved set.

    Args:
        passages:        Retrieved passages with 'passage_id', 'score'.
        extractions:     Dict mapping passage_id → {answer, qa_score}.
        evidence_weight: Evidence weight for Opinion.from_evidence.
        prior_weight:    Non-informative prior weight.

    Returns:
        Dict with keys from SCALAR_DETECTION_SIGNALS + SL_DETECTION_SIGNALS.
    """
    n = len(passages)
    cosine_scores = [p["score"] for p in passages]
    qa_scores = [
        extractions.get(p["passage_id"], {}).get("qa_score", 0.0)
        for p in passages
    ]
    answers = [
        extractions.get(p["passage_id"], {}).get("answer", "").strip().lower()
        for p in passages
    ]

    # ── Scalar signals ──

    score_variance = _variance(cosine_scores)
    score_range = max(cosine_scores) - min(cosine_scores) if n > 0 else 0.0

    # Answer disagreement: (n_unique - 1) / n, where 0 = all agree
    non_empty_answers = [a for a in answers if a]
    if len(non_empty_answers) > 0:
        n_unique = len(set(non_empty_answers))
        answer_disagreement = (n_unique - 1) / len(non_empty_answers)
    else:
        answer_disagreement = 0.0

    qa_score_variance = _variance(qa_scores)

    # Top-bottom gap: mean of top-3 minus mean of bottom-3 scores
    sorted_scores = sorted(cosine_scores, reverse=True)
    k = min(3, n)
    top_mean = sum(sorted_scores[:k]) / k if k > 0 else 0.0
    bottom_mean = sum(sorted_scores[-k:]) / k if k > 0 else 0.0
    top_bottom_gap = top_mean - bottom_mean

    # ── SL signals ──

    # Build opinions from combined cosine × QA scores
    opinions: List[Opinion] = []
    for p in passages:
        pid = p["passage_id"]
        cos = p["score"]
        qa = extractions.get(pid, {}).get("qa_score", 0.0)
        combined = (max(0.0, min(1.0, cos)) * max(0.0, min(1.0, qa))) ** 0.5
        opinions.append(
            _score_to_opinion(combined, evidence_weight, prior_weight)
        )

    # Pairwise conflicts
    conflicts: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            conflicts.append(pairwise_conflict(opinions[i], opinions[j]))

    sl_max_conflict = max(conflicts) if conflicts else 0.0
    sl_mean_conflict = sum(conflicts) / len(conflicts) if conflicts else 0.0

    # Fused opinion
    if n == 1:
        fused = opinions[0]
    else:
        fused = cumulative_fuse(*opinions)

    # Belief spread
    beliefs = [op.belief for op in opinions]
    sl_belief_disbelief_spread = max(beliefs) - min(beliefs) if beliefs else 0.0

    # Conflict weighted by answer disagreement: max conflict between
    # passages with DIFFERENT extracted answers
    sl_conflict_answer_weighted = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ai = answers[i]
            aj = answers[j]
            if ai and aj and ai != aj:
                c = pairwise_conflict(opinions[i], opinions[j])
                if c > sl_conflict_answer_weighted:
                    sl_conflict_answer_weighted = c

    return {
        # Scalar
        "score_variance": score_variance,
        "score_range": score_range,
        "answer_disagreement": answer_disagreement,
        "qa_score_variance": qa_score_variance,
        "top_bottom_gap": top_bottom_gap,
        # SL
        "sl_max_conflict": sl_max_conflict,
        "sl_mean_conflict": sl_mean_conflict,
        "sl_fused_uncertainty": fused.uncertainty,
        "sl_fused_disbelief": fused.disbelief,
        "sl_belief_disbelief_spread": sl_belief_disbelief_spread,
        "sl_conflict_answer_weighted": sl_conflict_answer_weighted,
    }


# ═══════════════════════════════════════════════════════════════════
# AUROC
# ═══════════════════════════════════════════════════════════════════

def compute_auroc(
    scores: List[float],
    labels: List[bool],
) -> float:
    """Compute Area Under the ROC Curve via the Wilcoxon-Mann-Whitney statistic.

    Higher score should correspond to positive label.

    Args:
        scores: Predicted scores (one per instance).
        labels: True labels (True = positive).

    Returns:
        AUROC in [0, 1]. Returns 0.5 if all labels are the same.
    """
    n = len(scores)
    positives = [(scores[i], i) for i in range(n) if labels[i]]
    negatives = [(scores[i], i) for i in range(n) if not labels[i]]

    n_pos = len(positives)
    n_neg = len(negatives)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Count concordant pairs: positive scored higher than negative
    concordant = 0.0
    for sp, _ in positives:
        for sn, _ in negatives:
            if sp > sn:
                concordant += 1.0
            elif sp == sn:
                concordant += 0.5

    return concordant / (n_pos * n_neg)


# ═══════════════════════════════════════════════════════════════════
# Precision at recall
# ═══════════════════════════════════════════════════════════════════

def precision_at_recall(
    scores: List[float],
    labels: List[bool],
    recall_levels: List[float],
) -> List[Tuple[float, float]]:
    """Compute precision at specified recall levels.

    Sorts by score descending, walks down the ranked list, and at each
    recall threshold reports the precision achieved.

    Args:
        scores:        Predicted scores (higher = more likely positive).
        labels:        True labels.
        recall_levels: Recall thresholds in (0, 1].

    Returns:
        List of (recall_level, precision) tuples.
    """
    n_pos = sum(1 for l in labels if l)
    if n_pos == 0:
        return [(r, 0.0) for r in recall_levels]

    # Sort by score descending, break ties by index
    indexed = sorted(enumerate(zip(scores, labels)), key=lambda x: (-x[1][0], x[0]))

    # Walk down ranked list, tracking TP and total examined
    tp_at_k = []
    cumulative_tp = 0
    for _, (_, label) in indexed:
        if label:
            cumulative_tp += 1
        tp_at_k.append(cumulative_tp)

    results: List[Tuple[float, float]] = []
    for target_recall in recall_levels:
        # Find smallest k where recall >= target
        needed_tp = max(1, int(target_recall * n_pos + 0.5))
        needed_tp = min(needed_tp, n_pos)

        # Find first k where we have needed_tp true positives
        found_k = len(indexed)  # default: need all
        for k in range(len(indexed)):
            if tp_at_k[k] >= needed_tp:
                found_k = k + 1  # 1-indexed count
                break

        precision = needed_tp / found_k if found_k > 0 else 0.0
        results.append((target_recall, precision))

    return results
