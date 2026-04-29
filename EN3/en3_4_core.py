"""EN3.4 Phase A1 — NER Fusion Evaluation Core.

NeurIPS 2026 D&B, Suite EN3 (ML Pipeline Integration), Experiment 4.

Provides the full pipeline from model predictions to hypothesis verdicts:
span alignment, opinion construction, six experimental conditions
(B1-B5 + SL fusion), entity-level evaluation with strict span matching,
and statistical testing (bootstrap CI, Cohen's h, Holm-Bonferroni).

All SL operations use the pip-installed jsonld_ex package — we test
the actual library, never a reimplementation.

References:
    Jøsang, A. (2016). Subjective Logic. Springer.
    Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
    Cohen, J. (1988). Statistical Power Analysis. 2nd ed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    conflict_metric,
    pairwise_conflict,
)


# =====================================================================
# 1. EntitySpan
# =====================================================================


@dataclass
class EntitySpan:
    """A predicted or gold-standard entity span.

    Attributes:
        start:       Character offset of span start (inclusive).
        end:         Character offset of span end (exclusive).
        entity_type: Entity type label (e.g. "Chemical", "Disease").
        score:       Model confidence score in [0, 1].
        source:      Identifier for the producing model or "gold".
        text:        Optional surface text of the span.
    """

    start: int
    end: int
    entity_type: str
    score: float
    source: str
    text: Optional[str] = None


# =====================================================================
# 2. Span IoU
# =====================================================================


def compute_span_iou(
    start_a: int, end_a: int,
    start_b: int, end_b: int,
) -> float:
    """Character-level Intersection-over-Union between two spans.

    Spans are half-open intervals [start, end).

    Returns:
        IoU in [0, 1]. Zero if no overlap.
    """
    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    intersection = max(0, inter_end - inter_start)

    if intersection == 0:
        return 0.0

    union = (end_a - start_a) + (end_b - start_b) - intersection
    if union == 0:
        return 0.0

    return intersection / union


# =====================================================================
# 3. Span Alignment
# =====================================================================


def align_spans(
    spans_a: List[EntitySpan],
    spans_b: List[EntitySpan],
    iou_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Align spans from two models by character-level IoU.

    Greedy matching: iterate over all (a, b) pairs sorted by IoU
    descending, match each span at most once.

    Returns:
        List of alignment groups, each a dict with:
            span_a:     EntitySpan from model A (or None)
            span_b:     EntitySpan from model B (or None)
            type_match: bool or None (None if single-source)
    """
    if not spans_a and not spans_b:
        return []

    # Compute all pairwise IoUs above threshold
    candidates: List[Tuple[float, int, int]] = []
    for i, sa in enumerate(spans_a):
        for j, sb in enumerate(spans_b):
            iou = compute_span_iou(sa.start, sa.end, sb.start, sb.end)
            if iou >= iou_threshold:
                candidates.append((iou, i, j))

    # Greedy matching: best IoU first
    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    groups: List[Dict[str, Any]] = []

    for iou, i, j in candidates:
        if i in matched_a or j in matched_b:
            continue
        matched_a.add(i)
        matched_b.add(j)
        sa, sb = spans_a[i], spans_b[j]
        groups.append({
            "span_a": sa,
            "span_b": sb,
            "type_match": sa.entity_type == sb.entity_type,
        })

    # Unmatched from A
    for i, sa in enumerate(spans_a):
        if i not in matched_a:
            groups.append({"span_a": sa, "span_b": None, "type_match": None})

    # Unmatched from B
    for j, sb in enumerate(spans_b):
        if j not in matched_b:
            groups.append({"span_a": None, "span_b": sb, "type_match": None})

    return groups


# =====================================================================
# 4. Opinion Construction
# =====================================================================


def build_opinion(score: float, model_uncertainty: float) -> Opinion:
    """Construct an SL Opinion from a GLiNER score and calibration-derived
    model uncertainty.

    Uses the pip-installed jsonld_ex.confidence_algebra.Opinion.

    Args:
        score:             Model confidence score in [0, 1].
        model_uncertainty: Calibration-derived uncertainty from Phase A0.

    Returns:
        Opinion with uncertainty = model_uncertainty.
    """
    return Opinion.from_confidence(score, uncertainty=model_uncertainty)


# =====================================================================
# 5. Experimental Conditions
# =====================================================================


def apply_condition_single_model(
    spans: List[EntitySpan],
    threshold: float,
) -> List[EntitySpan]:
    """B1/B2: Accept entities from a single model above threshold."""
    return [s for s in spans if s.score >= threshold]


def apply_condition_union(
    groups: List[Dict[str, Any]],
    threshold_a: float,
    threshold_b: float,
) -> List[EntitySpan]:
    """B3: Union ensemble — accept if EITHER model predicts above threshold.

    Entity type from higher-confidence model when both present.
    """
    accepted: List[EntitySpan] = []

    for g in groups:
        sa: Optional[EntitySpan] = g["span_a"]
        sb: Optional[EntitySpan] = g["span_b"]

        a_above = sa is not None and sa.score >= threshold_a
        b_above = sb is not None and sb.score >= threshold_b

        if a_above or b_above:
            # Pick the representative span
            if a_above and b_above:
                winner = sa if sa.score >= sb.score else sb
            elif a_above:
                winner = sa
            else:
                winner = sb
            accepted.append(winner)

    return accepted


def apply_condition_intersection(
    groups: List[Dict[str, Any]],
    threshold_a: float,
    threshold_b: float,
) -> List[EntitySpan]:
    """B4: Intersection ensemble — accept only if BOTH models predict
    above their respective thresholds."""
    accepted: List[EntitySpan] = []

    for g in groups:
        sa: Optional[EntitySpan] = g["span_a"]
        sb: Optional[EntitySpan] = g["span_b"]

        if (sa is not None and sa.score >= threshold_a and
                sb is not None and sb.score >= threshold_b):
            winner = sa if sa.score >= sb.score else sb
            accepted.append(winner)

    return accepted


def apply_condition_scalar_average(
    groups: List[Dict[str, Any]],
    threshold: float,
) -> List[EntitySpan]:
    """B5: Scalar average — average confidence scores, threshold."""
    accepted: List[EntitySpan] = []

    for g in groups:
        sa: Optional[EntitySpan] = g["span_a"]
        sb: Optional[EntitySpan] = g["span_b"]

        if sa is not None and sb is not None:
            avg = (sa.score + sb.score) / 2.0
            if avg >= threshold:
                winner = sa if sa.score >= sb.score else sb
                accepted.append(EntitySpan(
                    start=winner.start, end=winner.end,
                    entity_type=winner.entity_type,
                    score=avg, source="scalar_avg",
                    text=winner.text,
                ))
        elif sa is not None:
            if sa.score >= threshold:
                accepted.append(sa)
        elif sb is not None:
            if sb.score >= threshold:
                accepted.append(sb)

    return accepted


def apply_condition_sl_fusion(
    groups: List[Dict[str, Any]],
    u_a: float,
    u_b: float,
    accept_threshold: float,
    conflict_threshold: float,
) -> Tuple[List[EntitySpan], List[EntitySpan]]:
    """SL fusion with conflict-based abstention.

    Returns:
        (accepted, abstained) — two lists of EntitySpan.
    """
    accepted: List[EntitySpan] = []
    abstained: List[EntitySpan] = []

    for g in groups:
        sa: Optional[EntitySpan] = g["span_a"]
        sb: Optional[EntitySpan] = g["span_b"]

        if sa is not None and sb is not None:
            # Both models predict — fuse
            op_a = build_opinion(sa.score, u_a)
            op_b = build_opinion(sb.score, u_b)
            fused = cumulative_fuse(op_a, op_b)
            conf = conflict_metric(fused)

            # Representative span from higher-confidence model
            winner = sa if sa.score >= sb.score else sb

            if conf > conflict_threshold:
                abstained.append(EntitySpan(
                    start=winner.start, end=winner.end,
                    entity_type=winner.entity_type,
                    score=fused.projected_probability(),
                    source="sl_abstained",
                    text=winner.text,
                ))
            elif fused.projected_probability() >= accept_threshold:
                accepted.append(EntitySpan(
                    start=winner.start, end=winner.end,
                    entity_type=winner.entity_type,
                    score=fused.projected_probability(),
                    source="sl_fused",
                    text=winner.text,
                ))
        elif sa is not None:
            # Single source A — no fusion, just threshold
            op = build_opinion(sa.score, u_a)
            if op.projected_probability() >= accept_threshold:
                accepted.append(sa)
        elif sb is not None:
            # Single source B — no fusion, just threshold
            op = build_opinion(sb.score, u_b)
            if op.projected_probability() >= accept_threshold:
                accepted.append(sb)

    return accepted, abstained


# =====================================================================
# 6. Entity-Level Evaluation (Strict Span Match)
# =====================================================================


@dataclass
class EvalMetrics:
    """Entity-level evaluation metrics."""

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def evaluate_entities(
    preds: List[EntitySpan],
    golds: List[EntitySpan],
) -> EvalMetrics:
    """Compute entity-level P/R/F1 with strict span matching.

    A prediction is a true positive if and only if:
        - start matches exactly
        - end matches exactly
        - entity_type matches exactly
    Each gold entity can be matched at most once (greedy, first match).

    Args:
        preds: Predicted entity spans.
        golds: Ground-truth entity spans.

    Returns:
        EvalMetrics with tp, fp, fn, precision, recall, f1.
    """
    matched_gold: set[int] = set()
    tp = 0

    for pred in preds:
        for j, gold in enumerate(golds):
            if j in matched_gold:
                continue
            if (pred.start == gold.start and
                    pred.end == gold.end and
                    pred.entity_type == gold.entity_type):
                tp += 1
                matched_gold.add(j)
                break

    fp = len(preds) - tp
    fn = len(golds) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)

    return EvalMetrics(
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
    )


# =====================================================================
# 7. Bootstrap CI on F1 Differences
# =====================================================================


def bootstrap_f1_difference_ci(
    samples_a: List[Tuple[int, int]],
    samples_b: List[Tuple[int, int]],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI on the F1 difference between two conditions.

    Each sample is a (tp_i, total_gold_i) pair for one sentence/document.
    We resample sentences with replacement and recompute F1 for each
    condition on each bootstrap replicate.

    Args:
        samples_a: Per-sentence (n_correct, n_gold) for condition A.
        samples_b: Per-sentence (n_correct, n_gold) for condition B.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 for 95% CI).
        seed: RNG seed.

    Returns:
        (lower, mean_diff, upper) for F1_A - F1_B.
    """
    assert len(samples_a) == len(samples_b), \
        "Paired bootstrap requires same number of samples"

    n = len(samples_a)
    arr_a = np.array(samples_a, dtype=np.float64)  # (n, 2)
    arr_b = np.array(samples_b, dtype=np.float64)

    rng = np.random.RandomState(seed)
    diffs = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        # Micro F1 for condition A on this resample
        tp_a = arr_a[idx, 0].sum()
        gold_a = arr_a[idx, 1].sum()
        f1_a = tp_a / gold_a if gold_a > 0 else 0.0
        # Micro F1 for condition B on this resample
        tp_b = arr_b[idx, 0].sum()
        gold_b = arr_b[idx, 1].sum()
        f1_b = tp_b / gold_b if gold_b > 0 else 0.0
        diffs[i] = f1_a - f1_b

    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    mean_diff = float(np.mean(diffs))

    return (lo, mean_diff, hi)


# =====================================================================
# 8. Cohen's h
# =====================================================================


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for comparing two proportions.

    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    Positive when p1 > p2.

    Args:
        p1: First proportion (e.g. F1 of condition A).
        p2: Second proportion (e.g. F1 of condition B).

    Returns:
        Cohen's h (can be negative).
    """
    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))


# =====================================================================
# 9. Holm-Bonferroni
# =====================================================================


def holm_bonferroni(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Holm-Bonferroni step-down correction for multiple testing.

    Procedure:
        1. Sort p-values ascending.
        2. For rank k (1-indexed), compare p_(k) to α / (m - k + 1).
        3. Reject if p < adjusted α. Stop rejecting at first failure.

    Args:
        p_values: List of raw p-values.
        alpha:    Family-wise error rate.

    Returns:
        List (same order as input) of dicts with:
            p_value, reject (bool), adjusted_alpha, rank
    """
    m = len(p_values)
    # Create indexed list and sort by p-value
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # Apply step-down procedure
    reject_flags = [False] * m
    adjusted_alphas = [0.0] * m
    still_rejecting = True

    for rank_0, (orig_idx, p) in enumerate(indexed):
        rank = rank_0 + 1  # 1-indexed
        adj_alpha = alpha / (m - rank + 1)
        adjusted_alphas[orig_idx] = adj_alpha

        if still_rejecting and p < adj_alpha:
            reject_flags[orig_idx] = True
        else:
            still_rejecting = False
            reject_flags[orig_idx] = False

    # Build results in original order
    results = []
    for i in range(m):
        results.append({
            "p_value": p_values[i],
            "reject": reject_flags[i],
            "adjusted_alpha": adjusted_alphas[i],
        })

    return results


# =====================================================================
# 10. Conflict-Error Correlation
# =====================================================================


def spearman_conflict_error(
    conflicts: List[float],
    is_error: List[bool],
) -> Tuple[float, float]:
    """Spearman rank correlation between conflict scores and error.

    Args:
        conflicts: Conflict scores per entity.
        is_error:  Whether each entity extraction was incorrect.

    Returns:
        (rho, p_value).
    """
    error_numeric = [1.0 if e else 0.0 for e in is_error]
    rho, p = spearmanr(conflicts, error_numeric)
    return (float(rho), float(p))


# =====================================================================
# 11. Threshold Optimization
# =====================================================================


def optimize_threshold(
    dev_spans: List[EntitySpan],
    dev_golds: List[EntitySpan],
    thresholds: List[float],
) -> Tuple[float, float]:
    """Find the threshold that maximizes F1 on dev data.

    Args:
        dev_spans: All predicted spans (with scores) on dev set.
        dev_golds: Ground-truth spans on dev set.
        thresholds: Candidate thresholds to sweep.

    Returns:
        (best_threshold, best_f1).
    """
    best_t = thresholds[0]
    best_f1 = -1.0

    for t in thresholds:
        filtered = apply_condition_single_model(dev_spans, threshold=t)
        m = evaluate_entities(filtered, dev_golds)
        if m.f1 > best_f1:
            best_f1 = m.f1
            best_t = t

    return (best_t, best_f1)
