"""EN1.3 Ablation Core — Comprehensive Byzantine-Robust Fusion Analysis.

Pure computation module (no I/O, no API calls).

Eight analysis dimensions:
  A. Heterogeneous honest source generation
  B. Extended adversary strategies (SUBTLE, COLLUDING)
  C. Additional scalar baselines (median, majority vote, oracle trimmed)
  D. Calibration analysis (ECE, Brier score)
  E. Probability extraction for all methods
  F. Uncertainty preservation tracking
  G. Breaking point analysis
  H. Pairwise McNemar matrix with Bonferroni correction
"""
from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import binomtest  # type: ignore

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    robust_fuse,
    trust_discount,
    pairwise_conflict,
)

from en1_3_core import (
    generate_ground_truth,
    generate_honest_opinions,
    generate_adversarial_opinions,
    AdversarialStrategy,
    _label_to_opinion,
    _transpose_opinions,
    _learn_trust_opinions,
    fuse_scalar_mean,
    fuse_scalar_trimmed_mean,
    fuse_sl_cumulative,
    fuse_sl_robust,
    fuse_sl_trust_discount,
    compute_accuracy,
    compute_f1,
    compute_detection_rate,
    _DEFAULT_EW,
    _DEFAULT_PW,
)


# ═══════════════════════════════════════════════════════════════════
# Extended adversary strategies
# ═══════════════════════════════════════════════════════════════════

class AdversarialStrategyExt(Enum):
    """Extended adversary types beyond the base three."""
    SUBTLE = "subtle"
    COLLUDING = "colluding"


# ═══════════════════════════════════════════════════════════════════
# A. Heterogeneous honest source generation
# ═══════════════════════════════════════════════════════════════════

def generate_heterogeneous_honest_opinions(
    ground_truth: List[bool],
    accuracies: List[float],
    seed: int = 42,
    evidence_weight: float = _DEFAULT_EW,
    prior_weight: float = _DEFAULT_PW,
) -> List[List[Opinion]]:
    """Generate honest sources with different accuracy levels.

    Each source has its own accuracy, producing a heterogeneous pool.
    """
    rng = np.random.RandomState(seed)
    sources = []
    for acc in accuracies:
        src_opinions = []
        for label in ground_truth:
            correct = bool(rng.random() < acc)
            op = _label_to_opinion(
                label, correct, acc, rng, evidence_weight, prior_weight,
            )
            src_opinions.append(op)
        sources.append(src_opinions)
    return sources


# ═══════════════════════════════════════════════════════════════════
# B. Extended adversary strategies
# ═══════════════════════════════════════════════════════════════════

def generate_subtle_adversarial_opinions(
    ground_truth: List[bool],
    n_sources: int,
    seed: int = 42,
    evidence_weight: float = _DEFAULT_EW,
    prior_weight: float = _DEFAULT_PW,
) -> List[List[Opinion]]:
    """Subtle adversary: inverts labels but with LOW confidence.

    Harder to detect via conflict because the opinions are uncertain,
    not strongly opposing. Mimics an unreliable-but-not-obviously-malicious
    source.
    """
    rng = np.random.RandomState(seed)
    sources = []
    for _ in range(n_sources):
        src_opinions = []
        for label in ground_truth:
            # Invert with low-to-moderate confidence (0.55-0.70)
            conf = rng.uniform(0.55, 0.70)
            op = _label_to_opinion(
                label, False, conf, rng, evidence_weight, prior_weight,
            )
            src_opinions.append(op)
        sources.append(src_opinions)
    return sources


def generate_colluding_adversarial_opinions(
    ground_truth: List[bool],
    n_sources: int,
    seed: int = 42,
    evidence_weight: float = _DEFAULT_EW,
    prior_weight: float = _DEFAULT_PW,
) -> List[List[Opinion]]:
    """Colluding adversaries: coordinate to produce identical wrong answers.

    All colluding sources produce nearly identical opinions (with tiny
    jitter) to appear as a cohesive group, making conflict-based
    detection harder — the adversaries don't conflict with each other.
    """
    rng = np.random.RandomState(seed)
    # Generate one "leader" adversary
    leader_opinions = []
    for label in ground_truth:
        conf = rng.uniform(0.80, 0.92)
        op = _label_to_opinion(
            label, False, conf, rng, evidence_weight, prior_weight,
        )
        leader_opinions.append(op)

    sources = []
    for s in range(n_sources):
        src_opinions = []
        for i, leader_op in enumerate(leader_opinions):
            # Tiny jitter around the leader's opinion
            jitter = rng.uniform(-0.02, 0.02)
            b = max(0.0, min(1.0 - 1e-9, leader_op.belief + jitter))
            d = max(0.0, min(1.0 - b - 1e-9, leader_op.disbelief - jitter))
            u = max(0.0, 1.0 - b - d)
            src_opinions.append(Opinion(b, d, u, leader_op.base_rate))
        sources.append(src_opinions)
    return sources


# ═══════════════════════════════════════════════════════════════════
# C. Additional scalar baselines
# ═══════════════════════════════════════════════════════════════════

def fuse_scalar_median(
    per_instance: List[List[Opinion]],
) -> List[bool]:
    """Scalar median of projected probabilities → threshold at 0.5."""
    predictions = []
    for instance_opinions in per_instance:
        probs = [op.projected_probability() for op in instance_opinions]
        median_p = float(np.median(probs))
        predictions.append(bool(median_p > 0.5))
    return predictions


def fuse_scalar_majority_vote(
    per_instance: List[List[Opinion]],
) -> List[bool]:
    """Simple majority vote: each source votes positive if P(ω) > 0.5."""
    predictions = []
    for instance_opinions in per_instance:
        votes = sum(
            1 for op in instance_opinions
            if op.projected_probability() > 0.5
        )
        predictions.append(bool(votes > len(instance_opinions) / 2))
    return predictions


def fuse_scalar_oracle_trimmed(
    per_instance: List[List[Opinion]],
    adversarial_indices: List[int],
) -> List[bool]:
    """Oracle trimmed mean: knows which sources are adversarial, removes them.

    Upper-bound baseline — no real method has this information.
    """
    adv_set = set(adversarial_indices)
    predictions = []
    for instance_opinions in per_instance:
        honest_probs = [
            op.projected_probability()
            for idx, op in enumerate(instance_opinions)
            if idx not in adv_set
        ]
        if not honest_probs:
            honest_probs = [op.projected_probability() for op in instance_opinions]
        mean_p = float(np.mean(honest_probs))
        predictions.append(bool(mean_p > 0.5))
    return predictions


# ═══════════════════════════════════════════════════════════════════
# D. Calibration metrics
# ═══════════════════════════════════════════════════════════════════

def compute_ece(
    labels: List[bool],
    probs: List[float],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    Partitions predictions into bins by confidence, computes the
    weighted average of |accuracy_bin - confidence_bin|.
    """
    if not labels:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = [(lo <= p < hi) if i < n_bins - 1 else (lo <= p <= hi)
                for p in probs]
        bin_labels = [l for l, m in zip(labels, mask) if m]
        bin_probs = [p for p, m in zip(probs, mask) if m]

        if not bin_labels:
            continue

        bin_acc = sum(bin_labels) / len(bin_labels)
        bin_conf = np.mean(bin_probs)
        ece += len(bin_labels) / n * abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(
    labels: List[bool],
    probs: List[float],
) -> float:
    """Brier score: mean squared error of probabilistic predictions."""
    if not labels:
        return 0.0
    return float(np.mean([
        (p - float(l)) ** 2
        for l, p in zip(labels, probs)
    ]))


# ═══════════════════════════════════════════════════════════════════
# E. Probability extraction (for calibration)
# ═══════════════════════════════════════════════════════════════════

def fuse_scalar_mean_probs(
    per_instance: List[List[Opinion]],
) -> List[float]:
    """Return fused scalar mean probabilities (not thresholded)."""
    return [
        float(np.mean([op.projected_probability() for op in ops]))
        for ops in per_instance
    ]


def fuse_sl_cumulative_probs(
    per_instance: List[List[Opinion]],
) -> List[float]:
    """Return SL cumulative fused projected probabilities."""
    probs = []
    for ops in per_instance:
        fused = cumulative_fuse(*ops)
        probs.append(fused.projected_probability())
    return probs


def fuse_sl_robust_probs(
    per_instance: List[List[Opinion]],
    threshold: float = 0.15,
) -> List[float]:
    """Return SL robust fused projected probabilities."""
    probs = []
    for ops in per_instance:
        fused, _ = robust_fuse(ops, threshold=threshold)
        probs.append(fused.projected_probability())
    return probs


def fuse_sl_trust_discount_probs(
    per_instance: List[List[Opinion]],
    trust_opinions: List[Opinion],
) -> List[float]:
    """Return SL trust-discounted fused projected probabilities."""
    probs = []
    for ops in per_instance:
        discounted = [
            trust_discount(trust_opinions[s], ops[s])
            for s in range(len(ops))
        ]
        fused = cumulative_fuse(*discounted)
        probs.append(fused.projected_probability())
    return probs


# ═══════════════════════════════════════════════════════════════════
# F. Uncertainty preservation
# ═══════════════════════════════════════════════════════════════════

def compute_mean_fused_uncertainty(
    per_instance: List[List[Opinion]],
    method: str = "cumulative",
    threshold: float = 0.15,
) -> float:
    """Compute mean uncertainty of fused opinions across instances.

    Tracks whether fusion under adversarial conditions produces
    falsely confident (low u) or appropriately uncertain (high u) results.
    """
    uncertainties = []
    for ops in per_instance:
        if method == "cumulative":
            fused = cumulative_fuse(*ops)
        elif method == "robust":
            fused, _ = robust_fuse(ops, threshold=threshold)
        else:
            raise ValueError(f"Unknown method: {method}")
        uncertainties.append(fused.uncertainty)
    return float(np.mean(uncertainties))


# ═══════════════════════════════════════════════════════════════════
# G. Breaking point analysis
# ═══════════════════════════════════════════════════════════════════

def find_breaking_point(
    n_honest: int,
    honest_accuracy: float,
    adversarial_strategy: AdversarialStrategy,
    method: str,
    baseline_drop: float = 0.05,
    max_adversarial: int = 10,
    n_instances: int = 500,
    seed: int = 42,
    robust_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Find the adversarial count k where a method's accuracy drops
    more than baseline_drop below its k=0 performance.

    Returns: {breaking_k, breaking_ratio, baseline_accuracy, accuracy_at_break,
              accuracy_curve}
    """
    from en1_3_core import evaluate_single_scenario

    # Get baseline (k=0)
    baseline_result = evaluate_single_scenario(
        n_instances=n_instances, n_honest=n_honest, n_adversarial=0,
        honest_accuracy=honest_accuracy,
        adversarial_strategy=adversarial_strategy,
        robust_thresholds=[robust_threshold],
        seed=seed,
    )

    # Resolve method name to baseline key
    if method.startswith("sl_robust"):
        baseline_key = f"sl_robust_t{robust_threshold:.2f}"
    else:
        baseline_key = method

    baseline_acc = baseline_result.get(baseline_key, baseline_result.get(method, {})).get("accuracy", 0.0)
    threshold_acc = baseline_acc - baseline_drop

    accuracy_curve = [(0, baseline_acc)]
    breaking_k = None
    breaking_acc = None

    for k in range(1, max_adversarial + 1):
        result = evaluate_single_scenario(
            n_instances=n_instances, n_honest=n_honest, n_adversarial=k,
            honest_accuracy=honest_accuracy,
            adversarial_strategy=adversarial_strategy,
            robust_thresholds=[robust_threshold],
            seed=seed,
        )
        acc = result.get(baseline_key, result.get(method, {})).get("accuracy", 0.0)
        accuracy_curve.append((k, acc))

        if acc < threshold_acc and breaking_k is None:
            breaking_k = k
            breaking_acc = acc

    return {
        "breaking_k": breaking_k,
        "breaking_ratio": (breaking_k / (n_honest + breaking_k)) if breaking_k is not None else None,
        "baseline_accuracy": baseline_acc,
        "accuracy_at_break": breaking_acc,
        "threshold_accuracy": threshold_acc,
        "accuracy_curve": accuracy_curve,
    }


# ═══════════════════════════════════════════════════════════════════
# H. Pairwise McNemar matrix with Bonferroni correction
# ═══════════════════════════════════════════════════════════════════

def compute_pairwise_mcnemar_matrix(
    per_question: List[Dict[str, Any]],
    methods: List[str],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Compute McNemar's exact test for all method pairs.

    Applies Bonferroni correction for multiple comparisons.

    Args:
        per_question: List of dicts, each mapping method_name → bool.
        methods: List of method names to compare.
        alpha: Significance level before correction.

    Returns:
        Dict keyed by "methodA_vs_methodB" with contingency counts,
        raw p-value, Bonferroni-corrected p, and significance flag.
    """
    n_comparisons = len(methods) * (len(methods) - 1) // 2
    matrix: Dict[str, Dict[str, Any]] = {}

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a_name, b_name = methods[i], methods[j]
            both_correct = 0
            a_only = 0
            b_only = 0
            both_wrong = 0

            for q in per_question:
                a_corr = bool(q.get(a_name, False))
                b_corr = bool(q.get(b_name, False))
                if a_corr and b_corr:
                    both_correct += 1
                elif a_corr and not b_corr:
                    a_only += 1
                elif not a_corr and b_corr:
                    b_only += 1
                else:
                    both_wrong += 1

            n_disc = a_only + b_only
            if n_disc == 0:
                p_val = 1.0
            else:
                p_val = binomtest(a_only, n_disc, 0.5).pvalue

            p_bonf = min(1.0, p_val * n_comparisons)

            matrix[f"{a_name}_vs_{b_name}"] = {
                "both_correct": both_correct,
                "a_only": a_only,
                "b_only": b_only,
                "both_wrong": both_wrong,
                "n_discordant": n_disc,
                "p_value": p_val,
                "p_bonferroni": p_bonf,
                "significant_bonferroni": p_bonf < alpha,
                "winner": a_name if a_only > b_only else (b_name if b_only > a_only else "tie"),
            }

    return matrix


# ═══════════════════════════════════════════════════════════════════
# I. Extended scenario evaluator
# ═══════════════════════════════════════════════════════════════════

def evaluate_extended_scenario(
    n_instances: int = 500,
    n_honest: int = 0,
    n_adversarial: int = 2,
    honest_accuracy: float = 0.85,
    honest_accuracies: Optional[List[float]] = None,
    adversarial_strategy: Optional[AdversarialStrategy] = None,
    adversarial_strategy_ext: Optional[AdversarialStrategyExt] = None,
    robust_thresholds: Optional[List[float]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Extended scenario evaluator with all analysis dimensions.

    Supports heterogeneous honest sources, extended adversary types,
    additional baselines, calibration, uncertainty tracking, and
    pairwise McNemar.

    Use either n_honest + honest_accuracy (homogeneous) or
    honest_accuracies (heterogeneous), not both.
    """
    if robust_thresholds is None:
        robust_thresholds = [0.15]

    # ── Generate honest sources ──
    gt = generate_ground_truth(n_instances, positive_rate=0.6, seed=seed)

    if honest_accuracies is not None:
        honest = generate_heterogeneous_honest_opinions(
            gt, accuracies=honest_accuracies, seed=seed + 1,
        )
        effective_n_honest = len(honest_accuracies)
    else:
        if n_honest == 0:
            n_honest = 5  # default
        honest = generate_honest_opinions(
            gt, n_honest, honest_accuracy, seed=seed + 1,
        )
        effective_n_honest = n_honest

    # ── Generate adversarial sources ──
    if n_adversarial > 0:
        if adversarial_strategy_ext == AdversarialStrategyExt.SUBTLE:
            adversarial = generate_subtle_adversarial_opinions(
                gt, n_adversarial, seed=seed + 2,
            )
        elif adversarial_strategy_ext == AdversarialStrategyExt.COLLUDING:
            adversarial = generate_colluding_adversarial_opinions(
                gt, n_adversarial, seed=seed + 2,
            )
        elif adversarial_strategy is not None:
            adversarial = generate_adversarial_opinions(
                gt, n_adversarial, adversarial_strategy, seed=seed + 2,
            )
        else:
            adversarial = generate_adversarial_opinions(
                gt, n_adversarial, AdversarialStrategy.INVERSION, seed=seed + 2,
            )
    else:
        adversarial = []

    all_sources = honest + adversarial
    n_total = len(all_sources)
    adversarial_indices = list(range(effective_n_honest, n_total))
    per_instance = _transpose_opinions(all_sources)

    results: Dict[str, Any] = {}
    # Per-instance predictions per method for McNemar
    per_q_records: List[Dict[str, bool]] = [{} for _ in range(n_instances)]

    def _register(method_name: str, preds: List[bool], probs: List[float],
                  extra: Optional[Dict] = None):
        acc = compute_accuracy(gt, preds)
        f1 = compute_f1(gt, preds)
        ece = compute_ece(gt, probs)
        brier = compute_brier_score(gt, probs)
        entry: Dict[str, Any] = {
            "accuracy": acc, "f1": f1, "ece": ece, "brier": brier,
        }
        if extra:
            entry.update(extra)
        results[method_name] = entry
        for i, p in enumerate(preds):
            per_q_records[i][method_name] = p

    # ── Scalar mean ──
    probs_sm = fuse_scalar_mean_probs(per_instance)
    preds_sm = [bool(p > 0.5) for p in probs_sm]
    _register("scalar_mean", preds_sm, probs_sm)

    # ── Scalar median ──
    preds_med = fuse_scalar_median(per_instance)
    probs_med = [float(np.median([op.projected_probability() for op in ops]))
                 for ops in per_instance]
    _register("scalar_median", preds_med, probs_med)

    # ── Scalar majority vote ──
    preds_mv = fuse_scalar_majority_vote(per_instance)
    # Majority vote doesn't produce probabilities; use vote fraction
    probs_mv = [
        sum(1 for op in ops if op.projected_probability() > 0.5) / len(ops)
        for ops in per_instance
    ]
    _register("scalar_majority_vote", preds_mv, probs_mv)

    # ── Scalar trimmed mean ──
    k_trim = max(1, n_adversarial)
    preds_tm = fuse_scalar_trimmed_mean(per_instance, k=k_trim)
    # Compute trimmed probs
    probs_tm = []
    for ops in per_instance:
        sorted_p = sorted([op.projected_probability() for op in ops])
        trim = min(k_trim, (len(sorted_p) - 1) // 2)
        trimmed = sorted_p[trim:-trim] if trim > 0 else sorted_p
        probs_tm.append(float(np.mean(trimmed)))
    _register("scalar_trimmed_mean", preds_tm, probs_tm, {"k": k_trim})

    # ── Scalar oracle trimmed ──
    preds_ot = fuse_scalar_oracle_trimmed(per_instance, adversarial_indices)
    probs_ot = []
    adv_set = set(adversarial_indices)
    for ops in per_instance:
        honest_p = [op.projected_probability() for idx, op in enumerate(ops)
                     if idx not in adv_set]
        probs_ot.append(float(np.mean(honest_p)) if honest_p else 0.5)
    _register("scalar_oracle_trimmed", preds_ot, probs_ot)

    # ── SL cumulative ──
    probs_slc = fuse_sl_cumulative_probs(per_instance)
    preds_slc = [bool(p > 0.5) for p in probs_slc]
    u_slc = compute_mean_fused_uncertainty(per_instance, method="cumulative")
    _register("sl_cumulative", preds_slc, probs_slc,
              {"mean_fused_uncertainty": u_slc})

    # ── SL robust (per threshold) ──
    for t in robust_thresholds:
        probs_slr = fuse_sl_robust_probs(per_instance, threshold=t)
        preds_slr = [bool(p > 0.5) for p in probs_slr]
        _, removed = fuse_sl_robust(per_instance, threshold=t)
        u_slr = compute_mean_fused_uncertainty(
            per_instance, method="robust", threshold=t,
        )

        detection = compute_detection_rate(
            [set(r) for r in removed], adversarial_indices, n_total,
        ) if n_adversarial > 0 else {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        mean_removed = float(np.mean([len(r) for r in removed]))
        _register(f"sl_robust_t{t:.2f}", preds_slr, probs_slr, {
            "mean_fused_uncertainty": u_slr,
            "mean_removed": round(mean_removed, 3),
            "detection_precision": detection["precision"],
            "detection_recall": detection["recall"],
            "detection_f1": detection["f1"],
        })

    # ── SL trust discount ──
    trust_opinions = _learn_trust_opinions(per_instance, n_total)
    probs_slt = fuse_sl_trust_discount_probs(per_instance, trust_opinions)
    preds_slt = [bool(p > 0.5) for p in probs_slt]
    # Compute uncertainty for trust discount
    u_slt_list = []
    for ops in per_instance:
        disc = [trust_discount(trust_opinions[s], ops[s]) for s in range(len(ops))]
        fused = cumulative_fuse(*disc)
        u_slt_list.append(fused.uncertainty)
    u_slt = float(np.mean(u_slt_list))

    _register("sl_trust_discount", preds_slt, probs_slt, {
        "mean_fused_uncertainty": u_slt,
        "learned_trust": [round(t.projected_probability(), 4) for t in trust_opinions],
    })

    # ── Pairwise McNemar matrix ──
    method_names = [k for k in results.keys() if not k.startswith("_")]
    mcnemar = compute_pairwise_mcnemar_matrix(per_q_records, method_names)
    results["_mcnemar_matrix"] = mcnemar

    return results
