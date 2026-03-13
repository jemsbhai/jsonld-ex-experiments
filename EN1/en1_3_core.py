"""EN1.3 Core — Byzantine-Robust Fusion.

Pure computation module (no I/O, no API calls).

Evaluates SL robust_fuse against scalar baselines when adversarial
sources inject corrupted opinions into a multi-source fusion scenario.

Three adversarial strategies:
  RANDOM:     Adversary produces random opinions (near chance)
  INVERSION:  Adversary systematically inverts the true label
  TARGETED:   Adversary flips only positives → negatives (precision attack)
"""
from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure sibling modules importable
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


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

class AdversarialStrategy(Enum):
    RANDOM = "random"
    INVERSION = "inversion"
    TARGETED = "targeted"


FUSION_METHODS = [
    "scalar_mean",
    "scalar_trimmed_mean",
    "sl_cumulative",
    "sl_robust",
    "sl_trust_discount",
]

# Default evidence weight for opinion construction
_DEFAULT_EW = 10.0
_DEFAULT_PW = 2.0


# ═══════════════════════════════════════════════════════════════════
# 1. Data generation
# ═══════════════════════════════════════════════════════════════════

def generate_ground_truth(
    n: int,
    positive_rate: float = 0.6,
    seed: int = 42,
) -> List[bool]:
    """Generate binary ground truth labels."""
    rng = np.random.RandomState(seed)
    return [bool(x) for x in rng.random(n) < positive_rate]


def _label_to_opinion(
    label: bool,
    correct: bool,
    confidence: float,
    rng: np.random.RandomState,
    evidence_weight: float = _DEFAULT_EW,
    prior_weight: float = _DEFAULT_PW,
) -> Opinion:
    """Create an opinion for a source's prediction on one instance.

    If correct=True, the opinion supports the true label.
    If correct=False, the opinion supports the wrong label.
    Confidence is jittered slightly for realism.
    """
    # Jitter confidence ±0.05
    conf = np.clip(confidence + rng.uniform(-0.05, 0.05), 0.05, 0.95)

    if label:
        # True label is positive
        if correct:
            pos_ev = conf * evidence_weight
            neg_ev = (1.0 - conf) * evidence_weight
        else:
            pos_ev = (1.0 - conf) * evidence_weight
            neg_ev = conf * evidence_weight
    else:
        # True label is negative
        if correct:
            pos_ev = (1.0 - conf) * evidence_weight
            neg_ev = conf * evidence_weight
        else:
            pos_ev = conf * evidence_weight
            neg_ev = (1.0 - conf) * evidence_weight

    return Opinion.from_evidence(pos_ev, neg_ev, prior_weight=prior_weight)


def generate_honest_opinions(
    ground_truth: List[bool],
    n_sources: int,
    accuracy: float = 0.85,
    seed: int = 42,
    evidence_weight: float = _DEFAULT_EW,
    prior_weight: float = _DEFAULT_PW,
) -> List[List[Opinion]]:
    """Generate opinions from honest sources.

    Each source independently gets each instance right with probability
    ``accuracy``, producing an opinion reflecting its prediction.

    Returns: List of n_sources lists, each with len(ground_truth) opinions.
    """
    rng = np.random.RandomState(seed)
    sources = []
    for s in range(n_sources):
        src_opinions = []
        for label in ground_truth:
            correct = bool(rng.random() < accuracy)
            op = _label_to_opinion(
                label, correct, accuracy, rng, evidence_weight, prior_weight,
            )
            src_opinions.append(op)
        sources.append(src_opinions)
    return sources


def generate_adversarial_opinions(
    ground_truth: List[bool],
    n_sources: int,
    strategy: AdversarialStrategy,
    seed: int = 42,
    evidence_weight: float = _DEFAULT_EW,
    prior_weight: float = _DEFAULT_PW,
) -> List[List[Opinion]]:
    """Generate opinions from adversarial sources.

    Strategies:
      RANDOM:     Random opinions with high confidence (misleading noise)
      INVERSION:  Systematically invert the true label with high confidence
      TARGETED:   Flip positives to negatives, leave negatives correct

    Note: Uses .value string comparison for robustness against Python's
    dual-module-loading issue (same enum imported via different paths
    creates non-identical enum classes).
    """
    rng = np.random.RandomState(seed)
    sv = strategy.value  # Compare by string value, not identity
    sources = []
    for s in range(n_sources):
        src_opinions = []
        for label in ground_truth:
            if sv == "random":
                # Random belief, high confidence to be maximally disruptive
                conf = rng.uniform(0.6, 0.95)
                random_label = bool(rng.random() < 0.5)
                correct = (random_label == label)
                op = _label_to_opinion(
                    label, correct, conf, rng, evidence_weight, prior_weight,
                )
            elif sv == "inversion":
                # Always invert the true label with high confidence
                op = _label_to_opinion(
                    label, False, 0.90, rng, evidence_weight, prior_weight,
                )
            elif sv == "targeted":
                # Flip positives → negative, leave negatives correct
                if label:
                    op = _label_to_opinion(
                        label, False, 0.90, rng, evidence_weight, prior_weight,
                    )
                else:
                    op = _label_to_opinion(
                        label, True, 0.85, rng, evidence_weight, prior_weight,
                    )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            src_opinions.append(op)
        sources.append(src_opinions)
    return sources


# ═══════════════════════════════════════════════════════════════════
# 2. Fusion methods
# ═══════════════════════════════════════════════════════════════════

def _transpose_opinions(
    source_opinions: List[List[Opinion]],
) -> List[List[Opinion]]:
    """Transpose from [sources][instances] to [instances][sources]."""
    n_instances = len(source_opinions[0])
    return [
        [source_opinions[s][i] for s in range(len(source_opinions))]
        for i in range(n_instances)
    ]


def fuse_scalar_mean(
    per_instance: List[List[Opinion]],
) -> List[bool]:
    """Scalar mean of projected probabilities → threshold at 0.5."""
    predictions = []
    for instance_opinions in per_instance:
        mean_p = np.mean([op.projected_probability() for op in instance_opinions])
        predictions.append(bool(mean_p > 0.5))
    return predictions


def fuse_scalar_trimmed_mean(
    per_instance: List[List[Opinion]],
    k: int = 1,
) -> List[bool]:
    """Scalar trimmed mean: remove top-k and bottom-k, then average."""
    predictions = []
    for instance_opinions in per_instance:
        probs = sorted([op.projected_probability() for op in instance_opinions])
        # Trim k from each end (but keep at least 1)
        trim = min(k, (len(probs) - 1) // 2)
        if trim > 0:
            trimmed = probs[trim:-trim]
        else:
            trimmed = probs
        mean_p = np.mean(trimmed)
        predictions.append(bool(mean_p > 0.5))
    return predictions


def fuse_sl_cumulative(
    per_instance: List[List[Opinion]],
) -> List[bool]:
    """SL cumulative fusion → threshold on projected probability."""
    predictions = []
    for instance_opinions in per_instance:
        fused = cumulative_fuse(*instance_opinions)
        predictions.append(bool(fused.projected_probability() > 0.5))
    return predictions


def fuse_sl_robust(
    per_instance: List[List[Opinion]],
    threshold: float = 0.15,
) -> Tuple[List[bool], List[List[int]]]:
    """SL robust fusion with conflict-based outlier removal.

    Returns: (predictions, per_instance_removed_indices)
    """
    predictions = []
    all_removed: List[List[int]] = []
    for instance_opinions in per_instance:
        fused, removed = robust_fuse(
            instance_opinions, threshold=threshold,
        )
        predictions.append(bool(fused.projected_probability() > 0.5))
        all_removed.append(removed)
    return predictions, all_removed


def fuse_sl_trust_discount(
    per_instance: List[List[Opinion]],
    trust_opinions: List[Opinion],
) -> List[bool]:
    """SL fusion with per-source trust discount before cumulative fusion."""
    predictions = []
    for instance_opinions in per_instance:
        discounted = [
            trust_discount(trust_opinions[s], instance_opinions[s])
            for s in range(len(instance_opinions))
        ]
        fused = cumulative_fuse(*discounted)
        predictions.append(bool(fused.projected_probability() > 0.5))
    return predictions


# ═══════════════════════════════════════════════════════════════════
# 3. Evaluation metrics
# ═══════════════════════════════════════════════════════════════════

def compute_accuracy(
    ground_truth: List[bool],
    predictions: List[bool],
) -> float:
    """Compute classification accuracy."""
    if not ground_truth:
        return 0.0
    correct = sum(g == p for g, p in zip(ground_truth, predictions))
    return correct / len(ground_truth)


def compute_f1(
    ground_truth: List[bool],
    predictions: List[bool],
) -> float:
    """Compute F1 score (positive class = True)."""
    tp = sum(g and p for g, p in zip(ground_truth, predictions))
    fp = sum((not g) and p for g, p in zip(ground_truth, predictions))
    fn = sum(g and (not p) for g, p in zip(ground_truth, predictions))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_detection_rate(
    per_instance_removed: List[Set[int]],
    adversarial_indices: List[int],
    n_sources: int,
) -> Dict[str, float]:
    """Compute adversarial detection precision and recall.

    Aggregates across instances: what fraction of removals were adversarial
    (precision) and what fraction of adversarial sources were ever removed
    (recall).

    For single-instance detection (all instances share the same source set),
    we compute per-instance and average.
    """
    if not adversarial_indices:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    adv_set = set(adversarial_indices)
    honest_set = set(range(n_sources)) - adv_set

    total_removed = 0
    total_correct_removed = 0
    adv_ever_removed = set()

    for removed in per_instance_removed:
        removed_set = removed if isinstance(removed, set) else set(removed)
        total_removed += len(removed_set)
        correct = removed_set & adv_set
        total_correct_removed += len(correct)
        adv_ever_removed |= correct

    precision = (total_correct_removed / total_removed) if total_removed > 0 else 1.0
    recall = len(adv_ever_removed) / len(adv_set) if adv_set else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


# ═══════════════════════════════════════════════════════════════════
# 4. Trust opinion learning (simple heuristic)
# ═══════════════════════════════════════════════════════════════════

def _learn_trust_opinions(
    per_instance: List[List[Opinion]],
    n_sources: int,
) -> List[Opinion]:
    """Learn per-source trust opinions from inter-source agreement.

    For each source, compute its mean pairwise conflict with all other
    sources. Sources with low conflict (high agreement) get high trust;
    sources with high conflict get low trust.

    This is an unsupervised heuristic — no ground truth needed.
    """
    n_instances = len(per_instance)
    mean_conflict = [0.0] * n_sources

    for instance_opinions in per_instance:
        for s in range(n_sources):
            conflicts = []
            for t in range(n_sources):
                if s != t:
                    c = pairwise_conflict(instance_opinions[s], instance_opinions[t])
                    conflicts.append(c)
            if conflicts:
                mean_conflict[s] += np.mean(conflicts)

    # Normalize by number of instances
    for s in range(n_sources):
        mean_conflict[s] /= n_instances

    # Convert conflict to trust: low conflict → high trust
    trust_opinions = []
    for s in range(n_sources):
        # Map conflict ∈ [0,1] to trust belief ∈ [0.1, 0.9]
        trust_belief = max(0.1, min(0.9, 1.0 - mean_conflict[s]))
        trust_opinions.append(
            Opinion.from_confidence(trust_belief, uncertainty=0.1)
        )

    return trust_opinions


# ═══════════════════════════════════════════════════════════════════
# 5. Full pipeline
# ═══════════════════════════════════════════════════════════════════

def evaluate_single_scenario(
    n_instances: int = 500,
    n_honest: int = 10,
    n_adversarial: int = 2,
    honest_accuracy: float = 0.85,
    adversarial_strategy: AdversarialStrategy = AdversarialStrategy.INVERSION,
    robust_thresholds: Optional[List[float]] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Run a single Byzantine fusion scenario and evaluate all methods.

    Returns: {method_name: {accuracy, f1, ...}}
    """
    if robust_thresholds is None:
        robust_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30]

    # Generate data
    gt = generate_ground_truth(n_instances, positive_rate=0.6, seed=seed)
    honest = generate_honest_opinions(
        gt, n_honest, honest_accuracy, seed=seed + 1,
    )
    adversarial = generate_adversarial_opinions(
        gt, n_adversarial, adversarial_strategy, seed=seed + 2,
    ) if n_adversarial > 0 else []

    # Combine sources: honest first, then adversarial
    all_sources = honest + adversarial
    n_total = len(all_sources)
    adversarial_indices = list(range(n_honest, n_total))

    # Transpose to per-instance format
    per_instance = _transpose_opinions(all_sources)

    results: Dict[str, Dict[str, Any]] = {}

    # ── Scalar mean ──
    preds = fuse_scalar_mean(per_instance)
    results["scalar_mean"] = {
        "accuracy": compute_accuracy(gt, preds),
        "f1": compute_f1(gt, preds),
    }

    # ── Scalar trimmed mean ──
    k_trim = max(1, n_adversarial)
    preds = fuse_scalar_trimmed_mean(per_instance, k=k_trim)
    results["scalar_trimmed_mean"] = {
        "accuracy": compute_accuracy(gt, preds),
        "f1": compute_f1(gt, preds),
        "k": k_trim,
    }

    # ── SL cumulative (no filtering) ──
    preds = fuse_sl_cumulative(per_instance)
    results["sl_cumulative"] = {
        "accuracy": compute_accuracy(gt, preds),
        "f1": compute_f1(gt, preds),
    }

    # ── SL robust (multiple thresholds) ──
    for t in robust_thresholds:
        preds, removed = fuse_sl_robust(per_instance, threshold=t)
        detection = compute_detection_rate(
            [set(r) for r in removed],
            adversarial_indices, n_total,
        ) if n_adversarial > 0 else {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        mean_removed = np.mean([len(r) for r in removed])
        results[f"sl_robust_t{t:.2f}"] = {
            "accuracy": compute_accuracy(gt, preds),
            "f1": compute_f1(gt, preds),
            "threshold": t,
            "mean_removed_per_instance": round(float(mean_removed), 3),
            "detection_precision": detection["precision"],
            "detection_recall": detection["recall"],
            "detection_f1": detection["f1"],
        }

    # ── SL trust discount (learned trust) ──
    trust_opinions = _learn_trust_opinions(per_instance, n_total)
    preds = fuse_sl_trust_discount(per_instance, trust_opinions)
    results["sl_trust_discount"] = {
        "accuracy": compute_accuracy(gt, preds),
        "f1": compute_f1(gt, preds),
        "learned_trust": [
            round(t.projected_probability(), 4)
            for t in trust_opinions
        ],
    }

    return results
