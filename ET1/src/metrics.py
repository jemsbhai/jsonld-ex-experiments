"""Evaluation metrics for ET1 experiments.

Implements all metrics specified in protocol §8.2:

H1 (Format Grounding):
  - json_validity_rate, jsonld_validity_rate, jsonldex_compliance_rate

H2 (Calibration Transfer):
  - expected_calibration_error (ECE)
  - maximum_calibration_error (MCE)
  - brier_score
  - confidence_auroc

H3 (Uncertainty Awareness):
  - hallucination_rate
  - selective_prediction_accuracy
  - abstention_appropriateness
"""

from __future__ import annotations

import json
import math
from typing import Optional


# ---------------------------------------------------------------------------
# H2: Calibration metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    confidences: list[float],
    correctness: list[int],
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (equal-width binning).

    ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|

    Args:
        confidences: Stated confidence per prediction, each in [0, 1].
        correctness: Binary correctness indicator (1=correct, 0=incorrect).
        n_bins: Number of equal-width bins.

    Returns:
        ECE in [0, 1]. Returns 0.0 for empty input.
    """
    n = len(confidences)
    if n == 0:
        return 0.0

    bin_width = 1.0 / n_bins
    ece = 0.0

    for i in range(n_bins):
        lo = i * bin_width
        hi = lo + bin_width

        # Collect predictions in this bin
        bin_confs = []
        bin_accs = []
        for c, a in zip(confidences, correctness):
            if lo <= c < hi or (i == n_bins - 1 and c == hi):
                bin_confs.append(c)
                bin_accs.append(a)

        if not bin_confs:
            continue

        avg_conf = sum(bin_confs) / len(bin_confs)
        avg_acc = sum(bin_accs) / len(bin_accs)
        ece += (len(bin_confs) / n) * abs(avg_acc - avg_conf)

    return ece


def maximum_calibration_error(
    confidences: list[float],
    correctness: list[int],
    n_bins: int = 15,
) -> float:
    """Maximum Calibration Error — worst-case bin gap.

    MCE = max_m |acc(B_m) - conf(B_m)|

    Returns:
        MCE in [0, 1]. Returns 0.0 for empty input.
    """
    n = len(confidences)
    if n == 0:
        return 0.0

    bin_width = 1.0 / n_bins
    mce = 0.0

    for i in range(n_bins):
        lo = i * bin_width
        hi = lo + bin_width

        bin_confs = []
        bin_accs = []
        for c, a in zip(confidences, correctness):
            if lo <= c < hi or (i == n_bins - 1 and c == hi):
                bin_confs.append(c)
                bin_accs.append(a)

        if not bin_confs:
            continue

        avg_conf = sum(bin_confs) / len(bin_confs)
        avg_acc = sum(bin_accs) / len(bin_accs)
        gap = abs(avg_acc - avg_conf)
        mce = max(mce, gap)

    return mce


def brier_score(
    confidences: list[float],
    correctness: list[int],
) -> float:
    """Brier Score = mean squared error between confidence and outcome.

    BS = (1/n) Σ (c_i - o_i)²

    Lower is better. Perfect = 0, worst = 1.

    Returns:
        Brier score in [0, 1]. Returns 0.0 for empty input.
    """
    n = len(confidences)
    if n == 0:
        return 0.0

    return sum((c - o) ** 2 for c, o in zip(confidences, correctness)) / n


def confidence_auroc(
    confidences: list[float],
    correctness: list[int],
) -> float:
    """AUROC treating confidence as a classifier score for correctness.

    A well-calibrated model assigns higher confidence to correct predictions,
    yielding AUROC near 1.0. Random → 0.5.

    Returns:
        AUROC in [0, 1]. Returns 0.5 if all predictions are same class.
    """
    n = len(confidences)
    if n == 0:
        return 0.5

    n_pos = sum(correctness)
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Wilcoxon-Mann-Whitney statistic with proper tie handling.
    # For each (positive, negative) pair:
    #   confidence_pos > confidence_neg  → count 1.0  (concordant)
    #   confidence_pos == confidence_neg → count 0.5  (tied)
    #   confidence_pos < confidence_neg  → count 0.0  (discordant)
    score = 0.0
    for i in range(n):
        if correctness[i] != 1:
            continue
        for j in range(n):
            if correctness[j] != 0:
                continue
            if confidences[i] > confidences[j]:
                score += 1.0
            elif confidences[i] == confidences[j]:
                score += 0.5

    auroc = score / (n_pos * n_neg)
    return auroc


# ---------------------------------------------------------------------------
# H3: Uncertainty awareness metrics
# ---------------------------------------------------------------------------

def hallucination_rate(
    confidences: list[float],
    correctness: list[int],
    threshold: float = 0.7,
) -> float:
    """Fraction of predictions that are confident AND incorrect.

    Hallucination = confidence > threshold AND correctness == 0.

    Args:
        confidences: Stated confidence per prediction.
        correctness: Binary correctness indicator.
        threshold: Confidence threshold for "confident".

    Returns:
        Rate in [0, 1]. Returns 0.0 for empty input.
    """
    n = len(confidences)
    if n == 0:
        return 0.0

    hallucinations = sum(
        1 for c, o in zip(confidences, correctness)
        if c > threshold and o == 0
    )
    return hallucinations / n


def selective_prediction_accuracy(
    correctness: list[int],
    abstained: list[bool],
) -> Optional[float]:
    """Accuracy computed only on non-abstained predictions.

    Args:
        correctness: Binary correctness for each item.
        abstained: Whether the model abstained for each item.

    Returns:
        Accuracy in [0, 1], or None if all predictions were abstained.
    """
    answered = [
        (c, a) for c, a in zip(correctness, abstained) if not a
    ]
    if not answered:
        return None

    correct_count = sum(c for c, _ in answered)
    return correct_count / len(answered)


def abstention_appropriateness(
    abstained: list[bool],
    tiers: list[str],
    uncertain_tiers: set[str],
) -> Optional[float]:
    """Among abstentions, fraction that involved genuinely uncertain facts.

    Args:
        abstained: Whether the model abstained for each item.
        tiers: Ground-truth confidence tier for each item.
        uncertain_tiers: Set of tier names considered "genuinely uncertain".

    Returns:
        Score in [0, 1], or None if no abstentions occurred.
    """
    abstention_tiers = [
        t for a, t in zip(abstained, tiers) if a
    ]
    if not abstention_tiers:
        return None

    appropriate = sum(1 for t in abstention_tiers if t in uncertain_tiers)
    return appropriate / len(abstention_tiers)


# ---------------------------------------------------------------------------
# H1: Format compliance metrics
# ---------------------------------------------------------------------------

def json_validity_rate(responses: list[str]) -> float:
    """Fraction of responses that parse as valid JSON."""
    if not responses:
        return 0.0

    valid = 0
    for resp in responses:
        try:
            obj = json.loads(resp.strip())
            if isinstance(obj, dict):
                valid += 1
        except (json.JSONDecodeError, ValueError):
            pass

    return valid / len(responses)


def jsonld_validity_rate(responses: list[str]) -> float:
    """Fraction of responses that are valid JSON-LD (have @context AND @type)."""
    if not responses:
        return 0.0

    valid = 0
    for resp in responses:
        try:
            obj = json.loads(resp.strip())
            if isinstance(obj, dict) and "@context" in obj and "@type" in obj:
                valid += 1
        except (json.JSONDecodeError, ValueError):
            pass

    return valid / len(responses)


def jsonldex_compliance_rate(responses: list[str]) -> float:
    """Fraction of responses with valid @opinion fields (b+d+u=1, all in [0,1])."""
    if not responses:
        return 0.0

    valid = 0
    for resp in responses:
        try:
            obj = json.loads(resp.strip())
            if not isinstance(obj, dict):
                continue
            opinion = _find_nested(obj, "@opinion")
            if opinion is None:
                continue
            b = opinion.get("belief", -1)
            d = opinion.get("disbelief", -1)
            u = opinion.get("uncertainty", -1)
            if all(0.0 <= v <= 1.0 for v in (b, d, u)):
                if abs(b + d + u - 1.0) < 1e-6:
                    valid += 1
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return valid / len(responses)


def _find_nested(obj, key):
    """Recursively find a key in nested dict/list."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_nested(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_nested(item, key)
            if found is not None:
                return found
    return None
