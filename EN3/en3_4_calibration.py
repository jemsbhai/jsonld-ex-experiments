"""EN3.4 Phase A0 — Calibration Analysis.

NeurIPS 2026 D&B, Suite EN3 (ML Pipeline Integration), Experiment 4.

Provides Expected Calibration Error (ECE) computation, reliability
diagram bin generation, temperature scaling, and principled per-model
uncertainty derivation for Subjective Logic opinion construction.

This module is a PREREQUISITE for Phase A1 (NER fusion evaluation).
Without calibration-derived uncertainty, the SL opinions would be
mechanical reparameterizations of scalar scores — a reviewer would
correctly reject the contribution.

References:
    Guo et al., 2017. "On Calibration of Modern Neural Networks." ICML.
    Naeini et al., 2015. "Obtaining Well Calibrated Probabilities Using
        Bayesian Binning into Quantiles." AAAI.
    Jøsang, A., 2016. "Subjective Logic." Springer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize_scalar


# =====================================================================
# ECE Computation
# =====================================================================


def compute_ece(
    scores: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (equal-width binning).

    ECE = Σ (n_b / N) × |accuracy_b − confidence_b|

    where the sum is over bins b with at least one prediction.

    Args:
        scores:  Predicted confidence scores in [0, 1].
        correct: Whether each prediction was correct (True/False).
        n_bins:  Number of equal-width bins (default 10).

    Returns:
        ECE in [0, 1].

    Raises:
        ValueError: If inputs are empty or mismatched in length.
    """
    if len(scores) == 0:
        raise ValueError("scores must be non-empty")
    if len(scores) != len(correct):
        raise ValueError(
            f"scores ({len(scores)}) and correct ({len(correct)}) "
            f"must have the same length"
        )

    scores_arr = np.asarray(scores, dtype=np.float64)
    correct_arr = np.asarray(correct, dtype=np.float64)
    n = len(scores_arr)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (scores_arr >= lo) & (scores_arr < hi)
        else:
            # Last bin includes upper boundary
            mask = (scores_arr >= lo) & (scores_arr <= hi)

        n_b = mask.sum()
        if n_b == 0:
            continue

        accuracy_b = correct_arr[mask].mean()
        confidence_b = scores_arr[mask].mean()
        ece += (n_b / n) * abs(accuracy_b - confidence_b)

    return float(ece)


# =====================================================================
# Reliability Diagram Bins
# =====================================================================


def reliability_diagram_bins(
    scores: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> List[Dict[str, Any]]:
    """Generate per-bin data for a reliability diagram.

    Args:
        scores:  Predicted confidence scores in [0, 1].
        correct: Whether each prediction was correct.
        n_bins:  Number of equal-width bins.

    Returns:
        List of dicts, one per bin, each containing:
            bin_lower, bin_upper, count, mean_confidence, accuracy, abs_diff
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    correct_arr = np.asarray(correct, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins = []
    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        if i < n_bins - 1:
            mask = (scores_arr >= lo) & (scores_arr < hi)
        else:
            mask = (scores_arr >= lo) & (scores_arr <= hi)

        n_b = int(mask.sum())
        if n_b == 0:
            bins.append({
                "bin_lower": lo,
                "bin_upper": hi,
                "count": 0,
                "mean_confidence": 0.0,
                "accuracy": 0.0,
                "abs_diff": 0.0,
            })
        else:
            acc = float(correct_arr[mask].mean())
            conf = float(scores_arr[mask].mean())
            bins.append({
                "bin_lower": lo,
                "bin_upper": hi,
                "count": n_b,
                "mean_confidence": conf,
                "accuracy": acc,
                "abs_diff": abs(acc - conf),
            })

    return bins


# =====================================================================
# Temperature Scaling
# =====================================================================


def _nll_at_temperature(
    t: float,
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Negative log-likelihood at temperature T (internal helper)."""
    eps = 1e-7
    scaled = logits / t
    # Numerically stable sigmoid
    calibrated = np.where(
        scaled >= 0,
        1.0 / (1.0 + np.exp(-scaled)),
        np.exp(scaled) / (1.0 + np.exp(scaled)),
    )
    calibrated = np.clip(calibrated, eps, 1.0 - eps)
    # Binary cross-entropy
    nll = -np.mean(
        labels * np.log(calibrated) + (1 - labels) * np.log(1 - calibrated)
    )
    return float(nll)


def fit_temperature(
    scores: Sequence[float],
    correct: Sequence[bool],
) -> float:
    """Fit temperature T that minimizes negative log-likelihood.

    calibrated = sigmoid(logit(score) / T)

    Uses scipy bounded scalar optimization (Brent's method) for
    robust convergence on this 1-D problem.

    Args:
        scores:   Predicted confidence scores in (0, 1).
        correct:  Whether each prediction was correct.

    Returns:
        Fitted temperature T > 0.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(correct, dtype=np.float64)

    # Clamp scores to avoid log(0)
    eps = 1e-7
    scores_arr = np.clip(scores_arr, eps, 1.0 - eps)

    # Convert to logits
    logits = np.log(scores_arr / (1.0 - scores_arr))

    result = minimize_scalar(
        _nll_at_temperature,
        bounds=(0.05, 20.0),
        args=(logits, labels),
        method="bounded",
    )

    return float(result.x)


# =====================================================================
# Per-Model Uncertainty Derivation
# =====================================================================


def derive_model_uncertainty(
    ece: float,
    epsilon: float = 0.02,
    floor: float = 0.02,
    ceiling: float = 0.50,
) -> float:
    """Derive SL uncertainty parameter from ECE.

    u = clamp(ECE + ε, floor, ceiling)

    A model with high calibration error genuinely has more epistemic
    uncertainty. This mapping is monotonic and bounded.

    Args:
        ece:     Expected Calibration Error in [0, 1].
        epsilon: Additive floor to prevent overconfident uncertainty.
        floor:   Minimum uncertainty (even perfect calibration has some).
        ceiling: Maximum uncertainty cap.

    Returns:
        Uncertainty value in [floor, ceiling].
    """
    u = ece + epsilon
    return float(max(floor, min(ceiling, u)))


# =====================================================================
# CalibrationReport
# =====================================================================


@dataclass
class CalibrationReport:
    """Container for calibration analysis results.

    Attributes:
        model_name:          Human-readable model identifier.
        ece:                 Expected Calibration Error (raw scores).
        ece_post_tempscale:  ECE after temperature scaling.
        temperature:         Fitted temperature T.
        derived_uncertainty: SL uncertainty derived from ECE.
        n_predictions:       Number of (score, correct) pairs analyzed.
        reliability_bins:    Per-bin data for reliability diagram.
    """

    model_name: str
    ece: float
    ece_post_tempscale: float
    temperature: float
    derived_uncertainty: float
    n_predictions: int
    reliability_bins: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)
