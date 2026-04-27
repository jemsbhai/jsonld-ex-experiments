"""Statistical tests for ET1 experiments.

Implements the analysis plan from protocol §9:

- Bootstrap CIs for metric differences between conditions
- McNemar's test for paired binary outcomes (hallucination rates)
- Holm-Šídák step-down correction for multiple comparisons
- Cohen's d (continuous) and Cohen's h (proportions) effect sizes
"""

from __future__ import annotations

import math
import random
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci_difference(
    group_a: list[float],
    group_b: list[float],
    metric_fn: Callable[[list[float]], float],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for the difference metric(A) - metric(B).

    Resamples paired observations with replacement and computes the
    metric on each resample to build a distribution of differences.

    Args:
        group_a: Metric values or raw observations for condition A.
        group_b: Metric values or raw observations for condition B.
        metric_fn: Function that computes a scalar metric from a list of values.
        n_resamples: Number of bootstrap resamples.
        ci: Confidence interval width (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound, mean_difference)
    """
    rng = random.Random(seed)
    n = len(group_a)
    assert len(group_b) == n, "Groups must be the same length for paired bootstrap"

    diffs = []
    for _ in range(n_resamples):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        sample_a = [group_a[i] for i in indices]
        sample_b = [group_b[i] for i in indices]
        diff = metric_fn(sample_a) - metric_fn(sample_b)
        diffs.append(diff)

    diffs.sort()
    alpha = 1.0 - ci
    lo_idx = int(math.floor((alpha / 2) * n_resamples))
    hi_idx = int(math.ceil((1.0 - alpha / 2) * n_resamples)) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))

    mean_diff = sum(diffs) / len(diffs)
    return diffs[lo_idx], diffs[hi_idx], mean_diff


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(
    a_correct: list[int],
    b_correct: list[int],
) -> tuple[float, float]:
    """McNemar's test for paired binary outcomes.

    Compares whether two models have the same error rate on paired data.
    Uses the chi-squared approximation with continuity correction.

    Args:
        a_correct: Binary correctness for model A (1=correct, 0=incorrect).
        b_correct: Binary correctness for model B.

    Returns:
        (test_statistic, p_value). p_value of 1.0 if no discordant pairs.
    """
    assert len(a_correct) == len(b_correct)

    # Count discordant pairs
    b_only = 0  # A wrong, B correct
    c_only = 0  # A correct, B wrong

    for a, b in zip(a_correct, b_correct):
        if a == 1 and b == 0:
            c_only += 1
        elif a == 0 and b == 1:
            b_only += 1

    # No discordant pairs → no difference
    if b_only + c_only == 0:
        return 0.0, 1.0

    # Chi-squared with continuity correction
    stat = (abs(b_only - c_only) - 1) ** 2 / (b_only + c_only)

    # p-value from chi-squared distribution with df=1
    p_value = _chi2_sf(stat, df=1)

    return float(stat), float(p_value)


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function (1-CDF) of chi-squared distribution.

    Uses the regularized incomplete gamma function approximation
    for df=1: P(X > x) = erfc(sqrt(x/2)).
    """
    if df != 1:
        raise NotImplementedError("Only df=1 supported")
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2))


# ---------------------------------------------------------------------------
# Holm-Šídák correction
# ---------------------------------------------------------------------------

def holm_sidak_correction(p_values: list[float]) -> list[float]:
    """Holm-Šídák step-down correction for multiple comparisons.

    Less conservative than Bonferroni. For k remaining tests,
    the adjusted p-value is: p_adj = 1 - (1 - p_raw)^k

    Enforces monotonicity: each adjusted p-value is at least as large
    as the previous one in the sorted order.

    Args:
        p_values: Raw p-values from individual tests.

    Returns:
        Adjusted p-values in the ORIGINAL order (not sorted).
    """
    m = len(p_values)
    if m == 0:
        return []
    if m == 1:
        return list(p_values)

    # Sort by raw p-value, keeping track of original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * m
    prev_adj = 0.0

    for rank, (orig_idx, raw_p) in enumerate(indexed):
        k = m - rank  # Number of remaining comparisons
        # Šídák adjustment for k comparisons
        adj_p = 1.0 - (1.0 - raw_p) ** k
        # Enforce monotonicity: can't be smaller than previous
        adj_p = max(adj_p, prev_adj)
        # Cap at 1.0
        adj_p = min(adj_p, 1.0)
        adjusted[orig_idx] = adj_p
        prev_adj = adj_p

    return adjusted


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d for independent groups (pooled standard deviation).

    d = (mean_a - mean_b) / s_pooled

    Positive d means group A has a higher mean.

    Returns:
        Cohen's d. Returns 0.0 if standard deviation is 0.
    """
    n_a = len(group_a)
    n_b = len(group_b)
    if n_a == 0 or n_b == 0:
        return 0.0

    mean_a = sum(group_a) / n_a
    mean_b = sum(group_b) / n_b

    var_a = sum((x - mean_a) ** 2 for x in group_a) / max(n_a - 1, 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / max(n_b - 1, 1)

    # Pooled standard deviation
    s_pooled = math.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
    )

    if s_pooled == 0:
        return 0.0

    return (mean_a - mean_b) / s_pooled


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for comparing two proportions.

    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    Positive h means p1 > p2.

    Returns:
        Cohen's h in [-π, π].
    """
    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))
