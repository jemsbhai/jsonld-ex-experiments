"""Statistical utilities for experiment analysis.

Provides bootstrap confidence intervals and related helpers
used across all experiment suites.

Usage::

    from experiments.infra.stats import bootstrap_ci

    data = [0.80, 0.85, 0.82, 0.88]
    lower, mean, upper = bootstrap_ci(data, n_bootstrap=1000, seed=42)
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def bootstrap_ci(
    data: Sequence[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Tuple[float, float, float]:
    """Compute a bootstrap confidence interval for the mean.

    Resamples ``data`` with replacement ``n_bootstrap`` times, computes
    the mean of each resample, and returns the percentile-based CI.

    Args:
        data:        Observed values (at least 1 element).
        n_bootstrap: Number of bootstrap resamples (default 1000).
        alpha:       Significance level.  ``alpha=0.05`` gives a 95% CI.
        seed:        Optional RNG seed for reproducibility.

    Returns:
        ``(lower, mean, upper)`` where ``lower`` and ``upper`` are the
        ``alpha/2`` and ``1-alpha/2`` percentiles of the bootstrap
        distribution, and ``mean`` is the sample mean.
    """
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)

    if n == 0:
        raise ValueError("data must be non-empty")

    sample_mean = float(np.mean(arr))

    if n == 1:
        return (sample_mean, sample_mean, sample_mean)

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        resample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(resample)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    lower = float(np.percentile(boot_means, lower_pct))
    upper = float(np.percentile(boot_means, upper_pct))

    return (lower, sample_mean, upper)
