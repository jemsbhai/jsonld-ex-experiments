#!/usr/bin/env python
"""EN1.2 — Temporal Regime Adaptation.

NeurIPS 2026 D&B, Suite EN1 (Confidence-Aware Knowledge Fusion), Experiment 2.

Hypothesis: SL opinions with temporal decay adapt to distributional shift
faster than recursive Bayesian updating and exponential moving average.

Demonstrates a core advantage of the SL opinion framework: when the
underlying data distribution changes over time, SL's explicit uncertainty
component (u) and principled decay operators enable faster adaptation
than methods that only track point estimates or conjugate posteriors.

Drift scenarios:
    1. Sudden drift:    P(positive) = 0.8 for t=1..500, then 0.2
    2. Gradual drift:   P linearly 0.8 -> 0.2 over t=200..700
    3. Recurring drift:  P oscillates 0.8/0.2 with period 200

Methods compared:
    - Bayesian Beta posterior with exponential forgetting
    - Exponential Moving Average (alpha in {0.01, 0.05, 0.1})
    - ADWIN adaptive windowing (river library)
    - SL: decay_opinion + cumulative_fuse (half_life in {25, 50, 100, 200, 400})

Metrics:
    - MAE from true P(t) at each timestep
    - Time-to-detection of drift (steps to cross halfway threshold)
    - KL divergence from true distribution
    - All metrics with bootstrap 95% CIs

Usage:
    pip install river  # if not installed
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN1/en1_2_temporal_adaptation.py

Output:
    experiments/EN1/results/en1_2_results.json
    experiments/EN1/results/en1_2_results_<timestamp>.json

References:
    Josang, A. (2016). Subjective Logic, S10.4 (Opinion Aging), S12.3 (Fusion).
    Bifet, A. & Gavalda, R. (2007). Learning from Time-Changing Data with
        Adaptive Windowing. SDM 2007.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "packages" / "python" / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from jsonld_ex.confidence_algebra import Opinion, cumulative_fuse
from jsonld_ex.confidence_decay import decay_opinion, exponential_decay

from experiments.infra.config import set_global_seed
from experiments.infra.results import ExperimentResult
from experiments.infra.env_log import log_environment
from experiments.infra.stats import bootstrap_ci

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

GLOBAL_SEED = 42
N_TIMESTEPS = 1000
BATCH_SIZE = 5           # Observations per timestep
N_BOOTSTRAP = 1000       # Bootstrap resamples for CIs
KL_EPSILON = 1e-10       # Clamping for KL divergence log(0) avoidance
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Method hyperparameters
EMA_ALPHAS = [0.01, 0.05, 0.1]
SL_HALF_LIVES = [25, 50, 100, 200, 400]
BAYESIAN_FORGETTING_FACTOR = 0.99   # Per-step retention factor
BAYESIAN_PRIOR_ALPHA = 1.0          # Uniform prior: Beta(1, 1)
BAYESIAN_PRIOR_BETA = 1.0


# ═══════════════════════════════════════════════════════════════════
# DRIFT SCENARIOS
# ═══════════════════════════════════════════════════════════════════


def ground_truth_sudden(t: int) -> float:
    """P(positive) = 0.8 for t<500, then 0.2."""
    return 0.8 if t < 500 else 0.2


def ground_truth_gradual(t: int) -> float:
    """P linearly 0.8 -> 0.2 over t=200..700."""
    if t < 200:
        return 0.8
    elif t > 700:
        return 0.2
    else:
        return 0.8 - 0.6 * (t - 200) / 500


def ground_truth_recurring(t: int) -> float:
    """P oscillates 0.8/0.2 with period 200 (square wave)."""
    cycle_pos = t % 200
    return 0.8 if cycle_pos < 100 else 0.2


DRIFT_SCENARIOS = {
    "sudden": ground_truth_sudden,
    "gradual": ground_truth_gradual,
    "recurring": ground_truth_recurring,
}


# ═══════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════


def generate_stream(
    ground_truth_fn: Callable[[int], float],
    n_timesteps: int,
    batch_size: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic streaming dataset.

    Args:
        ground_truth_fn: Function mapping timestep -> true P(positive).
        n_timesteps: Number of timesteps.
        batch_size: Observations per timestep.
        rng: Random state for reproducibility.

    Returns:
        (true_p, batches) where:
            true_p: shape (n_timesteps,) — true P(t)
            batches: shape (n_timesteps, batch_size) — 0/1 observations
    """
    true_p = np.array([ground_truth_fn(t) for t in range(n_timesteps)])
    batches = np.zeros((n_timesteps, batch_size), dtype=np.int32)
    for t in range(n_timesteps):
        batches[t] = rng.binomial(1, true_p[t], size=batch_size)
    return true_p, batches


# ═══════════════════════════════════════════════════════════════════
# ESTIMATION METHODS
# ═══════════════════════════════════════════════════════════════════


def run_bayesian_beta(
    batches: np.ndarray,
    forgetting: float = BAYESIAN_FORGETTING_FACTOR,
    prior_a: float = BAYESIAN_PRIOR_ALPHA,
    prior_b: float = BAYESIAN_PRIOR_BETA,
) -> np.ndarray:
    """Bayesian Beta posterior with exponential forgetting.

    At each step:
        1. Shrink toward prior: alpha = alpha * f + prior_a * (1-f)
        2. Update: alpha += n_positive, beta += n_negative
        3. Estimate: alpha / (alpha + beta)

    The forgetting factor f < 1 ensures the posterior doesn't lock
    onto stale evidence, but adaptation speed is limited by the
    fixed shrinkage rate.
    """
    n_steps = len(batches)
    estimates = np.zeros(n_steps)
    alpha, beta = prior_a, prior_b

    for t in range(n_steps):
        # Shrink toward prior (exponential forgetting)
        alpha = alpha * forgetting + prior_a * (1.0 - forgetting)
        beta = beta * forgetting + prior_b * (1.0 - forgetting)

        # Update with new evidence
        pos = int(np.sum(batches[t]))
        neg = len(batches[t]) - pos
        alpha += pos
        beta += neg

        # Point estimate
        estimates[t] = alpha / (alpha + beta)

    return estimates


def run_ema(
    batches: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Exponential moving average of batch means.

    estimate_{t+1} = alpha * batch_mean_t + (1 - alpha) * estimate_t
    """
    n_steps = len(batches)
    estimates = np.zeros(n_steps)
    estimate = 0.5  # Initial estimate (no prior knowledge)

    for t in range(n_steps):
        batch_mean = np.mean(batches[t])
        estimate = alpha * batch_mean + (1.0 - alpha) * estimate
        estimates[t] = estimate

    return estimates


def run_adwin(
    batches: np.ndarray,
) -> np.ndarray:
    """ADWIN adaptive windowing estimator.

    Uses the river library's ADWIN drift detector.  When drift is
    detected, the internal statistics are automatically adjusted
    to discard obsolete data.

    The estimate is the windowed mean of observations.
    """
    from river.drift import ADWIN

    n_steps = len(batches)
    estimates = np.zeros(n_steps)
    adwin = ADWIN()

    # Track running windowed mean manually alongside ADWIN
    window: list[float] = []

    for t in range(n_steps):
        for obs in batches[t]:
            adwin.update(float(obs))
            window.append(float(obs))

            if adwin.drift_detected:
                # ADWIN detected drift — trim the window
                # Keep only the most recent portion
                # ADWIN's internal width tells us how far back is valid
                width = max(1, int(adwin.width))
                if len(window) > width:
                    window = window[-width:]

        if window:
            estimates[t] = np.mean(window)
        else:
            estimates[t] = 0.5

    return estimates


def run_sl(
    batches: np.ndarray,
    half_life: float,
    base_rate: float = 0.5,
    prior_weight: float = 2.0,
) -> np.ndarray:
    """SL temporal decay + cumulative fusion.

    At each timestep:
        1. Create batch opinion from evidence: Opinion.from_evidence(pos, neg)
        2. Decay accumulated opinion: decay_opinion(acc, elapsed=1, half_life)
        3. Fuse: cumulative_fuse(decayed_acc, batch_opinion)
        4. Record projected_probability()

    The key advantage: decay explicitly migrates mass from b/d into u,
    so the accumulated opinion "knows" its old evidence is stale and
    gives proportionally more weight to fresh evidence during fusion.
    """
    n_steps = len(batches)
    estimates = np.zeros(n_steps)

    # Start with vacuous opinion (total ignorance)
    accumulated = Opinion(0.0, 0.0, 1.0, base_rate=base_rate)

    for t in range(n_steps):
        pos = int(np.sum(batches[t]))
        neg = len(batches[t]) - pos

        # Create batch opinion from evidence
        batch_opinion = Opinion.from_evidence(
            pos, neg, prior_weight=prior_weight, base_rate=base_rate,
        )

        # Decay accumulated opinion (1 timestep elapsed)
        decayed = decay_opinion(accumulated, elapsed=1.0, half_life=half_life)

        # Fuse decayed accumulator with fresh evidence
        accumulated = cumulative_fuse(decayed, batch_opinion)

        # Record estimate
        estimates[t] = accumulated.projected_probability()

    return estimates


# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════


def compute_mae(true_p: np.ndarray, estimates: np.ndarray) -> float:
    """Mean absolute error across all timesteps."""
    return float(np.mean(np.abs(true_p - estimates)))


def compute_per_step_ae(true_p: np.ndarray, estimates: np.ndarray) -> np.ndarray:
    """Absolute error at each timestep (for bootstrap CIs and curves)."""
    return np.abs(true_p - estimates)


def compute_kl_divergence(true_p: np.ndarray, estimates: np.ndarray) -> float:
    """Mean KL(P_true || P_hat) over all timesteps.

    KL(p || q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))

    Estimates are clamped to [epsilon, 1-epsilon] to avoid log(0).
    """
    p = np.clip(true_p, KL_EPSILON, 1.0 - KL_EPSILON)
    q = np.clip(estimates, KL_EPSILON, 1.0 - KL_EPSILON)

    kl = p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))
    return float(np.mean(kl))


def compute_time_to_detection_sudden(
    estimates: np.ndarray,
    change_point: int = 500,
    old_p: float = 0.8,
    new_p: float = 0.2,
) -> int | None:
    """Time-to-detection for sudden drift.

    Detection = first t > change_point where estimate crosses the
    midpoint between old and new regime values.

    Returns number of steps after change_point, or None if never detected.
    """
    midpoint = (old_p + new_p) / 2.0
    going_down = new_p < old_p

    for t in range(change_point, len(estimates)):
        if going_down and estimates[t] < midpoint:
            return t - change_point
        elif not going_down and estimates[t] > midpoint:
            return t - change_point

    return None  # Never detected


def compute_time_to_detection_recurring(
    estimates: np.ndarray,
    period: int = 200,
) -> list[int | None]:
    """Time-to-detection for each regime change in recurring drift.

    Regime changes occur at t = 100, 200, 300, ... (every half-period).
    """
    half_period = period // 2
    detections: list[int | None] = []

    for change_t in range(half_period, len(estimates), half_period):
        # Determine old and new regime
        cycle_pos_before = (change_t - 1) % period
        old_p = 0.8 if cycle_pos_before < half_period else 0.2
        new_p = 0.2 if old_p == 0.8 else 0.8
        midpoint = (old_p + new_p) / 2.0
        going_down = new_p < old_p

        detected = None
        end_t = min(change_t + half_period, len(estimates))
        for t in range(change_t, end_t):
            if going_down and estimates[t] < midpoint:
                detected = t - change_t
                break
            elif not going_down and estimates[t] > midpoint:
                detected = t - change_t
                break

        detections.append(detected)

    return detections


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════


def run_scenario(
    scenario_name: str,
    ground_truth_fn: Callable[[int], float],
    rng: np.random.RandomState,
) -> dict[str, Any]:
    """Run all methods on a single drift scenario.

    Returns a dict with per-method results including MAE, KL, detection
    times, and per-timestep estimate curves.
    """
    print(f"\n  Scenario: {scenario_name}")
    true_p, batches = generate_stream(
        ground_truth_fn, N_TIMESTEPS, BATCH_SIZE, rng,
    )

    results: dict[str, Any] = {
        "scenario": scenario_name,
        "n_timesteps": N_TIMESTEPS,
        "batch_size": BATCH_SIZE,
        "true_p": true_p.tolist(),
        "methods": {},
    }

    def evaluate_method(
        name: str,
        estimates: np.ndarray,
    ) -> dict[str, Any]:
        """Compute all metrics for a method."""
        mae = compute_mae(true_p, estimates)
        kl = compute_kl_divergence(true_p, estimates)
        per_step_ae = compute_per_step_ae(true_p, estimates)

        # Bootstrap CI for MAE
        mae_lo, mae_mean, mae_hi = bootstrap_ci(
            per_step_ae.tolist(), n_bootstrap=N_BOOTSTRAP, seed=GLOBAL_SEED,
        )

        method_result: dict[str, Any] = {
            "mae": mae,
            "mae_ci_95": [mae_lo, mae_hi],
            "kl_divergence": kl,
            "estimates": estimates.tolist(),
        }

        # Time-to-detection (scenario-specific)
        if scenario_name == "sudden":
            ttd = compute_time_to_detection_sudden(estimates)
            method_result["time_to_detection"] = ttd
        elif scenario_name == "recurring":
            ttds = compute_time_to_detection_recurring(estimates)
            valid_ttds = [d for d in ttds if d is not None]
            method_result["time_to_detection_per_change"] = ttds
            method_result["mean_time_to_detection"] = (
                float(np.mean(valid_ttds)) if valid_ttds else None
            )
            method_result["n_detected"] = len(valid_ttds)
            method_result["n_changes"] = len(ttds)
        elif scenario_name == "gradual":
            # For gradual drift, measure MAE within the transition window
            transition_ae = per_step_ae[200:701]
            method_result["transition_mae"] = float(np.mean(transition_ae))

        print(f"    {name:30s}  MAE={mae:.4f} [{mae_lo:.4f}, {mae_hi:.4f}]  KL={kl:.4f}", end="")
        if "time_to_detection" in method_result and method_result["time_to_detection"] is not None:
            print(f"  TTD={method_result['time_to_detection']}", end="")
        elif "mean_time_to_detection" in method_result and method_result["mean_time_to_detection"] is not None:
            print(f"  mean_TTD={method_result['mean_time_to_detection']:.1f}", end="")
        print()

        return method_result

    # ── Bayesian Beta ──
    t0 = time.perf_counter()
    est_bayes = run_bayesian_beta(batches)
    results["methods"]["bayesian_beta"] = evaluate_method("Bayesian Beta (f=0.99)", est_bayes)
    results["methods"]["bayesian_beta"]["wall_time"] = round(time.perf_counter() - t0, 4)

    # ── EMA variants ──
    for alpha in EMA_ALPHAS:
        name = f"EMA (alpha={alpha})"
        key = f"ema_alpha_{alpha}"
        t0 = time.perf_counter()
        est_ema = run_ema(batches, alpha=alpha)
        results["methods"][key] = evaluate_method(name, est_ema)
        results["methods"][key]["alpha"] = alpha
        results["methods"][key]["wall_time"] = round(time.perf_counter() - t0, 4)

    # ── ADWIN ──
    t0 = time.perf_counter()
    est_adwin = run_adwin(batches)
    results["methods"]["adwin"] = evaluate_method("ADWIN", est_adwin)
    results["methods"]["adwin"]["wall_time"] = round(time.perf_counter() - t0, 4)

    # ── SL variants ──
    for hl in SL_HALF_LIVES:
        name = f"SL (half_life={hl})"
        key = f"sl_hl_{hl}"
        t0 = time.perf_counter()
        est_sl = run_sl(batches, half_life=hl)
        results["methods"][key] = evaluate_method(name, est_sl)
        results["methods"][key]["half_life"] = hl
        results["methods"][key]["wall_time"] = round(time.perf_counter() - t0, 4)

    return results


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════


def compute_summary(scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute cross-scenario summary statistics.

    For each method, aggregate MAE and KL across scenarios.
    Identify the best SL half_life per scenario and overall.
    """
    summary: dict[str, Any] = {"per_scenario": {}, "rankings": {}}

    all_methods: set[str] = set()
    for sr in scenario_results:
        all_methods.update(sr["methods"].keys())

    # Per-scenario rankings by MAE
    for sr in scenario_results:
        scenario = sr["scenario"]
        method_maes = {
            m: sr["methods"][m]["mae"]
            for m in sr["methods"]
        }
        ranked = sorted(method_maes.items(), key=lambda x: x[1])
        summary["per_scenario"][scenario] = {
            "ranking": [(m, round(mae, 5)) for m, mae in ranked],
            "best_method": ranked[0][0],
            "best_mae": round(ranked[0][1], 5),
        }

        # Best SL variant for this scenario
        sl_methods = {m: mae for m, mae in method_maes.items() if m.startswith("sl_")}
        if sl_methods:
            best_sl = min(sl_methods, key=sl_methods.get)
            summary["per_scenario"][scenario]["best_sl"] = best_sl
            summary["per_scenario"][scenario]["best_sl_mae"] = round(sl_methods[best_sl], 5)

    # Cross-scenario average MAE per method
    avg_maes: dict[str, float] = {}
    for m in all_methods:
        maes = [sr["methods"][m]["mae"] for sr in scenario_results if m in sr["methods"]]
        if maes:
            avg_maes[m] = float(np.mean(maes))

    ranked_overall = sorted(avg_maes.items(), key=lambda x: x[1])
    summary["rankings"]["by_avg_mae"] = [(m, round(mae, 5)) for m, mae in ranked_overall]

    # SL vs best non-SL comparison
    sl_avg = {m: mae for m, mae in avg_maes.items() if m.startswith("sl_")}
    non_sl_avg = {m: mae for m, mae in avg_maes.items() if not m.startswith("sl_")}

    if sl_avg and non_sl_avg:
        best_sl_key = min(sl_avg, key=sl_avg.get)
        best_non_sl_key = min(non_sl_avg, key=non_sl_avg.get)
        summary["rankings"]["best_sl_overall"] = {
            "method": best_sl_key,
            "avg_mae": round(sl_avg[best_sl_key], 5),
        }
        summary["rankings"]["best_non_sl_overall"] = {
            "method": best_non_sl_key,
            "avg_mae": round(non_sl_avg[best_non_sl_key], 5),
        }
        improvement = non_sl_avg[best_non_sl_key] - sl_avg[best_sl_key]
        relative = improvement / non_sl_avg[best_non_sl_key] if non_sl_avg[best_non_sl_key] > 0 else 0
        summary["rankings"]["sl_improvement_over_best_baseline"] = {
            "absolute": round(improvement, 5),
            "relative_pct": round(relative * 100, 2),
            "sl_wins": improvement > 0,
        }

    return summary


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    print("=" * 70)
    print("EN1.2 — Temporal Regime Adaptation")
    print("=" * 70)

    set_global_seed(GLOBAL_SEED)
    env = log_environment()
    print(f"  Python:      {env['python_version']}")
    print(f"  Platform:    {env['platform']}")
    print(f"  jsonld-ex:   {env.get('jsonld_ex_version', 'unknown')}")
    print(f"  Seed:        {GLOBAL_SEED}")
    print(f"  Timesteps:   {N_TIMESTEPS}")
    print(f"  Batch size:  {BATCH_SIZE}")
    print(f"  Bootstrap:   {N_BOOTSTRAP}")
    print("-" * 70)

    # Check river availability
    try:
        import river
        print(f"  river:       {river.__version__}")
    except ImportError:
        print("  ERROR: river library not found. Install with: pip install river")
        sys.exit(1)

    rng = np.random.RandomState(GLOBAL_SEED)
    t_start = time.perf_counter()

    # ── Run all scenarios ──
    scenario_results: list[dict[str, Any]] = []
    # Deterministic per-scenario seeds (hash() is non-deterministic in Python 3.3+)
    scenario_seeds = {
        "sudden": GLOBAL_SEED + 1,
        "gradual": GLOBAL_SEED + 2,
        "recurring": GLOBAL_SEED + 3,
    }

    for scenario_name, gt_fn in DRIFT_SCENARIOS.items():
        # Each scenario gets a fresh RNG derived from a fixed seed
        # to ensure reproducibility regardless of execution order
        scenario_rng = np.random.RandomState(scenario_seeds[scenario_name])
        result = run_scenario(scenario_name, gt_fn, scenario_rng)
        scenario_results.append(result)

    # ── Summary analysis ──
    summary = compute_summary(scenario_results)
    total_time = time.perf_counter() - t_start

    # ── Print summary ──
    print("\n" + "-" * 70)
    print("  SUMMARY")
    print("-" * 70)

    for scenario, info in summary["per_scenario"].items():
        print(f"\n  {scenario}:")
        print(f"    Best overall:  {info['best_method']} (MAE={info['best_mae']})")
        if "best_sl" in info:
            print(f"    Best SL:       {info['best_sl']} (MAE={info['best_sl_mae']})")

    print(f"\n  Cross-scenario ranking (by avg MAE):")
    for rank, (method, mae) in enumerate(summary["rankings"]["by_avg_mae"][:5], 1):
        print(f"    {rank}. {method:30s} avg_MAE={mae:.5f}")

    sl_info = summary["rankings"].get("sl_improvement_over_best_baseline", {})
    if sl_info:
        if sl_info["sl_wins"]:
            print(f"\n  SL improves over best baseline by {sl_info['relative_pct']:.1f}%")
        else:
            print(f"\n  Best baseline outperforms best SL by {-sl_info['relative_pct']:.1f}%")

    print(f"\n  Total wall time: {total_time:.1f}s")

    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "en1_2_results.json"

    # Strip per-timestep arrays from saved results to keep file manageable
    # (they can be regenerated deterministically from the seed)
    saved_scenarios = []
    for sr in scenario_results:
        sr_copy = dict(sr)
        sr_copy.pop("true_p", None)
        methods_copy = {}
        for m, md in sr_copy["methods"].items():
            md_copy = dict(md)
            md_copy.pop("estimates", None)
            methods_copy[m] = md_copy
        sr_copy["methods"] = methods_copy
        saved_scenarios.append(sr_copy)

    experiment_result = ExperimentResult(
        experiment_id="EN1.2",
        parameters={
            "global_seed": GLOBAL_SEED,
            "n_timesteps": N_TIMESTEPS,
            "batch_size": BATCH_SIZE,
            "n_bootstrap": N_BOOTSTRAP,
            "ema_alphas": EMA_ALPHAS,
            "sl_half_lives": SL_HALF_LIVES,
            "bayesian_forgetting_factor": BAYESIAN_FORGETTING_FACTOR,
            "bayesian_prior": [BAYESIAN_PRIOR_ALPHA, BAYESIAN_PRIOR_BETA],
            "drift_scenarios": list(DRIFT_SCENARIOS.keys()),
        },
        metrics={
            "total_wall_time_seconds": round(total_time, 4),
            "summary": summary,
        },
        raw_data={
            "scenario_results": saved_scenarios,
        },
        environment=env,
        notes=(
            f"EN1.2: Temporal regime adaptation. {len(DRIFT_SCENARIOS)} drift "
            f"scenarios x {1 + len(EMA_ALPHAS) + 1 + len(SL_HALF_LIVES)} methods. "
            f"SL vs best baseline: "
            f"{'SL wins' if sl_info.get('sl_wins') else 'baseline wins'} "
            f"({sl_info.get('relative_pct', 'N/A')}%)."
        ),
    )
    experiment_result.save_json(str(output_path))
    print(f"\n  Results saved to: {output_path}")

    # Timestamped archive
    ts = experiment_result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    archive_path = RESULTS_DIR / f"en1_2_results_{ts}.json"
    experiment_result.save_json(str(archive_path))
    print(f"  Archived to:      {archive_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()