#!/usr/bin/env python
"""EN1.2b — Temporal Regime Adaptation (Extended NeurIPS Evaluation).

Extends EN1.2 with:
    1. Multi-seed replications (20 seeds) for statistical power
    2. Batch size sweep (5, 10, 25, 50) to characterize signal-to-noise
    3. Robustness metrics (worst-case MAE, minimax regret)
    4. Paired Wilcoxon signed-rank tests for SL vs each baseline
    5. Cohen's d effect sizes

Design: 20 seeds x 4 batch sizes x 3 scenarios x 10 methods = 2,400 runs.

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN1/en1_2b_extended.py

Output:
    experiments/EN1/results/en1_2b_results.json
    experiments/EN1/results/en1_2b_results_<timestamp>.json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

# ── Path setup ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "packages" / "python" / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from experiments.infra.config import set_global_seed
from experiments.infra.results import ExperimentResult
from experiments.infra.env_log import log_environment
from experiments.infra.stats import bootstrap_ci

# Import core experiment functions from EN1.2
from experiments.EN1.en1_2_temporal_adaptation import (
    DRIFT_SCENARIOS,
    generate_stream,
    run_bayesian_beta,
    run_ema,
    run_adwin,
    run_sl,
    compute_mae,
    compute_kl_divergence,
    compute_time_to_detection_sudden,
    compute_time_to_detection_recurring,
    BAYESIAN_FORGETTING_FACTOR,
    BAYESIAN_PRIOR_ALPHA,
    BAYESIAN_PRIOR_BETA,
    KL_EPSILON,
)

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

GLOBAL_SEED = 42
N_SEEDS = 20
N_TIMESTEPS = 1000
BATCH_SIZES = [5, 10, 25, 50]
N_BOOTSTRAP = 1000
RESULTS_DIR = Path(__file__).resolve().parent / "results"

EMA_ALPHAS = [0.01, 0.05, 0.1]
SL_HALF_LIVES = [25, 50, 100, 200, 400]

# Method registry: (key, display_name, runner_factory)
# runner_factory: (batches) -> estimates
def _make_methods():
    """Build method registry. Each entry: (key, name, runner_fn)."""
    methods = []
    methods.append(("bayesian_beta", "Bayesian Beta", lambda b: run_bayesian_beta(b)))
    for a in EMA_ALPHAS:
        methods.append((f"ema_{a}", f"EMA(α={a})", lambda b, _a=a: run_ema(b, alpha=_a)))
    methods.append(("adwin", "ADWIN", lambda b: run_adwin(b)))
    for hl in SL_HALF_LIVES:
        methods.append((f"sl_{hl}", f"SL(hl={hl})", lambda b, _hl=hl: run_sl(b, half_life=_hl)))
    return methods

METHODS = _make_methods()
METHOD_KEYS = [m[0] for m in METHODS]
METHOD_NAMES = {m[0]: m[1] for m in METHODS}
SL_KEYS = [k for k in METHOD_KEYS if k.startswith("sl_")]
BASELINE_KEYS = [k for k in METHOD_KEYS if not k.startswith("sl_")]


# ═══════════════════════════════════════════════════════════════════
# CORE LOOP
# ═══════════════════════════════════════════════════════════════════


def run_single(
    scenario_name: str,
    ground_truth_fn,
    batch_size: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Run all methods on one (scenario, batch_size, seed) config.

    Returns: {method_key: {"mae": ..., "kl": ..., "ttd": ...}}
    """
    rng = np.random.RandomState(seed)
    true_p, batches = generate_stream(ground_truth_fn, N_TIMESTEPS, batch_size, rng)

    results: dict[str, dict[str, float]] = {}
    for key, name, runner in METHODS:
        estimates = runner(batches)
        mae = compute_mae(true_p, estimates)
        kl = compute_kl_divergence(true_p, estimates)

        entry: dict[str, Any] = {"mae": mae, "kl": kl}

        if scenario_name == "sudden":
            ttd = compute_time_to_detection_sudden(estimates)
            entry["ttd"] = ttd if ttd is not None else N_TIMESTEPS
        elif scenario_name == "recurring":
            ttds = compute_time_to_detection_recurring(estimates)
            valid = [d for d in ttds if d is not None]
            entry["mean_ttd"] = float(np.mean(valid)) if valid else N_TIMESTEPS

        results[key] = entry

    return results


# ═══════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d effect size for paired samples."""
    diff = x - y
    d_mean = np.mean(diff)
    d_std = np.std(diff, ddof=1)
    if d_std == 0:
        return 0.0 if d_mean == 0 else float("inf") * np.sign(d_mean)
    return float(d_mean / d_std)


def wilcoxon_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test (paired, two-sided).

    Returns (statistic, p_value). If all differences are zero,
    returns (0.0, 1.0).
    """
    diff = x - y
    if np.all(diff == 0):
        return 0.0, 1.0
    try:
        stat, p = scipy_stats.wilcoxon(x, y, alternative="two-sided")
        return float(stat), float(p)
    except ValueError:
        # All differences are the same sign or too few samples
        return 0.0, 1.0


def analyze_pairwise(
    mae_matrix: dict[str, np.ndarray],
    best_sl_key: str,
) -> dict[str, Any]:
    """Pairwise statistical comparison of best SL vs each baseline.

    Args:
        mae_matrix: {method_key: array of shape (n_seeds,)} with MAE values
        best_sl_key: key of the best SL method

    Returns dict of per-baseline test results.
    """
    sl_maes = mae_matrix[best_sl_key]
    comparisons = {}
    for bk in BASELINE_KEYS:
        base_maes = mae_matrix[bk]
        stat, p_val = wilcoxon_test(sl_maes, base_maes)
        d = cohens_d(base_maes, sl_maes)  # positive d = SL better
        comparisons[bk] = {
            "sl_mean_mae": float(np.mean(sl_maes)),
            "baseline_mean_mae": float(np.mean(base_maes)),
            "improvement_abs": float(np.mean(base_maes) - np.mean(sl_maes)),
            "improvement_pct": float(
                (np.mean(base_maes) - np.mean(sl_maes)) / np.mean(base_maes) * 100
            ) if np.mean(base_maes) > 0 else 0.0,
            "wilcoxon_stat": stat,
            "wilcoxon_p": p_val,
            "significant_005": p_val < 0.05,
            "significant_001": p_val < 0.01,
            "cohens_d": d,
            "effect_size_label": (
                "large" if abs(d) >= 0.8 else
                "medium" if abs(d) >= 0.5 else
                "small" if abs(d) >= 0.2 else
                "negligible"
            ),
        }
    return comparisons


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    print("=" * 75)
    print("EN1.2b — Temporal Regime Adaptation (Extended NeurIPS Evaluation)")
    print("=" * 75)

    set_global_seed(GLOBAL_SEED)
    env = log_environment()
    print(f"  Python:      {env['python_version']}")
    print(f"  Platform:    {env['platform']}")
    print(f"  jsonld-ex:   {env.get('jsonld_ex_version', 'unknown')}")
    print(f"  Seeds:       {N_SEEDS}")
    print(f"  Batch sizes: {BATCH_SIZES}")
    print(f"  Scenarios:   {list(DRIFT_SCENARIOS.keys())}")
    print(f"  Methods:     {len(METHODS)}")
    n_total = N_SEEDS * len(BATCH_SIZES) * len(DRIFT_SCENARIOS) * len(METHODS)
    print(f"  Total runs:  {n_total:,}")
    print("-" * 75)

    try:
        import river
        print(f"  river:       {river.__version__}")
    except ImportError:
        print("  ERROR: river not found. pip install river")
        sys.exit(1)

    t_start = time.perf_counter()
    seeds = [GLOBAL_SEED + i for i in range(N_SEEDS)]

    # ── Collect raw results ──
    # Structure: raw[batch_size][scenario][method_key] = list of MAE (len=N_SEEDS)
    raw: dict[int, dict[str, dict[str, list[float]]]] = {}
    raw_kl: dict[int, dict[str, dict[str, list[float]]]] = {}
    raw_ttd: dict[int, dict[str, dict[str, list[float]]]] = {}

    run_count = 0
    for bs in BATCH_SIZES:
        raw[bs] = {}
        raw_kl[bs] = {}
        raw_ttd[bs] = {}
        for sname, gt_fn in DRIFT_SCENARIOS.items():
            raw[bs][sname] = {k: [] for k in METHOD_KEYS}
            raw_kl[bs][sname] = {k: [] for k in METHOD_KEYS}
            raw_ttd[bs][sname] = {k: [] for k in METHOD_KEYS}

            for seed in seeds:
                result = run_single(sname, gt_fn, bs, seed)
                for mk in METHOD_KEYS:
                    raw[bs][sname][mk].append(result[mk]["mae"])
                    raw_kl[bs][sname][mk].append(result[mk]["kl"])
                    ttd_val = result[mk].get("ttd", result[mk].get("mean_ttd"))
                    if ttd_val is not None:
                        raw_ttd[bs][sname][mk].append(ttd_val)
                run_count += len(METHODS)

            # Progress
            elapsed = time.perf_counter() - t_start
            pct = run_count / n_total * 100
            print(f"  [{pct:5.1f}%] bs={bs:2d}, {sname:10s}: "
                  f"{run_count:4d}/{n_total} runs ({elapsed:.0f}s)")

    total_time = time.perf_counter() - t_start

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    analysis: dict[str, Any] = {
        "per_batch_size": {},
        "robustness": {},
        "batch_size_effect": {},
    }

    for bs in BATCH_SIZES:
        bs_analysis: dict[str, Any] = {"per_scenario": {}, "cross_scenario": {}}

        # ── Per-scenario analysis ──
        for sname in DRIFT_SCENARIOS:
            mae_arrays = {mk: np.array(raw[bs][sname][mk]) for mk in METHOD_KEYS}

            # Mean MAE with bootstrap CI
            scenario_results: dict[str, Any] = {}
            for mk in METHOD_KEYS:
                arr = mae_arrays[mk]
                lo, mean, hi = bootstrap_ci(arr.tolist(), N_BOOTSTRAP, seed=GLOBAL_SEED)
                scenario_results[mk] = {
                    "mean_mae": round(float(np.mean(arr)), 5),
                    "std_mae": round(float(np.std(arr, ddof=1)), 5),
                    "ci_95": [round(lo, 5), round(hi, 5)],
                    "mean_kl": round(float(np.mean(raw_kl[bs][sname][mk])), 6),
                }
                if raw_ttd[bs][sname][mk]:
                    scenario_results[mk]["mean_ttd"] = round(
                        float(np.mean(raw_ttd[bs][sname][mk])), 1
                    )

            # Rank by mean MAE
            ranked = sorted(scenario_results.items(), key=lambda x: x[1]["mean_mae"])
            scenario_results["_ranking"] = [r[0] for r in ranked]

            # Best SL for this scenario
            sl_means = {mk: scenario_results[mk]["mean_mae"] for mk in SL_KEYS}
            best_sl = min(sl_means, key=sl_means.get)
            scenario_results["_best_sl"] = best_sl

            # Pairwise tests: best SL vs all baselines
            scenario_results["_pairwise"] = analyze_pairwise(mae_arrays, best_sl)

            bs_analysis["per_scenario"][sname] = scenario_results

        # ── Cross-scenario average MAE ──
        cross_avg: dict[str, float] = {}
        for mk in METHOD_KEYS:
            avg = float(np.mean([
                bs_analysis["per_scenario"][sn][mk]["mean_mae"]
                for sn in DRIFT_SCENARIOS
            ]))
            cross_avg[mk] = round(avg, 5)

        ranked_overall = sorted(cross_avg.items(), key=lambda x: x[1])
        bs_analysis["cross_scenario"]["avg_mae"] = dict(ranked_overall)
        bs_analysis["cross_scenario"]["ranking"] = [r[0] for r in ranked_overall]

        # Best SL overall vs best baseline overall
        best_sl_overall = min(
            (k for k in SL_KEYS), key=lambda k: cross_avg[k]
        )
        best_base_overall = min(
            (k for k in BASELINE_KEYS), key=lambda k: cross_avg[k]
        )
        improvement = cross_avg[best_base_overall] - cross_avg[best_sl_overall]
        rel_pct = improvement / cross_avg[best_base_overall] * 100 if cross_avg[best_base_overall] > 0 else 0

        bs_analysis["cross_scenario"]["best_sl"] = best_sl_overall
        bs_analysis["cross_scenario"]["best_baseline"] = best_base_overall
        bs_analysis["cross_scenario"]["sl_vs_baseline"] = {
            "sl_avg_mae": cross_avg[best_sl_overall],
            "baseline_avg_mae": cross_avg[best_base_overall],
            "improvement_abs": round(improvement, 5),
            "improvement_pct": round(rel_pct, 2),
            "sl_wins": improvement > 0,
        }

        # ── Robustness: worst-case MAE across scenarios ──
        worst_case: dict[str, float] = {}
        for mk in METHOD_KEYS:
            worst = max(
                bs_analysis["per_scenario"][sn][mk]["mean_mae"]
                for sn in DRIFT_SCENARIOS
            )
            worst_case[mk] = round(worst, 5)

        ranked_robust = sorted(worst_case.items(), key=lambda x: x[1])
        bs_analysis["cross_scenario"]["worst_case_mae"] = dict(ranked_robust)
        bs_analysis["cross_scenario"]["robustness_ranking"] = [r[0] for r in ranked_robust]

        best_sl_robust = min((k for k in SL_KEYS), key=lambda k: worst_case[k])
        best_base_robust = min((k for k in BASELINE_KEYS), key=lambda k: worst_case[k])
        robust_improvement = worst_case[best_base_robust] - worst_case[best_sl_robust]
        robust_rel = robust_improvement / worst_case[best_base_robust] * 100 if worst_case[best_base_robust] > 0 else 0

        bs_analysis["cross_scenario"]["robustness_sl_vs_baseline"] = {
            "sl_worst_mae": worst_case[best_sl_robust],
            "baseline_worst_mae": worst_case[best_base_robust],
            "improvement_abs": round(robust_improvement, 5),
            "improvement_pct": round(robust_rel, 2),
            "sl_wins": robust_improvement > 0,
            "best_sl": best_sl_robust,
            "best_baseline": best_base_robust,
        }

        analysis["per_batch_size"][bs] = bs_analysis

    # ── Batch size effect: how does SL advantage scale? ──
    for bs in BATCH_SIZES:
        bsa = analysis["per_batch_size"][bs]["cross_scenario"]
        analysis["batch_size_effect"][bs] = {
            "avg_mae_improvement_pct": bsa["sl_vs_baseline"]["improvement_pct"],
            "worst_case_improvement_pct": bsa["robustness_sl_vs_baseline"]["improvement_pct"],
            "best_sl": bsa["best_sl"],
            "best_baseline": bsa["best_baseline"],
        }

    # ═══════════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 75)
    print("  RESULTS SUMMARY")
    print("=" * 75)

    for bs in BATCH_SIZES:
        bsa = analysis["per_batch_size"][bs]["cross_scenario"]
        sl_info = bsa["sl_vs_baseline"]
        rob_info = bsa["robustness_sl_vs_baseline"]
        print(f"\n  batch_size={bs}:")
        print(f"    Avg MAE:    SL {sl_info['sl_avg_mae']:.5f} vs "
              f"baseline {sl_info['baseline_avg_mae']:.5f} "
              f"({'SL wins' if sl_info['sl_wins'] else 'baseline wins'} "
              f"by {abs(sl_info['improvement_pct']):.1f}%)")
        print(f"    Worst-case: SL {rob_info['sl_worst_mae']:.5f} vs "
              f"baseline {rob_info['baseline_worst_mae']:.5f} "
              f"({'SL wins' if rob_info['sl_wins'] else 'baseline wins'} "
              f"by {abs(rob_info['improvement_pct']):.1f}%)")
        print(f"    Ranking:    {bsa['ranking'][:3]}")

    # Per-scenario detail for best batch size
    print(f"\n{'─' * 75}")
    print("  PAIRWISE STATISTICAL TESTS (batch_size=25, best SL vs baselines)")
    print(f"{'─' * 75}")
    bs_detail = analysis["per_batch_size"].get(25, analysis["per_batch_size"][BATCH_SIZES[-1]])
    for sname in DRIFT_SCENARIOS:
        pw = bs_detail["per_scenario"][sname].get("_pairwise", {})
        best_sl = bs_detail["per_scenario"][sname].get("_best_sl", "?")
        print(f"\n  {sname} (SL={best_sl}):")
        for bk, info in pw.items():
            sig = "**" if info["significant_001"] else "*" if info["significant_005"] else "ns"
            print(f"    vs {METHOD_NAMES[bk]:20s}: "
                  f"Δ={info['improvement_pct']:+.1f}% "
                  f"d={info['cohens_d']:.2f}({info['effect_size_label']}) "
                  f"p={info['wilcoxon_p']:.4f} {sig}")

    # Batch size scaling
    print(f"\n{'─' * 75}")
    print("  BATCH SIZE SCALING")
    print(f"{'─' * 75}")
    for bs in BATCH_SIZES:
        bse = analysis["batch_size_effect"][bs]
        print(f"  bs={bs:2d}: avg_mae +{bse['avg_mae_improvement_pct']:.1f}%, "
              f"worst_case +{bse['worst_case_improvement_pct']:.1f}%")

    print(f"\n  Total wall time: {total_time:.1f}s ({run_count:,} runs)")

    # ═══════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "en1_2b_results.json"

    experiment_result = ExperimentResult(
        experiment_id="EN1.2b",
        parameters={
            "global_seed": GLOBAL_SEED,
            "n_seeds": N_SEEDS,
            "seeds": seeds,
            "n_timesteps": N_TIMESTEPS,
            "batch_sizes": BATCH_SIZES,
            "n_bootstrap": N_BOOTSTRAP,
            "ema_alphas": EMA_ALPHAS,
            "sl_half_lives": SL_HALF_LIVES,
            "bayesian_forgetting_factor": BAYESIAN_FORGETTING_FACTOR,
            "drift_scenarios": list(DRIFT_SCENARIOS.keys()),
            "n_methods": len(METHODS),
            "method_keys": METHOD_KEYS,
            "total_runs": run_count,
        },
        metrics={
            "total_wall_time_seconds": round(total_time, 4),
            "analysis": analysis,
        },
        raw_data={
            "mae_by_bs_scenario_method_seed": {
                str(bs): {
                    sn: {mk: vals for mk, vals in raw[bs][sn].items()}
                    for sn in DRIFT_SCENARIOS
                }
                for bs in BATCH_SIZES
            },
        },
        environment=env,
        notes=(
            f"EN1.2b: Extended temporal regime adaptation. "
            f"{N_SEEDS} seeds x {len(BATCH_SIZES)} batch sizes x "
            f"{len(DRIFT_SCENARIOS)} scenarios x {len(METHODS)} methods = "
            f"{run_count} total runs. "
            f"Includes Wilcoxon signed-rank tests and Cohen's d effect sizes."
        ),
    )
    experiment_result.save_json(str(output_path))
    print(f"\n  Results saved to: {output_path}")

    ts = experiment_result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    archive_path = RESULTS_DIR / f"en1_2b_results_{ts}.json"
    experiment_result.save_json(str(archive_path))
    print(f"  Archived to:      {archive_path.name}")
    print("=" * 75)


if __name__ == "__main__":
    main()
