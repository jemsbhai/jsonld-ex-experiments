"""EN1.3 Ablation — Comprehensive Byzantine-Robust Fusion Experiment.

Full experimental grid:
  A. Stress configs: n_honest ∈ {3,5,7,10}, accuracy ∈ {0.60,0.70,0.80,0.90}
  B. Breaking point analysis per method
  C. Heterogeneous honest quality
  D. All 5 adversary types (random, inversion, targeted, subtle, colluding)
  E. 8 fusion methods with calibration + uncertainty tracking
  F. Pairwise McNemar with Bonferroni correction
  G. 20 seeds per configuration for bootstrap CIs

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex

    python experiments/EN1/en1_3_ablation.py --dry-run
    python experiments/EN1/en1_3_ablation.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from infra.config import set_global_seed
from infra.env_log import log_environment
from infra.results import ExperimentResult
from infra.stats import bootstrap_ci

from en1_3_core import AdversarialStrategy
from en1_3_ablation_core import (
    AdversarialStrategyExt,
    evaluate_extended_scenario,
    find_breaking_point,
)

GLOBAL_SEED = 42
N_BOOTSTRAP = 1000
N_SEEDS = 20


# ═══════════════════════════════════════════════════════════════════
# Experimental grid
# ═══════════════════════════════════════════════════════════════════

# Part 1: Core grid — stress configurations × adversary types × k levels
HONEST_COUNTS = [3, 5, 7, 10]
HONEST_ACCURACIES = [0.60, 0.70, 0.80, 0.90]
ADVERSARIAL_K = [0, 1, 2, 3, 4, 5]
ROBUST_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30]

# All adversary types (base + extended)
BASE_STRATEGIES = [
    AdversarialStrategy.INVERSION,
    AdversarialStrategy.TARGETED,
    AdversarialStrategy.RANDOM,
]
EXT_STRATEGIES = [
    AdversarialStrategyExt.SUBTLE,
    AdversarialStrategyExt.COLLUDING,
]

# Part 2: Heterogeneous honest configurations
HETERO_CONFIGS = [
    {"name": "uniform_low", "accs": [0.65, 0.65, 0.65, 0.65, 0.65]},
    {"name": "uniform_high", "accs": [0.90, 0.90, 0.90, 0.90, 0.90]},
    {"name": "spread_5", "accs": [0.60, 0.70, 0.80, 0.85, 0.95]},
    {"name": "spread_3", "accs": [0.65, 0.80, 0.95]},
    {"name": "one_strong_rest_weak", "accs": [0.95, 0.60, 0.60, 0.60, 0.60]},
    {"name": "one_weak_rest_strong", "accs": [0.60, 0.90, 0.90, 0.90, 0.90]},
]

# Part 3: Breaking point — methods to test
BREAKING_METHODS = [
    "scalar_mean", "scalar_trimmed_mean",
    "sl_cumulative", "sl_robust", "sl_trust_discount",
]


def _strat_label(
    base: Optional[AdversarialStrategy],
    ext: Optional[AdversarialStrategyExt],
) -> str:
    if ext is not None:
        return ext.value
    if base is not None:
        return base.value
    return "none"


def run_core_grid(
    n_instances: int,
    n_seeds: int,
    n_bootstrap: int,
    honest_counts: List[int],
    honest_accuracies: List[float],
    adversarial_k: List[int],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Part 1: Core stress grid — homogeneous honest, all adversary types."""
    results = {}
    total_configs = (
        len(honest_counts) * len(honest_accuracies)
        * (len(BASE_STRATEGIES) + len(EXT_STRATEGIES))
        * len(adversarial_k)
    )
    config_idx = 0

    for n_h in honest_counts:
        for acc in honest_accuracies:
            for strat_base, strat_ext in (
                [(s, None) for s in BASE_STRATEGIES]
                + [(None, s) for s in EXT_STRATEGIES]
            ):
                strat_name = _strat_label(strat_base, strat_ext)

                for k in adversarial_k:
                    config_idx += 1
                    config_key = f"nh{n_h}_acc{int(acc*100)}_k{k}_{strat_name}"

                    if verbose and config_idx % 20 == 1:
                        print(f"  [{config_idx}/{total_configs}] {config_key}")

                    seed_metrics: Dict[str, List[Dict]] = {}

                    for seed_idx in range(n_seeds):
                        seed = GLOBAL_SEED + seed_idx * 1000
                        result = evaluate_extended_scenario(
                            n_instances=n_instances,
                            n_honest=n_h,
                            n_adversarial=k,
                            honest_accuracy=acc,
                            adversarial_strategy=strat_base,
                            adversarial_strategy_ext=strat_ext,
                            robust_thresholds=ROBUST_THRESHOLDS,
                            seed=seed,
                        )
                        for method, metrics in result.items():
                            if method.startswith("_"):
                                continue
                            if method not in seed_metrics:
                                seed_metrics[method] = []
                            seed_metrics[method].append(metrics)

                    # Aggregate
                    agg: Dict[str, Dict] = {}
                    for method, metrics_list in seed_metrics.items():
                        accs_list = [m["accuracy"] for m in metrics_list]
                        f1s = [m["f1"] for m in metrics_list]
                        eces = [m["ece"] for m in metrics_list]
                        briers = [m["brier"] for m in metrics_list]

                        ci_lo, _, ci_hi = bootstrap_ci(
                            accs_list, n_bootstrap=n_bootstrap, seed=GLOBAL_SEED,
                        )

                        entry: Dict[str, Any] = {
                            "mean_accuracy": round(float(np.mean(accs_list)), 6),
                            "ci_accuracy": [round(ci_lo, 6), round(ci_hi, 6)],
                            "mean_f1": round(float(np.mean(f1s)), 6),
                            "mean_ece": round(float(np.mean(eces)), 6),
                            "mean_brier": round(float(np.mean(briers)), 6),
                        }

                        # Uncertainty for SL methods
                        if any("mean_fused_uncertainty" in m for m in metrics_list):
                            us = [m["mean_fused_uncertainty"] for m in metrics_list
                                  if "mean_fused_uncertainty" in m]
                            entry["mean_uncertainty"] = round(float(np.mean(us)), 6)

                        # Detection for robust methods
                        if any("detection_recall" in m for m in metrics_list):
                            entry["mean_det_recall"] = round(float(np.mean([
                                m.get("detection_recall", 0) for m in metrics_list
                            ])), 4)
                            entry["mean_det_precision"] = round(float(np.mean([
                                m.get("detection_precision", 0) for m in metrics_list
                            ])), 4)

                        agg[method] = entry

                    results[config_key] = agg

    return results


def run_heterogeneous_grid(
    n_instances: int,
    n_seeds: int,
    n_bootstrap: int,
    adversarial_k: List[int],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Part 2: Heterogeneous honest quality configurations."""
    results = {}

    for hconfig in HETERO_CONFIGS:
        for strat_base in [AdversarialStrategy.INVERSION, AdversarialStrategy.TARGETED]:
            for k in adversarial_k:
                config_key = f"{hconfig['name']}_k{k}_{strat_base.value}"

                if verbose:
                    print(f"  Hetero: {config_key}")

                seed_metrics: Dict[str, List[Dict]] = {}
                for seed_idx in range(n_seeds):
                    seed = GLOBAL_SEED + seed_idx * 1000
                    result = evaluate_extended_scenario(
                        n_instances=n_instances,
                        honest_accuracies=hconfig["accs"],
                        n_adversarial=k,
                        adversarial_strategy=strat_base,
                        robust_thresholds=ROBUST_THRESHOLDS,
                        seed=seed,
                    )
                    for method, metrics in result.items():
                        if method.startswith("_"):
                            continue
                        if method not in seed_metrics:
                            seed_metrics[method] = []
                        seed_metrics[method].append(metrics)

                agg: Dict[str, Dict] = {}
                for method, metrics_list in seed_metrics.items():
                    accs_list = [m["accuracy"] for m in metrics_list]
                    ci_lo, _, ci_hi = bootstrap_ci(
                        accs_list, n_bootstrap=n_bootstrap, seed=GLOBAL_SEED,
                    )
                    agg[method] = {
                        "mean_accuracy": round(float(np.mean(accs_list)), 6),
                        "ci_accuracy": [round(ci_lo, 6), round(ci_hi, 6)],
                        "mean_f1": round(float(np.mean([m["f1"] for m in metrics_list])), 6),
                        "mean_ece": round(float(np.mean([m["ece"] for m in metrics_list])), 6),
                        "mean_brier": round(float(np.mean([m["brier"] for m in metrics_list])), 6),
                    }
                results[config_key] = agg

    return results


def run_breaking_point_analysis(
    n_instances: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Part 3: Find breaking points for each method."""
    results = {}

    configs = [
        (5, 0.80, AdversarialStrategy.INVERSION),
        (5, 0.80, AdversarialStrategy.TARGETED),
        (5, 0.70, AdversarialStrategy.INVERSION),
        (10, 0.80, AdversarialStrategy.INVERSION),
        (10, 0.90, AdversarialStrategy.INVERSION),
        (3, 0.80, AdversarialStrategy.INVERSION),
    ]

    for n_h, acc, strat in configs:
        for method in BREAKING_METHODS:
            config_key = f"nh{n_h}_acc{int(acc*100)}_{strat.value}_{method}"
            if verbose:
                print(f"  Breaking: {config_key}")

            bp = find_breaking_point(
                n_honest=n_h,
                honest_accuracy=acc,
                adversarial_strategy=strat,
                method=method,
                baseline_drop=0.05,
                max_adversarial=min(n_h + 2, 10),
                n_instances=n_instances,
                seed=GLOBAL_SEED,
            )
            results[config_key] = bp

    return results


def main():
    parser = argparse.ArgumentParser(description="EN1.3 Ablation — Full Grid")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--part", type=int, default=None,
                        help="Run only part 1, 2, or 3")
    args = parser.parse_args()

    set_global_seed(GLOBAL_SEED)

    if args.dry_run:
        n_inst = 100
        n_seeds = 3
        n_boot = 100
        h_counts = [5]
        h_accs = [0.80]
        k_levels = [0, 2]
        print("=== DRY RUN ===\n")
    else:
        n_inst = 500
        n_seeds = N_SEEDS
        n_boot = N_BOOTSTRAP
        h_counts = HONEST_COUNTS
        h_accs = HONEST_ACCURACIES
        k_levels = ADVERSARIAL_K

    t_start = time.time()
    env = log_environment()
    all_results: Dict[str, Any] = {}

    # Part 1: Core stress grid
    if args.part is None or args.part == 1:
        print(f"\n{'='*60}")
        print("PART 1: Core stress grid")
        print(f"{'='*60}")
        t1 = time.time()
        all_results["core_grid"] = run_core_grid(
            n_instances=n_inst, n_seeds=n_seeds, n_bootstrap=n_boot,
            honest_counts=h_counts, honest_accuracies=h_accs,
            adversarial_k=k_levels, verbose=True,
        )
        print(f"  Part 1 done: {time.time()-t1:.1f}s, "
              f"{len(all_results['core_grid'])} configs")

    # Part 2: Heterogeneous honest
    if args.part is None or args.part == 2:
        print(f"\n{'='*60}")
        print("PART 2: Heterogeneous honest quality")
        print(f"{'='*60}")
        t2 = time.time()
        all_results["heterogeneous"] = run_heterogeneous_grid(
            n_instances=n_inst, n_seeds=n_seeds, n_bootstrap=n_boot,
            adversarial_k=k_levels, verbose=True,
        )
        print(f"  Part 2 done: {time.time()-t2:.1f}s, "
              f"{len(all_results['heterogeneous'])} configs")

    # Part 3: Breaking point analysis
    if args.part is None or args.part == 3:
        print(f"\n{'='*60}")
        print("PART 3: Breaking point analysis")
        print(f"{'='*60}")
        t3 = time.time()
        all_results["breaking_points"] = run_breaking_point_analysis(
            n_instances=n_inst, verbose=True,
        )
        print(f"  Part 3 done: {time.time()-t3:.1f}s, "
              f"{len(all_results['breaking_points'])} configs")

    elapsed = time.time() - t_start
    mode = "dry" if args.dry_run else "full"
    if args.part is not None:
        mode += f"_part{args.part}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_name = f"en1_3_ablation_{mode}_results"

    experiment_result = ExperimentResult(
        experiment_id="EN1.3-ablation",
        parameters={
            "n_instances": n_inst,
            "honest_counts": h_counts,
            "honest_accuracies": h_accs,
            "adversarial_k": k_levels,
            "robust_thresholds": ROBUST_THRESHOLDS,
            "hetero_configs": [h["name"] for h in HETERO_CONFIGS],
            "n_seeds": n_seeds,
            "n_bootstrap": n_boot,
            "global_seed": GLOBAL_SEED,
            "mode": mode,
        },
        metrics=all_results,
        environment=env,
        notes=(
            f"EN1.3 comprehensive ablation: core grid "
            f"({len(h_counts)}×{len(h_accs)}×5strats×{len(k_levels)}k), "
            f"heterogeneous ({len(HETERO_CONFIGS)} configs), "
            f"breaking points ({len(BREAKING_METHODS)} methods). "
            f"{elapsed:.1f}s."
        ),
    )

    primary = RESULTS_DIR / f"{result_name}.json"
    archive = RESULTS_DIR / f"{result_name}_{timestamp}.json"
    experiment_result.save_json(str(primary))
    experiment_result.save_json(str(archive))

    print(f"\n{'='*60}")
    print(f"DONE — {elapsed:.1f}s")
    print(f"Results saved to:")
    print(f"  {primary}")
    print(f"  {archive}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
