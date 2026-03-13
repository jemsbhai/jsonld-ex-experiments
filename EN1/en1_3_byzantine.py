"""EN1.3 — Byzantine-Robust Fusion Experiment.

Evaluates SL robust_fuse against scalar baselines under adversarial
source injection across multiple configurations.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex

    python experiments/EN1/en1_3_byzantine.py
    python experiments/EN1/en1_3_byzantine.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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

from en1_3_core import (
    AdversarialStrategy,
    evaluate_single_scenario,
    generate_ground_truth,
    generate_honest_opinions,
    generate_adversarial_opinions,
    fuse_scalar_mean,
    fuse_scalar_trimmed_mean,
    fuse_sl_cumulative,
    fuse_sl_robust,
    fuse_sl_trust_discount,
    compute_accuracy,
    compute_f1,
    _transpose_opinions,
    _learn_trust_opinions,
    compute_detection_rate,
)

GLOBAL_SEED = 42
N_BOOTSTRAP = 1000
N_SEEDS = 20  # Run each config with 20 seeds for bootstrap CIs

# Experimental grid
N_INSTANCES = 500
N_HONEST = 10
HONEST_ACCURACY = 0.85
N_ADVERSARIAL_LEVELS = [0, 1, 2, 3, 4, 5]
ADVERSARIAL_STRATEGIES = [
    AdversarialStrategy.RANDOM,
    AdversarialStrategy.INVERSION,
    AdversarialStrategy.TARGETED,
]
ROBUST_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30]


def run_experiment(
    n_instances: int = N_INSTANCES,
    n_honest: int = N_HONEST,
    honest_accuracy: float = HONEST_ACCURACY,
    n_adversarial_levels: List[int] | None = None,
    strategies: List[AdversarialStrategy] | None = None,
    robust_thresholds: List[float] | None = None,
    n_seeds: int = N_SEEDS,
    n_bootstrap: int = N_BOOTSTRAP,
    verbose: bool = True,
) -> Dict[str, Any]:

    if n_adversarial_levels is None:
        n_adversarial_levels = N_ADVERSARIAL_LEVELS
    if strategies is None:
        strategies = ADVERSARIAL_STRATEGIES
    if robust_thresholds is None:
        robust_thresholds = ROBUST_THRESHOLDS

    all_results: Dict[str, Any] = {}

    for strategy in strategies:
        strat_name = strategy.value
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Strategy: {strat_name}")
            print(f"{'=' * 60}")

        strat_results: Dict[str, Any] = {}

        for n_adv in n_adversarial_levels:
            config_key = f"k{n_adv}"
            if verbose:
                print(f"\n  k={n_adv} adversarial sources "
                      f"({n_honest} honest, acc={honest_accuracy})")

            # Run across multiple seeds
            seed_results: Dict[str, List[Dict[str, Any]]] = {}

            for seed_idx in range(n_seeds):
                seed = GLOBAL_SEED + seed_idx * 1000
                result = evaluate_single_scenario(
                    n_instances=n_instances,
                    n_honest=n_honest,
                    n_adversarial=n_adv,
                    honest_accuracy=honest_accuracy,
                    adversarial_strategy=strategy,
                    robust_thresholds=robust_thresholds,
                    seed=seed,
                )
                for method, metrics in result.items():
                    if method not in seed_results:
                        seed_results[method] = []
                    seed_results[method].append(metrics)

            # Aggregate across seeds with bootstrap CIs
            aggregated: Dict[str, Dict[str, Any]] = {}
            for method, seed_metrics_list in seed_results.items():
                accs = [m["accuracy"] for m in seed_metrics_list]
                f1s = [m["f1"] for m in seed_metrics_list]

                mean_acc = float(np.mean(accs))
                mean_f1 = float(np.mean(f1s))
                ci_acc_lo, _, ci_acc_hi = bootstrap_ci(
                    accs, n_bootstrap=n_bootstrap, seed=GLOBAL_SEED,
                )
                ci_f1_lo, _, ci_f1_hi = bootstrap_ci(
                    f1s, n_bootstrap=n_bootstrap, seed=GLOBAL_SEED,
                )

                entry: Dict[str, Any] = {
                    "mean_accuracy": round(mean_acc, 6),
                    "ci_accuracy": [round(ci_acc_lo, 6), round(ci_acc_hi, 6)],
                    "mean_f1": round(mean_f1, 6),
                    "ci_f1": [round(ci_f1_lo, 6), round(ci_f1_hi, 6)],
                    "n_seeds": n_seeds,
                }

                # Aggregate detection metrics for robust methods
                if any("detection_recall" in m for m in seed_metrics_list):
                    det_recalls = [
                        m.get("detection_recall", 0.0)
                        for m in seed_metrics_list
                    ]
                    det_precs = [
                        m.get("detection_precision", 0.0)
                        for m in seed_metrics_list
                    ]
                    entry["mean_detection_recall"] = round(
                        float(np.mean(det_recalls)), 4)
                    entry["mean_detection_precision"] = round(
                        float(np.mean(det_precs)), 4)
                    entry["mean_removed"] = round(float(np.mean([
                        m.get("mean_removed_per_instance", 0.0)
                        for m in seed_metrics_list
                    ])), 3)

                # Aggregate trust for trust discount
                if any("learned_trust" in m for m in seed_metrics_list):
                    all_trusts = [m["learned_trust"] for m in seed_metrics_list]
                    mean_trusts = np.mean(all_trusts, axis=0).tolist()
                    entry["mean_learned_trust"] = [
                        round(t, 4) for t in mean_trusts
                    ]

                aggregated[method] = entry

            strat_results[config_key] = aggregated

            if verbose:
                print(f"  {'Method':<25} {'Acc':>7} {'F1':>7} "
                      f"{'95% CI(Acc)':>18}")
                print(f"  {'-' * 60}")
                sorted_methods = sorted(
                    aggregated.items(),
                    key=lambda x: -x[1]["mean_accuracy"],
                )
                for method, info in sorted_methods:
                    ci = info["ci_accuracy"]
                    print(f"  {method:<25} {info['mean_accuracy']:>7.3f} "
                          f"{info['mean_f1']:>7.3f} "
                          f"[{ci[0]:.3f}, {ci[1]:.3f}]")

        all_results[strat_name] = strat_results

    return all_results


def main():
    parser = argparse.ArgumentParser(description="EN1.3 Byzantine-Robust Fusion")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    set_global_seed(GLOBAL_SEED)

    if args.dry_run:
        n_inst = 50
        n_seeds = 3
        n_boot = 100
        n_adv_levels = [0, 2]
        strats = [AdversarialStrategy.INVERSION]
        thresholds = [0.10, 0.20]
        print("=== DRY RUN ===\n")
    else:
        n_inst = N_INSTANCES
        n_seeds = N_SEEDS
        n_boot = N_BOOTSTRAP
        n_adv_levels = N_ADVERSARIAL_LEVELS
        strats = ADVERSARIAL_STRATEGIES
        thresholds = ROBUST_THRESHOLDS

    t_start = time.time()
    env = log_environment()

    results = run_experiment(
        n_instances=n_inst,
        n_honest=N_HONEST,
        honest_accuracy=HONEST_ACCURACY,
        n_adversarial_levels=n_adv_levels,
        strategies=strats,
        robust_thresholds=thresholds,
        n_seeds=n_seeds,
        n_bootstrap=n_boot,
        verbose=True,
    )

    elapsed = time.time() - t_start
    mode = "dry" if args.dry_run else "full"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_name = f"en1_3_{mode}_results"

    experiment_result = ExperimentResult(
        experiment_id="EN1.3",
        parameters={
            "n_instances": n_inst,
            "n_honest": N_HONEST,
            "honest_accuracy": HONEST_ACCURACY,
            "n_adversarial_levels": n_adv_levels,
            "strategies": [s.value for s in strats],
            "robust_thresholds": thresholds,
            "n_seeds": n_seeds,
            "n_bootstrap": n_boot,
            "global_seed": GLOBAL_SEED,
            "mode": mode,
        },
        metrics=results,
        environment=env,
        notes=(
            f"Byzantine-robust fusion: {len(strats)} strategies × "
            f"{len(n_adv_levels)} adversarial levels × {n_seeds} seeds. "
            f"{elapsed:.1f}s."
        ),
    )

    primary = RESULTS_DIR / f"{result_name}.json"
    archive = RESULTS_DIR / f"{result_name}_{timestamp}.json"
    experiment_result.save_json(str(primary))
    experiment_result.save_json(str(archive))

    print(f"\n{'=' * 60}")
    print(f"DONE — {elapsed:.1f}s")
    print(f"Results saved to:")
    print(f"  {primary}")
    print(f"  {archive}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
