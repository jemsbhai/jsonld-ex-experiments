"""EN1.3 Wrong-k Analysis — The Practitioner's Dilemma.

Core argument: scalar_trimmed_mean requires choosing k_hat (number of
sources to trim). In practice, k_true is unknown. SL trust_discount
doesn't need this hyperparameter.

This experiment runs trimmed_mean with FIXED k_hat ∈ {0,1,2,3,4} across
ALL true adversarial counts k_true ∈ {0..5}, demonstrating that no single
k_hat choice works well everywhere, while SL trust_discount degrades
gracefully without any k input.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN1/en1_3_wrong_k.py
"""
from __future__ import annotations

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
    generate_ground_truth,
    generate_honest_opinions,
    generate_adversarial_opinions,
    _transpose_opinions,
    _learn_trust_opinions,
    fuse_scalar_mean,
    fuse_scalar_trimmed_mean,
    fuse_sl_cumulative,
    fuse_sl_trust_discount,
    compute_accuracy,
    compute_f1,
)

GLOBAL_SEED = 42
N_SEEDS = 20
N_INSTANCES = 500
N_BOOTSTRAP = 1000

# Configurations to test
CONFIGS = [
    {"nh": 5, "acc": 0.80, "label": "nh5_acc80"},
    {"nh": 5, "acc": 0.90, "label": "nh5_acc90"},
    {"nh": 7, "acc": 0.80, "label": "nh7_acc80"},
    {"nh": 10, "acc": 0.80, "label": "nh10_acc80"},
    {"nh": 10, "acc": 0.90, "label": "nh10_acc90"},
    {"nh": 3, "acc": 0.80, "label": "nh3_acc80"},
]

K_TRUES = [0, 1, 2, 3, 4, 5]
K_HATS = [0, 1, 2, 3, 4]  # fixed trim levels to test
STRATEGIES = [AdversarialStrategy.INVERSION, AdversarialStrategy.TARGETED]


def run_wrong_k_analysis() -> Dict[str, Any]:
    results = {}

    for config in CONFIGS:
        nh = config["nh"]
        acc = config["acc"]
        label = config["label"]

        for strat in STRATEGIES:
            strat_name = strat.value

            for k_true in K_TRUES:
                config_key = f"{label}_k{k_true}_{strat_name}"
                print(f"  {config_key}")

                # Collect per-seed results for each method
                method_accs: Dict[str, List[float]] = {}

                for seed_idx in range(N_SEEDS):
                    seed = GLOBAL_SEED + seed_idx * 1000

                    # Generate data
                    gt = generate_ground_truth(N_INSTANCES, 0.6, seed=seed)
                    honest = generate_honest_opinions(gt, nh, acc, seed=seed+1)
                    adv = generate_adversarial_opinions(
                        gt, k_true, strat, seed=seed+2,
                    ) if k_true > 0 else []

                    all_sources = honest + adv
                    per_instance = _transpose_opinions(all_sources)
                    n_total = len(all_sources)

                    # SL trust_discount (no k needed)
                    trust = _learn_trust_opinions(per_instance, n_total)
                    preds_td = fuse_sl_trust_discount(per_instance, trust)
                    acc_td = compute_accuracy(gt, preds_td)
                    method_accs.setdefault("sl_trust_discount", []).append(acc_td)

                    # SL cumulative (no k needed)
                    from en1_3_core import fuse_sl_cumulative
                    preds_cum = fuse_sl_cumulative(per_instance)
                    method_accs.setdefault("sl_cumulative", []).append(
                        compute_accuracy(gt, preds_cum))

                    # Scalar mean (no k needed)
                    preds_mean = fuse_scalar_mean(per_instance)
                    method_accs.setdefault("scalar_mean", []).append(
                        compute_accuracy(gt, preds_mean))

                    # Trimmed mean with EACH fixed k_hat
                    for k_hat in K_HATS:
                        preds_trim = fuse_scalar_trimmed_mean(
                            per_instance, k=k_hat,
                        )
                        method_accs.setdefault(
                            f"trimmed_k{k_hat}", []
                        ).append(compute_accuracy(gt, preds_trim))

                    # Oracle trimmed (knows k_true)
                    preds_oracle = fuse_scalar_trimmed_mean(
                        per_instance, k=max(1, k_true),
                    )
                    method_accs.setdefault("trimmed_oracle", []).append(
                        compute_accuracy(gt, preds_oracle))

                # Aggregate
                agg: Dict[str, Dict[str, Any]] = {}
                for method, acc_list in method_accs.items():
                    mean_a = float(np.mean(acc_list))
                    ci_lo, _, ci_hi = bootstrap_ci(
                        acc_list, n_bootstrap=N_BOOTSTRAP, seed=GLOBAL_SEED,
                    )
                    agg[method] = {
                        "mean_accuracy": round(mean_a, 6),
                        "ci_accuracy": [round(ci_lo, 6), round(ci_hi, 6)],
                    }

                results[config_key] = agg

    return results


def print_summary(results: Dict[str, Any]):
    """Print the key wrong-k table."""
    for config in CONFIGS:
        label = config["label"]
        for strat in ["inversion"]:
            print(f"\n{'='*90}")
            print(f"  {label}, {strat}: Fixed k_hat accuracy across k_true")
            print(f"{'='*90}")
            header = f"  {'k_true':>6}"
            for k_hat in K_HATS:
                header += f"  trim_k{k_hat:>1}"
            header += "  oracle  SL_TD    Δ(best_fixed-SL)"
            print(header)
            print(f"  {'-'*85}")

            for k_true in K_TRUES:
                key = f"{label}_k{k_true}_{strat}"
                if key not in results:
                    continue
                data = results[key]
                sl_td = data.get("sl_trust_discount", {}).get("mean_accuracy", 0)
                oracle = data.get("trimmed_oracle", {}).get("mean_accuracy", 0)

                row = f"  {k_true:>6}"
                trim_accs = []
                for k_hat in K_HATS:
                    a = data.get(f"trimmed_k{k_hat}", {}).get("mean_accuracy", 0)
                    trim_accs.append(a)
                    row += f"  {a:>7.3f}"
                # For each k_true, find the BEST fixed k_hat
                best_fixed = max(trim_accs)
                delta = best_fixed - sl_td
                row += f"  {oracle:>6.3f}  {sl_td:.3f}  {delta:>+.3f}"
                print(row)

            # Now the KEY table: for each fixed k_hat, what's its WORST across k_true?
            print(f"\n  Worst-case (minimax) across k_true:")
            print(f"  {'Method':<20} {'worst':>7} {'mean':>7}")
            print(f"  {'-'*35}")
            for k_hat in K_HATS:
                accs = []
                for k_true in K_TRUES:
                    key = f"{label}_k{k_true}_{strat}"
                    if key in results:
                        accs.append(results[key].get(
                            f"trimmed_k{k_hat}", {}).get("mean_accuracy", 0))
                print(f"  trimmed_k{k_hat:<9} {min(accs):>7.3f} {np.mean(accs):>7.3f}")

            sl_accs = []
            for k_true in K_TRUES:
                key = f"{label}_k{k_true}_{strat}"
                if key in results:
                    sl_accs.append(results[key].get(
                        "sl_trust_discount", {}).get("mean_accuracy", 0))
            print(f"  {'SL_trust_discount':<20} {min(sl_accs):>7.3f} {np.mean(sl_accs):>7.3f}")


def main():
    set_global_seed(GLOBAL_SEED)
    t_start = time.time()
    env = log_environment()

    print("EN1.3 Wrong-k Analysis")
    print("=" * 60)

    results = run_wrong_k_analysis()
    elapsed = time.time() - t_start

    print_summary(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_result = ExperimentResult(
        experiment_id="EN1.3-wrong-k",
        parameters={
            "configs": CONFIGS,
            "k_trues": K_TRUES,
            "k_hats": K_HATS,
            "strategies": [s.value for s in STRATEGIES],
            "n_seeds": N_SEEDS,
            "n_instances": N_INSTANCES,
            "n_bootstrap": N_BOOTSTRAP,
        },
        metrics=results,
        environment=env,
        notes=f"Wrong-k analysis: {len(CONFIGS)} configs × {len(STRATEGIES)} "
              f"strats × {len(K_TRUES)} k_true × {len(K_HATS)} k_hat × "
              f"{N_SEEDS} seeds. {elapsed:.1f}s.",
    )

    primary = RESULTS_DIR / "en1_3_wrong_k_results.json"
    archive = RESULTS_DIR / f"en1_3_wrong_k_results_{timestamp}.json"
    experiment_result.save_json(str(primary))
    experiment_result.save_json(str(archive))

    print(f"\nDONE — {elapsed:.1f}s")
    print(f"Results: {primary}")


if __name__ == "__main__":
    main()
