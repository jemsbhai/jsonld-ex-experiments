"""EN1.5 Supplementary -- Prior Weight Sweep.

Varies prior_weight W in {1, 2, 5, 10, 20} to show the
shrinkage-accuracy tradeoff. Predicted MAE reduction = W/(N+W).

At N=5: W=1 -> 16.7%, W=2 -> 28.6%, W=5 -> 50.0%, W=10 -> 66.7%, W=20 -> 80.0%

Higher W = more shrinkage = lower MAE at low N, but also more bias
at high N (over-regularization). This sweep demonstrates the
practitioner-tunable tradeoff.

Usage:
    cd experiments
    python EN1/en1_5_prior_weight_sweep.py
    python EN1/en1_5_prior_weight_sweep.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_EXPERIMENTS_ROOT))

from en1_5_core import (
    load_all_bn_kbs,
    load_synthea_kb,
    DeductionKB,
    DeductionEdge,
    compute_calibration_metrics,
    _CONDITIONAL_EVIDENCE,
)
from jsonld_ex.confidence_algebra import Opinion, deduce
from infra.env_log import log_environment


RESULTS_DIR = _SCRIPT_DIR / "results"
PRIOR_WEIGHTS = [1, 2, 5, 10, 20]
N_VALUES = [5, 10, 50, 100, 1000]
DEFAULT_N_REPS = 1000
SEED = 42


def run_trial_with_prior_weight(
    edge: DeductionEdge,
    n_evidence: int,
    prior_weight: float,
    seed: int,
) -> Dict[str, Any]:
    """Run a single deduction trial with a specific prior_weight."""
    rng = np.random.RandomState(seed)

    observations = rng.binomial(1, edge.p_parent, size=n_evidence)
    r = int(observations.sum())
    s = n_evidence - r

    # Scalar baseline (unchanged by prior_weight)
    p_hat = r / n_evidence
    scalar_pred = (
        p_hat * edge.p_child_given_parent
        + (1 - p_hat) * edge.p_child_given_not_parent
    )

    # SL with variable prior_weight
    opinion_x = Opinion.from_evidence(
        positive=r, negative=s,
        prior_weight=prior_weight,
        base_rate=edge.p_parent,
    )

    cond_r = edge.p_child_given_parent * _CONDITIONAL_EVIDENCE
    cond_s = _CONDITIONAL_EVIDENCE - cond_r
    opinion_y_given_x = Opinion.from_evidence(
        positive=cond_r, negative=cond_s,
        prior_weight=2.0,  # conditionals always W=2
        base_rate=edge.p_child,
    )

    cond_nr = edge.p_child_given_not_parent * _CONDITIONAL_EVIDENCE
    cond_ns = _CONDITIONAL_EVIDENCE - cond_nr
    opinion_y_given_not_x = Opinion.from_evidence(
        positive=cond_nr, negative=cond_ns,
        prior_weight=2.0,
        base_rate=edge.p_child,
    )

    deduced = deduce(opinion_x, opinion_y_given_x, opinion_y_given_not_x)
    sl_pred = deduced.projected_probability()

    ground_truth_p = edge.p_child
    outcome = int(rng.binomial(1, ground_truth_p))

    return {
        "scalar_pred": float(scalar_pred),
        "sl_pred": float(sl_pred),
        "ground_truth_p": float(ground_truth_p),
        "outcome": outcome,
    }


def run_prior_weight_sweep(
    kbs: List[DeductionKB],
    prior_weights: List[float],
    n_values: List[int],
    n_reps: int,
    seed: int,
) -> Dict[str, Any]:
    """Sweep prior_weight across all KBs."""
    results = {}

    for w in prior_weights:
        print(f"\n  prior_weight={w}:", end="", flush=True)
        w_results = {}
        t0 = time.perf_counter()

        for n_ev in n_values:
            all_scalar = []
            all_sl = []
            all_outcomes = []
            all_gt = []

            for kb in kbs:
                for edge_idx, edge in enumerate(kb.edges):
                    for rep in range(n_reps):
                        trial_seed = (
                            seed + hash(f"{w}_{edge_idx}_{n_ev}_{rep}") % (2**31)
                        )
                        trial = run_trial_with_prior_weight(
                            edge, n_evidence=n_ev,
                            prior_weight=w, seed=abs(trial_seed),
                        )
                        all_scalar.append(trial["scalar_pred"])
                        all_sl.append(trial["sl_pred"])
                        all_outcomes.append(trial["outcome"])
                        all_gt.append(trial["ground_truth_p"])

            scalar_preds = np.array(all_scalar)
            sl_preds = np.array(all_sl)
            outcomes = np.array(all_outcomes)
            gt = np.array(all_gt)

            sc_m = compute_calibration_metrics(
                scalar_preds, outcomes, n_bins=10, ground_truth_probs=gt,
            )
            sl_m = compute_calibration_metrics(
                sl_preds, outcomes, n_bins=10, ground_truth_probs=gt,
            )

            predicted_reduction = w / (n_ev + w)
            actual_reduction = (
                (sc_m["mae_vs_true"] - sl_m["mae_vs_true"]) / sc_m["mae_vs_true"]
                if sc_m["mae_vs_true"] > 1e-12 else 0
            )

            w_results[n_ev] = {
                "scalar_mae": sc_m["mae_vs_true"],
                "sl_mae": sl_m["mae_vs_true"],
                "scalar_ece": sc_m["ece"],
                "sl_ece": sl_m["ece"],
                "scalar_brier": sc_m["brier"],
                "sl_brier": sl_m["brier"],
                "delta_mae": sc_m["mae_vs_true"] - sl_m["mae_vs_true"],
                "predicted_reduction_pct": predicted_reduction * 100,
                "actual_reduction_pct": actual_reduction * 100,
                "n_trials": len(all_scalar),
            }

            print(f" N={n_ev}({actual_reduction*100:.1f}%)", end="", flush=True)

        elapsed = time.perf_counter() - t0
        print(f"  [{elapsed:.0f}s]")
        results[str(w)] = w_results

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EN1.5 Supplementary: Prior Weight Sweep"
    )
    parser.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quick", action="store_true",
                        help="Quick: 50 reps, 3 models, W in {1,2,10}")
    args = parser.parse_args()

    n_reps = args.n_reps
    prior_weights = PRIOR_WEIGHTS
    n_values = N_VALUES
    bn_models = None

    if args.quick:
        n_reps = 50
        prior_weights = [1, 2, 10]
        n_values = [5, 100]
        bn_models = ["asia", "alarm", "sachs"]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"{'=' * 70}")
    print(f"  EN1.5 Supplementary: Prior Weight Sweep")
    print(f"  Prior weights: {prior_weights}")
    print(f"  N values: {n_values}")
    print(f"  Reps: {n_reps}")
    print(f"{'=' * 70}")

    env_info = log_environment()

    # Load KBs
    print("\nLoading KBs ...")
    kbs = load_all_bn_kbs(model_names=bn_models)
    try:
        kbs.append(load_synthea_kb())
    except Exception as e:
        print(f"  Synthea failed: {e}")

    total_edges = sum(len(kb.edges) for kb in kbs)
    total_trials = total_edges * len(prior_weights) * len(n_values) * n_reps
    print(f"  {len(kbs)} datasets, {total_edges} edges")
    print(f"  Estimated trials: {total_trials:,}")

    # Run sweep
    t0 = time.perf_counter()
    results = run_prior_weight_sweep(
        kbs, prior_weights=prior_weights, n_values=n_values,
        n_reps=n_reps, seed=args.seed,
    )
    total_elapsed = time.perf_counter() - t0

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"  PRIOR WEIGHT SWEEP RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  Predicted vs Actual MAE Reduction (%):")
    print(f"  {'W':>4}", end="")
    for n in n_values:
        print(f"  {'Pred@'+str(n):>8}  {'Act@'+str(n):>8}", end="")
    print()
    print(f"  {'-' * (4 + len(n_values) * 20)}")
    for w in prior_weights:
        print(f"  {w:>4}", end="")
        for n in n_values:
            r = results[str(w)][n]
            print(f"  {r['predicted_reduction_pct']:>8.1f}  "
                  f"{r['actual_reduction_pct']:>8.1f}", end="")
        print()

    print(f"\n  SL MAE by (W, N):")
    print(f"  {'W':>4}", end="")
    for n in n_values:
        print(f"  {'N='+str(n):>10}", end="")
    print()
    print(f"  {'-' * (4 + len(n_values) * 12)}")
    for w in prior_weights:
        print(f"  {w:>4}", end="")
        for n in n_values:
            r = results[str(w)][n]
            print(f"  {r['sl_mae']:>10.6f}", end="")
        print()

    # Bias at high N: which W gives lowest MAE at N=1000?
    if 1000 in n_values:
        print(f"\n  Over-regularization check at N=1000:")
        for w in prior_weights:
            r = results[str(w)][1000]
            print(f"    W={w:>2}: SL MAE={r['sl_mae']:.6f}  "
                  f"Scalar MAE={r['scalar_mae']:.6f}  "
                  f"SL {'better' if r['sl_mae'] < r['scalar_mae'] else 'WORSE'}")

    print(f"\n  Total runtime: {total_elapsed:.0f}s")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment_id": "EN1.5-supp",
        "experiment_name": "Prior Weight Sweep",
        "timestamp": timestamp,
        "config": {
            "prior_weights": prior_weights,
            "n_values": n_values,
            "n_reps": n_reps,
            "seed": args.seed,
            "n_datasets": len(kbs),
            "total_edges": total_edges,
        },
        "environment": env_info,
        "results": results,
        "total_elapsed_seconds": total_elapsed,
    }

    primary = RESULTS_DIR / "en1_5_prior_weight_results.json"
    with open(primary, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results: {primary}")

    archive = RESULTS_DIR / f"en1_5_prior_weight_results_{timestamp}.json"
    with open(archive, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Archive: {archive}")

    print(f"\n{'=' * 70}")
    print(f"  EN1.5 SUPPLEMENTARY COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
