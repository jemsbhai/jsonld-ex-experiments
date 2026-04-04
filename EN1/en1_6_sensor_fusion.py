"""EN1.6 Runner -- Multi-Source Sensor Fusion.

Full experiment across 5 signal scenarios, 4 fusion methods (+naive
SL baseline), multiple repetitions for statistical stability.

Usage:
    cd experiments
    python EN1/en1_6_sensor_fusion.py              # full run
    python EN1/en1_6_sensor_fusion.py --quick       # quick validation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_EXPERIMENTS_ROOT))

from en1_6_core import (
    SCENARIOS, SENSOR_SPECS,
    generate_signal, generate_sensor_readings,
    fuse_scalar_weighted, fuse_kalman,
    fuse_sl, fuse_sl_naive, fuse_sl_temporal,
    compute_fusion_metrics, run_full_experiment,
)
from infra.env_log import log_environment

EXPERIMENT_ID = "EN1.6"
EXPERIMENT_NAME = "Multi-Source Sensor Fusion"
RESULTS_DIR = _SCRIPT_DIR / "results"
DEFAULT_N_STEPS = 500
DEFAULT_N_REPS = 200
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_ID}: {EXPERIMENT_NAME}"
    )
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quick", action="store_true",
                        help="Quick: 100 steps, 20 reps")
    args = parser.parse_args()

    n_steps = args.n_steps
    n_reps = args.n_reps
    if args.quick:
        n_steps = 100
        n_reps = 20

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"{'=' * 70}")
    print(f"  {EXPERIMENT_ID}: {EXPERIMENT_NAME}")
    print(f"  Steps: {n_steps}, Reps: {n_reps}, Seed: {args.seed}")
    print(f"  Scenarios: {list(SCENARIOS.keys())}")
    print(f"  Sensors: {[s.name for s in SENSOR_SPECS]}")
    print(f"{'=' * 70}")

    env_info = log_environment()

    # ---- Run full experiment (calibrated SL) ----
    print("\n  Running calibrated SL experiment ...")
    t0 = time.perf_counter()
    calibrated_results = run_full_experiment(
        n_steps=n_steps, n_reps=n_reps, seed=args.seed
    )
    elapsed_cal = time.perf_counter() - t0
    print(f"  Done ({elapsed_cal:.1f}s)")

    # ---- Run naive SL baseline for comparison ----
    print("\n  Running naive SL baseline ...")
    t0 = time.perf_counter()
    naive_results = {}
    for scenario_name in SCENARIOS:
        naive_maes = []
        for rep in range(n_reps):
            rep_seed = args.seed + rep * 1000
            signal = generate_signal(scenario_name, n_steps=n_steps, seed=rep_seed)
            readings = generate_sensor_readings(signal, SENSOR_SPECS, seed=rep_seed + 1)
            result_naive = fuse_sl_naive(readings, SENSOR_SPECS)
            m = compute_fusion_metrics(
                result_naive["predictions"], signal["true_state"],
                uncertainties=result_naive["uncertainties"],
            )
            naive_maes.append(m)
        # Average
        avg = {}
        for key in naive_maes[0]:
            vals = [m[key] for m in naive_maes if isinstance(m.get(key), (int, float))]
            if vals:
                avg[key] = float(np.mean(vals))
        avg["n_reps"] = n_reps
        naive_results[scenario_name] = avg
    elapsed_naive = time.perf_counter() - t0
    print(f"  Done ({elapsed_naive:.1f}s)")

    total_elapsed = elapsed_cal + elapsed_naive

    # ---- Summary tables ----
    methods = ["scalar_weighted", "kalman", "sl_fusion", "sl_temporal"]
    print(f"\n{'=' * 70}")
    print(f"  MAE COMPARISON (lower is better)")
    print(f"{'=' * 70}")
    print(f"  {'Scenario':<16} {'Scalar':>8} {'Kalman':>8} "
          f"{'SL_cal':>8} {'SL+temp':>8} {'SL_naive':>8}  Best")
    print(f"  {'-' * 75}")

    sl_vs_scalar_wins = 0
    sl_vs_kalman_wins = 0
    total_scenarios = 0

    for sc in SCENARIOS:
        cr = calibrated_results[sc]
        nr = naive_results[sc]
        vals = {
            "scalar": cr["scalar_weighted"]["mae"],
            "kalman": cr["kalman"]["mae"],
            "sl_cal": cr["sl_fusion"]["mae"],
            "sl_temp": cr["sl_temporal"]["mae"],
            "sl_naive": nr["mae"],
        }
        best = min(vals, key=vals.get)
        print(f"  {sc:<16} {vals['scalar']:>8.4f} {vals['kalman']:>8.4f} "
              f"{vals['sl_cal']:>8.4f} {vals['sl_temp']:>8.4f} "
              f"{vals['sl_naive']:>8.4f}  {best}")
        total_scenarios += 1
        if vals["sl_cal"] < vals["scalar"]:
            sl_vs_scalar_wins += 1
        if vals["sl_cal"] < vals["kalman"]:
            sl_vs_kalman_wins += 1

    print(f"\n  SL calibrated vs scalar: {sl_vs_scalar_wins}/{total_scenarios} wins")
    print(f"  SL calibrated vs Kalman: {sl_vs_kalman_wins}/{total_scenarios} wins")

    # ---- Improvement: calibrated vs naive ----
    print(f"\n  {'Scenario':<16} {'Naive MAE':>10} {'Cal MAE':>10} {'Improve':>10}")
    print(f"  {'-' * 50}")
    for sc in SCENARIOS:
        naive_mae = naive_results[sc]["mae"]
        cal_mae = calibrated_results[sc]["sl_fusion"]["mae"]
        imp = (naive_mae - cal_mae) / naive_mae * 100
        print(f"  {sc:<16} {naive_mae:>10.4f} {cal_mae:>10.4f} {imp:>+9.1f}%")

    # ---- Uncertainty analysis ----
    print(f"\n{'=' * 70}")
    print(f"  UNCERTAINTY ANALYSIS (SL unique capability)")
    print(f"{'=' * 70}")
    for sc in SCENARIOS:
        cr = calibrated_results[sc]
        sl_u = cr["sl_fusion"].get("mean_uncertainty", 0)
        slt_u = cr["sl_temporal"].get("mean_uncertainty", 0)
        cov = cr["sl_fusion"].get("coverage", 0)
        cov_t = cr["sl_temporal"].get("coverage", 0)
        print(f"  {sc:<16} SL u={sl_u:.4f} cov={cov:.3f}  "
              f"SL+temp u={slt_u:.4f} cov={cov_t:.3f}")

    # ---- ECE comparison ----
    print(f"\n{'=' * 70}")
    print(f"  ECE COMPARISON (calibration quality, lower is better)")
    print(f"{'=' * 70}")
    print(f"  {'Scenario':<16} {'Scalar':>8} {'Kalman':>8} "
          f"{'SL_cal':>8} {'SL+temp':>8}  Best")
    print(f"  {'-' * 60}")
    for sc in SCENARIOS:
        cr = calibrated_results[sc]
        vals = {
            "scalar": cr["scalar_weighted"]["ece"],
            "kalman": cr["kalman"]["ece"],
            "sl_cal": cr["sl_fusion"]["ece"],
            "sl_temp": cr["sl_temporal"]["ece"],
        }
        best = min(vals, key=vals.get)
        print(f"  {sc:<16} {vals['scalar']:>8.4f} {vals['kalman']:>8.4f} "
              f"{vals['sl_cal']:>8.4f} {vals['sl_temp']:>8.4f}  {best}")

    print(f"\n  Total runtime: {total_elapsed:.1f}s")

    # ---- Save results ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": timestamp,
        "config": {
            "n_steps": n_steps,
            "n_reps": n_reps,
            "seed": args.seed,
            "sensors": [
                {"name": s.name, "tpr": s.tpr, "fpr": s.fpr,
                 "availability": s.availability,
                 "saturation_tpr": s.saturation_tpr}
                for s in SENSOR_SPECS
            ],
        },
        "environment": env_info,
        "calibrated_results": calibrated_results,
        "naive_results": naive_results,
        "total_elapsed_seconds": total_elapsed,
    }

    primary = RESULTS_DIR / "en1_6_results.json"
    with open(primary, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results: {primary}")

    archive = RESULTS_DIR / f"en1_6_results_{timestamp}.json"
    with open(archive, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Archive: {archive}")

    print(f"\n{'=' * 70}")
    print(f"  {EXPERIMENT_ID} COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
