#!/usr/bin/env python
"""EN8.2 — IoT Sensor Pipeline: Full Comparison Runner.

Phases:
    A: Sensor fusion comparison on Intel Lab data (6 methods, 2 clusters)
    B: SSN/SOSA round-trip gap analysis
    C: Code complexity comparison

Usage:
    python experiments/EN8/run_en8_2.py --phase a
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# -- Path Setup --
_EN8_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _EN8_DIR.parent
_PKG_SRC = _EXPERIMENTS_ROOT.parent / "packages" / "python" / "src"

for p in [str(_EN8_DIR), str(_EXPERIMENTS_ROOT), str(_PKG_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from en8_2_core import (
    load_intel_lab_data, load_mote_locations, select_sensor_cluster,
    time_align_readings, leave_one_out_eval,
    scalar_average, weighted_average, kalman_fuse_1d,
    sl_fuse_sensors, sl_fuse_sensors_with_decay, ds_fuse_sensors,
    compute_rmse, compute_mae,
)
from infra.env_log import log_environment

RESULTS_DIR = _EN8_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42
N_BOOTSTRAP = 2000

# -- Clusters --
CLUSTER_PRIMARY = [4, 6, 7, 9]      # within 7m, NW quadrant
CLUSTER_VALIDATION = [22, 23, 24]    # SW corner, all >55k readings
TEMP_RANGE = (15.0, 35.0)           # outlier filter


# =====================================================================
# Helpers
# =====================================================================

def filter_outliers(data: list, field: str = "temperature",
                    lo: float = 10.0, hi: float = 45.0) -> list:
    """Remove physically implausible sensor readings."""
    return [r for r in data
            if r[field] is not None
            and lo <= r[field] <= hi]


def estimate_per_sensor_sigma(aligned: dict, sensor_ids: list) -> dict:
    """Estimate per-sensor noise via MAD of deviations from consensus.

    For each sensor, compute its deviation from the mean of all OTHER
    sensors at each time step. Use MAD (robust to outliers) scaled to
    sigma equivalent.
    """
    sigmas = {}
    for sid in sensor_ids:
        others = [s for s in sensor_ids if s != sid]
        deviations = []
        n = len(aligned["timestamps"])
        for t in range(n):
            if np.isnan(aligned[sid][t]):
                continue
            other_vals = [aligned[s][t] for s in others
                          if not np.isnan(aligned[s][t])]
            if len(other_vals) >= 2:
                consensus = np.mean(other_vals)
                deviations.append(aligned[sid][t] - consensus)
        if len(deviations) >= 10:
            dev_arr = np.array(deviations)
            mad = np.median(np.abs(dev_arr - np.median(dev_arr)))
            sigmas[sid] = max(0.05, mad * 1.4826)  # MAD -> sigma
        else:
            sigmas[sid] = 0.5  # fallback
    return sigmas


def bootstrap_rmse_ci(errors_a: np.ndarray, errors_b: np.ndarray,
                       n_bootstrap: int = 2000, seed: int = 42,
                       ) -> Tuple[float, float, float]:
    """Bootstrap CI on RMSE difference (A - B).

    Returns (ci_lower, mean_diff, ci_upper).
    """
    rng = np.random.default_rng(seed)
    n = len(errors_a)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rmse_a = np.sqrt(np.mean(errors_a[idx] ** 2))
        rmse_b = np.sqrt(np.mean(errors_b[idx] ** 2))
        diffs.append(rmse_a - rmse_b)
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.mean(diffs)), \
           float(np.percentile(diffs, 97.5))


def run_loo_with_errors(aligned: dict, sigmas: dict,
                         sensor_ids: list, value_range: tuple,
                         ) -> Dict[str, np.ndarray]:
    """LOO evaluation returning raw squared errors per method.

    Needed for bootstrap CI computation.
    """
    n_steps = len(aligned["timestamps"])
    method_sq_errors: Dict[str, list] = {
        "scalar_avg": [], "weighted_avg": [], "kalman": [],
        "sl_fuse": [], "sl_fuse_decay": [], "ds_fuse": [],
    }

    for held_out in sensor_ids:
        others = [s for s in sensor_ids if s != held_out]
        if len(others) < 2:
            continue
        truth = aligned[held_out]
        sigma_arr = np.array([sigmas[s] for s in others])
        readings = np.column_stack([aligned[s] for s in others])
        valid_t = ~np.isnan(truth)
        # Also need at least 2 valid readings from others
        for t in range(n_steps):
            if not valid_t[t]:
                continue
            n_valid_others = np.sum(~np.isnan(readings[t]))
            if n_valid_others < 2:
                valid_t[t] = False
        if np.sum(valid_t) < 10:
            continue

        # Scalar average
        sa_pred = np.array([scalar_average(readings[t]) for t in range(n_steps)])
        errs = (sa_pred[valid_t] - truth[valid_t]) ** 2
        method_sq_errors["scalar_avg"].extend(errs.tolist())

        # Weighted average
        wa_pred = np.array([weighted_average(readings[t], sigma_arr)
                            for t in range(n_steps)])
        errs = (wa_pred[valid_t] - truth[valid_t]) ** 2
        method_sq_errors["weighted_avg"].extend(errs.tolist())

        # Kalman
        k_pred = kalman_fuse_1d(readings, sigma_arr)
        errs = (k_pred[valid_t] - truth[valid_t]) ** 2
        method_sq_errors["kalman"].extend(errs.tolist())

        # SL
        sl_pred = sl_fuse_sensors(readings, sigma_arr, value_range)
        errs = (sl_pred[valid_t] - truth[valid_t]) ** 2
        method_sq_errors["sl_fuse"].extend(errs.tolist())

        # SL + decay (inject staleness: 0h for fresh, simulated gaps)
        staleness = np.zeros(len(others))
        sl_d_pred = sl_fuse_sensors_with_decay(
            readings, sigma_arr, staleness, value_range)
        errs = (sl_d_pred[valid_t] - truth[valid_t]) ** 2
        method_sq_errors["sl_fuse_decay"].extend(errs.tolist())

        # DS
        ds_pred = ds_fuse_sensors(readings, sigma_arr, value_range)
        errs = (ds_pred[valid_t] - truth[valid_t]) ** 2
        method_sq_errors["ds_fuse"].extend(errs.tolist())

    return {k: np.array(v) for k, v in method_sq_errors.items()}


def run_misspecified_loo(aligned: dict, true_sigmas: dict,
                          sensor_ids: list, value_range: tuple,
                          ) -> Dict[str, Dict[str, float]]:
    """LOO with WRONG sigma estimates (robustness test).

    Shuffles sigma assignments so each sensor gets another's noise profile.
    """
    ids = list(true_sigmas.keys())
    shuffled = ids[1:] + ids[:1]  # rotate by 1
    wrong_sigmas = {ids[i]: true_sigmas[shuffled[i]] for i in range(len(ids))}

    results = leave_one_out_eval(aligned, wrong_sigmas, value_range)
    return results


# =====================================================================
# Phase A: Sensor Fusion Comparison
# =====================================================================

def run_phase_a() -> Dict[str, Any]:
    """Full 6-method comparison on Intel Lab data."""
    print("\n" + "=" * 60)
    print("Phase A: Sensor Fusion Comparison (Intel Lab)")
    print("=" * 60)

    # -- Load and filter --
    t0 = time.time()
    raw_data = load_intel_lab_data()
    data = filter_outliers(raw_data, lo=10.0, hi=45.0)
    print(f"  Loaded {len(raw_data):,} records, {len(data):,} after outlier filter "
          f"({len(data)/len(raw_data):.1%}) in {time.time()-t0:.1f}s")

    results = {"clusters": {}}

    for cluster_name, sensor_ids in [
        ("primary", CLUSTER_PRIMARY),
        ("validation", CLUSTER_VALIDATION),
    ]:
        print(f"\n  --- Cluster: {cluster_name} ({sensor_ids}) ---")
        cluster = select_sensor_cluster(data, sensor_ids)
        aligned = time_align_readings(cluster, sensor_ids, interval_seconds=300)
        n = len(aligned["timestamps"])
        print(f"  {len(cluster):,} records -> {n} aligned steps")

        for sid in sensor_ids:
            valid = np.sum(~np.isnan(aligned[sid]))
            print(f"    Sensor {sid}: {valid}/{n} valid ({valid/n:.1%})")

        # -- Estimate per-sensor noise --
        sigmas = estimate_per_sensor_sigma(aligned, sensor_ids)
        print(f"  Estimated sigmas:")
        for sid in sensor_ids:
            print(f"    Sensor {sid}: sigma={sigmas[sid]:.4f}")

        # -- Full LOO evaluation --
        print(f"\n  Running full LOO ({n} steps)...")
        t1 = time.time()
        sq_errors = run_loo_with_errors(aligned, sigmas, sensor_ids, TEMP_RANGE)
        elapsed = time.time() - t1
        print(f"  Done in {elapsed:.1f}s")

        # Compute RMSE and MAE per method
        method_results = {}
        for method, errs in sorted(sq_errors.items()):
            rmse = float(np.sqrt(np.mean(errs)))
            mae = float(np.mean(np.sqrt(errs)))  # sqrt of sq errors = abs errors
            method_results[method] = {
                "rmse": rmse, "mae": mae, "n_points": len(errs),
            }
            print(f"    {method:20s}  RMSE={rmse:.4f}  n={len(errs)}")

        # -- Bootstrap CIs (SL vs each baseline) --
        print(f"\n  Bootstrap CIs (n={N_BOOTSTRAP})...")
        ci_results = []
        sl_errs = sq_errors["sl_fuse"]
        for baseline in ["scalar_avg", "weighted_avg", "kalman", "ds_fuse"]:
            bl_errs = sq_errors[baseline]
            n_min = min(len(sl_errs), len(bl_errs))
            lo, md, hi = bootstrap_rmse_ci(
                np.sqrt(sl_errs[:n_min]), np.sqrt(bl_errs[:n_min]),
                n_bootstrap=N_BOOTSTRAP, seed=SEED)
            ci = {
                "comparison": f"sl_fuse - {baseline}",
                "mean_diff": md, "ci_lower": lo, "ci_upper": hi,
                "significant": (lo > 0) or (hi < 0),
            }
            ci_results.append(ci)
            print(f"    SL - {baseline:15s}: {md:+.4f} CI[{lo:+.4f},{hi:+.4f}] "
                  f"{'SIG' if ci['significant'] else 'ns'}")

        # -- Misspecified sigmas (robustness) --
        print(f"\n  Misspecified sigma robustness test...")
        mis_results = run_misspecified_loo(aligned, sigmas, sensor_ids, TEMP_RANGE)
        print(f"  Correct vs misspecified RMSE:")
        for method in sorted(method_results.keys()):
            correct = method_results[method]["rmse"]
            mis = mis_results.get(method, {}).get("rmse", float("nan"))
            if not np.isnan(mis):
                degradation = (mis - correct) / correct * 100
                print(f"    {method:20s}  correct={correct:.4f}  "
                      f"misspec={mis:.4f}  degrad={degradation:+.1f}%")

        results["clusters"][cluster_name] = {
            "sensor_ids": sensor_ids,
            "n_steps": n,
            "sigmas": {str(k): v for k, v in sigmas.items()},
            "methods": method_results,
            "bootstrap_ci": ci_results,
            "misspecified": {m: r for m, r in mis_results.items()},
        }

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="EN8.2 — IoT Sensor Pipeline")
    parser.add_argument("--phase", choices=["a", "b", "c", "all"], default="a")
    args = parser.parse_args()

    print("=" * 60)
    print("EN8.2 — IoT Sensor Pipeline")
    print("=" * 60)

    all_results = {
        "experiment": "EN8.2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "n_bootstrap": N_BOOTSTRAP,
        "dataset": "Intel Berkeley Research Lab (Bodik et al. 2004)",
        "outlier_filter": "temperature in [10, 45] C",
    }

    if args.phase in ("a", "all"):
        all_results["phase_a"] = run_phase_a()

    if args.phase in ("b", "all"):
        print("\n  Phase B (SSN/SOSA): to be implemented after Phase A validated")

    if args.phase in ("c", "all"):
        print("\n  Phase C (code complexity): to be implemented after Phase A validated")

    primary = RESULTS_DIR / "EN8_2_results.json"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = RESULTS_DIR / f"EN8_2_results_{ts}.json"
    for p in [primary, archive]:
        with open(p, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {primary}")
    print(f"  Archive: {archive}")


if __name__ == "__main__":
    main()
