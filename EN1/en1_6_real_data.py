"""EN1.6 Real Data -- Intel Berkeley Research Lab Sensor Fusion.

Downloads and processes the Intel Berkeley Research Lab dataset
(Bodik et al. 2004): 54 Mica2Dot sensors, 2.3M readings over 36 days.
Real sensor failures, missing data, voltage-induced anomalies.

Compares SL fusion (with conflict detection) vs scalar baselines
on REAL sensor data with REAL failures.

Usage:
    cd experiments
    python EN1/en1_6_real_data.py              # full run
    python EN1/en1_6_real_data.py --quick       # quick (3 clusters)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_EXPERIMENTS_ROOT))

from jsonld_ex.confidence_algebra import (
    Opinion, cumulative_fuse, pairwise_conflict, conflict_metric,
)
from jsonld_ex.confidence_decay import decay_opinion
from infra.env_log import log_environment

RESULTS_DIR = _SCRIPT_DIR / "results"
DATA_DIR = _SCRIPT_DIR.parent.parent / "data" / "intel_lab"
DATA_URL = "https://raw.githubusercontent.com/linsea423/Intel_Lab_Data/master/data.zip"

# Sensor clusters (discovered via correlation > 0.7, see en1_6_diagnostic)
CLUSTERS = [
    {"id": 0, "sensors": [1, 2, 3, 4, 7],       "region": "NE corner"},
    {"id": 1, "sensors": [6, 19, 35, 50],        "region": "scattered"},
    {"id": 2, "sensors": [9, 10, 11, 12, 13],    "region": "E wall"},
    {"id": 3, "sensors": [14, 21, 22, 23, 24],   "region": "center-N"},
    {"id": 4, "sensors": [25, 26, 27, 28, 29],   "region": "center"},
    {"id": 5, "sensors": [30, 31, 32, 33, 34],   "region": "center-S"},
    {"id": 6, "sensors": [36, 37, 38, 39, 40],   "region": "SW area"},
    {"id": 7, "sensors": [41, 43, 44, 45, 47],   "region": "S wall"},
    {"id": 8, "sensors": [42, 46, 49, 51, 52],   "region": "SE corner"},
    {"id": 9, "sensors": [48, 53, 54],            "region": "S edge"},
]

THRESHOLD = 25.0       # Binary task: is temperature above 25C?
NORMAL_RANGE = (10, 40) # Readings outside this are anomalous


# ===================================================================
# 1. Data download and preparation
# ===================================================================

def download_intel_lab_data() -> Path:
    """Download Intel Lab data if not already present."""
    data_file = DATA_DIR / "data.txt"
    if data_file.exists():
        print(f"  Data already exists: {data_file}")
        return data_file

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading Intel Lab data from {DATA_URL} ...")
    zip_path = DATA_DIR / "data.zip"
    urllib.request.urlretrieve(DATA_URL, zip_path)

    print(f"  Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    print(f"  Done: {data_file}")
    return data_file


def load_and_prepare(data_file: Path, bin_minutes: int = 5) -> pd.DataFrame:
    """Load raw data and create time-binned pivot table.

    Returns:
        DataFrame with time_bin as index, sensor IDs as columns,
        mean temperature per bin as values.
    """
    df = pd.read_csv(
        data_file, sep=r"\s+", header=None,
        names=["date", "time", "epoch", "moteid",
               "temperature", "humidity", "light", "voltage"],
        on_bad_lines="skip",
    )
    # Filter to valid sensor IDs
    valid = df[(df["moteid"] >= 1) & (df["moteid"] <= 54)].copy()
    valid["moteid"] = valid["moteid"].astype(int)
    valid["timestamp"] = pd.to_datetime(
        valid["date"] + " " + valid["time"], errors="coerce"
    )
    valid = valid.dropna(subset=["timestamp", "temperature"])
    valid["time_bin"] = valid["timestamp"].dt.floor(f"{bin_minutes}min")

    pivot = valid.pivot_table(
        values="temperature", index="time_bin",
        columns="moteid", aggfunc="mean",
    )
    return pivot


# ===================================================================
# 2. SL sensor fusion with conflict detection
# ===================================================================

def reading_to_opinion(
    temp: float,
    threshold: float,
    base_rate: float,
) -> Opinion:
    """Convert a temperature reading to an SL opinion about 'above threshold'.

    Evidence strength is proportional to distance from threshold,
    capped at 10 to prevent extreme opinions from single readings.
    """
    distance = abs(temp - threshold)
    strength = min(distance / 5.0, 10.0)

    if temp > threshold:
        return Opinion.from_evidence(
            positive=strength, negative=1.0,
            prior_weight=2.0, base_rate=base_rate,
        )
    else:
        return Opinion.from_evidence(
            positive=1.0, negative=strength,
            prior_weight=2.0, base_rate=base_rate,
        )


def sl_fuse_with_conflict_exclusion(
    readings: List[float],
    base_rate: float,
    conflict_threshold: float = 0.15,
) -> Dict[str, Any]:
    """SL fusion with conflict-based outlier exclusion.

    Strategy: for each sensor, compute its mean pairwise conflict
    with all other sensors. A sensor that disagrees with the majority
    will have high conflict. Exclude sensors above the threshold.

    The threshold should be calibrated from the data. Clean sensor
    groups produce conflict ~0.10; faulty sensors push to ~0.18+.
    Default threshold 0.15 is set at the midpoint.

    Returns dict with: predicted_prob, uncertainty, n_excluded,
    max_conflict, per_sensor_conflicts, excluded_indices, kept_readings.
    """
    if not readings:
        return {
            "predicted_prob": base_rate, "uncertainty": 1.0,
            "n_excluded": 0, "max_conflict": 0.0,
            "per_sensor_conflicts": [],
            "excluded_indices": [], "kept_readings": [],
        }

    opinions = [reading_to_opinion(r, THRESHOLD, base_rate) for r in readings]

    excluded_indices = []
    per_sensor_conflicts = []
    if len(opinions) >= 3:
        # Per-sensor mean conflict with all others
        conflicts = np.zeros(len(opinions))
        for i in range(len(opinions)):
            for j in range(len(opinions)):
                if i != j:
                    conflicts[i] += pairwise_conflict(opinions[i], opinions[j])
            conflicts[i] /= (len(opinions) - 1)
        per_sensor_conflicts = conflicts.tolist()

        # Exclude the sensor with highest conflict IF it exceeds threshold
        # and the remaining sensors agree (their max conflict is below threshold).
        # This is conservative: only exclude clear outliers.
        sorted_idx = np.argsort(conflicts)[::-1]  # highest first
        remaining = list(range(len(opinions)))
        for idx in sorted_idx:
            if conflicts[idx] <= conflict_threshold:
                break
            # Check: if we exclude this sensor, do the rest agree?
            test_remaining = [i for i in remaining if i != idx]
            if len(test_remaining) < 2:
                break
            # Verify remaining sensors have low mutual conflict
            remaining_conflicts = [
                conflicts[i] for i in test_remaining
            ]
            if max(remaining_conflicts) <= conflict_threshold:
                excluded_indices.append(idx)
                remaining = test_remaining
            else:
                # Multiple sensors in conflict -- don't exclude, flag uncertainty
                break

        max_conf = float(conflicts.max())
    else:
        max_conf = 0.0

    # Rebuild opinions without excluded
    kept_opinions = [opinions[i] for i in range(len(opinions)) if i not in excluded_indices]

    # Fuse
    if len(kept_opinions) >= 2:
        fused = cumulative_fuse(*kept_opinions)
    elif kept_opinions:
        fused = kept_opinions[0]
    else:
        fused = Opinion(0.0, 0.0, 1.0, base_rate)

    kept_readings = [r for i, r in enumerate(readings) if i not in excluded_indices]

    return {
        "predicted_prob": fused.projected_probability(),
        "uncertainty": fused.uncertainty,
        "n_excluded": len(excluded_indices),
        "max_conflict": max_conf,
        "per_sensor_conflicts": per_sensor_conflicts,
        "excluded_indices": excluded_indices,
        "kept_readings": kept_readings,
    }


# ===================================================================
# 3. Baseline methods
# ===================================================================

def scalar_mean(readings: List[float]) -> float:
    return float(np.mean(readings)) if readings else THRESHOLD


def scalar_trimmed_mean(readings: List[float]) -> float:
    if len(readings) <= 2:
        return float(np.mean(readings)) if readings else THRESHOLD
    s = sorted(readings)
    return float(np.mean(s[1:-1]))


def scalar_median(readings: List[float]) -> float:
    return float(np.median(readings)) if readings else THRESHOLD


def kalman_fuse(readings: List[float], Q: float = 0.5) -> float:
    """Simple Kalman filter across sensors at a single timestep.

    Treats each sensor reading as an independent measurement.
    Process noise Q models unknown variability.
    """
    x = THRESHOLD  # initial estimate
    P = 25.0       # initial variance (wide)

    for r in readings:
        R = 4.0  # measurement noise variance (2C std)
        K = P / (P + R)
        x = x + K * (r - x)
        P = (1.0 - K) * P

    P = P + Q  # process noise for next step
    return float(np.clip(x, 0, 60))


# ===================================================================
# 4. Run experiment on a single cluster
# ===================================================================

def run_cluster_experiment(
    pivot: pd.DataFrame,
    cluster: Dict[str, Any],
    base_rate: float = 0.4,
    conflict_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Run full comparison on one sensor cluster."""
    sensors = cluster["sensors"]
    cluster_data = pivot[sensors].dropna()

    if len(cluster_data) < 100:
        return {"error": f"Only {len(cluster_data)} complete bins"}

    methods = {
        "scalar_mean": {"correct": 0, "temp_errors": [], "preds": [], "states": []},
        "trimmed_mean": {"correct": 0, "temp_errors": [], "preds": [], "states": []},
        "median": {"correct": 0, "temp_errors": [], "preds": [], "states": []},
        "kalman": {"correct": 0, "temp_errors": [], "preds": [], "states": []},
        "sl_conflict": {"correct": 0, "temp_errors": [], "preds": [], "states": []},
    }
    n_total = 0
    total_anomalies = 0
    total_sl_excluded = 0
    total_sl_correct_exclusions = 0
    conflict_pre_fault = []
    conflict_during_fault = []

    for _, row in cluster_data.iterrows():
        readings = row.values
        healthy = (readings >= NORMAL_RANGE[0]) & (readings <= NORMAL_RANGE[1])

        if healthy.sum() < 2:
            continue

        gt_temp = float(np.median(readings[healthy]))
        gt_state = int(gt_temp > THRESHOLD)
        n_anomalous = int((~healthy).sum())
        total_anomalies += n_anomalous

        readings_list = list(readings)

        # --- Scalar Mean ---
        pred = scalar_mean(readings_list)
        methods["scalar_mean"]["temp_errors"].append(abs(pred - gt_temp))
        pred_state = int(pred > THRESHOLD)
        methods["scalar_mean"]["preds"].append(pred)
        methods["scalar_mean"]["states"].append(pred_state)
        if pred_state == gt_state:
            methods["scalar_mean"]["correct"] += 1

        # --- Trimmed Mean ---
        pred = scalar_trimmed_mean(readings_list)
        methods["trimmed_mean"]["temp_errors"].append(abs(pred - gt_temp))
        pred_state = int(pred > THRESHOLD)
        methods["trimmed_mean"]["preds"].append(pred)
        methods["trimmed_mean"]["states"].append(pred_state)
        if pred_state == gt_state:
            methods["trimmed_mean"]["correct"] += 1

        # --- Median ---
        pred = scalar_median(readings_list)
        methods["median"]["temp_errors"].append(abs(pred - gt_temp))
        pred_state = int(pred > THRESHOLD)
        methods["median"]["preds"].append(pred)
        methods["median"]["states"].append(pred_state)
        if pred_state == gt_state:
            methods["median"]["correct"] += 1

        # --- Kalman ---
        pred = kalman_fuse(readings_list)
        methods["kalman"]["temp_errors"].append(abs(pred - gt_temp))
        pred_state = int(pred > THRESHOLD)
        methods["kalman"]["preds"].append(pred)
        methods["kalman"]["states"].append(pred_state)
        if pred_state == gt_state:
            methods["kalman"]["correct"] += 1

        # --- SL with Conflict Detection ---
        sl_result = sl_fuse_with_conflict_exclusion(
            readings_list, base_rate=base_rate,
            conflict_threshold=conflict_threshold,
        )
        # Temperature estimate: median of kept readings
        if sl_result["kept_readings"]:
            sl_temp = float(np.median(sl_result["kept_readings"]))
        else:
            sl_temp = float(np.median(readings))

        methods["sl_conflict"]["temp_errors"].append(abs(sl_temp - gt_temp))
        # Binary: use the kept median for classification
        sl_state = int(sl_temp > THRESHOLD)
        methods["sl_conflict"]["preds"].append(sl_temp)
        methods["sl_conflict"]["states"].append(sl_state)
        if sl_state == gt_state:
            methods["sl_conflict"]["correct"] += 1

        total_sl_excluded += sl_result["n_excluded"]

        # Track whether excluded sensors were actually anomalous
        for idx in sl_result["excluded_indices"]:
            if not healthy[idx]:
                total_sl_correct_exclusions += 1

        # Track conflict levels
        if n_anomalous > 0:
            conflict_during_fault.append(sl_result["max_conflict"])
        else:
            conflict_pre_fault.append(sl_result["max_conflict"])

        n_total += 1

    # Aggregate
    result = {
        "cluster_id": cluster["id"],
        "sensors": sensors,
        "region": cluster["region"],
        "n_timesteps": n_total,
        "total_anomalies": total_anomalies,
        "anomaly_rate": total_anomalies / (n_total * len(sensors)) * 100,
    }

    for method_name, m_data in methods.items():
        result[f"{method_name}_accuracy"] = m_data["correct"] / n_total if n_total else 0
        result[f"{method_name}_mae"] = float(np.mean(m_data["temp_errors"])) if m_data["temp_errors"] else 0

    result["sl_excluded_total"] = total_sl_excluded
    result["sl_correct_exclusions"] = total_sl_correct_exclusions
    result["sl_exclusion_precision"] = (
        total_sl_correct_exclusions / total_sl_excluded
        if total_sl_excluded > 0 else 0
    )
    result["conflict_clean_mean"] = float(np.mean(conflict_pre_fault)) if conflict_pre_fault else 0
    result["conflict_fault_mean"] = float(np.mean(conflict_during_fault)) if conflict_during_fault else 0
    result["conflict_increase"] = (
        (result["conflict_fault_mean"] / result["conflict_clean_mean"] - 1) * 100
        if result["conflict_clean_mean"] > 0 else 0
    )

    return result


# ===================================================================
# 5. Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EN1.6 Real Data: Intel Berkeley Lab Sensor Fusion"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Run on 3 clusters only")
    parser.add_argument("--conflict-threshold", type=float, default=0.15)
    parser.add_argument("--base-rate", type=float, default=0.4)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("  EN1.6 REAL DATA: Intel Berkeley Research Lab")
    print("  54 sensors, 2.3M readings, real failures")
    print("  Bodik et al. 2004 (MIT CSAIL)")
    print(f"  Conflict threshold: {args.conflict_threshold}")
    print(f"  Base rate: {args.base_rate}")
    print("=" * 70)

    env_info = log_environment()

    # Download data
    print("\n  Step 1: Data preparation")
    data_file = download_intel_lab_data()
    print("  Loading and binning (5-min windows) ...")
    pivot = load_and_prepare(data_file)
    print(f"  Pivot: {pivot.shape[0]} time bins x {pivot.shape[1]} sensors")

    # Select clusters
    clusters_to_run = CLUSTERS
    if args.quick:
        clusters_to_run = CLUSTERS[:3]
        print(f"  Quick mode: {len(clusters_to_run)} clusters")

    # Run
    print(f"\n  Step 2: Running {len(clusters_to_run)} cluster experiments ...")
    t0 = time.perf_counter()
    all_results = []
    for cluster in clusters_to_run:
        print(f"    Cluster {cluster['id']} ({cluster['region']}, "
              f"sensors {cluster['sensors']}) ...", end=" ", flush=True)
        result = run_cluster_experiment(
            pivot, cluster,
            base_rate=args.base_rate,
            conflict_threshold=args.conflict_threshold,
        )
        all_results.append(result)
        if "error" in result:
            print(f"SKIP: {result['error']}")
        else:
            print(f"done ({result['n_timesteps']} bins, "
                  f"{result['anomaly_rate']:.1f}% anomalies)")

    elapsed = time.perf_counter() - t0
    valid_results = [r for r in all_results if "error" not in r]

    # ---- Results tables ----
    methods = ["scalar_mean", "trimmed_mean", "median", "kalman", "sl_conflict"]
    method_labels = {"scalar_mean": "Mean", "trimmed_mean": "Trim",
                     "median": "Median", "kalman": "Kalman", "sl_conflict": "SL"}

    print(f"\n{'=' * 70}")
    print(f"  TEMPERATURE MAE (lower is better)")
    print(f"{'=' * 70}")
    header = f"  {'C#':>2} {'Region':<12} {'Anom%':>6}"
    for m in methods:
        header += f"  {method_labels[m]:>7}"
    header += "  Best"
    print(header)
    print(f"  {'-' * (30 + len(methods) * 9 + 6)}")

    wins = {m: 0 for m in methods}
    for r in valid_results:
        maes = {m: r[f"{m}_mae"] for m in methods}
        best_m = min(maes, key=maes.get)
        wins[best_m] += 1
        line = f"  {r['cluster_id']:>2} {r['region']:<12} {r['anomaly_rate']:>5.1f}%"
        for m in methods:
            marker = " *" if m == best_m else "  "
            line += f"  {maes[m]:>5.2f}{marker}"
        print(line)

    print(f"\n  Wins: ", end="")
    for m in methods:
        print(f"{method_labels[m]}={wins[m]}  ", end="")
    print()

    # Aggregate
    total_n = sum(r["n_timesteps"] for r in valid_results)
    print(f"\n  AGGREGATE (weighted by timesteps, N={total_n:,}):")
    for m in methods:
        agg = sum(r[f"{m}_mae"] * r["n_timesteps"] for r in valid_results) / total_n
        print(f"    {method_labels[m]:<8}: MAE = {agg:.4f}")

    print(f"\n{'=' * 70}")
    print(f"  BINARY ACCURACY (is temp > 25C?)")
    print(f"{'=' * 70}")
    header = f"  {'C#':>2} {'Anom%':>6}"
    for m in methods:
        header += f"  {method_labels[m]:>7}"
    header += "  Best"
    print(header)
    print(f"  {'-' * (16 + len(methods) * 9 + 6)}")

    acc_wins = {m: 0 for m in methods}
    for r in valid_results:
        accs = {m: r[f"{m}_accuracy"] for m in methods}
        best_m = max(accs, key=accs.get)
        acc_wins[best_m] += 1
        line = f"  {r['cluster_id']:>2} {r['anomaly_rate']:>5.1f}%"
        for m in methods:
            marker = " *" if m == best_m else "  "
            line += f"  {accs[m]:>.4f}{marker}"
        print(line)

    print(f"\n  Wins: ", end="")
    for m in methods:
        print(f"{method_labels[m]}={acc_wins[m]}  ", end="")
    print()

    # Aggregate accuracy
    print(f"\n  AGGREGATE ACCURACY:")
    for m in methods:
        agg = sum(r[f"{m}_accuracy"] * r["n_timesteps"] for r in valid_results) / total_n
        print(f"    {method_labels[m]:<8}: {agg:.4f}")

    # Conflict detection analysis
    print(f"\n{'=' * 70}")
    print(f"  CONFLICT DETECTION ANALYSIS (SL unique)")
    print(f"{'=' * 70}")
    total_excluded = sum(r["sl_excluded_total"] for r in valid_results)
    total_correct_excl = sum(r["sl_correct_exclusions"] for r in valid_results)
    total_actual_anom = sum(r["total_anomalies"] for r in valid_results)
    print(f"  Anomalous readings (total): {total_actual_anom:,}")
    print(f"  SL exclusions (total):      {total_excluded:,}")
    print(f"  Correct exclusions:         {total_correct_excl:,}")
    print(f"  Exclusion precision:        "
          f"{total_correct_excl/total_excluded*100:.1f}%" if total_excluded else "  N/A")
    print(f"\n  Per-cluster conflict analysis:")
    print(f"  {'C#':>2} {'Clean Conf':>11} {'Fault Conf':>11} {'Increase':>10} {'Excl Prec':>10}")
    print(f"  {'-' * 50}")
    for r in valid_results:
        print(f"  {r['cluster_id']:>2} {r['conflict_clean_mean']:>11.4f} "
              f"{r['conflict_fault_mean']:>11.4f} "
              f"{r['conflict_increase']:>+9.1f}% "
              f"{r['sl_exclusion_precision']*100:>9.1f}%")

    # ---- Threshold sweep ----
    print(f"\n{'=' * 70}")
    print(f"  CONFLICT THRESHOLD SWEEP")
    print(f"{'=' * 70}")
    print(f"  {'Thresh':>7} {'Excluded':>9} {'Correct':>8} {'Precision':>10} "
          f"{'Recall':>8} {'SL MAE':>8} {'Median':>8}")
    print(f"  {'-' * 65}")

    total_actual = sum(r['total_anomalies'] for r in valid_results)
    median_mae_agg = sum(r['median_mae'] * r['n_timesteps'] for r in valid_results) / total_n

    for sweep_thresh in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        sweep_excluded = 0
        sweep_correct = 0
        sweep_errors = []
        sweep_n = 0

        for cluster in clusters_to_run:
            sensors = cluster['sensors']
            cdata = pivot[sensors].dropna()
            for _, row in cdata.iterrows():
                readings = row.values
                healthy = (readings >= NORMAL_RANGE[0]) & (readings <= NORMAL_RANGE[1])
                if healthy.sum() < 2:
                    continue
                gt_temp = float(np.median(readings[healthy]))

                sl_result = sl_fuse_with_conflict_exclusion(
                    list(readings), base_rate=args.base_rate,
                    conflict_threshold=sweep_thresh,
                )
                kept = sl_result['kept_readings']
                sl_temp = float(np.median(kept)) if kept else float(np.median(readings))
                sweep_errors.append(abs(sl_temp - gt_temp))
                sweep_excluded += sl_result['n_excluded']
                for idx in sl_result['excluded_indices']:
                    if not healthy[idx]:
                        sweep_correct += 1
                sweep_n += 1

        prec = sweep_correct / sweep_excluded * 100 if sweep_excluded > 0 else 0
        recall = sweep_correct / total_actual * 100 if total_actual > 0 else 0
        sl_mae = float(np.mean(sweep_errors)) if sweep_errors else 0
        marker = " <--" if abs(sweep_thresh - args.conflict_threshold) < 0.001 else ""
        print(f"  {sweep_thresh:>7.2f} {sweep_excluded:>9,} {sweep_correct:>8,} "
              f"{prec:>9.1f}% {recall:>7.1f}% {sl_mae:>8.4f} {median_mae_agg:>8.4f}{marker}")

    print(f"\n  Total time: {elapsed:.1f}s")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment_id": "EN1.6-real",
        "experiment_name": "Intel Berkeley Lab Sensor Fusion",
        "timestamp": timestamp,
        "config": {
            "dataset": "Intel Berkeley Research Lab (Bodik et al. 2004)",
            "dataset_url": DATA_URL,
            "n_sensors": 54,
            "n_readings": "~2.3M",
            "date_range": "2004-02-28 to 2004-04-05",
            "bin_minutes": 5,
            "threshold": THRESHOLD,
            "normal_range": list(NORMAL_RANGE),
            "conflict_threshold": args.conflict_threshold,
            "base_rate": args.base_rate,
            "n_clusters": len(valid_results),
        },
        "environment": env_info,
        "cluster_results": valid_results,
        "elapsed_seconds": elapsed,
    }

    primary = RESULTS_DIR / "en1_6_real_results.json"
    with open(primary, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results: {primary}")

    archive = RESULTS_DIR / f"en1_6_real_results_{timestamp}.json"
    with open(archive, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Archive: {archive}")

    print(f"\n{'=' * 70}")
    print(f"  EN1.6 REAL DATA COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
