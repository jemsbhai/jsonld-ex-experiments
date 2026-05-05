#!/usr/bin/env python
"""EN8.6 -- Graph Merge and Diff Operations: Full Runner (Phase 1).

Phases:
    sweep:   Parameter sweep (48 configs x 5 seeds = 240 runs)
    scaling: Throughput scaling analysis (4 sizes x 5 seeds = 20 runs)
    all:     Both phases

Usage:
    python experiments/EN8/run_en8_6.py --phase sweep
    python experiments/EN8/run_en8_6.py --phase scaling
    python experiments/EN8/run_en8_6.py --phase all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_EN8_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _EN8_DIR.parent
_PKG_SRC = _EXPERIMENTS_ROOT.parent / "packages" / "python" / "src"
for p in [str(_EN8_DIR), str(_EXPERIMENTS_ROOT), str(_PKG_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from en8_6_core import (
    run_single_config,
    CalibrationRegime,
    ExperimentConfig,
    MergeResult,
)
from infra.stats import bootstrap_ci
from infra.results import ExperimentResult

RESULTS_DIR = _EN8_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Sweep parameters ──────────────────────────────────────────────

CORRUPTION_RATES = [0.05, 0.10, 0.20, 0.40]
CALIBRATIONS = list(CalibrationRegime)
N_SOURCES_LIST = [2, 3, 5]
N_SEEDS = 5
BASE_SEED = 42

# Fixed for sweep
SWEEP_GRAPH_SIZE = 500
SWEEP_OVERLAP = 0.4

# Scaling parameters
SCALING_SIZES = [100, 500, 1000, 5000]
SCALING_CORRUPTION = 0.10
SCALING_CALIBRATION = CalibrationRegime.IDEAL
SCALING_N_SOURCES = 3


# ═══════════════════════════════════════════════════════════════════
# PHASE: PARAMETER SWEEP
# ═══════════════════════════════════════════════════════════════════


def run_sweep() -> dict[str, Any]:
    """Run the full parameter sweep: 48 configs x 5 seeds = 240 runs."""
    print("\n" + "=" * 60)
    print("PHASE 1: Parameter Sweep")
    print(f"  {len(CORRUPTION_RATES)} corruption rates x "
          f"{len(CALIBRATIONS)} calibrations x "
          f"{len(N_SOURCES_LIST)} source counts = "
          f"{len(CORRUPTION_RATES) * len(CALIBRATIONS) * len(N_SOURCES_LIST)} configs")
    print(f"  {N_SEEDS} seeds per config = "
          f"{len(CORRUPTION_RATES) * len(CALIBRATIONS) * len(N_SOURCES_LIST) * N_SEEDS} total runs")
    print("=" * 60)

    all_results: list[dict[str, Any]] = []
    run_count = 0
    total_runs = len(CORRUPTION_RATES) * len(CALIBRATIONS) * len(N_SOURCES_LIST) * N_SEEDS
    t_start = time.perf_counter()

    for cr in CORRUPTION_RATES:
        for cal in CALIBRATIONS:
            for ns in N_SOURCES_LIST:
                seed_results: list[MergeResult] = []
                for s in range(N_SEEDS):
                    seed = BASE_SEED + s * 1000
                    run_count += 1
                    config = ExperimentConfig(
                        n_entities=SWEEP_GRAPH_SIZE,
                        n_sources=ns,
                        corruption_rate=cr,
                        calibration=cal,
                        overlap_rate=SWEEP_OVERLAP,
                        seed=seed,
                    )
                    result = run_single_config(config)
                    seed_results.append(result)

                    elapsed = time.perf_counter() - t_start
                    rate = run_count / elapsed if elapsed > 0 else 0
                    eta = (total_runs - run_count) / rate if rate > 0 else 0
                    print(f"  [{run_count:3d}/{total_runs}] "
                          f"cr={cr:.2f} cal={cal.value:13s} ns={ns} "
                          f"seed={seed:5d} "
                          f"acc_h={result.jsonldex_highest_accuracy:.3f} "
                          f"acc_mv={result.majority_vote_accuracy:.3f} "
                          f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

                # Aggregate across seeds
                agg = _aggregate_seed_results(seed_results, cr, cal, ns)
                all_results.append(agg)

    print(f"\nSweep complete: {run_count} runs in "
          f"{time.perf_counter() - t_start:.1f}s")

    return {"sweep_results": all_results, "n_runs": run_count}


def _aggregate_seed_results(
    results: list[MergeResult],
    corruption_rate: float,
    calibration: CalibrationRegime,
    n_sources: int,
) -> dict[str, Any]:
    """Aggregate results across seeds with bootstrap CIs."""
    def _ci(values: list[float]) -> dict[str, float]:
        if not values or all(v == 0.0 for v in values):
            return {"lower": 0.0, "mean": 0.0, "upper": 0.0}
        lo, mn, hi = bootstrap_ci(values, n_bootstrap=2000, seed=42)
        return {"lower": round(lo, 5), "mean": round(mn, 5), "upper": round(hi, 5)}

    return {
        "corruption_rate": corruption_rate,
        "calibration": calibration.value,
        "n_sources": n_sources,
        "n_seeds": len(results),
        # H8.6a: Overall accuracy
        "jsonldex_highest_acc": _ci([r.jsonldex_highest_accuracy for r in results]),
        "jsonldex_weighted_vote_acc": _ci([r.jsonldex_weighted_vote_accuracy for r in results]),
        "rdflib_union_acc": _ci([r.rdflib_union_accuracy for r in results]),
        "random_acc": _ci([r.random_accuracy for r in results]),
        "majority_vote_acc": _ci([r.majority_vote_accuracy for r in results]),
        "most_recent_acc": _ci([r.most_recent_accuracy for r in results]),
        "b5_conf_argmax_acc": _ci([r.b5_confidence_argmax_accuracy for r in results]),
        # H8.6c: Delta (confidence-aware - majority vote)
        "delta_highest_vs_majority": _ci([
            r.jsonldex_highest_accuracy - r.majority_vote_accuracy
            for r in results
        ]),
        # H8.6b: Partitioned accuracy
        "highest_majority_correct_acc": _ci([r.highest_majority_correct_acc for r in results]),
        "weighted_vote_majority_correct_acc": _ci([r.weighted_vote_majority_correct_acc for r in results]),
        "highest_standard_acc": _ci([r.highest_standard_acc for r in results]),
        "weighted_vote_standard_acc": _ci([r.weighted_vote_standard_acc for r in results]),
        # H8.6d: ECE
        "ece_noisy_or": _ci([r.ece_noisy_or for r in results]),
        "ece_max": _ci([r.ece_max for r in results]),
        # H8.6e/f
        "diff_precision": _ci([r.diff_precision for r in results]),
        "diff_recall": _ci([r.diff_recall for r in results]),
        "audit_completeness": _ci([r.audit_completeness for r in results]),
        # H8.6g: Throughput
        "throughput_p50_ms": _ci([r.throughput_p50_ms for r in results]),
        # Metadata
        "n_conflicts_total": [r.n_conflicts_total for r in results],
        "n_majority_correct_conflicts": [r.n_majority_correct_conflicts for r in results],
    }


# ═══════════════════════════════════════════════════════════════════
# PHASE: SCALING ANALYSIS
# ═══════════════════════════════════════════════════════════════════


def run_scaling() -> dict[str, Any]:
    """Run scaling analysis: 4 sizes x 5 seeds = 20 runs."""
    print("\n" + "=" * 60)
    print("PHASE 1b: Scaling Analysis")
    print(f"  Sizes: {SCALING_SIZES}")
    print(f"  {N_SEEDS} seeds each = {len(SCALING_SIZES) * N_SEEDS} runs")
    print("=" * 60)

    scaling_results: list[dict[str, Any]] = []
    run_count = 0
    total_runs = len(SCALING_SIZES) * N_SEEDS
    t_start = time.perf_counter()

    for size in SCALING_SIZES:
        seed_results: list[MergeResult] = []
        for s in range(N_SEEDS):
            seed = BASE_SEED + s * 1000
            run_count += 1
            config = ExperimentConfig(
                n_entities=size,
                n_sources=SCALING_N_SOURCES,
                corruption_rate=SCALING_CORRUPTION,
                calibration=SCALING_CALIBRATION,
                overlap_rate=SWEEP_OVERLAP,
                seed=seed,
            )
            result = run_single_config(config)
            seed_results.append(result)

            elapsed = time.perf_counter() - t_start
            print(f"  [{run_count:2d}/{total_runs}] "
                  f"size={size:5d} seed={seed:5d} "
                  f"p50={result.throughput_p50_ms:.2f}ms "
                  f"acc={result.jsonldex_highest_accuracy:.3f} "
                  f"({elapsed:.0f}s elapsed)")

        # Aggregate
        agg = {
            "graph_size": size,
            "throughput_p50_ms": _ci_list([r.throughput_p50_ms for r in seed_results]),
            "throughput_p95_ms": _ci_list([r.throughput_p95_ms for r in seed_results]),
            "throughput_p99_ms": _ci_list([r.throughput_p99_ms for r in seed_results]),
            "accuracy": _ci_list([r.jsonldex_highest_accuracy for r in seed_results]),
        }
        scaling_results.append(agg)

    # Compute scaling exponent: log(time) vs log(size)
    log_sizes = np.log([r["graph_size"] for r in scaling_results])
    log_times = np.log([r["throughput_p50_ms"]["mean"] for r in scaling_results])
    if len(log_sizes) >= 2:
        coeffs = np.polyfit(log_sizes, log_times, 1)
        scaling_exponent = float(coeffs[0])
    else:
        scaling_exponent = float("nan")

    print(f"\nScaling exponent: {scaling_exponent:.3f}")
    print(f"Scaling complete: {run_count} runs in "
          f"{time.perf_counter() - t_start:.1f}s")

    return {
        "scaling_results": scaling_results,
        "scaling_exponent": scaling_exponent,
        "n_runs": run_count,
    }


def _ci_list(values: list[float]) -> dict[str, float]:
    """Bootstrap CI helper for scaling results."""
    if not values:
        return {"lower": 0.0, "mean": 0.0, "upper": 0.0}
    lo, mn, hi = bootstrap_ci(values, n_bootstrap=2000, seed=42)
    return {"lower": round(lo, 5), "mean": round(mn, 5), "upper": round(hi, 5)}


# ═══════════════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════════


def print_summary(sweep_data: dict, scaling_data: dict | None) -> None:
    """Print a human-readable summary of results."""
    print("\n" + "=" * 70)
    print("EN8.6 RESULTS SUMMARY")
    print("=" * 70)

    if "sweep_results" in sweep_data:
        results = sweep_data["sweep_results"]

        # H8.6a: Best accuracy by calibration regime
        print("\n--- H8.6a: Conflict Resolution Accuracy (ideal calibration) ---")
        ideal_results = [r for r in results
                         if r["calibration"] == "ideal" and r["n_sources"] == 3]
        for r in ideal_results:
            cr = r["corruption_rate"]
            h = r["jsonldex_highest_acc"]
            mv = r["majority_vote_acc"]
            b5 = r["b5_conf_argmax_acc"]
            print(f"  cr={cr:.2f}: highest={h['mean']:.3f} [{h['lower']:.3f},{h['upper']:.3f}]"
                  f"  majority={mv['mean']:.3f}  B5={b5['mean']:.3f}")

        # H8.6c: Delta heatmap summary
        print("\n--- H8.6c: Delta (highest - majority_vote) by regime ---")
        for cal in CalibrationRegime:
            cal_results = [r for r in results
                           if r["calibration"] == cal.value and r["n_sources"] == 3]
            if cal_results:
                deltas = [r["delta_highest_vs_majority"]["mean"] for r in cal_results]
                avg_delta = np.mean(deltas)
                print(f"  {cal.value:13s}: avg delta = {avg_delta:+.3f} "
                      f"(per corruption_rate: {[f'{d:+.3f}' for d in deltas]})")

        # H8.6d: ECE
        print("\n--- H8.6d: ECE (ideal calibration, n_sources=3, cr=0.10) ---")
        ece_results = [r for r in results
                       if r["calibration"] == "ideal" and r["n_sources"] == 3
                       and r["corruption_rate"] == 0.10]
        if ece_results:
            r = ece_results[0]
            print(f"  noisy-OR ECE: {r['ece_noisy_or']['mean']:.4f}")
            print(f"  max ECE:      {r['ece_max']['mean']:.4f}")

        # H8.6e/f: Diff and audit
        print("\n--- H8.6e/f: Diff & Audit (all configs) ---")
        all_diff_p = [r["diff_precision"]["mean"] for r in results]
        all_diff_r = [r["diff_recall"]["mean"] for r in results]
        all_audit = [r["audit_completeness"]["mean"] for r in results]
        print(f"  Diff precision: {np.mean(all_diff_p):.3f} (min={min(all_diff_p):.3f})")
        print(f"  Diff recall:    {np.mean(all_diff_r):.3f} (min={min(all_diff_r):.3f})")
        print(f"  Audit complete: {np.mean(all_audit):.3f} (min={min(all_audit):.3f})")

    if scaling_data and "scaling_results" in scaling_data:
        print("\n--- H8.6g: Scaling ---")
        for r in scaling_data["scaling_results"]:
            t = r["throughput_p50_ms"]
            print(f"  {r['graph_size']:5d} nodes: "
                  f"p50={t['mean']:.2f}ms [{t['lower']:.2f},{t['upper']:.2f}]")
        print(f"  Scaling exponent: {scaling_data['scaling_exponent']:.3f}")


# ═══════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════


def save_results(sweep_data: dict, scaling_data: dict | None) -> None:
    """Save results as primary JSON + timestamped archive."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    combined = {
        "experiment_id": "EN8.6",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "synthetic_phase1",
        "sweep": sweep_data,
        "scaling": scaling_data,
    }

    # Primary
    primary_path = RESULTS_DIR / "en8_6_phase1.json"
    with open(primary_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {primary_path}")

    # Timestamped archive
    archive_path = RESULTS_DIR / f"en8_6_phase1_{timestamp}.json"
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {archive_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="EN8.6 Phase 1 Runner")
    parser.add_argument(
        "--phase", choices=["sweep", "scaling", "all"],
        default="all", help="Which phase to run (default: all)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    sweep_data: dict = {}
    scaling_data: dict | None = None

    if args.phase in ("sweep", "all"):
        sweep_data = run_sweep()

    if args.phase in ("scaling", "all"):
        scaling_data = run_scaling()

    print_summary(sweep_data, scaling_data)
    save_results(sweep_data, scaling_data)

    total_time = time.perf_counter() - t0
    print(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f}min)")


if __name__ == "__main__":
    main()
