"""EN1.5 Runner -- Deduction Under Uncertainty.

Full three-part experiment:
  Part A: Single-edge deduction across 15 BN models + Synthea
  Part B: Multi-hop chained deduction across all models with paths
  Evidence levels N in {5, 10, 50, 100, 1000}, 1000 reps per (edge/path, N).

Usage:
    cd experiments
    python EN1/en1_5_deduction.py                # full run (~20-30 min)
    python EN1/en1_5_deduction.py --quick         # quick validation (~2 min)
    python EN1/en1_5_deduction.py --n-reps 100    # medium run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_EXPERIMENTS_ROOT))

from en1_5_core import (
    ALL_BN_MODELS,
    load_all_bn_kbs,
    load_synthea_kb,
    run_evidence_sweep,
    extract_paths,
    run_multihop_sweep,
    DeductionKB,
)
from infra.env_log import log_environment
from infra.results import ExperimentResult


# ===================================================================
# Configuration
# ===================================================================

EXPERIMENT_ID = "EN1.5"
EXPERIMENT_NAME = "Deduction Under Uncertainty"
N_VALUES = [5, 10, 50, 100, 1000]
DEFAULT_N_REPS = 1000
SEED = 42
MAX_PATH_LENGTH = 4
MAX_PATHS_PER_LENGTH = 200

RESULTS_DIR = _SCRIPT_DIR / "results"


# ===================================================================
# Part A: Single-Edge Deduction
# ===================================================================

def run_single_edge_part(
    kbs: list[DeductionKB],
    n_values: list[int],
    n_reps: int,
    seed: int,
) -> dict:
    """Run single-edge deduction across all KBs."""
    print(f"\n{'=' * 70}")
    print(f"  PART A: Single-Edge Deduction ({len(kbs)} datasets)")
    print(f"{'=' * 70}")

    total_edges = sum(len(kb.edges) for kb in kbs)
    total_trials = total_edges * len(n_values) * n_reps
    print(f"  Total edges: {total_edges:,}")
    print(f"  Total trials: {total_trials:,}")

    all_results = {}
    t0 = time.perf_counter()

    for kb in kbs:
        print(f"\n  {kb.name:>12}: {len(kb.edges)} edges ...", end=" ", flush=True)
        t_kb = time.perf_counter()
        sweep = run_evidence_sweep(kb, n_values=n_values, n_reps=n_reps, seed=seed)
        elapsed = time.perf_counter() - t_kb
        print(f"done ({elapsed:.1f}s)")

        all_results[kb.name] = {
            "n_nodes": len(kb.nodes),
            "n_edges": len(kb.edges),
            "elapsed_seconds": elapsed,
            "sweep": {str(n): r for n, r in sweep.items()},
        }

    total_elapsed = time.perf_counter() - t0
    print(f"\n  Part A total: {total_elapsed:.1f}s")

    # -- Summary table --
    print(f"\n  {'Dataset':>12}  {'Edges':>5}", end="")
    for n in n_values:
        print(f"  {'ScMAE@'+str(n):>10}  {'SLMAE@'+str(n):>10}", end="")
    print()
    print(f"  {'-' * (18 + len(n_values) * 24)}")
    for kb in kbs:
        r = all_results[kb.name]
        print(f"  {kb.name:>12}  {r['n_edges']:>5}", end="")
        for n in n_values:
            sr = r["sweep"][str(n)]
            print(f"  {sr['scalar_mae']:>10.6f}  {sr['sl_mae']:>10.6f}", end="")
        print()

    # -- Delta summary at N=5 --
    print(f"\n  SL advantage at N=5 (DMAE = Scalar-SL, positive = SL wins):")
    for kb in kbs:
        r5 = all_results[kb.name]["sweep"]["5"]
        delta = r5["scalar_mae"] - r5["sl_mae"]
        pct = (delta / r5["scalar_mae"] * 100) if r5["scalar_mae"] > 1e-9 else 0
        winner = "SL *" if delta > 0 else "Scalar"
        print(f"    {kb.name:>12}: D={delta:+.6f} ({pct:+.1f}%)  {winner}")

    all_results["_elapsed_seconds"] = total_elapsed
    return all_results


# ===================================================================
# Part B: Multi-Hop Chained Deduction
# ===================================================================

def run_multihop_part(
    kbs: list[DeductionKB],
    n_values: list[int],
    n_reps: int,
    seed: int,
    max_path_length: int,
    max_paths_per_length: int,
) -> dict:
    """Run multi-hop chained deduction across all KBs with paths."""
    print(f"\n{'=' * 70}")
    print(f"  PART B: Multi-Hop Chained Deduction (max length {max_path_length})")
    print(f"{'=' * 70}")

    # Pre-check which KBs have multi-hop paths
    kbs_with_paths = []
    for kb in kbs:
        paths = extract_paths(kb, max_length=max_path_length)
        if paths:
            from collections import Counter
            length_counts = Counter(len(p) for p in paths)
            summary = ", ".join(f"L{k}={v}" for k, v in sorted(length_counts.items()))
            print(f"  {kb.name:>12}: {len(paths):>6} paths  ({summary})")
            kbs_with_paths.append(kb)
        else:
            print(f"  {kb.name:>12}: no multi-hop paths, skipping")

    all_results = {}
    t0 = time.perf_counter()

    for kb in kbs_with_paths:
        print(f"\n  Running {kb.name} ...", end=" ", flush=True)
        t_kb = time.perf_counter()
        sweep = run_multihop_sweep(
            kb, n_values=n_values, n_reps=n_reps, seed=seed,
            max_path_length=max_path_length,
            max_paths_per_length=max_paths_per_length,
        )
        elapsed = time.perf_counter() - t_kb
        print(f"done ({elapsed:.1f}s)")

        all_results[kb.name] = {
            "elapsed_seconds": elapsed,
            "sweep_by_length": {k: {str(n): v for n, v in ndict.items()}
                                for k, ndict in sweep.items()},
        }

    total_elapsed = time.perf_counter() - t0
    print(f"\n  Part B total: {total_elapsed:.1f}s")

    # -- Summary: MAE by path length at N=5 --
    print(f"\n  Multi-hop MAE at N=5 by path length:")
    print(f"  {'Dataset':>12}  {'Length':>6}  {'Paths':>5}  "
          f"{'ScalarMAE':>10}  {'SL_MAE':>10}  {'DMAE':>10}  Winner")
    print(f"  {'-' * 75}")
    for kb_name, data in all_results.items():
        if kb_name.startswith("_"):
            continue
        for length_key in sorted(data["sweep_by_length"].keys()):
            r = data["sweep_by_length"][length_key].get("5")
            if r is None:
                continue
            delta = r["scalar_mae"] - r["sl_mae"]
            winner = "SL *" if delta > 0 else "Scalar"
            length = length_key.replace("length_", "")
            print(f"  {kb_name:>12}  {length:>6}  {r['n_paths']:>5}  "
                  f"{r['scalar_mae']:>10.6f}  {r['sl_mae']:>10.6f}  "
                  f"{delta:>+10.6f}  {winner}")

    all_results["_elapsed_seconds"] = total_elapsed
    return all_results


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT_ID}: {EXPERIMENT_NAME}")
    parser.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS,
                        help=f"Reps per (edge, N). Default: {DEFAULT_N_REPS}")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 50 reps, N in {5, 100}, 3 models")
    parser.add_argument("--no-multihop", action="store_true",
                        help="Skip multi-hop analysis")
    parser.add_argument("--max-path-length", type=int, default=MAX_PATH_LENGTH)
    parser.add_argument("--max-paths-per-length", type=int, default=MAX_PATHS_PER_LENGTH)
    args = parser.parse_args()

    n_reps = args.n_reps
    n_values = N_VALUES
    bn_models = None  # all

    if args.quick:
        n_reps = 50
        n_values = [5, 100]
        bn_models = ["asia", "alarm", "sachs"]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"{'=' * 70}")
    print(f"  {EXPERIMENT_ID}: {EXPERIMENT_NAME}")
    print(f"  Timestamp: {timestamp}")
    print(f"  N values: {n_values}")
    print(f"  Reps per (edge/path, N): {n_reps}")
    print(f"  Seed: {args.seed}")
    print(f"  Max path length: {args.max_path_length}")
    print(f"  Max paths/length: {args.max_paths_per_length}")
    print(f"{'=' * 70}")

    # Log environment
    env_info = log_environment()

    # -- Load all KBs --
    print("\nLoading knowledge bases ...")
    t_load = time.perf_counter()

    kbs = []
    # BN models
    bn_kbs = load_all_bn_kbs(model_names=bn_models)
    kbs.extend(bn_kbs)
    print(f"  BN models: {len(bn_kbs)} loaded "
          f"({sum(len(kb.edges) for kb in bn_kbs)} edges)")

    # Synthea
    try:
        synthea_kb = load_synthea_kb()
        kbs.append(synthea_kb)
        print(f"  Synthea: {len(synthea_kb.edges)} edges")
    except Exception as e:
        print(f"  Synthea: FAILED - {e}")

    print(f"  Total: {len(kbs)} datasets, "
          f"{sum(len(kb.edges) for kb in kbs)} edges")
    print(f"  Loading took {time.perf_counter() - t_load:.1f}s")

    # -- Part A: Single-edge --
    single_results = run_single_edge_part(
        kbs, n_values=n_values, n_reps=n_reps, seed=args.seed,
    )

    # -- Part B: Multi-hop --
    multihop_results = {}
    if not args.no_multihop:
        multihop_results = run_multihop_part(
            kbs, n_values=n_values, n_reps=n_reps, seed=args.seed,
            max_path_length=args.max_path_length,
            max_paths_per_length=args.max_paths_per_length,
        )

    # -- Aggregate cross-dataset statistics --
    print(f"\n{'=' * 70}")
    print(f"  AGGREGATE STATISTICS")
    print(f"{'=' * 70}")

    for n in n_values:
        sl_wins = 0
        total = 0
        sl_delta_sum = 0.0
        for kb in kbs:
            r = single_results.get(kb.name, {}).get("sweep", {}).get(str(n))
            if r:
                delta = r["scalar_mae"] - r["sl_mae"]
                if delta > 0:
                    sl_wins += 1
                sl_delta_sum += delta
                total += 1
        avg_delta = sl_delta_sum / total if total else 0
        print(f"  N={n:>5}: SL wins {sl_wins}/{total} datasets, "
              f"avg DMAE={avg_delta:+.6f}")

    # -- Save results --
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": timestamp,
        "config": {
            "n_values": n_values,
            "n_reps": n_reps,
            "seed": args.seed,
            "conditional_evidence": 10_000,
            "prior_weight": 2.0,
            "max_path_length": args.max_path_length,
            "max_paths_per_length": args.max_paths_per_length,
            "bn_models_used": [kb.name for kb in bn_kbs],
            "synthea_included": any(kb.name == "synthea" for kb in kbs),
        },
        "environment": env_info,
        "single_edge_results": single_results,
        "multihop_results": multihop_results,
    }

    primary = RESULTS_DIR / "en1_5_results.json"
    with open(primary, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {primary}")

    archive = RESULTS_DIR / f"en1_5_results_{timestamp}.json"
    with open(archive, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Archive saved: {archive}")

    print(f"\n{'=' * 70}")
    print(f"  {EXPERIMENT_ID} COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
