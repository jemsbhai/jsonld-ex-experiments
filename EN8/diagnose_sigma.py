"""Diagnostic: estimate per-sensor noise from Intel Lab data."""
import sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "python" / "src"))

from en8_2_core import (load_intel_lab_data, select_sensor_cluster,
    time_align_readings, leave_one_out_eval)

data = load_intel_lab_data()

# Two clusters for primary + validation
clusters = {
    "primary": [4, 6, 7, 9],
    "validation": [25, 26, 27, 28],
}

for name, sensor_ids in clusters.items():
    print(f"\n=== Cluster: {name} ({sensor_ids}) ===")
    cluster = select_sensor_cluster(data, sensor_ids)
    aligned = time_align_readings(cluster, sensor_ids, interval_seconds=300)
    n = len(aligned["timestamps"])

    # Estimate per-sensor noise via consensus: for each time step,
    # compute deviation from the mean of OTHER sensors
    print(f"  {n} time steps")
    sigmas = {}
    for sid in sensor_ids:
        others = [s for s in sensor_ids if s != sid]
        deviations = []
        for t in range(n):
            if np.isnan(aligned[sid][t]):
                continue
            other_vals = [aligned[s][t] for s in others if not np.isnan(aligned[s][t])]
            if len(other_vals) >= 2:
                consensus = np.mean(other_vals)
                deviations.append(aligned[sid][t] - consensus)
        if deviations:
            dev_arr = np.array(deviations)
            # Use MAD (robust) instead of std
            mad = np.median(np.abs(dev_arr - np.median(dev_arr)))
            sigma_mad = mad * 1.4826  # MAD to sigma conversion
            sigma_std = np.std(dev_arr)
            sigmas[sid] = sigma_mad
            print(f"  Sensor {sid}: n={len(deviations):,}  "
                  f"std={sigma_std:.4f}  MAD_sigma={sigma_mad:.4f}  "
                  f"mean_dev={np.mean(dev_arr):+.4f}")
        else:
            sigmas[sid] = 0.5
            print(f"  Sensor {sid}: no consensus data, using default")

    # Now run LOO with estimated sigmas
    print(f"\n  LOO with estimated sigmas (first 1000 steps):")
    small = {k: v[:1000] if isinstance(v, np.ndarray) else v
             for k, v in aligned.items()}
    t1 = time.time()
    results = leave_one_out_eval(small, sigmas, value_range=(15, 35))
    print(f"  Done in {time.time()-t1:.1f}s")
    for method, r in sorted(results.items()):
        print(f"    {method:20s}  RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  n={r['n_points']}")

    # Also run with equal sigmas for comparison
    equal_sigmas = {sid: 0.5 for sid in sensor_ids}
    results_eq = leave_one_out_eval(small, equal_sigmas, value_range=(15, 35))
    print(f"\n  LOO with EQUAL sigmas (control):")
    for method, r in sorted(results_eq.items()):
        print(f"    {method:20s}  RMSE={r['rmse']:.4f}")
