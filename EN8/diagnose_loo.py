"""Quick diagnostic: test LOO eval on Intel Lab data."""
import sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "python" / "src"))

from en8_2_core import (load_intel_lab_data, select_sensor_cluster,
    time_align_readings, leave_one_out_eval)

t0 = time.time()
data = load_intel_lab_data()
print(f"Loaded {len(data):,} records in {time.time()-t0:.1f}s")

cluster = select_sensor_cluster(data, [4, 6, 7, 9])
print(f"Cluster: {len(cluster):,} records")

aligned = time_align_readings(cluster, [4, 6, 7, 9], interval_seconds=300)
n = len(aligned["timestamps"])
print(f"Aligned: {n} time steps")
for sid in [4, 6, 7, 9]:
    valid = np.sum(~np.isnan(aligned[sid]))
    print(f"  Sensor {sid}: {valid}/{n} valid ({valid/n:.1%})")

# Quick LOO on first 500 steps
print(f"\nRunning LOO on first 500 steps...")
small = {k: v[:500] if isinstance(v, np.ndarray) else v for k, v in aligned.items()}
sigmas = {4: 0.5, 6: 0.5, 7: 0.5, 9: 0.5}
t1 = time.time()
results = leave_one_out_eval(small, sigmas, value_range=(15, 35))
print(f"LOO done in {time.time()-t1:.1f}s")
for method, r in sorted(results.items()):
    print(f"  {method:20s}  RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  n={r['n_points']}")
