"""Diagnose Phase D failures: outlier recall and inverted correlation."""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "python" / "src"))

from en8_2_core import (load_intel_lab_data, select_sensor_cluster,
                         time_align_readings)
from en8_2_pipeline import detect_outliers_by_annotation


def filter_outliers(data, lo=10.0, hi=45.0):
    return [r for r in data if r["temperature"] is not None
            and lo <= r["temperature"] <= hi]


raw = load_intel_lab_data()
data = filter_outliers(raw)

# ── Problem 1: Why is outlier recall so low? ──
print("=" * 55)
print("Problem 1: Outlier Detection Recall")
print("=" * 55)

sids = [4, 6, 7, 9]
raw_c = select_sensor_cluster(raw, sids)
raw_al = time_align_readings(raw_c, sids, interval_seconds=300)

for sid in sids:
    v = raw_al[sid]
    valid = ~np.isnan(v)
    is_out = valid & ((v < 10) | (v > 45))
    n_out = int(np.sum(is_out))
    if n_out == 0:
        print(f"  Sensor {sid}: no outliers")
        continue

    # What do the outlier values look like?
    out_vals = v[is_out]
    print(f"\n  Sensor {sid}: {n_out} outliers")
    print(f"    Outlier value range: [{np.min(out_vals):.1f}, {np.max(out_vals):.1f}]")
    print(f"    Normal value range: [{np.min(v[valid & ~is_out]):.1f}, {np.max(v[valid & ~is_out]):.1f}]")

    # Are outliers clustered in time?
    out_indices = np.where(is_out)[0]
    gaps = np.diff(out_indices)
    if len(gaps) > 0:
        print(f"    Consecutive outliers: {np.sum(gaps == 1)} pairs")
        # Find longest run
        run_len = 1
        max_run = 1
        for g in gaps:
            if g == 1:
                run_len += 1
                max_run = max(max_run, run_len)
            else:
                run_len = 1
        print(f"    Longest consecutive run: {max_run}")
        print(f"    Median gap between outliers: {np.median(gaps):.0f} steps")

    # Test detection at various thresholds
    sigma = 0.3
    for thresh in [3.0, 5.0, 10.0, 50.0]:
        flagged = detect_outliers_by_annotation(v, sigma=sigma, threshold_sigmas=thresh)
        tp = int(np.sum(flagged & is_out))
        fp = int(np.sum(flagged & ~is_out & valid))
        recall = tp / n_out if n_out > 0 else 0
        print(f"    thresh={thresh:5.1f}sigma: recall={recall:.1%} ({tp}/{n_out}) FP={fp}")

    # Manual check: what does the detection see at an outlier position?
    first_out = out_indices[0]
    window = 20
    start = max(0, first_out - window // 2)
    end = min(len(v), first_out + window // 2 + 1)
    local = v[start:end]
    local_clean = local[~np.isnan(local)]
    local_median = np.median(local_clean)
    deviation = abs(v[first_out] - local_median)
    print(f"    First outlier at idx={first_out}: val={v[first_out]:.1f}")
    print(f"      Local window [{start}:{end}] median={local_median:.1f}")
    print(f"      |val - median| = {deviation:.1f}")
    print(f"      threshold at 5sigma = {5*sigma:.1f}")
    print(f"      Would flag: {deviation > 5*sigma}")

    # Count how many outliers are in the local window of other outliers
    n_window_contaminated = 0
    for idx in out_indices[:50]:  # check first 50
        s = max(0, idx - 10)
        e = min(len(v), idx + 11)
        local = v[s:e]
        n_out_in_window = np.sum((~np.isnan(local)) & ((local < 10) | (local > 45)))
        if n_out_in_window > len(local_clean) // 2:
            n_window_contaminated += 1
    print(f"    Windows contaminated (>50% outliers): {n_window_contaminated}/50")


# ── Problem 2: Inverted correlation in validation cluster ──
print("\n" + "=" * 55)
print("Problem 2: Inverted Bias-Disagreement Correlation")
print("=" * 55)

val_sids = [22, 23, 24]
val_c = select_sensor_cluster(data, val_sids)
val_al = time_align_readings(val_c, val_sids, interval_seconds=300)
n = len(val_al["timestamps"])

for sid in val_sids:
    others = [s for s in val_sids if s != sid]
    devs = []
    for t in range(n):
        if np.isnan(val_al[sid][t]):
            continue
        ovals = [val_al[s][t] for s in others if not np.isnan(val_al[s][t])]
        if len(ovals) >= 1:
            devs.append(val_al[sid][t] - np.mean(ovals))
    d = np.array(devs)
    bias = np.median(d)
    sigma = np.median(np.abs(d - bias)) * 1.4826
    # Mean absolute deviation from consensus (our disagreement signal)
    mad_from_consensus = np.mean(np.abs(d))
    print(f"  Sensor {sid}: bias={bias:+.3f}  sigma={sigma:.3f}  "
          f"mean|dev|={mad_from_consensus:.3f}  n={len(devs)}")

print("\n  Issue: with only 3 sensors, 'consensus' of 2 others is fragile.")
print("  If sensor 22 (bias=-0.37) deviates from mean(23,24),")
print("  but 23 and 24 themselves have spread, the disagreement")
print("  of the high-bias sensor may not be the largest.")
print("  With N=3, correlation of |bias| vs disagreement is")
print("  unreliable (only 3 data points for Pearson r).")
