"""Diagnose: fix outlier detection threshold using inter-sensor baseline.

Root cause: intra-sensor sigma (0.3C) is wrong for cross-sensor mode
because co-located sensors legitimately disagree by 0.5-1C due to
microenvironments. We need inter-sensor disagreement as the baseline.

Approach:
1. Compute per-sensor inter-sensor disagreement distribution on CLEAN data
2. Use MAD of that distribution as the per-sensor baseline
3. Flag readings where disagreement exceeds k * baseline
4. Sweep k to find optimal recall/FPR tradeoff
"""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "python" / "src"))

from en8_2_core import (load_intel_lab_data, select_sensor_cluster,
                         time_align_readings)


def compute_inter_sensor_baseline(aligned, sids):
    """Compute per-sensor disagreement baseline from clean data.

    For each sensor, compute its absolute deviation from neighbor
    median at each time step. Use the distribution of these deviations
    to establish what 'normal disagreement' looks like.

    Returns per-sensor: {sid: {"median": ..., "mad_sigma": ..., "p95": ...}}
    """
    n = len(aligned["timestamps"])
    stats = {}
    for sid in sids:
        others = [s for s in sids if s != sid]
        abs_devs = []
        for t in range(n):
            if np.isnan(aligned[sid][t]):
                continue
            ovals = [aligned[s][t] for s in others if not np.isnan(aligned[s][t])]
            if len(ovals) < 2:
                continue
            consensus = np.median(ovals)
            abs_devs.append(abs(aligned[sid][t] - consensus))
        if abs_devs:
            d = np.array(abs_devs)
            stats[sid] = {
                "median": float(np.median(d)),
                "mad_sigma": float(np.median(np.abs(d - np.median(d))) * 1.4826),
                "p95": float(np.percentile(d, 95)),
                "p99": float(np.percentile(d, 99)),
                "mean": float(np.mean(d)),
                "n": len(d),
            }
    return stats


def cross_sensor_outlier_detect(raw_aligned, sid, sids, threshold):
    """Detect outliers using cross-sensor consensus with adaptive threshold.

    Flags readings where |value - neighbor_median| > threshold.
    """
    v = raw_aligned[sid]
    others = [s for s in sids if s != sid]
    n = len(v)
    flagged = np.zeros(n, dtype=bool)

    for t in range(n):
        if np.isnan(v[t]):
            continue
        ovals = [raw_aligned[s][t] for s in others
                 if t < len(raw_aligned[s]) and not np.isnan(raw_aligned[s][t])]
        if len(ovals) < 1:
            continue
        consensus = np.median(ovals)
        if abs(v[t] - consensus) > threshold:
            flagged[t] = True

    return flagged


def main():
    raw = load_intel_lab_data()
    # Filter for clean data baseline estimation
    clean = [r for r in raw if r["temperature"] is not None
             and 10 <= r["temperature"] <= 45]

    clusters = {"primary": [4, 6, 7, 9], "validation": [22, 23, 24, 25, 26]}

    for cname, sids in clusters.items():
        print(f"\n{'='*55}")
        print(f"Cluster: {cname} ({sids})")
        print(f"{'='*55}")

        # Step 1: Baseline from CLEAN data
        clean_c = select_sensor_cluster(clean, sids)
        clean_al = time_align_readings(clean_c, sids, interval_seconds=300)
        baseline = compute_inter_sensor_baseline(clean_al, sids)

        print(f"\n  Inter-sensor disagreement baseline (clean data):")
        for sid in sids:
            b = baseline.get(sid, {})
            print(f"    Sensor {sid}: median={b.get('median',0):.3f}  "
                  f"MAD_sigma={b.get('mad_sigma',0):.3f}  "
                  f"P95={b.get('p95',0):.3f}  P99={b.get('p99',0):.3f}")

        # Step 2: Load RAW (unfiltered) for outlier detection
        raw_c = select_sensor_cluster(raw, sids)
        raw_al = time_align_readings(raw_c, sids, interval_seconds=300)

        # Step 3: Sweep threshold strategies
        print(f"\n  Threshold sweep:")
        print(f"  {'Strategy':45s}  {'Recall':>8s}  {'FPR':>8s}  {'F1':>8s}")
        print(f"  {'-'*45}  {'-'*8}  {'-'*8}  {'-'*8}")

        strategies = []

        # Strategy A: Fixed absolute thresholds
        for thresh in [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
            total_tp, total_out, total_fp, total_clean = 0, 0, 0, 0
            for sid in sids:
                v = raw_al[sid]
                valid = ~np.isnan(v)
                is_out = valid & ((v < 10) | (v > 45))
                n_out = int(np.sum(is_out))
                if n_out == 0:
                    continue
                flagged = cross_sensor_outlier_detect(raw_al, sid, sids, thresh)
                tp = int(np.sum(flagged & is_out))
                fp = int(np.sum(flagged & ~is_out & valid))
                total_tp += tp
                total_out += n_out
                total_fp += fp
                total_clean += int(np.sum(~is_out & valid))
            recall = total_tp / total_out if total_out > 0 else 0
            fpr = total_fp / total_clean if total_clean > 0 else 0
            prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
            strategies.append(("fixed", thresh, recall, fpr, f1))
            print(f"  Fixed |dev| > {thresh:5.1f}C                        "
                  f"  {recall:7.1%}  {fpr:7.1%}  {f1:7.3f}")

        # Strategy B: Per-sensor adaptive (k * P95 of clean baseline)
        for k in [1.0, 1.5, 2.0, 3.0, 5.0]:
            total_tp, total_out, total_fp, total_clean = 0, 0, 0, 0
            for sid in sids:
                v = raw_al[sid]
                valid = ~np.isnan(v)
                is_out = valid & ((v < 10) | (v > 45))
                n_out = int(np.sum(is_out))
                if n_out == 0:
                    continue
                b = baseline.get(sid, {})
                thresh = k * b.get("p95", 3.0)
                flagged = cross_sensor_outlier_detect(raw_al, sid, sids, thresh)
                tp = int(np.sum(flagged & is_out))
                fp = int(np.sum(flagged & ~is_out & valid))
                total_tp += tp
                total_out += n_out
                total_fp += fp
                total_clean += int(np.sum(~is_out & valid))
            recall = total_tp / total_out if total_out > 0 else 0
            fpr = total_fp / total_clean if total_clean > 0 else 0
            prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
            strategies.append(("adaptive_p95", k, recall, fpr, f1))
            print(f"  Adaptive k={k:.1f} * P95(clean baseline)           "
                  f"  {recall:7.1%}  {fpr:7.1%}  {f1:7.3f}")

        # Strategy C: Per-sensor adaptive (k * MAD_sigma of clean baseline)
        for k in [3, 5, 8, 10, 15, 20]:
            total_tp, total_out, total_fp, total_clean = 0, 0, 0, 0
            for sid in sids:
                v = raw_al[sid]
                valid = ~np.isnan(v)
                is_out = valid & ((v < 10) | (v > 45))
                n_out = int(np.sum(is_out))
                if n_out == 0:
                    continue
                b = baseline.get(sid, {})
                thresh = k * b.get("mad_sigma", 0.3)
                flagged = cross_sensor_outlier_detect(raw_al, sid, sids, thresh)
                tp = int(np.sum(flagged & is_out))
                fp = int(np.sum(flagged & ~is_out & valid))
                total_tp += tp
                total_out += n_out
                total_fp += fp
                total_clean += int(np.sum(~is_out & valid))
            recall = total_tp / total_out if total_out > 0 else 0
            fpr = total_fp / total_clean if total_clean > 0 else 0
            prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
            strategies.append(("adaptive_mad", k, recall, fpr, f1))
            print(f"  Adaptive k={k:2d} * MAD_sigma(clean baseline)      "
                  f"  {recall:7.1%}  {fpr:7.1%}  {f1:7.3f}")

        # Best F1
        best = max(strategies, key=lambda x: x[4])
        print(f"\n  Best F1: {best[0]} k={best[1]} -> recall={best[2]:.1%} "
              f"FPR={best[3]:.1%} F1={best[4]:.3f}")


if __name__ == "__main__":
    main()
