"""Deep investigation: what quality signals exist in Intel Lab data?

The current approach fails because:
1. Cross-sensor consensus fails for correlated failures (48% of outliers)
2. Single-sensor rolling median fails for sustained failures (runs of 300+)

Key insight: sensor failures are NOT instantaneous. They involve
TRANSITIONS. When sensor 4 goes from 22C to 122C, the RATE OF CHANGE
is enormous (100C in 5 minutes). Real temperature never changes this fast.

This diagnostic explores:
1. Rate-of-change signal: does it detect failure transitions?
2. Composite scoring: can we combine multiple signals?
3. SL reliability tracking: cumulative quality opinion per sensor
"""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "python" / "src"))

from en8_2_core import (load_intel_lab_data, select_sensor_cluster,
                         time_align_readings)
from jsonld_ex.confidence_algebra import Opinion, cumulative_fuse
from jsonld_ex.confidence_decay import decay_opinion


def main():
    raw = load_intel_lab_data()
    clean = [r for r in raw if r["temperature"] is not None
             and 10 <= r["temperature"] <= 45]

    sids = [4, 6, 7, 9]

    # Load raw (unfiltered) aligned data
    raw_c = select_sensor_cluster(raw, sids)
    raw_al = time_align_readings(raw_c, sids, interval_seconds=300)
    n = len(raw_al["timestamps"])

    # Load clean aligned data for baseline
    clean_c = select_sensor_cluster(clean, sids)
    clean_al = time_align_readings(clean_c, sids, interval_seconds=300)

    print("=" * 60)
    print("Signal 1: Rate of Change")
    print("=" * 60)

    # For each sensor, compute |delta_t| = |reading[t] - reading[t-1]|
    # Normal temperature: delta < 0.5C per 5 min
    # Failure transition: delta > 10C per 5 min
    for sid in sids:
        v = raw_al[sid]
        valid = ~np.isnan(v)
        is_out = valid & ((v < 10) | (v > 45))
        n_out = int(np.sum(is_out))

        deltas = np.full(n, np.nan)
        for t in range(1, n):
            if not np.isnan(v[t]) and not np.isnan(v[t-1]):
                deltas[t] = abs(v[t] - v[t-1])

        valid_deltas = deltas[~np.isnan(deltas)]
        print(f"\n  Sensor {sid}: {n_out} outliers")
        print(f"    Delta stats (all): median={np.median(valid_deltas):.3f}  "
              f"P95={np.percentile(valid_deltas, 95):.3f}  "
              f"P99={np.percentile(valid_deltas, 99):.3f}  "
              f"max={np.max(valid_deltas):.1f}")

        # Delta at outlier transitions
        out_idx = np.where(is_out)[0]
        transition_deltas = []
        for idx in out_idx:
            if idx > 0 and not np.isnan(deltas[idx]):
                transition_deltas.append(deltas[idx])
        if transition_deltas:
            td = np.array(transition_deltas)
            print(f"    Delta at outlier positions: median={np.median(td):.3f}  "
                  f"P95={np.percentile(td, 95):.3f}  max={np.max(td):.1f}")
            # How many outlier transitions have delta > threshold?
            for thresh in [0.5, 1.0, 2.0, 5.0]:
                caught = np.sum(td > thresh)
                print(f"      delta > {thresh:.1f}C: {caught}/{len(td)} ({caught/len(td):.1%})")

    print("\n" + "=" * 60)
    print("Signal 2: Cross-sensor disagreement (existing)")
    print("=" * 60)
    # Already explored — ceiling at 52%

    print("\n" + "=" * 60)
    print("Signal 3: Composite (rate + consensus + range)")
    print("=" * 60)

    # Composite: flag if ANY of these is true:
    # A) |value - neighbor_median| > 5C  (cross-sensor)
    # B) |delta| > 2C per 5 min          (rate of change)
    # C) value outside [10, 45]           (range check - but this IS the label, so unfair)
    #
    # We need a composite that works WITHOUT the range check
    # because the range check IS the ground truth definition.
    #
    # Fair composite: A OR B

    for sid in sids:
        v = raw_al[sid]
        valid = ~np.isnan(v)
        is_out = valid & ((v < 10) | (v > 45))
        n_out = int(np.sum(is_out))
        if n_out == 0:
            continue
        n_clean = int(np.sum(valid & ~is_out))

        others = [s for s in sids if s != sid]

        # Signal A: cross-sensor
        flag_consensus = np.zeros(n, dtype=bool)
        for t in range(n):
            if np.isnan(v[t]):
                continue
            ovals = [raw_al[s][t] for s in others if not np.isnan(raw_al[s][t])]
            if len(ovals) >= 1:
                consensus = np.median(ovals)
                if abs(v[t] - consensus) > 5.0:
                    flag_consensus[t] = True

        # Signal B: rate of change
        flag_rate = np.zeros(n, dtype=bool)
        for t in range(1, n):
            if not np.isnan(v[t]) and not np.isnan(v[t-1]):
                if abs(v[t] - v[t-1]) > 2.0:
                    flag_rate[t] = True

        # Signal C: rate of change with forward propagation
        # Once a big jump is detected, flag the next K readings too
        flag_rate_prop = flag_rate.copy()
        propagate_steps = 50  # ~4 hours at 5-min intervals
        for t in range(n):
            if flag_rate[t]:
                for k in range(1, propagate_steps + 1):
                    if t + k < n:
                        flag_rate_prop[t + k] = True

        # Composite: A OR B
        flag_ab = flag_consensus | flag_rate
        flag_abc = flag_consensus | flag_rate_prop

        print(f"\n  Sensor {sid} ({n_out} outliers, {n_clean} clean):")

        for label, flagged in [
            ("Consensus only", flag_consensus),
            ("Rate only (delta>2C)", flag_rate),
            ("Rate propagated (50 steps)", flag_rate_prop),
            ("Consensus OR Rate", flag_ab),
            ("Consensus OR Rate-propagated", flag_abc),
        ]:
            tp = int(np.sum(flagged & is_out))
            fp = int(np.sum(flagged & ~is_out & valid))
            recall = tp / n_out if n_out > 0 else 0
            fpr = fp / n_clean if n_clean > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
            print(f"    {label:35s}  recall={recall:6.1%}  FPR={fpr:6.1%}  "
              f"prec={prec:6.1%}  F1={f1:.3f}")

    print("\n" + "=" * 60)
    print("Signal 4: SL reliability tracking")
    print("=" * 60)
    print("  Build cumulative opinion per sensor from agreement history.")
    print("  When sensor starts failing, opinion should degrade.")

    for sid in sids[:2]:  # Just first 2 for speed
        v = raw_al[sid]
        valid = ~np.isnan(v)
        is_out = valid & ((v < 10) | (v > 45))
        others = [s for s in sids if s != sid]

        # Track reliability opinion over time
        # At each step: if agrees with consensus -> positive evidence
        # If disagrees -> negative evidence
        reliability_u = []  # uncertainty track
        agree_thresh = 2.0  # within 2C = agree

        # Sliding window approach: opinion from last W steps
        W = 50  # ~4 hours
        for t in range(n):
            if np.isnan(v[t]):
                reliability_u.append(np.nan)
                continue
            ovals = [raw_al[s][t] for s in others if not np.isnan(raw_al[s][t])]
            if len(ovals) < 1:
                reliability_u.append(np.nan)
                continue

            # Count agrees/disagrees in window [t-W, t]
            n_agree, n_disagree = 0, 0
            for w in range(max(0, t - W), t + 1):
                if np.isnan(v[w]):
                    continue
                wvals = [raw_al[s][w] for s in others if not np.isnan(raw_al[s][w])]
                if not wvals:
                    continue
                if abs(v[w] - np.median(wvals)) <= agree_thresh:
                    n_agree += 1
                else:
                    n_disagree += 1

            total = n_agree + n_disagree
            if total > 0:
                op = Opinion.from_evidence(
                    positive=n_agree, negative=n_disagree, base_rate=0.5)
                reliability_u.append(op.uncertainty)
            else:
                reliability_u.append(1.0)

        reliability_u = np.array(reliability_u)

        # How well does reliability_u predict outliers?
        valid_both = ~np.isnan(reliability_u) & valid
        if np.sum(valid_both) < 100:
            continue

        # AUROC: does high uncertainty predict outlier?
        from sklearn.metrics import roc_auc_score
        y_true = is_out[valid_both].astype(int)
        y_score = reliability_u[valid_both]
        if len(np.unique(y_true)) < 2:
            print(f"\n  Sensor {sid}: no outliers in valid_both range")
            continue
        auroc = roc_auc_score(y_true, y_score)
        print(f"\n  Sensor {sid}: SL reliability AUROC = {auroc:.3f}")

        # At what uncertainty threshold do we get good recall?
        for u_thresh in [0.3, 0.5, 0.7, 0.9]:
            flagged = reliability_u > u_thresh
            tp = int(np.sum(flagged & is_out))
            fp = int(np.sum(flagged & ~is_out & valid))
            n_out_s = int(np.sum(is_out))
            n_clean_s = int(np.sum(valid & ~is_out))
            recall = tp / n_out_s if n_out_s > 0 else 0
            fpr = fp / n_clean_s if n_clean_s > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
            print(f"    u > {u_thresh:.1f}: recall={recall:.1%} FPR={fpr:.1%} "
                  f"prec={prec:.1%} F1={f1:.3f}")


if __name__ == "__main__":
    main()
