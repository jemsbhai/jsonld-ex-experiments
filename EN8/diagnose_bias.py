"""EN8.2 Diagnostic: Bias structure, de-biased fusion, and conflict detection.

Three questions:
1. How large are sensor biases relative to noise? Does bias dominate?
2. After de-biasing, which fusion method wins?
3. Can SL conflict_metric detect the biased sensors?
"""
import sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "python" / "src"))

from en8_2_core import (
    load_intel_lab_data, select_sensor_cluster, time_align_readings,
    scalar_average, weighted_average, kalman_fuse_1d,
    compute_rmse,
)
from jsonld_ex.confidence_algebra import Opinion, cumulative_fuse
from jsonld_ex.confidence_decay import decay_opinion
from EN4.en4_2_ds_comparison import dempster_combine, compute_conflict_K


def filter_outliers(data, lo=10.0, hi=45.0):
    return [r for r in data if r["temperature"] is not None
            and lo <= r["temperature"] <= hi]


def estimate_bias_and_sigma(aligned, sensor_ids):
    """Estimate per-sensor bias AND noise separately.

    Bias: median deviation from consensus (robust)
    Sigma: MAD of deviations AFTER removing bias (pure noise)
    """
    stats = {}
    n = len(aligned["timestamps"])
    for sid in sensor_ids:
        others = [s for s in sensor_ids if s != sid]
        deviations = []
        for t in range(n):
            if np.isnan(aligned[sid][t]):
                continue
            other_vals = [aligned[s][t] for s in others
                          if not np.isnan(aligned[s][t])]
            if len(other_vals) >= 2:
                consensus = np.mean(other_vals)
                deviations.append(aligned[sid][t] - consensus)
        if len(deviations) >= 10:
            dev = np.array(deviations)
            bias = np.median(dev)
            centered = dev - bias
            mad = np.median(np.abs(centered))
            sigma = max(0.05, mad * 1.4826)
            stats[sid] = {"bias": float(bias), "sigma": float(sigma),
                          "n": len(deviations),
                          "bias_to_sigma": abs(float(bias)) / float(sigma)}
        else:
            stats[sid] = {"bias": 0.0, "sigma": 0.5, "n": 0,
                          "bias_to_sigma": 0.0}
    return stats


def debias_aligned(aligned, sensor_ids, biases):
    """Create de-biased copy of aligned data."""
    result = {"timestamps": aligned["timestamps"].copy()}
    for sid in sensor_ids:
        result[sid] = aligned[sid] - biases.get(sid, 0.0)
    return result


def evidence_opinion(sigma, W):
    """Compute SL uncertainty from evidence model."""
    return W * sigma**2 / (W * sigma**2 + 1)


def sl_fuse_evidence(readings, sigmas, W):
    """SL evidence-based fusion."""
    if readings.ndim == 1:
        readings = readings.reshape(1, -1)
    n_steps, n_sensors = readings.shape
    estimates = np.zeros(n_steps)
    for t in range(n_steps):
        valid = ~np.isnan(readings[t])
        idx = np.where(valid)[0]
        if len(idx) == 0:
            estimates[t] = np.nan
            continue
        if len(idx) == 1:
            estimates[t] = readings[t, idx[0]]
            continue
        weights = np.array([1.0 - evidence_opinion(sigmas[i], W) for i in idx])
        w_sum = np.sum(weights)
        if w_sum > 0:
            estimates[t] = np.sum(weights * readings[t, idx]) / w_sum
        else:
            estimates[t] = np.mean(readings[t, idx])
    return estimates


def ds_fuse_evidence(readings, sigmas, W):
    """DS evidence-based fusion."""
    if readings.ndim == 1:
        readings = readings.reshape(1, -1)
    n_steps, n_sensors = readings.shape
    estimates = np.zeros(n_steps)
    for t in range(n_steps):
        valid = ~np.isnan(readings[t])
        idx = np.where(valid)[0]
        if len(idx) == 0:
            estimates[t] = np.nan
            continue
        if len(idx) == 1:
            estimates[t] = readings[t, idx[0]]
            continue
        masses = []
        for i in idx:
            u = evidence_opinion(sigmas[i], W)
            b = (1 - u) * 0.5
            d = (1 - u) * 0.5
            masses.append({"b": b, "d": d, "u": u})
        fused = masses[0].copy()
        for m in masses[1:]:
            K = compute_conflict_K(fused, m)
            if K >= 0.999:
                continue
            fused = dempster_combine(fused, m)
        weights = np.array([m["b"] + 0.5 * m["u"] for m in masses])
        w_sum = np.sum(weights)
        if w_sum > 0:
            estimates[t] = np.sum(weights * readings[t, idx]) / w_sum
        else:
            estimates[t] = np.mean(readings[t, idx])
    return estimates


def sl_fuse_with_decay_gaps(readings, sigmas, W, gap_mask, half_life=2.0):
    """SL fusion with temporal decay on gapped sensors.

    gap_mask: (n_steps, n_sensors) float array of hours since last fresh reading.
              0 = fresh, >0 = stale.
    """
    if readings.ndim == 1:
        readings = readings.reshape(1, -1)
    n_steps, n_sensors = readings.shape
    estimates = np.zeros(n_steps)
    for t in range(n_steps):
        valid = ~np.isnan(readings[t])
        idx = np.where(valid)[0]
        if len(idx) == 0:
            estimates[t] = np.nan
            continue
        if len(idx) == 1:
            estimates[t] = readings[t, idx[0]]
            continue
        opinions = []
        for i in idx:
            u = evidence_opinion(sigmas[i], W)
            op = Opinion(belief=(1-u)*0.5, disbelief=(1-u)*0.5,
                         uncertainty=u, base_rate=0.5)
            stale = gap_mask[t, i] if gap_mask is not None else 0
            if stale > 0:
                op = decay_opinion(op, elapsed=stale, half_life=half_life)
            opinions.append(op)
        weights = np.array([1.0 - op.uncertainty for op in opinions])
        w_sum = np.sum(weights)
        if w_sum > 0:
            estimates[t] = np.sum(weights * readings[t, idx]) / w_sum
        else:
            estimates[t] = np.mean(readings[t, idx])
    return estimates


def loo_eval_method(aligned, sensor_ids, sigmas_dict, method_fn):
    """LOO evaluation returning (RMSE, raw_squared_errors)."""
    n_steps = len(aligned["timestamps"])
    sq_errs = []
    for held_out in sensor_ids:
        others = [s for s in sensor_ids if s != held_out]
        if len(others) < 2:
            continue
        truth = aligned[held_out]
        sigma_arr = np.array([sigmas_dict[s] for s in others])
        readings = np.column_stack([aligned[s] for s in others])
        valid_t = ~np.isnan(truth)
        for t in range(n_steps):
            if valid_t[t] and np.sum(~np.isnan(readings[t])) < 2:
                valid_t[t] = False
        if np.sum(valid_t) < 10:
            continue
        pred = method_fn(readings, sigma_arr)
        errs = (pred[valid_t] - truth[valid_t]) ** 2
        sq_errs.extend(errs.tolist())
    arr = np.array(sq_errs)
    return float(np.sqrt(np.mean(arr))), arr


def loo_with_decay(aligned, sensor_ids, sigmas_dict, W, half_life, gap_prob=0.1):
    """LOO with artificial gap injection for decay testing.

    Randomly drops sensor readings and fills them with the last known value
    (simulating stale data). Tests whether decay helps downweight stale readings.
    """
    rng = np.random.default_rng(42)
    n_steps = len(aligned["timestamps"])
    interval_hours = 5.0 / 60.0  # 5 min intervals

    sq_errs_no_decay = []
    sq_errs_decay = []

    for held_out in sensor_ids:
        others = [s for s in sensor_ids if s != held_out]
        if len(others) < 2:
            continue
        truth = aligned[held_out]
        sigma_arr = np.array([sigmas_dict[s] for s in others])
        readings_orig = np.column_stack([aligned[s] for s in others])

        valid_t = ~np.isnan(truth)
        for t in range(n_steps):
            if valid_t[t] and np.sum(~np.isnan(readings_orig[t])) < 2:
                valid_t[t] = False
        if np.sum(valid_t) < 10:
            continue

        # Inject gaps: for each sensor, randomly drop readings and carry forward
        readings = readings_orig.copy()
        gap_mask = np.zeros_like(readings)  # hours since last fresh
        for s_idx in range(len(others)):
            in_gap = False
            last_val = np.nan
            hours_stale = 0.0
            for t in range(n_steps):
                if np.isnan(readings_orig[t, s_idx]):
                    continue
                if not in_gap and rng.random() < gap_prob:
                    in_gap = True
                    hours_stale = 0.0
                if in_gap:
                    gap_duration = rng.geometric(0.3)  # avg ~3 steps
                    if hours_stale > gap_duration * interval_hours:
                        in_gap = False
                    else:
                        if not np.isnan(last_val):
                            readings[t, s_idx] = last_val  # carry forward
                            hours_stale += interval_hours
                            gap_mask[t, s_idx] = hours_stale
                        else:
                            readings[t, s_idx] = np.nan
                if not in_gap or np.isnan(gap_mask[t, s_idx]):
                    last_val = readings_orig[t, s_idx]
                    gap_mask[t, s_idx] = 0.0

        # SL without decay
        pred_no = sl_fuse_evidence(readings, sigma_arr, W)
        sq_errs_no_decay.extend(
            ((pred_no[valid_t] - truth[valid_t]) ** 2).tolist())

        # SL with decay
        pred_d = sl_fuse_with_decay_gaps(readings, sigma_arr, W,
                                          gap_mask, half_life=half_life)
        sq_errs_decay.extend(
            ((pred_d[valid_t] - truth[valid_t]) ** 2).tolist())

    no_d = np.array(sq_errs_no_decay)
    d = np.array(sq_errs_decay)
    return float(np.sqrt(np.mean(no_d))), float(np.sqrt(np.mean(d)))


def conflict_detection_analysis(aligned, sensor_ids, stats, W):
    """Can SL conflict_metric detect biased sensors?"""
    n = len(aligned["timestamps"])
    results = {}

    for target in sensor_ids:
        others = [s for s in sensor_ids if s != target]
        conflicts = []

        for t in range(min(n, 3000)):  # sample for speed
            if np.isnan(aligned[target][t]):
                continue
            vals = {}
            for s in [target] + others:
                if not np.isnan(aligned[s][t]):
                    vals[s] = aligned[s][t]
            if len(vals) < 3:
                continue

            # Build opinions for target vs each other sensor
            target_u = evidence_opinion(stats[target]["sigma"], W)
            target_conf = 0.5  # neutral
            target_op = Opinion(belief=(1-target_u)*0.5,
                                disbelief=(1-target_u)*0.5,
                                uncertainty=target_u, base_rate=0.5)

            pair_conflicts = []
            for other in others:
                if other not in vals:
                    continue
                other_u = evidence_opinion(stats[other]["sigma"], W)
                other_op = Opinion(belief=(1-other_u)*0.5,
                                   disbelief=(1-other_u)*0.5,
                                   uncertainty=other_u, base_rate=0.5)
                # Disagreement: absolute difference in readings
                disagree = abs(vals[target] - vals[other])
                pair_conflicts.append(disagree)

            if pair_conflicts:
                conflicts.append(np.mean(pair_conflicts))

        if conflicts:
            results[target] = {
                "mean_disagreement": float(np.mean(conflicts)),
                "median_disagreement": float(np.median(conflicts)),
                "p95_disagreement": float(np.percentile(conflicts, 95)),
                "bias": stats[target]["bias"],
                "sigma": stats[target]["sigma"],
                "bias_to_sigma": stats[target]["bias_to_sigma"],
            }
    return results


def main():
    print("=" * 60)
    print("EN8.2: Bias Structure, De-biased Fusion, Conflict Detection")
    print("=" * 60)

    raw = load_intel_lab_data()
    data = filter_outliers(raw)
    print(f"Loaded {len(data):,} records")

    clusters = {
        "primary": [4, 6, 7, 9],
        "validation": [22, 23, 24],
    }

    for cname, sids in clusters.items():
        print(f"\n{'='*55}")
        print(f"Cluster: {cname} ({sids})")
        print(f"{'='*55}")

        cdata = select_sensor_cluster(data, sids)
        aligned = time_align_readings(cdata, sids, interval_seconds=300)
        n = len(aligned["timestamps"])

        # ── Question 1: Bias structure ────────────────────────────
        print(f"\n  Q1: Bias vs Noise Structure")
        stats = estimate_bias_and_sigma(aligned, sids)
        for sid in sids:
            s = stats[sid]
            print(f"    Sensor {sid}: bias={s['bias']:+.3f}  sigma={s['sigma']:.3f}  "
                  f"bias/sigma={s['bias_to_sigma']:.1f}x  n={s['n']}")

        max_bias_ratio = max(s["bias_to_sigma"] for s in stats.values())
        print(f"    Max |bias|/sigma: {max_bias_ratio:.1f}x")
        if max_bias_ratio > 2:
            print(f"    WARNING: Bias dominates noise. Equal-weight averaging")
            print(f"    will win because it averages out biases. Precision-based")
            print(f"    methods will overweight precise-but-biased sensors.")

        # ── Question 2: Biased vs de-biased comparison ────────────
        print(f"\n  Q2: Fusion Comparison (biased vs de-biased)")
        biases = {sid: stats[sid]["bias"] for sid in sids}
        sigmas = {sid: stats[sid]["sigma"] for sid in sids}
        aligned_db = debias_aligned(aligned, sids, biases)

        # Choose W from median sigma
        med_sig = np.median([stats[s]["sigma"] for s in sids])
        W = 1.0 / med_sig**2
        print(f"    W = 1/sigma_median^2 = {W:.1f}")

        methods = {
            "scalar_avg": lambda r, s: np.array(
                [scalar_average(r[t]) for t in range(len(r))]),
            "weighted_avg": lambda r, s: np.array(
                [weighted_average(r[t], s) for t in range(len(r))]),
            "kalman": lambda r, s: kalman_fuse_1d(r, s),
            "sl_evidence": lambda r, s: sl_fuse_evidence(r, s, W),
            "ds_evidence": lambda r, s: ds_fuse_evidence(r, s, W),
        }

        print(f"\n    {'Method':20s}  {'BIASED':>10s}  {'DE-BIASED':>10s}  {'Improvement':>12s}")
        print(f"    {'─'*20}  {'─'*10}  {'─'*10}  {'─'*12}")
        for mname, mfn in methods.items():
            rmse_b, _ = loo_eval_method(aligned, sids, sigmas, mfn)
            rmse_db, _ = loo_eval_method(aligned_db, sids, sigmas, mfn)
            improv = (rmse_b - rmse_db) / rmse_b * 100
            print(f"    {mname:20s}  {rmse_b:10.4f}  {rmse_db:10.4f}  {improv:+10.1f}%")

        # ── W sweep on de-biased data ────────────────────────────
        print(f"\n  De-biased W sweep:")
        W_vals = [0.5, 1, 2, 5, 10, 20, W, 50, 100]
        W_vals = sorted(set(round(w, 1) for w in W_vals))
        best_W, best_rmse = 0.5, 999
        for w in W_vals:
            rmse, _ = loo_eval_method(
                aligned_db, sids, sigmas,
                lambda r, s, _w=w: sl_fuse_evidence(r, s, _w))
            marker = " <-- principled" if abs(w - W) < 0.5 else ""
            if rmse < best_rmse:
                best_rmse = rmse
                best_W = w
            print(f"    W={w:8.1f}  RMSE={rmse:.4f}{marker}")
        print(f"    Best W={best_W:.1f} (RMSE={best_rmse:.4f})")

        # ── Question 3: Decay with artificial gaps ────────────────
        print(f"\n  Q3: Temporal Decay (artificial gap injection, p=0.1)")
        for hl in [0.5, 1.0, 2.0, 4.0]:
            rmse_no, rmse_d = loo_with_decay(
                aligned_db, sids, sigmas, W=best_W, half_life=hl, gap_prob=0.1)
            delta = rmse_d - rmse_no
            print(f"    half_life={hl:.1f}h: no_decay={rmse_no:.4f}  "
                  f"decay={rmse_d:.4f}  delta={delta:+.4f}  "
                  f"{'DECAY HELPS' if delta < -0.001 else 'no effect' if abs(delta) < 0.001 else 'DECAY HURTS'}")

        # ── Question 4: Conflict detection of bias ────────────────
        print(f"\n  Q4: Can disagreement detect biased sensors?")
        conflict_results = conflict_detection_analysis(aligned, sids, stats, W)
        print(f"    {'Sensor':>8s}  {'Bias':>8s}  {'Sigma':>8s}  "
              f"{'|Bias|/σ':>8s}  {'Mean Disagr':>12s}  {'P95 Disagr':>12s}")
        for sid in sids:
            r = conflict_results.get(sid, {})
            s = stats[sid]
            print(f"    {sid:8d}  {s['bias']:+8.3f}  {s['sigma']:8.3f}  "
                  f"{s['bias_to_sigma']:8.1f}  "
                  f"{r.get('mean_disagreement', 0):12.3f}  "
                  f"{r.get('p95_disagreement', 0):12.3f}")

        # Correlation between |bias| and mean disagreement
        biases_abs = [abs(stats[s]["bias"]) for s in sids]
        disagrs = [conflict_results.get(s, {}).get("mean_disagreement", 0)
                   for s in sids]
        if len(sids) >= 3:
            corr = np.corrcoef(biases_abs, disagrs)[0, 1]
            print(f"    Correlation(|bias|, disagreement): r={corr:.3f}")
            if corr > 0.7:
                print(f"    --> Disagreement IS a bias detector")
            elif corr > 0.3:
                print(f"    --> Weak signal")
            else:
                print(f"    --> NOT a reliable bias detector")


if __name__ == "__main__":
    main()
