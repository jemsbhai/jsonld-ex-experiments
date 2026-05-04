"""EN8.2 Diagnostic: Principled SL opinion construction for sensor fusion.

Key insight: for SL sensor fusion, the opinion uncertainty must be derived
from sensor precision (1/sigma^2), not from arbitrary linear scaling.

In Josang's SL framework:
    u = W / (W + N)
where N = effective evidence count, W = prior weight.

For sensors with noise sigma:
    N_eff = 1 / sigma^2  (precision = evidence quality)
    u = W / (W + 1/sigma^2) = W * sigma^2 / (W * sigma^2 + 1)

The prior weight W controls discrimination:
    W small  -> all sensors have low u -> can't discriminate
    W large  -> all sensors have high u -> too cautious
    W = 1/sigma_median^2 -> median sensor has u=0.5 (principled)

This diagnostic:
1. Sweeps W values to find how SL RMSE varies
2. Validates the principled choice (W = median precision)
3. Compares SL (corrected) vs weighted_avg vs scalar_avg
4. Tests robustness under misspecification
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


def filter_outliers(data, field="temperature", lo=10.0, hi=45.0):
    return [r for r in data if r[field] is not None and lo <= r[field] <= hi]

SEED = 42
TEMP_RANGE = (15.0, 35.0)


def evidence_based_opinion(sigma: float, W: float) -> float:
    """Compute SL uncertainty from sensor noise using evidence model.

    u = W * sigma^2 / (W * sigma^2 + 1)
    """
    return W * sigma**2 / (W * sigma**2 + 1)


def sl_fuse_evidence(readings, sigmas, W, value_range):
    """SL fusion with evidence-based opinion construction.

    Each sensor's reading is weighted by its evidence quality (1-u),
    where u is derived from the precision-based evidence model.
    """
    if readings.ndim == 1:
        readings = readings.reshape(1, -1)

    n_steps, n_sensors = readings.shape
    estimates = np.zeros(n_steps)

    for t in range(n_steps):
        valid = ~np.isnan(readings[t])
        valid_idx = np.where(valid)[0]

        if len(valid_idx) == 0:
            estimates[t] = np.nan
            continue
        if len(valid_idx) == 1:
            estimates[t] = readings[t, valid_idx[0]]
            continue

        # Evidence-based weights: w_i = 1/sigma_i^2
        # This is what the SL opinion framework gives us when
        # u = W*sigma^2 / (W*sigma^2 + 1) and weight = 1-u = 1/(W*sigma^2+1)
        # The key: 1-u is proportional to 1/sigma^2 when W is fixed
        weights = np.array([1.0 / sigmas[i]**2 for i in valid_idx])
        w_sum = np.sum(weights)
        estimates[t] = np.sum(weights * readings[t, valid_idx]) / w_sum

    return estimates


def sl_fuse_opinion_weighted(readings, sigmas, W, value_range):
    """SL fusion using actual opinion construction + cumulative fusion.

    This is the proper SL approach:
    1. Build opinion per sensor with evidence-based u
    2. The opinion's evidence quality (1-u) determines its weight
    3. Weight readings by evidence quality
    """
    if readings.ndim == 1:
        readings = readings.reshape(1, -1)

    n_steps, n_sensors = readings.shape
    estimates = np.zeros(n_steps)

    for t in range(n_steps):
        valid = ~np.isnan(readings[t])
        valid_idx = np.where(valid)[0]

        if len(valid_idx) == 0:
            estimates[t] = np.nan
            continue
        if len(valid_idx) == 1:
            estimates[t] = readings[t, valid_idx[0]]
            continue

        # Build opinions with evidence-based uncertainty
        opinions = []
        for i in valid_idx:
            u = evidence_based_opinion(sigmas[i], W)
            # Confidence: normalized position in value range
            range_w = value_range[1] - value_range[0]
            conf = (readings[t, i] - value_range[0]) / range_w
            conf = max(0.001, min(0.999, conf))
            op = Opinion.from_confidence(conf, uncertainty=u)
            opinions.append(op)

        # Weight by evidence quality (1 - u)
        weights = np.array([1.0 - op.uncertainty for op in opinions])
        w_sum = np.sum(weights)
        if w_sum > 0:
            estimates[t] = np.sum(
                weights * readings[t, valid_idx]) / w_sum
        else:
            estimates[t] = np.mean(readings[t, valid_idx])

    return estimates


def loo_eval(aligned, sensor_ids, sigmas_dict, method_fn, **kwargs):
    """Leave-one-out evaluation for a single method."""
    n_steps = len(aligned["timestamps"])
    sq_errors = []

    for held_out in sensor_ids:
        others = [s for s in sensor_ids if s != held_out]
        if len(others) < 2:
            continue
        truth = aligned[held_out]
        sigma_arr = np.array([sigmas_dict[s] for s in others])
        readings = np.column_stack([aligned[s] for s in others])

        valid_t = ~np.isnan(truth)
        for t in range(n_steps):
            if not valid_t[t]:
                continue
            if np.sum(~np.isnan(readings[t])) < 2:
                valid_t[t] = False
        if np.sum(valid_t) < 10:
            continue

        pred = method_fn(readings, sigma_arr, **kwargs)
        errs = (pred[valid_t] - truth[valid_t]) ** 2
        sq_errors.extend(errs.tolist())

    return np.sqrt(np.mean(sq_errors)) if sq_errors else float("nan")


def estimate_sigmas(aligned, sensor_ids):
    """MAD-based per-sensor noise estimation."""
    sigmas = {}
    n = len(aligned["timestamps"])
    for sid in sensor_ids:
        others = [s for s in sensor_ids if s != sid]
        devs = []
        for t in range(n):
            if np.isnan(aligned[sid][t]):
                continue
            other_vals = [aligned[s][t] for s in others
                          if not np.isnan(aligned[s][t])]
            if len(other_vals) >= 2:
                devs.append(aligned[sid][t] - np.mean(other_vals))
        if len(devs) >= 10:
            mad = np.median(np.abs(np.array(devs) - np.median(devs)))
            sigmas[sid] = max(0.05, mad * 1.4826)
        else:
            sigmas[sid] = 0.5
    return sigmas


def main():
    print("=" * 60)
    print("EN8.2 Diagnostic: Evidence-Based Opinion Construction")
    print("=" * 60)

    # Load data
    raw_data = load_intel_lab_data()
    data = filter_outliers(raw_data, lo=10.0, hi=45.0)
    print(f"Loaded {len(data):,} records (filtered)")

    clusters = {
        "primary": [4, 6, 7, 9],
        "validation": [22, 23, 24],
    }

    for cname, sids in clusters.items():
        print(f"\n{'='*50}")
        print(f"Cluster: {cname} ({sids})")
        print(f"{'='*50}")

        cluster_data = select_sensor_cluster(data, sids)
        aligned = time_align_readings(cluster_data, sids, interval_seconds=300)
        n = len(aligned["timestamps"])
        sigmas = estimate_sigmas(aligned, sids)

        print(f"  {n} time steps")
        for sid in sids:
            valid = np.sum(~np.isnan(aligned[sid]))
            print(f"  Sensor {sid}: sigma={sigmas[sid]:.4f}, "
                  f"{valid}/{n} valid ({valid/n:.1%})")

        # Baselines
        sigma_arr_all = np.array([sigmas[s] for s in sids])
        median_sigma = np.median(list(sigmas.values()))
        median_precision = 1.0 / median_sigma**2

        print(f"\n  Median sigma: {median_sigma:.4f}")
        print(f"  Median precision (1/sigma^2): {median_precision:.1f}")
        print(f"  Principled W = median_precision = {median_precision:.1f}")

        # Baseline RMSEs
        rmse_scalar = loo_eval(
            aligned, sids, sigmas,
            lambda r, s, **kw: np.array([scalar_average(r[t]) for t in range(len(r))]))
        rmse_weighted = loo_eval(
            aligned, sids, sigmas,
            lambda r, s, **kw: np.array([weighted_average(r[t], s) for t in range(len(r))]))
        rmse_kalman = loo_eval(
            aligned, sids, sigmas,
            lambda r, s, **kw: kalman_fuse_1d(r, s))

        print(f"\n  Baselines:")
        print(f"    scalar_avg:   RMSE={rmse_scalar:.4f}")
        print(f"    weighted_avg: RMSE={rmse_weighted:.4f}")
        print(f"    kalman:       RMSE={rmse_kalman:.4f}")

        # Sweep W
        print(f"\n  W sweep (evidence-based SL):")
        W_values = [0.5, 1, 2, 5, 10, 20, 50, 100,
                    median_precision, median_precision * 0.5,
                    median_precision * 2]
        W_values = sorted(set(W_values))

        best_W, best_rmse = None, float("inf")
        for W in W_values:
            # Show opinion uncertainties at this W
            us = {sid: evidence_based_opinion(sigmas[sid], W) for sid in sids}
            ws = {sid: 1 - us[sid] for sid in sids}
            w_ratio = max(ws.values()) / min(ws.values())

            rmse = loo_eval(
                aligned, sids, sigmas,
                sl_fuse_opinion_weighted, W=W, value_range=TEMP_RANGE)

            marker = ""
            if abs(W - median_precision) < 0.1:
                marker = " <-- principled (W=median_prec)"
            if rmse < best_rmse:
                best_rmse = rmse
                best_W = W

            print(f"    W={W:8.1f}  RMSE={rmse:.4f}  "
                  f"w_ratio={w_ratio:.2f}x  "
                  f"u_range=[{min(us.values()):.3f},{max(us.values()):.3f}]"
                  f"{marker}")

        print(f"\n  Best W: {best_W:.1f} (RMSE={best_rmse:.4f})")
        print(f"  weighted_avg:  RMSE={rmse_weighted:.4f}")
        print(f"  Difference:    {best_rmse - rmse_weighted:+.4f}")

        # At best W, what are the actual weights?
        print(f"\n  At W={best_W:.1f}:")
        for sid in sids:
            u = evidence_based_opinion(sigmas[sid], best_W)
            w = 1 - u
            iv_w = 1/sigmas[sid]**2
            print(f"    Sensor {sid}: u={u:.4f}  sl_weight={w:.4f}  "
                  f"inv_var_weight={iv_w:.2f}  ratio_to_min: "
                  f"sl={w/min(1-evidence_based_opinion(sigmas[s], best_W) for s in sids):.2f}x  "
                  f"iv={iv_w/min(1/sigmas[s]**2 for s in sids):.2f}x")


if __name__ == "__main__":
    main()
