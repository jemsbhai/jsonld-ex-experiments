"""Diagnose WHY scalar_avg keeps winning over precision-based methods.

Hypotheses:
H1: Sigma estimates are wrong (consensus-deviation doesn't capture true noise)
H2: Noise is temporally correlated (violates independence assumption)
H3: Noise is non-Gaussian (heavy tails from residual outliers)
H4: Sensors measure slightly different microenvironments (irreducible)

Also check: is ds_evidence identical to scalar_avg due to a bug?
"""
import sys, numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "python" / "src"))

from en8_2_core import (
    load_intel_lab_data, select_sensor_cluster, time_align_readings,
    scalar_average, weighted_average,
)


def filter_outliers(data, lo=10.0, hi=45.0):
    return [r for r in data if r["temperature"] is not None
            and lo <= r["temperature"] <= hi]


def main():
    raw = load_intel_lab_data()
    data = filter_outliers(raw)

    sids = [4, 6, 7, 9]
    cdata = select_sensor_cluster(data, sids)
    aligned = time_align_readings(cdata, sids, interval_seconds=300)
    n = len(aligned["timestamps"])

    # De-bias first
    biases = {}
    sigmas = {}
    for sid in sids:
        others = [s for s in sids if s != sid]
        devs = []
        for t in range(n):
            if np.isnan(aligned[sid][t]):
                continue
            ovals = [aligned[s][t] for s in others if not np.isnan(aligned[s][t])]
            if len(ovals) >= 2:
                devs.append(aligned[sid][t] - np.mean(ovals))
        dev = np.array(devs)
        biases[sid] = np.median(dev)
        centered = dev - biases[sid]
        sigmas[sid] = max(0.05, np.median(np.abs(centered)) * 1.4826)

    aligned_db = {sid: aligned[sid] - biases[sid] for sid in sids}
    aligned_db["timestamps"] = aligned["timestamps"]

    print("=== H1: Are sigma estimates accurate? ===")
    print("Cross-validate: estimate sigma from sensor pairs, not consensus")
    for sid in sids:
        pair_sigmas = []
        for other in [s for s in sids if s != sid]:
            diffs = []
            for t in range(n):
                if not np.isnan(aligned_db[sid][t]) and not np.isnan(aligned_db[other][t]):
                    diffs.append(aligned_db[sid][t] - aligned_db[other][t])
            if len(diffs) > 100:
                diff_arr = np.array(diffs)
                # var(A-B) = var(A) + var(B) if independent
                # So sigma_pair = std(A-B) / sqrt(2) if equal sigmas
                # Or use MAD-based estimate
                mad = np.median(np.abs(diff_arr - np.median(diff_arr))) * 1.4826
                pair_sigmas.append(mad)
        if pair_sigmas:
            # sigma_A = sqrt(var(A-B) - var(B)) but we don't know var(B)
            # With 3 others, we can triangulate
            avg_pair_sigma = np.mean(pair_sigmas)
            print(f"  Sensor {sid}: consensus_sigma={sigmas[sid]:.4f}  "
                  f"mean_pairwise_mad={avg_pair_sigma:.4f}  "
                  f"ratio={avg_pair_sigma/sigmas[sid]:.2f}x")

    print("\n=== H2: Is noise temporally correlated? ===")
    for sid in sids:
        residuals = []
        for t in range(n):
            if np.isnan(aligned_db[sid][t]):
                continue
            ovals = [aligned_db[s][t] for s in sids
                     if s != sid and not np.isnan(aligned_db[s][t])]
            if len(ovals) >= 2:
                residuals.append(aligned_db[sid][t] - np.mean(ovals))
        res = np.array(residuals)
        # Autocorrelation at lag 1
        if len(res) > 10:
            ac1 = np.corrcoef(res[:-1], res[1:])[0, 1]
            ac5 = np.corrcoef(res[:-5], res[5:])[0, 1] if len(res) > 10 else 0
            print(f"  Sensor {sid}: autocorr(lag=1)={ac1:.3f}  "
                  f"autocorr(lag=5)={ac5:.3f}  "
                  f"{'CORRELATED' if ac1 > 0.3 else 'OK'}")

    print("\n=== H3: Is noise Gaussian? ===")
    for sid in sids:
        residuals = []
        for t in range(n):
            if np.isnan(aligned_db[sid][t]):
                continue
            ovals = [aligned_db[s][t] for s in sids
                     if s != sid and not np.isnan(aligned_db[s][t])]
            if len(ovals) >= 2:
                residuals.append(aligned_db[sid][t] - np.mean(ovals))
        res = np.array(residuals)
        if len(res) > 100:
            kurtosis = scipy_stats.kurtosis(res)
            skew = scipy_stats.skew(res)
            # Fraction beyond 3-sigma
            sigma = np.std(res)
            beyond_3sig = np.mean(np.abs(res) > 3*sigma) * 100
            # Expected for Gaussian: 0.27%
            print(f"  Sensor {sid}: kurtosis={kurtosis:.2f}  skew={skew:.2f}  "
                  f"beyond_3sigma={beyond_3sig:.2f}% (Gaussian: 0.27%)  "
                  f"{'HEAVY TAILS' if kurtosis > 3 else 'OK'}")

    print("\n=== H4: Time-varying bias (drift)? ===")
    # Split time series into 10 chunks, compute bias per chunk
    for sid in sids:
        chunk_biases = []
        chunk_size = n // 10
        for c in range(10):
            start, end = c * chunk_size, (c+1) * chunk_size
            devs = []
            for t in range(start, min(end, n)):
                if np.isnan(aligned[sid][t]):
                    continue
                ovals = [aligned[s][t] for s in sids
                         if s != sid and not np.isnan(aligned[s][t])]
                if len(ovals) >= 2:
                    devs.append(aligned[sid][t] - np.mean(ovals))
            if devs:
                chunk_biases.append(np.median(devs))
        if chunk_biases:
            bias_range = max(chunk_biases) - min(chunk_biases)
            print(f"  Sensor {sid}: bias_range_across_chunks={bias_range:.3f}  "
                  f"global_sigma={sigmas[sid]:.3f}  "
                  f"drift/sigma={bias_range/sigmas[sid]:.1f}x  "
                  f"{'DRIFTING' if bias_range > sigmas[sid] else 'STABLE'}")

    print("\n=== Summary: What's breaking precision-based fusion? ===")
    print("If H2 (correlation) is strong: independence assumption violated,")
    print("  inverse-variance weights are wrong.")
    print("If H3 (heavy tails): outlier-robust estimation helps but Gaussian")
    print("  model is wrong, making precise-looking sensors unreliable.")
    print("If H4 (drift): static bias removal is insufficient, need adaptive.")


if __name__ == "__main__":
    main()
