"""Validate SL belief quality detection on validation cluster + bootstrap CIs."""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "python" / "src"))

from en8_2_core import (load_intel_lab_data, select_sensor_cluster,
                         time_align_readings)
from jsonld_ex.confidence_algebra import Opinion
from sklearn.metrics import roc_auc_score


def sl_belief_quality(raw_al, sid, sids, W=50, agree_thresh=2.0):
    """Compute SL belief-based quality score for a sensor."""
    v = raw_al[sid]
    n = len(v)
    others = [s for s in sids if s != sid]
    scores = np.full(n, np.nan)  # 1 - projected_probability

    for t in range(n):
        if np.isnan(v[t]):
            continue
        ovals = [raw_al[s][t] for s in others if not np.isnan(raw_al[s][t])]
        if not ovals:
            continue
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
            op = Opinion.from_evidence(positive=n_agree, negative=n_disagree,
                                        base_rate=0.5)
            scores[t] = 1 - op.projected_probability()
    return scores


def consensus_quality(raw_al, sid, sids):
    """Pointwise consensus deviation score."""
    v = raw_al[sid]
    n = len(v)
    others = [s for s in sids if s != sid]
    scores = np.full(n, 0.0)
    for t in range(n):
        if np.isnan(v[t]):
            scores[t] = np.nan
            continue
        ovals = [raw_al[s][t] for s in others if not np.isnan(raw_al[s][t])]
        if ovals:
            scores[t] = abs(v[t] - np.median(ovals))
        else:
            scores[t] = np.nan
    return scores


def bootstrap_auroc_ci(y_true, y_score, n_boot=2000, seed=42):
    """Bootstrap CI on AUROC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(roc_auc_score(yt, ys))
    aurocs = np.array(aurocs)
    return float(np.percentile(aurocs, 2.5)), float(np.mean(aurocs)), \
           float(np.percentile(aurocs, 97.5))


def bootstrap_auroc_diff_ci(y_true, score_a, score_b, n_boot=2000, seed=42):
    """Bootstrap CI on AUROC difference (A - B)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        a = roc_auc_score(yt, score_a[idx])
        b = roc_auc_score(yt, score_b[idx])
        diffs.append(a - b)
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.mean(diffs)), \
           float(np.percentile(diffs, 97.5))


def evaluate_cluster(raw_al, sids, cluster_name, W=50):
    """Full evaluation of SL belief vs consensus on a cluster."""
    n = len(raw_al["timestamps"])
    print(f"\n{'='*55}")
    print(f"Cluster: {cluster_name} ({sids}), W={W}")
    print(f"{'='*55}")

    pooled_yt, pooled_sl, pooled_cd = [], [], []

    for sid in sids:
        v = raw_al[sid]
        valid = ~np.isnan(v)
        is_out = valid & ((v < 10) | (v > 45))
        n_out = int(np.sum(is_out))
        if n_out < 10:
            print(f"  Sensor {sid}: {n_out} outliers (skipped)")
            continue

        sl_scores = sl_belief_quality(raw_al, sid, sids, W=W)
        cd_scores = consensus_quality(raw_al, sid, sids)

        valid_both = ~np.isnan(sl_scores) & ~np.isnan(cd_scores) & valid
        y_true = is_out[valid_both].astype(int)
        sl_s = sl_scores[valid_both]
        cd_s = cd_scores[valid_both]

        if len(np.unique(y_true)) < 2:
            print(f"  Sensor {sid}: no class variation (skipped)")
            continue

        auroc_sl = roc_auc_score(y_true, sl_s)
        auroc_cd = roc_auc_score(y_true, cd_s)

        # Bootstrap CI on difference
        lo, md, hi = bootstrap_auroc_diff_ci(y_true, sl_s, cd_s, n_boot=2000)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"

        # Best F1 for each
        best_f1_sl, best_f1_cd = 0, 0
        for pct in range(50, 100):
            for scores, label in [(sl_s, "sl"), (cd_s, "cd")]:
                thresh = np.percentile(scores, pct)
                flagged = scores > thresh
                tp = np.sum(flagged & (y_true == 1))
                fp = np.sum(flagged & (y_true == 0))
                fn = np.sum(~flagged & (y_true == 1))
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
                if label == "sl" and f1 > best_f1_sl:
                    best_f1_sl = f1
                elif label == "cd" and f1 > best_f1_cd:
                    best_f1_cd = f1

        print(f"  Sensor {sid:2d}: outliers={n_out:5d}  "
              f"AUROC sl={auroc_sl:.3f} cd={auroc_cd:.3f}  "
              f"diff={md:+.3f} CI[{lo:+.3f},{hi:+.3f}] {sig}  "
              f"F1 sl={best_f1_sl:.3f} cd={best_f1_cd:.3f}")

        pooled_yt.extend(y_true.tolist())
        pooled_sl.extend(sl_s.tolist())
        pooled_cd.extend(cd_s.tolist())

    # Pooled analysis
    if pooled_yt:
        yt = np.array(pooled_yt)
        sl = np.array(pooled_sl)
        cd = np.array(pooled_cd)
        auroc_sl = roc_auc_score(yt, sl)
        auroc_cd = roc_auc_score(yt, cd)
        lo, md, hi = bootstrap_auroc_diff_ci(yt, sl, cd, n_boot=2000)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        print(f"\n  POOLED: AUROC sl={auroc_sl:.3f} cd={auroc_cd:.3f}  "
              f"diff={md:+.3f} CI[{lo:+.3f},{hi:+.3f}] {sig}")
        print(f"  n={len(yt):,} ({np.sum(yt):,} outliers, {np.sum(1-yt):,} clean)")


def main():
    raw = load_intel_lab_data()
    clusters = {
        "primary": [4, 6, 7, 9],
        "validation": [22, 23, 24, 25, 26],
    }

    for cname, sids in clusters.items():
        raw_c = select_sensor_cluster(raw, sids)
        raw_al = time_align_readings(raw_c, sids, interval_seconds=300)
        evaluate_cluster(raw_al, sids, cname, W=50)


if __name__ == "__main__":
    main()
