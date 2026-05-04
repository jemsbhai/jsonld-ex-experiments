"""Fix: use BELIEF not UNCERTAINTY for SL reliability tracking.

Root cause: Opinion.from_evidence(pos=5, neg=45) gives:
  u = W/(W+N) = 2/52 = 0.038  (low — lots of evidence)
  b = 5/50 * (1-u) = 0.096    (low — evidence says unreliable)
  d = 45/50 * (1-u) = 0.866   (high — strong disbelief)

Uncertainty says "how much evidence?" — always low when W=50
Belief says "is this sensor reliable?" — exactly what we want
"""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "python" / "src"))

from en8_2_core import (load_intel_lab_data, select_sensor_cluster,
                         time_align_readings)
from jsonld_ex.confidence_algebra import Opinion
from sklearn.metrics import roc_auc_score


def main():
    raw = load_intel_lab_data()
    sids = [4, 6, 7, 9]
    raw_c = select_sensor_cluster(raw, sids)
    raw_al = time_align_readings(raw_c, sids, interval_seconds=300)
    n = len(raw_al["timestamps"])

    print("=" * 60)
    print("SL Reliability: Belief vs Uncertainty as Quality Signal")
    print("=" * 60)

    # Quick check: what does the opinion look like?
    for pos, neg in [(48, 2), (25, 25), (5, 45), (1, 49), (0, 50)]:
        op = Opinion.from_evidence(positive=pos, negative=neg, base_rate=0.5)
        pp = op.projected_probability()
        print(f"  from_evidence({pos:2d}, {neg:2d}): "
              f"b={op.belief:.4f} d={op.disbelief:.4f} u={op.uncertainty:.4f} "
              f"P={pp:.4f}")

    print()

    # For each sensor: compute belief-based reliability
    agree_thresh = 2.0
    for W in [20, 50, 100]:
        print(f"\n  Window W={W}")
        for sid in sids:
            v = raw_al[sid]
            valid = ~np.isnan(v)
            is_out = valid & ((v < 10) | (v > 45))
            n_out = int(np.sum(is_out))
            if n_out == 0:
                continue
            others = [s for s in sids if s != sid]

            beliefs = np.full(n, np.nan)
            proj_probs = np.full(n, np.nan)

            for t in range(n):
                if np.isnan(v[t]):
                    continue
                ovals = [raw_al[s][t] for s in others if not np.isnan(raw_al[s][t])]
                if len(ovals) < 1:
                    continue

                # Count agrees/disagrees in window
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
                    beliefs[t] = op.belief
                    proj_probs[t] = op.projected_probability()

            # AUROC with different signals
            valid_both = ~np.isnan(beliefs) & valid
            y_true = is_out[valid_both].astype(int)
            if len(np.unique(y_true)) < 2:
                continue

            # Low belief = unreliable, so score = 1 - belief
            auroc_belief = roc_auc_score(y_true, 1 - beliefs[valid_both])
            auroc_pp = roc_auc_score(y_true, 1 - proj_probs[valid_both])

            print(f"    Sensor {sid}: AUROC(1-belief)={auroc_belief:.3f}  "
                  f"AUROC(1-P)={auroc_pp:.3f}  n_out={n_out}")

            # Threshold sweep on (1 - projected_probability)
            score = 1 - proj_probs[valid_both]
            best_f1, best_thresh = 0, 0
            for pct in [50, 60, 70, 80, 90, 95, 99]:
                thresh = np.percentile(score, pct)
                flagged = score > thresh
                tp = np.sum(flagged & (y_true == 1))
                fp = np.sum(flagged & (y_true == 0))
                fn = np.sum(~flagged & (y_true == 1))
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
                if pct in [70, 80, 90, 95]:
                    fpr = fp / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0
                    print(f"      P{pct}: thresh={thresh:.4f} recall={recall:.1%} "
                          f"FPR={fpr:.1%} prec={prec:.1%} F1={f1:.3f}")
            print(f"      Best F1={best_f1:.3f} at thresh={best_thresh:.4f}")

    # Also try combining SL belief with consensus signal
    print("\n" + "=" * 60)
    print("Composite: SL belief + cross-sensor consensus")
    print("=" * 60)

    W = 50
    for sid in sids:
        v = raw_al[sid]
        valid = ~np.isnan(v)
        is_out = valid & ((v < 10) | (v > 45))
        n_out = int(np.sum(is_out))
        if n_out == 0:
            continue
        n_clean = int(np.sum(valid & ~is_out))
        others = [s for s in sids if s != sid]

        # Compute both signals
        sl_score = np.full(n, 0.0)  # 1 - P (higher = worse)
        consensus_dev = np.full(n, 0.0)  # |value - median(neighbors)|

        for t in range(n):
            if np.isnan(v[t]):
                continue
            ovals = [raw_al[s][t] for s in others if not np.isnan(raw_al[s][t])]
            if not ovals:
                continue
            consensus_dev[t] = abs(v[t] - np.median(ovals))

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
                sl_score[t] = 1 - op.projected_probability()

        # Composite: normalize both to [0,1] and average
        cd_max = np.percentile(consensus_dev[valid], 99)
        cd_norm = np.clip(consensus_dev / max(cd_max, 1e-6), 0, 1)
        composite = 0.5 * sl_score + 0.5 * cd_norm

        valid_mask = valid
        y_true = is_out[valid_mask].astype(int)
        if len(np.unique(y_true)) < 2:
            continue

        auroc_cd = roc_auc_score(y_true, consensus_dev[valid_mask])
        auroc_sl = roc_auc_score(y_true, sl_score[valid_mask])
        auroc_comp = roc_auc_score(y_true, composite[valid_mask])

        print(f"\n  Sensor {sid} ({n_out} outliers):")
        print(f"    AUROC: consensus_dev={auroc_cd:.3f}  "
              f"sl_belief={auroc_sl:.3f}  composite={auroc_comp:.3f}")

        # Best F1 for each signal
        for label, scores in [("consensus", consensus_dev[valid_mask]),
                               ("sl_belief", sl_score[valid_mask]),
                               ("composite", composite[valid_mask])]:
            best_f1 = 0
            best_info = ""
            for pct in range(50, 100):
                thresh = np.percentile(scores, pct)
                flagged = scores > thresh
                tp = np.sum(flagged & (y_true == 1))
                fp = np.sum(flagged & (y_true == 0))
                fn = np.sum(~flagged & (y_true == 1))
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
                fpr = fp / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0
                if f1 > best_f1:
                    best_f1 = f1
                    best_info = f"P{pct} recall={recall:.1%} FPR={fpr:.1%} prec={prec:.1%}"
            print(f"    {label:12s} best F1={best_f1:.3f} ({best_info})")


if __name__ == "__main__":
    main()
