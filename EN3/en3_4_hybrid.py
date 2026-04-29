"""EN3.4 Hybrid Condition: Intersection + SL Conflict Abstention.

Tests on both BC5CDR and MedMentions (corrected) using cached predictions.
No GPU needed.

Run: python experiments/EN3/en3_4_hybrid.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _SCRIPT_DIR.parent
for p in [str(_SCRIPT_DIR), str(_EXPERIMENTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from infra.config import set_global_seed
from infra.results import ExperimentResult
from infra.env_log import log_environment

from EN3.en3_4_calibration import compute_ece, reliability_diagram_bins
from EN3.en3_4_core import (
    EntitySpan, align_spans,
    apply_condition_single_model,
    apply_condition_union,
    apply_condition_intersection,
    apply_condition_scalar_average,
    apply_condition_sl_fusion_per_bin,
    apply_condition_hybrid_intersection_conflict,
    evaluate_entities,
    bootstrap_f1_difference_ci,
    cohens_h,
    spearman_conflict_error,
    optimize_threshold,
    build_opinion_per_bin,
)
from jsonld_ex.confidence_algebra import cumulative_fuse, conflict_metric

SEED = 42
RESULTS_DIR = _SCRIPT_DIR / "results"
CHECKPOINT_DIR = _SCRIPT_DIR / "checkpoints"

ACCEPT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ABSTENTION_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def _banner(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")


def _load_checkpoint(name):
    path = CHECKPOINT_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"  Checkpoint: {path.name}")
        return data
    return None


def _ser_to_preds(data):
    return [[EntitySpan(d["start"], d["end"], d["entity_type"],
                        d["score"], d["source"], d.get("text", ""))
             for d in sd] for sd in data]


def run_hybrid_evaluation(
    dataset_name: str,
    dev_sentences: List[Dict],
    test_sentences: List[Dict],
    dev_g2: List[List[EntitySpan]],
    dev_bm: List[List[EntitySpan]],
    test_g2: List[List[EntitySpan]],
    test_bm: List[List[EntitySpan]],
    bins_g2: List[Dict],
    bins_bm: List[Dict],
) -> Dict[str, Any]:
    """Run full evaluation including hybrid condition."""
    _banner(f"Hybrid Evaluation: {dataset_name}")
    t0 = time.time()

    # ── Threshold optimization on dev ──
    dev_golds = [s["gold_spans"] for s in dev_sentences]
    all_dev_gold = [g for sg in dev_golds for g in sg]
    all_dev_g2 = [p for sp in dev_g2 for p in sp]
    all_dev_bm = [p for sp in dev_bm for p in sp]

    t_g2, _ = optimize_threshold(all_dev_g2, all_dev_gold, ACCEPT_THRESHOLDS)
    t_bm, _ = optimize_threshold(all_dev_bm, all_dev_gold, ACCEPT_THRESHOLDS)
    t_scalar, _ = optimize_threshold(all_dev_g2 + all_dev_bm, all_dev_gold,
                                     ACCEPT_THRESHOLDS)

    # Optimize hybrid thresholds on dev
    best_hy_ct = 0.5
    best_hy_f1 = -1.0
    for ct in ABSTENTION_THRESHOLDS:
        acc = []
        for g2p, bmp in zip(dev_g2, dev_bm):
            groups = align_spans(g2p, bmp, iou_threshold=0.5)
            a, _ = apply_condition_hybrid_intersection_conflict(
                groups, t_g2, t_bm, bins_g2, bins_bm, ct)
            acc.extend(a)
        m = evaluate_entities(acc, all_dev_gold)
        if m.f1 > best_hy_f1:
            best_hy_f1 = m.f1
            best_hy_ct = ct
    print(f"  Hybrid best conflict threshold: {best_hy_ct} (dev F1={best_hy_f1:.4f})")

    # Optimize SL per-bin thresholds on dev
    best_sl_at, best_sl_ct = 0.5, 0.5
    best_sl_f1 = -1.0
    for at in ACCEPT_THRESHOLDS:
        for ct in ABSTENTION_THRESHOLDS:
            acc = []
            for g2p, bmp in zip(dev_g2, dev_bm):
                groups = align_spans(g2p, bmp, iou_threshold=0.5)
                a, _ = apply_condition_sl_fusion_per_bin(
                    groups, bins_g2, bins_bm, at, ct)
                acc.extend(a)
            m = evaluate_entities(acc, all_dev_gold)
            if m.f1 > best_sl_f1:
                best_sl_f1 = m.f1
                best_sl_at, best_sl_ct = at, ct

    # ── Evaluate on test ──
    test_golds = [s["gold_spans"] for s in test_sentences]
    all_test_gold = [g for sg in test_golds for g in sg]

    per_sent = {"B1": [], "B2": [], "B3": [], "B4": [], "B5": [],
                "SL_pb": [], "HY": []}
    all_abst_sl = []
    all_abst_hy = []
    conf_scores_hy, conf_errors_hy = [], []

    for g2p, bmp, sg in zip(test_g2, test_bm, test_golds):
        per_sent["B1"].append(apply_condition_single_model(g2p, t_g2))
        per_sent["B2"].append(apply_condition_single_model(bmp, t_bm))
        groups = align_spans(g2p, bmp, iou_threshold=0.5)
        per_sent["B3"].append(apply_condition_union(groups, t_g2, t_bm))
        per_sent["B4"].append(apply_condition_intersection(groups, t_g2, t_bm))
        per_sent["B5"].append(apply_condition_scalar_average(groups, t_scalar))

        acc_sl, abst_sl = apply_condition_sl_fusion_per_bin(
            groups, bins_g2, bins_bm, best_sl_at, best_sl_ct)
        per_sent["SL_pb"].append(acc_sl)
        all_abst_sl.extend(abst_sl)

        acc_hy, abst_hy = apply_condition_hybrid_intersection_conflict(
            groups, t_g2, t_bm, bins_g2, bins_bm, best_hy_ct)
        per_sent["HY"].append(acc_hy)
        all_abst_hy.extend(abst_hy)

        # Conflict-error pairs for hybrid (only intersection-agreed entities)
        for g in groups:
            if (g["span_a"] is not None and g["span_b"] is not None
                    and g["span_a"].score >= t_g2 and g["span_b"].score >= t_bm):
                op_a = build_opinion_per_bin(g["span_a"].score, bins_g2)
                op_b = build_opinion_per_bin(g["span_b"].score, bins_bm)
                fused = cumulative_fuse(op_a, op_b)
                conf = conflict_metric(fused)
                winner = g["span_a"] if g["span_a"].score >= g["span_b"].score \
                    else g["span_b"]
                is_err = not any(
                    winner.start == gld.start and winner.end == gld.end
                    and winner.entity_type == gld.entity_type for gld in sg)
                conf_scores_hy.append(conf)
                conf_errors_hy.append(is_err)

    # ── Results table ──
    cond_keys = [("B1: GLiNER2", "B1"), ("B2: BioMed", "B2"),
                 ("B3: Union", "B3"), ("B4: Intersection", "B4"),
                 ("B5: Scalar Avg", "B5"), ("SL: Per-bin", "SL_pb"),
                 ("HYBRID", "HY")]
    conditions = []
    for name, key in cond_keys:
        flat = [s for sp in per_sent[key] for s in sp]
        m = evaluate_entities(flat, all_test_gold)
        conditions.append((name, m))

    print(f"\n  {'Condition':<20s} {'P':>8s} {'R':>8s} {'F1':>8s} "
          f"{'TP':>6s} {'FP':>6s} {'FN':>6s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")
    for name, m in conditions:
        marker = " <<<" if name == "HYBRID" else ""
        print(f"  {name:<20s} {m.precision:8.4f} {m.recall:8.4f} {m.f1:8.4f} "
              f"{m.tp:6d} {m.fp:6d} {m.fn:6d}{marker}")
    print(f"  SL abstained: {len(all_abst_sl)}, Hybrid abstained: {len(all_abst_hy)}")

    # ── Bootstrap CI ──
    def _tp_fp_fn(key):
        return [(evaluate_entities(sp, sg).tp,
                 evaluate_entities(sp, sg).fp,
                 evaluate_entities(sp, sg).fn)
                for sp, sg in zip(per_sent[key], test_golds)]

    boot = {k: _tp_fp_fn(k) for _, k in cond_keys}

    # Hybrid vs plain intersection (B4)
    m_hy = conditions[6][1]
    m_b4 = conditions[3][1]
    ci_lo_b4, ci_mean_b4, ci_hi_b4 = bootstrap_f1_difference_ci(
        boot["HY"], boot["B4"], n_bootstrap=2000, seed=SEED)
    v_b4 = "ACCEPTED" if ci_lo_b4 > 0 else ("REJECTED" if ci_hi_b4 < 0 else "INCONCLUSIVE")

    # Hybrid vs best of B1-B5
    best_bl = max(conditions[:5], key=lambda x: x[1].f1)
    best_name, best_m = best_bl
    best_key = [k for n, k in cond_keys if n == best_name][0]
    ci_lo_best, ci_mean_best, ci_hi_best = bootstrap_f1_difference_ci(
        boot["HY"], boot[best_key], n_bootstrap=2000, seed=SEED)
    v_best = "ACCEPTED" if ci_lo_best > 0 else ("REJECTED" if ci_hi_best < 0 else "INCONCLUSIVE")

    # Hybrid vs SL per-bin
    ci_lo_sl, ci_mean_sl, ci_hi_sl = bootstrap_f1_difference_ci(
        boot["HY"], boot["SL_pb"], n_bootstrap=2000, seed=SEED)

    # Hybrid vs scalar avg
    ci_lo_sc, _, ci_hi_sc = bootstrap_f1_difference_ci(
        boot["HY"], boot["B5"], n_bootstrap=2000, seed=SEED)

    print(f"\n  Hypothesis testing:")
    print(f"    Best baseline: {best_name} (F1={best_m.f1:.4f})")
    print(f"    HYBRID - Intersection = {m_hy.f1 - m_b4.f1:+.4f}, "
          f"CI [{ci_lo_b4:+.4f}, {ci_hi_b4:+.4f}] → {v_b4}")
    print(f"    HYBRID - Best baseline = {m_hy.f1 - best_m.f1:+.4f}, "
          f"CI [{ci_lo_best:+.4f}, {ci_hi_best:+.4f}] → {v_best}")
    print(f"    HYBRID - SL Per-bin = {m_hy.f1 - conditions[5][1].f1:+.4f}, "
          f"CI [{ci_lo_sl:+.4f}, {ci_hi_sl:+.4f}]")
    print(f"    HYBRID - Scalar Avg = {m_hy.f1 - conditions[4][1].f1:+.4f}, "
          f"CI [{ci_lo_sc:+.4f}, {ci_hi_sc:+.4f}]")

    # Conflict analysis on intersection-agreed entities
    if len(conf_scores_hy) >= 10:
        rho, p = spearman_conflict_error(conf_scores_hy, conf_errors_hy)
        print(f"    Conflict-error (intersection-agreed) rho={rho:.4f}, p={p:.6f}")
        try:
            from sklearn.metrics import roc_auc_score
            labels = [1 if e else 0 for e in conf_errors_hy]
            if len(set(labels)) == 2:
                auroc = roc_auc_score(labels, conf_scores_hy)
                print(f"    Conflict AUROC (intersection-agreed): {auroc:.4f}")
        except ImportError:
            pass

    # Abstention quality
    if all_abst_hy:
        abst_err = sum(1 for a in all_abst_hy if not any(
            a.start == g.start and a.end == g.end and a.entity_type == g.entity_type
            for g in all_test_gold)) / len(all_abst_hy)
        acc_err = m_hy.fp / (m_hy.tp + m_hy.fp) if (m_hy.tp + m_hy.fp) > 0 else 0
        print(f"    Hybrid abstained: {len(all_abst_hy)}, "
              f"abstain error rate={abst_err:.3f}, accept error rate={acc_err:.3f}")

    result = {
        "dataset": dataset_name,
        "conditions": {n: {"p": m.precision, "r": m.recall, "f1": m.f1,
                           "tp": m.tp, "fp": m.fp, "fn": m.fn}
                       for n, m in conditions},
        "hybrid_vs_intersection": {"diff": m_hy.f1 - m_b4.f1,
                                    "ci_lo": ci_lo_b4, "ci_hi": ci_hi_b4,
                                    "verdict": v_b4},
        "hybrid_vs_best": {"best": best_name, "diff": m_hy.f1 - best_m.f1,
                            "ci_lo": ci_lo_best, "ci_hi": ci_hi_best,
                            "verdict": v_best},
        "thresholds": {"g2": t_g2, "bm": t_bm, "hybrid_ct": best_hy_ct},
    }

    print(f"\n  Complete ({time.time()-t0:.1f}s)")
    return result


def main():
    set_global_seed(SEED)
    env = log_environment()

    all_results = {}

    # ═══════════════════════════════════════════
    # Dataset 1: BC5CDR
    # ═══════════════════════════════════════════
    from EN3.run_en3_4 import load_bc5cdr, _serializable_to_preds

    dev_data = load_bc5cdr("validation")
    test_data = load_bc5cdr("test")

    g2_dev = _ser_to_preds(_load_checkpoint("en3_4_bc5cdr_dev_gliner2"))
    g2_test = _ser_to_preds(_load_checkpoint("en3_4_bc5cdr_test_gliner2"))
    bm_dev = _ser_to_preds(_load_checkpoint("en3_4_bc5cdr_dev_biomed"))
    bm_test = _ser_to_preds(_load_checkpoint("en3_4_bc5cdr_test_biomed"))

    with open(RESULTS_DIR / "EN3_4_calibration.json") as f:
        cal = json.load(f)
    bins_g2_bc = cal["metrics"]["gliner2"]["reliability_bins"]
    bins_bm_bc = cal["metrics"]["biomed"]["reliability_bins"]

    bc5_results = run_hybrid_evaluation(
        "BC5CDR", dev_data, test_data,
        g2_dev, bm_dev, g2_test, bm_test,
        bins_g2_bc, bins_bm_bc,
    )
    all_results["bc5cdr"] = bc5_results

    # ═══════════════════════════════════════════
    # Dataset 2: MedMentions corrected
    # ═══════════════════════════════════════════
    from EN3.en3_4_mm_corrected import load_mm_corrected

    mm_dev = load_mm_corrected("validation")
    mm_test = load_mm_corrected("test")

    mm_g2_dev = _ser_to_preds(_load_checkpoint("en3_4_mm_corrected_dev_g2"))
    mm_g2_test = _ser_to_preds(_load_checkpoint("en3_4_mm_corrected_test_g2"))
    mm_bm_dev = _ser_to_preds(_load_checkpoint("en3_4_mm_corrected_dev_bm"))
    mm_bm_test = _ser_to_preds(_load_checkpoint("en3_4_mm_corrected_test_bm"))

    # Compute MM calibration bins
    for model_name, preds in [("gliner2", mm_g2_dev), ("biomed", mm_bm_dev)]:
        scores, correct = [], []
        for sent, sp in zip(mm_dev, preds):
            for p in sp:
                scores.append(p.score)
                is_correct = any(
                    p.start == g.start and p.end == g.end
                    and p.entity_type == g.entity_type
                    for g in sent["gold_spans"])
                correct.append(is_correct)
        if scores:
            bins = reliability_diagram_bins(scores, correct, n_bins=10)
            if model_name == "gliner2":
                bins_g2_mm = bins
            else:
                bins_bm_mm = bins

    mm_results = run_hybrid_evaluation(
        "MedMentions-corrected", mm_dev, mm_test,
        mm_g2_dev, mm_bm_dev, mm_g2_test, mm_bm_test,
        bins_g2_mm, bins_bm_mm,
    )
    all_results["medmentions"] = mm_results

    # ═══════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = ExperimentResult(
        experiment_id="EN3.4-hybrid",
        parameters={"seed": SEED, "method": "intersection+conflict"},
        metrics=all_results,
        environment=env,
    )
    result.save_json(str(RESULTS_DIR / "EN3_4_hybrid.json"))
    print(f"\nSaved: EN3_4_hybrid.json")


if __name__ == "__main__":
    main()
