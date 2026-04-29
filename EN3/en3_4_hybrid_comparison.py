"""EN3.4 Hybrid: Conflict vs Confidence Abstention Comparison.

Tests whether SL conflict-based abstention on intersection-agreed entities
outperforms scalar confidence-based abstention. This is the key experiment
that separates "SL adds unique value" from "just use scalars."

Three abstention strategies on intersection-agreed entities:
  1. SL conflict: abstain when conflict_metric(fused) > τ
  2. Min-confidence: abstain when min(score_a, score_b) < τ
  3. Score-gap: abstain when |score_a - score_b| > τ (disagreement proxy)

Run: python experiments/EN3/en3_4_hybrid_comparison.py
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
    apply_condition_intersection,
    build_opinion_per_bin,
    evaluate_entities,
    bootstrap_f1_difference_ci,
    optimize_threshold,
)
from jsonld_ex.confidence_algebra import cumulative_fuse, conflict_metric

SEED = 42
RESULTS_DIR = _SCRIPT_DIR / "results"
CHECKPOINT_DIR = _SCRIPT_DIR / "checkpoints"
ACCEPT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def _banner(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")


def _load_checkpoint(name):
    path = CHECKPOINT_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _ser_to_preds(data):
    return [[EntitySpan(d["start"], d["end"], d["entity_type"],
                        d["score"], d["source"], d.get("text", ""))
             for d in sd] for sd in data]


def run_abstention_comparison(
    dataset_name: str,
    test_sentences: List[Dict],
    test_g2: List[List[EntitySpan]],
    test_bm: List[List[EntitySpan]],
    t_g2: float,
    t_bm: float,
    bins_g2: List[Dict],
    bins_bm: List[Dict],
) -> Dict[str, Any]:
    """Compare conflict vs confidence abstention on intersection-agreed entities."""
    _banner(f"Abstention Comparison: {dataset_name}")

    test_golds = [s["gold_spans"] for s in test_sentences]
    all_test_gold = [g for sg in test_golds for g in sg]

    # Collect all intersection-agreed entity pairs with metadata
    agreed_entities = []
    for g2p, bmp, sg in zip(test_g2, test_bm, test_golds):
        groups = align_spans(g2p, bmp, iou_threshold=0.5)
        for g in groups:
            sa, sb = g["span_a"], g["span_b"]
            if (sa is None or sb is None
                    or sa.score < t_g2 or sb.score < t_bm):
                continue

            op_a = build_opinion_per_bin(sa.score, bins_g2)
            op_b = build_opinion_per_bin(sb.score, bins_bm)
            fused = cumulative_fuse(op_a, op_b)
            conf = conflict_metric(fused)

            winner = sa if sa.score >= sb.score else sb
            is_correct = any(
                winner.start == gld.start and winner.end == gld.end
                and winner.entity_type == gld.entity_type
                for gld in sg)

            agreed_entities.append({
                "span": winner,
                "score_a": sa.score,
                "score_b": sb.score,
                "min_score": min(sa.score, sb.score),
                "max_score": max(sa.score, sb.score),
                "score_gap": abs(sa.score - sb.score),
                "avg_score": (sa.score + sb.score) / 2,
                "conflict": conf,
                "fused_pp": fused.projected_probability(),
                "correct": is_correct,
                "sent_golds": sg,
            })

    n_total = len(agreed_entities)
    n_correct = sum(1 for e in agreed_entities if e["correct"])
    n_error = n_total - n_correct
    print(f"  Intersection-agreed entities: {n_total}")
    print(f"  Correct: {n_correct} ({100*n_correct/n_total:.1f}%)")
    print(f"  Errors:  {n_error} ({100*n_error/n_total:.1f}%)")

    # Three abstention strategies + intersection baseline
    strategies = {
        "None (intersection)": lambda e, tau: False,  # never abstain
        "SL conflict > τ": lambda e, tau: e["conflict"] > tau,
        "Min-score < τ": lambda e, tau: e["min_score"] < tau,
        "Score-gap > τ": lambda e, tau: e["score_gap"] > tau,
        "1 - avg_score > τ": lambda e, tau: (1 - e["avg_score"]) > tau,
    }

    # Sweep thresholds
    tau_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]

    print(f"\n  {'Strategy':<25s} {'τ':>6s} {'Kept':>6s} {'Abst':>6s} "
          f"{'P_kept':>8s} {'R_kept':>8s} {'F1_kept':>8s} "
          f"{'Err%_abst':>10s}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    results_by_strategy = {}

    for strat_name, abstain_fn in strategies.items():
        strat_points = []
        for tau in tau_values:
            kept = [e for e in agreed_entities if not abstain_fn(e, tau)]
            abst = [e for e in agreed_entities if abstain_fn(e, tau)]

            n_kept = len(kept)
            n_abst = len(abst)

            if n_kept == 0:
                continue

            # Evaluate kept entities
            kept_preds = [e["span"] for e in kept]
            # We need to evaluate against the FULL gold set, not just per-sentence
            # But we need per-sentence matching... simplify: count TP among kept
            tp_kept = sum(1 for e in kept if e["correct"])
            fp_kept = n_kept - tp_kept
            fn_kept = len(all_test_gold) - tp_kept  # all gold minus matched

            p_kept = tp_kept / n_kept if n_kept > 0 else 0
            r_kept = tp_kept / len(all_test_gold) if all_test_gold else 0
            f1_kept = (2 * p_kept * r_kept / (p_kept + r_kept)
                       if (p_kept + r_kept) > 0 else 0)

            err_abst = (sum(1 for e in abst if not e["correct"]) / n_abst
                        if n_abst > 0 else 0)

            strat_points.append({
                "tau": tau, "n_kept": n_kept, "n_abst": n_abst,
                "p": p_kept, "r": r_kept, "f1": f1_kept,
                "err_rate_abst": err_abst,
            })

        results_by_strategy[strat_name] = strat_points

        # Print best F1 point
        if strat_points:
            best = max(strat_points, key=lambda x: x["f1"])
            print(f"  {strat_name:<25s} {best['tau']:6.2f} {best['n_kept']:6d} "
                  f"{best['n_abst']:6d} {best['p']:8.4f} {best['r']:8.4f} "
                  f"{best['f1']:8.4f} {best['err_rate_abst']:10.4f}")

    # ── Detailed comparison at matched abstention rates ──
    _banner(f"Matched-Rate Comparison: {dataset_name}")
    print("  At similar abstention counts, which method has higher precision?")

    # Find comparable operating points
    target_abst_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"\n  {'Abst%':>6s} | {'SL Conflict':^30s} | {'Min-Score':^30s} | {'Score-Gap':^30s}")
    print(f"  {'-'*6} | {'P':>8s} {'R':>8s} {'F1':>8s} {'Err%ab':>8s} | "
          f"{'P':>8s} {'R':>8s} {'F1':>8s} {'Err%ab':>8s} | "
          f"{'P':>8s} {'R':>8s} {'F1':>8s} {'Err%ab':>8s}")

    comparison_data = []
    for target_rate in target_abst_rates:
        target_n = int(n_total * target_rate)
        row = {"target_rate": target_rate}

        for strat_name in ["SL conflict > τ", "Min-score < τ", "Score-gap > τ"]:
            points = results_by_strategy[strat_name]
            # Find point closest to target abstention count
            closest = min(points, key=lambda p: abs(p["n_abst"] - target_n)) if points else None
            if closest:
                short_name = strat_name.split()[0] + "_" + strat_name.split()[1]
                row[strat_name] = closest
            else:
                row[strat_name] = {"p": 0, "r": 0, "f1": 0, "err_rate_abst": 0, "n_abst": 0}

        sl = row.get("SL conflict > τ", {})
        ms = row.get("Min-score < τ", {})
        sg = row.get("Score-gap > τ", {})

        print(f"  {target_rate*100:5.0f}% | "
              f"{sl.get('p',0):8.4f} {sl.get('r',0):8.4f} {sl.get('f1',0):8.4f} {sl.get('err_rate_abst',0):8.4f} | "
              f"{ms.get('p',0):8.4f} {ms.get('r',0):8.4f} {ms.get('f1',0):8.4f} {ms.get('err_rate_abst',0):8.4f} | "
              f"{sg.get('p',0):8.4f} {sg.get('r',0):8.4f} {sg.get('f1',0):8.4f} {sg.get('err_rate_abst',0):8.4f}")

        comparison_data.append(row)

    # ── AUROC comparison ──
    _banner(f"AUROC Comparison: {dataset_name}")
    try:
        from sklearn.metrics import roc_auc_score
        error_labels = [0 if e["correct"] else 1 for e in agreed_entities]

        if len(set(error_labels)) == 2:
            for signal_name, signal_key in [
                ("SL conflict", "conflict"),
                ("1 - min_score", lambda e: 1 - e["min_score"]),
                ("score_gap", "score_gap"),
                ("1 - avg_score", lambda e: 1 - e["avg_score"]),
            ]:
                if callable(signal_key):
                    scores = [signal_key(e) for e in agreed_entities]
                else:
                    scores = [e[signal_key] for e in agreed_entities]
                auroc = roc_auc_score(error_labels, scores)
                print(f"  {signal_name:<20s} AUROC = {auroc:.4f}")
    except ImportError:
        print("  scikit-learn not available for AUROC")

    return {
        "dataset": dataset_name,
        "n_intersection_agreed": n_total,
        "n_correct": n_correct,
        "n_error": n_error,
        "strategies": {k: v for k, v in results_by_strategy.items()},
    }


def main():
    set_global_seed(SEED)
    env = log_environment()
    all_results = {}

    # ═══ BC5CDR ═══
    from EN3.run_en3_4 import load_bc5cdr

    test_data = load_bc5cdr("test")
    g2_test = _ser_to_preds(_load_checkpoint("en3_4_bc5cdr_test_gliner2"))
    bm_test = _ser_to_preds(_load_checkpoint("en3_4_bc5cdr_test_biomed"))

    with open(RESULTS_DIR / "EN3_4_calibration.json") as f:
        cal = json.load(f)
    bins_g2_bc = cal["metrics"]["gliner2"]["reliability_bins"]
    bins_bm_bc = cal["metrics"]["biomed"]["reliability_bins"]

    all_results["bc5cdr"] = run_abstention_comparison(
        "BC5CDR", test_data, g2_test, bm_test,
        t_g2=0.7, t_bm=0.7, bins_g2=bins_g2_bc, bins_bm=bins_bm_bc)

    # ═══ MedMentions corrected ═══
    from EN3.en3_4_mm_corrected import load_mm_corrected

    mm_test = load_mm_corrected("test")
    mm_g2_test = _ser_to_preds(_load_checkpoint("en3_4_mm_corrected_test_g2"))
    mm_bm_test = _ser_to_preds(_load_checkpoint("en3_4_mm_corrected_test_bm"))

    # Compute MM calibration bins from dev
    mm_dev = load_mm_corrected("validation")
    mm_g2_dev = _ser_to_preds(_load_checkpoint("en3_4_mm_corrected_dev_g2"))
    mm_bm_dev = _ser_to_preds(_load_checkpoint("en3_4_mm_corrected_dev_bm"))
    for mn, preds in [("gliner2", mm_g2_dev), ("biomed", mm_bm_dev)]:
        sc, co = [], []
        for sent, sp in zip(mm_dev, preds):
            for p in sp:
                sc.append(p.score)
                co.append(any(p.start == g.start and p.end == g.end
                              and p.entity_type == g.entity_type
                              for g in sent["gold_spans"]))
        bins = reliability_diagram_bins(sc, co, n_bins=10) if sc else []
        if mn == "gliner2":
            bins_g2_mm = bins
        else:
            bins_bm_mm = bins

    all_results["medmentions"] = run_abstention_comparison(
        "MedMentions-corrected", mm_test, mm_g2_test, mm_bm_test,
        t_g2=0.3, t_bm=0.3, bins_g2=bins_g2_mm, bins_bm=bins_bm_mm)

    # Save
    result = ExperimentResult(
        experiment_id="EN3.4-abstention-comparison",
        parameters={"seed": SEED},
        metrics=all_results,
        environment=env,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.save_json(str(RESULTS_DIR / "EN3_4_abstention_comparison.json"))
    print(f"\nSaved: EN3_4_abstention_comparison.json")


if __name__ == "__main__":
    main()
