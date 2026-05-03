#!/usr/bin/env python
"""EN4.2 — Dempster-Shafer vs Subjective Logic: Full Comparison Runner.

Phases:
    A: Zadeh paradox demonstration (synthetic, no data deps)
    B: BC5CDR DS vs SL comparison (requires gliner venv for gold labels)

Usage:
    python experiments/EN4/run_en4_2.py --phase a
    experiments/.venv-gliner/Scripts/python experiments/EN4/run_en4_2.py --phase b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Path Setup ──────────────────────────────────────────────────────
_EN4_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _EN4_DIR.parent
_REPO_ROOT = _EXPERIMENTS_ROOT.parent
_PKG_SRC = _REPO_ROOT / "packages" / "python" / "src"

for p in [str(_EN4_DIR), str(_EXPERIMENTS_ROOT), str(_PKG_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from en4_2_ds_comparison import (
    dempster_combine, yager_combine,
    dempster_combine_multi, yager_combine_multi,
    confidence_to_mass, mass_to_decision, sl_to_decision,
    compute_conflict_K, compare_fusion_methods_binary,
)
from jsonld_ex.confidence_algebra import (
    Opinion, cumulative_fuse, averaging_fuse, conflict_metric,
)
from infra.results import ExperimentResult
from infra.env_log import log_environment

# ── Import EN3.4 utilities ──────────────────────────────────────────
sys.path.insert(0, str(_EXPERIMENTS_ROOT / "EN3"))
from en3_4_core import (
    EntitySpan, align_spans, evaluate_entities, EvalMetrics,
    bootstrap_f1_difference_ci, build_opinion_per_bin, cohens_h,
)
from run_en3_4 import (
    load_bc5cdr, _serializable_to_preds,
    _load_checkpoint as _load_en3_checkpoint,
)
from en3_4_mm_corrected import load_mm_corrected

# ── Constants ───────────────────────────────────────────────────────
RESULTS_DIR = _EN4_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
EN3_CHECKPOINT_DIR = _EXPERIMENTS_ROOT / "EN3" / "checkpoints"
SEED = 42
N_BOOTSTRAP = 2000


# =====================================================================
# Helpers
# =====================================================================

def _load_checkpoint(name: str) -> list:
    """Load cached predictions from EN3 checkpoints."""
    path = EN3_CHECKPOINT_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path) as f:
        return json.load(f)


def _load_calibration_bins() -> Optional[Dict[str, list]]:
    """Load calibration bins from EN3.4 Phase A0."""
    cal_path = EN3_CHECKPOINT_DIR / "en3_4_calibration_values.json"
    if not cal_path.exists():
        print("  WARNING: No calibration bins found. Using defaults.")
        return None
    with open(cal_path) as f:
        return json.load(f)


def _apply_ds_condition(
    groups: List[Dict[str, Any]], combine_fn, bins_a, bins_b,
    threshold: float, use_base_rate: bool = False, base_rate: float = 0.5,
) -> Tuple[List[EntitySpan], List[Dict]]:
    """Apply a DS combination rule to aligned span groups."""
    accepted, diagnostics = [], []
    for g in groups:
        sa, sb = g.get("span_a"), g.get("span_b")
        if sa is not None and sb is not None:
            op_a = build_opinion_per_bin(sa.score, bins_a)
            op_b = build_opinion_per_bin(sb.score, bins_b)
            m_a = {"b": op_a.belief, "d": op_a.disbelief, "u": op_a.uncertainty}
            m_b = {"b": op_b.belief, "d": op_b.disbelief, "u": op_b.uncertainty}
            K = compute_conflict_K(m_a, m_b)
            fused = combine_fn(m_a, m_b)
            if use_base_rate:
                proj_p = fused["b"] + base_rate * fused["u"]
                decision = proj_p > threshold
            else:
                decision = mass_to_decision(fused)
            winner = sa if sa.score >= sb.score else sb
            score_out = fused["b"] + base_rate * fused["u"] if use_base_rate else fused["b"]
            if decision:
                accepted.append(EntitySpan(
                    start=winner.start, end=winner.end,
                    entity_type=winner.entity_type,
                    score=score_out, source="ds_fused", text=winner.text,
                ))
            diagnostics.append({"K": K, "fused_b": fused["b"],
                                "fused_d": fused["d"], "fused_u": fused["u"],
                                "decision": decision})
        elif sa is not None and sa.score >= threshold:
            accepted.append(sa)
        elif sb is not None and sb.score >= threshold:
            accepted.append(sb)
    return accepted, diagnostics


def _apply_sl_condition(
    groups: List[Dict[str, Any]], bins_a, bins_b,
    threshold: float, base_rate: float = 0.5, use_averaging: bool = False,
) -> Tuple[List[EntitySpan], List[Dict]]:
    """Apply SL fusion to aligned span groups."""
    accepted, diagnostics = [], []
    fuse_fn = averaging_fuse if use_averaging else cumulative_fuse
    for g in groups:
        sa, sb = g.get("span_a"), g.get("span_b")
        if sa is not None and sb is not None:
            op_a = build_opinion_per_bin(sa.score, bins_a)
            op_b = build_opinion_per_bin(sb.score, bins_b)
            op_a_adj = Opinion(belief=op_a.belief, disbelief=op_a.disbelief,
                               uncertainty=op_a.uncertainty, base_rate=base_rate)
            op_b_adj = Opinion(belief=op_b.belief, disbelief=op_b.disbelief,
                               uncertainty=op_b.uncertainty, base_rate=base_rate)
            fused = fuse_fn(op_a_adj, op_b_adj)
            proj_p = fused.projected_probability()
            conf = conflict_metric(fused)
            decision = proj_p > threshold
            winner = sa if sa.score >= sb.score else sb
            if decision:
                accepted.append(EntitySpan(
                    start=winner.start, end=winner.end,
                    entity_type=winner.entity_type,
                    score=proj_p, source="sl_fused", text=winner.text,
                ))
            diagnostics.append({"fused_b": fused.belief, "fused_d": fused.disbelief,
                                "fused_u": fused.uncertainty, "conflict": conf,
                                "proj_p": proj_p, "decision": decision})
        elif sa is not None and sa.score >= threshold:
            accepted.append(sa)
        elif sb is not None and sb.score >= threshold:
            accepted.append(sb)
    return accepted, diagnostics


def _optimize_threshold(apply_fn, dev_groups, dev_golds, thresholds, **kwargs):
    """Find threshold maximizing micro-F1 on dev set."""
    all_dev_gold = [span for sent in dev_golds for span in sent]
    best_t, best_f1 = 0.3, 0.0
    for t in thresholds:
        flat = [g for sent in dev_groups for g in sent]
        accepted, _ = apply_fn(flat, threshold=t, **kwargs)
        m = evaluate_entities(accepted, all_dev_gold)
        if m.f1 > best_f1:
            best_f1 = m.f1
            best_t = t
    return best_t, best_f1


# =====================================================================
# Phase A: Zadeh Paradox (Synthetic)
# =====================================================================

def run_phase_a() -> Dict[str, Any]:
    """Zadeh's paradox demonstration + systematic conflict sweep."""
    print("\n" + "=" * 60)
    print("Phase A: Zadeh Paradox Demonstration")
    print("=" * 60)
    results = {}

    # A1: Classic Zadeh (dogmatic)
    print("\n  A1: Classic Zadeh scenario (dogmatic conflict)")
    m1, m2 = {"b": 0.99, "d": 0.01, "u": 0.0}, {"b": 0.01, "d": 0.99, "u": 0.0}
    K = compute_conflict_K(m1, m2)
    ds = dempster_combine(m1, m2)
    sl = cumulative_fuse(
        Opinion(belief=0.99, disbelief=0.01, uncertainty=0.0, base_rate=0.5),
        Opinion(belief=0.01, disbelief=0.99, uncertainty=0.0, base_rate=0.5))
    results["zadeh_classic"] = {"K": K, "ds": ds,
        "sl": {"b": sl.belief, "d": sl.disbelief, "u": sl.uncertainty}}
    print(f"    K={K:.4f}  DS: b={ds['b']:.4f} d={ds['d']:.4f} u={ds['u']:.4f}")
    print(f"              SL: b={sl.belief:.4f} d={sl.disbelief:.4f} u={sl.uncertainty:.4f}")

    # A2: Realistic Zadeh (with uncertainty)
    print("\n  A2: Realistic Zadeh (with residual uncertainty)")
    m1, m2 = {"b": 0.85, "d": 0.05, "u": 0.10}, {"b": 0.05, "d": 0.85, "u": 0.10}
    K = compute_conflict_K(m1, m2)
    ds = dempster_combine(m1, m2)
    yg = yager_combine(m1, m2)
    sl = cumulative_fuse(
        Opinion(belief=0.85, disbelief=0.05, uncertainty=0.10, base_rate=0.5),
        Opinion(belief=0.05, disbelief=0.85, uncertainty=0.10, base_rate=0.5))
    results["zadeh_realistic"] = {"K": K, "ds": ds, "yager": yg,
        "sl": {"b": sl.belief, "d": sl.disbelief, "u": sl.uncertainty}}
    print(f"    K={K:.4f}")
    print(f"    DS:    b={ds['b']:.4f} d={ds['d']:.4f} u={ds['u']:.4f}")
    print(f"    Yager: b={yg['b']:.4f} d={yg['d']:.4f} u={yg['u']:.4f}")
    print(f"    SL:    b={sl.belief:.4f} d={sl.disbelief:.4f} u={sl.uncertainty:.4f}")

    # A3: Systematic conflict sweep
    print("\n  A3: Systematic conflict sweep (10,000 pairs)")
    rng = np.random.default_rng(SEED)
    sweep = []
    for _ in range(10_000):
        raw1, raw2 = rng.dirichlet([1,1,1]), rng.dirichlet([1,1,1])
        m1 = {"b": raw1[0], "d": raw1[1], "u": raw1[2]}
        m2 = {"b": raw2[0], "d": raw2[1], "u": raw2[2]}
        K = compute_conflict_K(m1, m2)
        if K >= 0.99:
            continue
        ds = dempster_combine(m1, m2)
        sl = cumulative_fuse(
            Opinion(belief=raw1[0], disbelief=raw1[1], uncertainty=raw1[2], base_rate=0.5),
            Opinion(belief=raw2[0], disbelief=raw2[1], uncertainty=raw2[2], base_rate=0.5))
        sweep.append({"K": K, "belief_diff": ds["b"] - sl.belief,
                       "uncertainty_diff": sl.uncertainty - ds["u"]})

    bins = [(0.0,0.1),(0.1,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,0.95)]
    binned = {}
    for lo, hi in bins:
        sub = [r for r in sweep if lo <= r["K"] < hi]
        if sub:
            binned[f"K_{lo:.1f}_{hi:.1f}"] = {
                "n": len(sub),
                "mean_belief_diff": float(np.mean([r["belief_diff"] for r in sub])),
                "frac_ds_higher": float(np.mean([r["belief_diff"] > 0 for r in sub])),
            }
            print(f"    K∈[{lo:.1f},{hi:.1f}): n={len(sub):5d}  "
                  f"mean_b_diff={np.mean([r['belief_diff'] for r in sub]):+.4f}  "
                  f"DS>SL={np.mean([r['belief_diff'] > 0 for r in sub]):.1%}")
    results["conflict_sweep"] = binned

    all_diffs = [r["belief_diff"] for r in sweep]
    results["structural_divergence"] = {
        "mean": float(np.mean(all_diffs)), "std": float(np.std(all_diffs)),
        "frac_ds_higher": float(np.mean([d > 0 for d in all_diffs])),
        "n": len(sweep),
    }
    print(f"\n  FINDING: DS belief > SL in {np.mean([d>0 for d in all_diffs]):.1%}")
    return results


# =====================================================================
# Phase B: BC5CDR DS vs SL (dev-optimized thresholds)
# =====================================================================

def run_phase_b() -> Dict[str, Any]:
    """H4.2a/c/d/e — DS vs SL on BC5CDR with dev-optimized thresholds."""
    print("\n" + "=" * 60)
    print("Phase B: BC5CDR DS vs SL (dev-optimized)")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    print("  Loading BC5CDR dev + test (EN3.4 loader)...")
    dev_data = load_bc5cdr("validation")
    test_data = load_bc5cdr("test")
    dev_golds = [s["gold_spans"] for s in dev_data]
    test_golds = [s["gold_spans"] for s in test_data]
    all_test_gold = [span for sent in test_golds for span in sent]
    print(f"  Dev:  {len(dev_data)} sent, {sum(len(g) for g in dev_golds)} gold")
    print(f"  Test: {len(test_data)} sent, {len(all_test_gold)} gold")

    # ── Load cached predictions ──────────────────────────────────
    print("  Loading cached predictions...")
    g2_dev = _serializable_to_preds(_load_checkpoint("en3_4_bc5cdr_dev_gliner2"))
    bm_dev = _serializable_to_preds(_load_checkpoint("en3_4_bc5cdr_dev_biomed"))
    g2_test = _serializable_to_preds(_load_checkpoint("en3_4_bc5cdr_test_gliner2"))
    bm_test = _serializable_to_preds(_load_checkpoint("en3_4_bc5cdr_test_biomed"))
    print(f"  Test: GLiNER2={sum(len(s) for s in g2_test)} BioMed={sum(len(s) for s in bm_test)}")

    # ── Calibration bins ─────────────────────────────────────────
    cal_data = _load_calibration_bins()
    if cal_data:
        bins_g2 = cal_data.get("gliner2_bins", cal_data.get("bins_g2", []))
        bins_bm = cal_data.get("biomed_bins", cal_data.get("bins_bm", []))
    else:
        bins_g2 = [{"bin_lower": i/10, "bin_upper": (i+1)/10,
                     "abs_diff": 0.10, "count": 100} for i in range(10)]
        bins_bm = bins_g2

    # ── Align ────────────────────────────────────────────────────
    print("  Aligning predictions...")
    dev_groups = [align_spans(g2, bm) for g2, bm in zip(g2_dev, bm_dev)]
    test_groups = [align_spans(g2, bm) for g2, bm in zip(g2_test, bm_test)]
    flat_test = [g for sent in test_groups for g in sent]

    # ── Threshold grid ───────────────────────────────────────────
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                  0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    base_rate = 0.5

    # ── Define conditions ────────────────────────────────────────
    condition_defs = [
        ("C1_ds_classical", _apply_ds_condition,
         dict(combine_fn=dempster_combine, bins_a=bins_g2, bins_b=bins_bm,
              use_base_rate=False, base_rate=base_rate)),
        ("C2_ds_base_rate", _apply_ds_condition,
         dict(combine_fn=dempster_combine, bins_a=bins_g2, bins_b=bins_bm,
              use_base_rate=True, base_rate=base_rate)),
        ("C3_yager", _apply_ds_condition,
         dict(combine_fn=yager_combine, bins_a=bins_g2, bins_b=bins_bm,
              use_base_rate=False, base_rate=base_rate)),
        ("C4_sl_cumulative", _apply_sl_condition,
         dict(bins_a=bins_g2, bins_b=bins_bm,
              base_rate=base_rate, use_averaging=False)),
        ("C5_sl_averaging", _apply_sl_condition,
         dict(bins_a=bins_g2, bins_b=bins_bm,
              base_rate=base_rate, use_averaging=True)),
    ]

    # ── Optimize on dev, evaluate on test ────────────────────────
    print("\n  Optimizing thresholds on dev set...")
    conditions = {}
    for label, apply_fn, kwargs in condition_defs:
        best_t, dev_f1 = _optimize_threshold(
            apply_fn, dev_groups, dev_golds, thresholds, **kwargs)
        accepted, _ = apply_fn(flat_test, threshold=best_t, **kwargs)
        m = evaluate_entities(accepted, all_test_gold)
        conditions[label] = {
            "P": m.precision, "R": m.recall, "F1": m.f1,
            "tp": m.tp, "fp": m.fp, "fn": m.fn,
            "n_accepted": len(accepted),
            "dev_threshold": best_t, "dev_f1": dev_f1,
        }
        print(f"    {label:25s}  t*={best_t:.2f}  dev_F1={dev_f1:.4f}  "
              f"test: P={m.precision:.4f} R={m.recall:.4f} F1={m.f1:.4f}")

    # ── Bootstrap CIs ────────────────────────────────────────────
    print("\n  Bootstrap CIs (n=2000)...")

    def per_sent_triples(apply_fn, threshold, **kwargs):
        triples = []
        for i, sg in enumerate(test_groups):
            acc, _ = apply_fn(sg, threshold=threshold, **kwargs)
            m = evaluate_entities(acc, test_golds[i])
            triples.append((m.tp, m.fp, m.fn))
        return triples

    cond_triples = {}
    for label, apply_fn, kwargs in condition_defs:
        t = conditions[label]["dev_threshold"]
        cond_triples[label] = per_sent_triples(apply_fn, threshold=t, **kwargs)

    comparisons = [
        ("C4_sl_cumulative", "C1_ds_classical"),
        ("C4_sl_cumulative", "C2_ds_base_rate"),
        ("C2_ds_base_rate", "C1_ds_classical"),
    ]
    ci_results = []
    for la, lb in comparisons:
        lo, md, hi = bootstrap_f1_difference_ci(
            cond_triples[la], cond_triples[lb], n_bootstrap=N_BOOTSTRAP, seed=SEED)
        ci = {"comparison": f"{la} - {lb}", "mean_diff": md,
              "ci_lower": lo, "ci_upper": hi,
              "significant": (lo > 0) or (hi < 0),
              "cohens_h": cohens_h(conditions[la]["F1"], conditions[lb]["F1"])}
        ci_results.append(ci)
        print(f"    {la} - {lb}: {md:+.4f} CI[{lo:+.4f},{hi:+.4f}] "
              f"h={ci['cohens_h']:+.4f} {'SIG' if ci['significant'] else 'ns'}")

    # ── Base rate sweep (H4.2c) ──────────────────────────────────
    print("\n  Base rate sweep (H4.2c)...")
    sl_t = conditions["C4_sl_cumulative"]["dev_threshold"]
    br_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    empirical_prior = len(all_test_gold) / max(1, len(flat_test))
    br_values.append(round(min(empirical_prior, 0.9), 3))
    print(f"    Empirical prior: {empirical_prior:.4f}")

    br_sweep = {}
    for br in sorted(set(br_values)):
        acc, _ = _apply_sl_condition(
            flat_test, bins_a=bins_g2, bins_b=bins_bm,
            threshold=sl_t, base_rate=br, use_averaging=False)
        m = evaluate_entities(acc, all_test_gold)
        br_sweep[str(br)] = {"F1": m.f1, "P": m.precision, "R": m.recall}
        print(f"    a={br:.3f}  F1={m.f1:.4f}  P={m.precision:.4f}  R={m.recall:.4f}")

    # ── Conflict-partitioned analysis (H4.2d) ────────────────────
    print("\n  Conflict-partitioned analysis (H4.2d)...")
    ds_t = conditions["C1_ds_classical"]["dev_threshold"]
    conflict_data = []
    for sent_groups, sent_golds in zip(test_groups, test_golds):
        for g in sent_groups:
            sa, sb = g.get("span_a"), g.get("span_b")
            if sa is None or sb is None:
                continue
            op_a = build_opinion_per_bin(sa.score, bins_g2)
            op_b = build_opinion_per_bin(sb.score, bins_bm)
            m_a = {"b": op_a.belief, "d": op_a.disbelief, "u": op_a.uncertainty}
            m_b = {"b": op_b.belief, "d": op_b.disbelief, "u": op_b.uncertainty}
            K = compute_conflict_K(m_a, m_b)
            winner = sa if sa.score >= sb.score else sb
            is_correct = any(
                winner.start == gg.start and winner.end == gg.end
                and winner.entity_type == gg.entity_type for gg in sent_golds)
            ds_fused = dempster_combine(m_a, m_b)
            op_a_sl = Opinion(belief=op_a.belief, disbelief=op_a.disbelief,
                              uncertainty=op_a.uncertainty, base_rate=base_rate)
            op_b_sl = Opinion(belief=op_b.belief, disbelief=op_b.disbelief,
                              uncertainty=op_b.uncertainty, base_rate=base_rate)
            sl_fused = cumulative_fuse(op_a_sl, op_b_sl)
            conflict_data.append({
                "K": K, "is_correct": is_correct,
                "ds_decision": mass_to_decision(ds_fused),
                "sl_decision": sl_fused.projected_probability() > sl_t,
                "sl_conflict": conflict_metric(sl_fused),
            })

    Ks = np.array([c["K"] for c in conflict_data])
    quartiles = np.percentile(Ks, [25, 50, 75])
    q_defs = [("Q1_low", 0, quartiles[0]), ("Q2", quartiles[0], quartiles[1]),
              ("Q3", quartiles[1], quartiles[2]), ("Q4_high", quartiles[2], 1.0)]
    conflict_results = {}
    for ql, lo, hi in q_defs:
        sub = [c for c in conflict_data if lo <= c["K"] < hi]
        if not sub:
            continue
        n = len(sub)
        ds_tp = sum(c["ds_decision"] and c["is_correct"] for c in sub)
        ds_fp = sum(c["ds_decision"] and not c["is_correct"] for c in sub)
        sl_tp = sum(c["sl_decision"] and c["is_correct"] for c in sub)
        sl_fp = sum(c["sl_decision"] and not c["is_correct"] for c in sub)
        ds_prec = ds_tp / (ds_tp + ds_fp) if (ds_tp + ds_fp) > 0 else 0.0
        sl_prec = sl_tp / (sl_tp + sl_fp) if (sl_tp + sl_fp) > 0 else 0.0
        conflict_results[ql] = {
            "n": n, "K_range": [float(lo), float(hi)],
            "pct_correct": sum(c["is_correct"] for c in sub) / n,
            "ds_precision": ds_prec, "sl_precision": sl_prec,
        }
        print(f"    {ql:10s} K∈[{lo:.3f},{hi:.3f}) n={n:5d} "
              f"DS_prec={ds_prec:.3f} SL_prec={sl_prec:.3f}")

    return {
        "dataset": "BC5CDR", "n_gold_test": len(all_test_gold),
        "n_sent_test": len(test_data), "n_sent_dev": len(dev_data),
        "conditions": conditions, "bootstrap_ci": ci_results,
        "base_rate_sweep": br_sweep, "empirical_prior": empirical_prior,
        "conflict_quartiles": conflict_results,
    }


# =====================================================================
# Phase C: MedMentions DS vs SL (sparse entities, base rate matters more)
# =====================================================================

def run_phase_c() -> Dict[str, Any]:
    """DS vs SL on MedMentions-corrected (5 types, sparse entities)."""
    print("\n" + "=" * 60)
    print("Phase C: MedMentions DS vs SL (dev-optimized)")
    print("=" * 60)

    # -- Load data --
    print("  Loading MedMentions corrected dev + test...")
    dev_data = load_mm_corrected("validation")
    test_data = load_mm_corrected("test")
    dev_golds = [s["gold_spans"] for s in dev_data]
    test_golds = [s["gold_spans"] for s in test_data]
    all_test_gold = [span for sent in test_golds for span in sent]
    print(f"  Dev:  {len(dev_data)} sent, {sum(len(g) for g in dev_golds)} gold")
    print(f"  Test: {len(test_data)} sent, {len(all_test_gold)} gold")

    # -- Load cached predictions --
    print("  Loading cached predictions...")
    g2_dev = _serializable_to_preds(_load_checkpoint("en3_4_mm_corrected_dev_g2"))
    bm_dev = _serializable_to_preds(_load_checkpoint("en3_4_mm_corrected_dev_bm"))
    g2_test = _serializable_to_preds(_load_checkpoint("en3_4_mm_corrected_test_g2"))
    bm_test = _serializable_to_preds(_load_checkpoint("en3_4_mm_corrected_test_bm"))
    print(f"  Test: GLiNER2={sum(len(s) for s in g2_test)} BioMed={sum(len(s) for s in bm_test)}")

    # -- Calibration bins (reuse BC5CDR bins -- model property, not domain) --
    cal_path = _EXPERIMENTS_ROOT / "EN3" / "results" / "EN3_4_calibration.json"
    if cal_path.exists():
        with open(cal_path) as f:
            cal_results = json.load(f)
        metrics = cal_results.get("metrics", cal_results)
        bins_g2 = metrics.get("gliner2", {}).get("reliability_bins", [])
        bins_bm = metrics.get("biomed", {}).get("reliability_bins", [])
        print(f"  Calibration bins: g2={len(bins_g2)} bm={len(bins_bm)}")
    else:
        print("  WARNING: No calibration bins. Using defaults.")
        bins_g2 = [{"bin_lower": i/10, "bin_upper": (i+1)/10,
                     "abs_diff": 0.10, "count": 100} for i in range(10)]
        bins_bm = bins_g2

    # -- Align --
    print("  Aligning predictions...")
    dev_groups = [align_spans(g2, bm) for g2, bm in zip(g2_dev, bm_dev)]
    test_groups = [align_spans(g2, bm) for g2, bm in zip(g2_test, bm_test)]
    flat_test = [g for sent in test_groups for g in sent]

    # -- Threshold grid --
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                  0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    base_rate = 0.5

    # -- Define conditions --
    condition_defs = [
        ("C1_ds_classical", _apply_ds_condition,
         dict(combine_fn=dempster_combine, bins_a=bins_g2, bins_b=bins_bm,
              use_base_rate=False, base_rate=base_rate)),
        ("C2_ds_base_rate", _apply_ds_condition,
         dict(combine_fn=dempster_combine, bins_a=bins_g2, bins_b=bins_bm,
              use_base_rate=True, base_rate=base_rate)),
        ("C3_yager", _apply_ds_condition,
         dict(combine_fn=yager_combine, bins_a=bins_g2, bins_b=bins_bm,
              use_base_rate=False, base_rate=base_rate)),
        ("C4_sl_cumulative", _apply_sl_condition,
         dict(bins_a=bins_g2, bins_b=bins_bm,
              base_rate=base_rate, use_averaging=False)),
        ("C5_sl_averaging", _apply_sl_condition,
         dict(bins_a=bins_g2, bins_b=bins_bm,
              base_rate=base_rate, use_averaging=True)),
    ]

    # -- Optimize on dev, evaluate on test --
    # NOTE: MedMentions-ZS validation has 0 gold entities for the corrected
    # 5 types (they only exist in the test split). We cannot optimize
    # thresholds on dev. Instead, we use BC5CDR-optimized thresholds from
    # Phase B, which is defensible because the thresholds are a property
    # of model confidence calibration, not the target domain.
    #
    # If Phase B results are available, use those thresholds. Otherwise
    # fall back to cross-validated threshold on test (less ideal).
    bc5cdr_results_path = RESULTS_DIR / "EN4_2_ds_comparison.json"
    bc5cdr_thresholds = {}
    if bc5cdr_results_path.exists():
        with open(bc5cdr_results_path) as f:
            prev = json.load(f)
        pb = prev.get("phase_b", {})
        for label in pb.get("conditions", {}):
            bc5cdr_thresholds[label] = pb["conditions"][label].get("dev_threshold", 0.5)
        print(f"  Using BC5CDR-optimized thresholds: {bc5cdr_thresholds}")
    else:
        print("  WARNING: No Phase B results. Using default threshold 0.50.")
        for label, _, _ in condition_defs:
            bc5cdr_thresholds[label] = 0.50

    has_dev_gold = sum(len(g) for g in dev_golds) > 0
    print(f"  Dev gold entities: {sum(len(g) for g in dev_golds)}")
    if not has_dev_gold:
        print("  NOTE: MedMentions-ZS validation has 0 gold entities for the")
        print("        corrected 5 types. Using BC5CDR thresholds (cross-domain).")

    conditions = {}
    for label, apply_fn, kwargs in condition_defs:
        if has_dev_gold:
            best_t, dev_f1 = _optimize_threshold(
                apply_fn, dev_groups, dev_golds, thresholds, **kwargs)
        else:
            best_t = bc5cdr_thresholds.get(label, 0.50)
            dev_f1 = float('nan')
        accepted, _ = apply_fn(flat_test, threshold=best_t, **kwargs)
        m = evaluate_entities(accepted, all_test_gold)
        conditions[label] = {
            "P": m.precision, "R": m.recall, "F1": m.f1,
            "tp": m.tp, "fp": m.fp, "fn": m.fn,
            "n_accepted": len(accepted),
            "dev_threshold": best_t,
            "threshold_source": "bc5cdr_transfer" if not has_dev_gold else "dev_optimized",
        }
        print(f"    {label:25s}  t*={best_t:.2f}  "
              f"test: P={m.precision:.4f} R={m.recall:.4f} F1={m.f1:.4f}")

    # -- Bootstrap CIs --
    print("\n  Bootstrap CIs (n=2000)...")

    def per_sent_triples(apply_fn, threshold, **kwargs):
        triples = []
        for i, sg in enumerate(test_groups):
            acc, _ = apply_fn(sg, threshold=threshold, **kwargs)
            m = evaluate_entities(acc, test_golds[i])
            triples.append((m.tp, m.fp, m.fn))
        return triples

    cond_triples = {}
    for label, apply_fn, kwargs in condition_defs:
        t = conditions[label]["dev_threshold"]
        cond_triples[label] = per_sent_triples(apply_fn, threshold=t, **kwargs)

    comparisons = [
        ("C4_sl_cumulative", "C1_ds_classical"),
        ("C4_sl_cumulative", "C2_ds_base_rate"),
        ("C2_ds_base_rate", "C1_ds_classical"),
    ]
    ci_results = []
    for la, lb in comparisons:
        lo, md, hi = bootstrap_f1_difference_ci(
            cond_triples[la], cond_triples[lb], n_bootstrap=N_BOOTSTRAP, seed=SEED)
        ci = {"comparison": f"{la} - {lb}", "mean_diff": md,
              "ci_lower": lo, "ci_upper": hi,
              "significant": (lo > 0) or (hi < 0),
              "cohens_h": cohens_h(conditions[la]["F1"], conditions[lb]["F1"])}
        ci_results.append(ci)
        print(f"    {la} - {lb}: {md:+.4f} CI[{lo:+.4f},{hi:+.4f}] "
              f"h={ci['cohens_h']:+.4f} {'SIG' if ci['significant'] else 'ns'}")

    # -- Base rate sweep (MedMentions is sparser -> base rate should matter more) --
    print("\n  Base rate sweep (H4.2c on MedMentions)...")
    sl_t = conditions["C4_sl_cumulative"]["dev_threshold"]
    br_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    empirical_prior = len(all_test_gold) / max(1, len(flat_test))
    br_values.append(round(min(empirical_prior, 0.9), 3))
    print(f"    Empirical prior: {empirical_prior:.4f}")

    br_sweep = {}
    for br in sorted(set(br_values)):
        acc, _ = _apply_sl_condition(
            flat_test, bins_a=bins_g2, bins_b=bins_bm,
            threshold=sl_t, base_rate=br, use_averaging=False)
        m = evaluate_entities(acc, all_test_gold)
        br_sweep[str(br)] = {"F1": m.f1, "P": m.precision, "R": m.recall}
        print(f"    a={br:.3f}  F1={m.f1:.4f}  P={m.precision:.4f}  R={m.recall:.4f}")

    # -- Conflict-partitioned --
    print("\n  Conflict-partitioned analysis...")
    conflict_data = []
    for sent_groups, sent_golds in zip(test_groups, test_golds):
        for g in sent_groups:
            sa, sb = g.get("span_a"), g.get("span_b")
            if sa is None or sb is None:
                continue
            op_a = build_opinion_per_bin(sa.score, bins_g2)
            op_b = build_opinion_per_bin(sb.score, bins_bm)
            m_a = {"b": op_a.belief, "d": op_a.disbelief, "u": op_a.uncertainty}
            m_b = {"b": op_b.belief, "d": op_b.disbelief, "u": op_b.uncertainty}
            K = compute_conflict_K(m_a, m_b)
            winner = sa if sa.score >= sb.score else sb
            is_correct = any(
                winner.start == gg.start and winner.end == gg.end
                and winner.entity_type == gg.entity_type for gg in sent_golds)
            ds_fused = dempster_combine(m_a, m_b)
            op_a_sl = Opinion(belief=op_a.belief, disbelief=op_a.disbelief,
                              uncertainty=op_a.uncertainty, base_rate=base_rate)
            op_b_sl = Opinion(belief=op_b.belief, disbelief=op_b.disbelief,
                              uncertainty=op_b.uncertainty, base_rate=base_rate)
            sl_fused = cumulative_fuse(op_a_sl, op_b_sl)
            conflict_data.append({
                "K": K, "is_correct": is_correct,
                "ds_decision": mass_to_decision(ds_fused),
                "sl_decision": sl_fused.projected_probability() > sl_t,
            })

    if conflict_data:
        Ks = np.array([c["K"] for c in conflict_data])
        quartiles = np.percentile(Ks, [25, 50, 75])
        q_defs = [("Q1_low", 0, quartiles[0]), ("Q2", quartiles[0], quartiles[1]),
                  ("Q3", quartiles[1], quartiles[2]), ("Q4_high", quartiles[2], 1.0)]
        conflict_results = {}
        for ql, lo, hi in q_defs:
            sub = [c for c in conflict_data if lo <= c["K"] < hi]
            if not sub:
                continue
            n = len(sub)
            ds_tp = sum(c["ds_decision"] and c["is_correct"] for c in sub)
            ds_fp = sum(c["ds_decision"] and not c["is_correct"] for c in sub)
            sl_tp = sum(c["sl_decision"] and c["is_correct"] for c in sub)
            sl_fp = sum(c["sl_decision"] and not c["is_correct"] for c in sub)
            ds_prec = ds_tp / (ds_tp + ds_fp) if (ds_tp + ds_fp) > 0 else 0.0
            sl_prec = sl_tp / (sl_tp + sl_fp) if (sl_tp + sl_fp) > 0 else 0.0
            conflict_results[ql] = {
                "n": n, "K_range": [float(lo), float(hi)],
                "pct_correct": sum(c["is_correct"] for c in sub) / n,
                "ds_precision": ds_prec, "sl_precision": sl_prec,
            }
            print(f"    {ql:10s} K\u2208[{lo:.3f},{hi:.3f}) n={n:5d} "
                  f"DS_prec={ds_prec:.3f} SL_prec={sl_prec:.3f}")
    else:
        conflict_results = {}
        print("    No aligned pairs for conflict analysis")

    return {
        "dataset": "MedMentions-corrected", "n_gold_test": len(all_test_gold),
        "n_sent_test": len(test_data), "n_sent_dev": len(dev_data),
        "conditions": conditions, "bootstrap_ci": ci_results,
        "base_rate_sweep": br_sweep, "empirical_prior": empirical_prior,
        "conflict_quartiles": conflict_results,
    }


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="EN4.2 — DS vs SL")
    parser.add_argument("--phase", choices=["a", "b", "c", "all"], default="a")
    args = parser.parse_args()

    print("=" * 60)
    print("EN4.2 — Dempster-Shafer vs Subjective Logic")
    print("=" * 60)

    all_results = {
        "experiment": "EN4.2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "n_bootstrap": N_BOOTSTRAP,
    }

    if args.phase in ("a", "all"):
        all_results["phase_a"] = run_phase_a()
    if args.phase in ("b", "all"):
        all_results["phase_b"] = run_phase_b()
    if args.phase in ("c", "all"):
        all_results["phase_c"] = run_phase_c()

    primary = RESULTS_DIR / "EN4_2_ds_comparison.json"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = RESULTS_DIR / f"EN4_2_ds_comparison_{ts}.json"
    for p in [primary, archive]:
        with open(p, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {primary}")
    print(f"  Archive: {archive}")


if __name__ == "__main__":
    main()
