"""EN3.4 MedMentions Corrected — use only labels present in gold set.

Fixes:
  1. Only use GLiNER labels for entity types that exist in the gold set
  2. Map labels precisely to the actual UMLS types present
  3. Run full per-bin evaluation

Run: python experiments/EN3/en3_4_mm_corrected.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import Counter
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

from EN3.en3_4_calibration import (
    compute_ece, reliability_diagram_bins,
)
from EN3.en3_4_core import (
    EntitySpan, align_spans,
    apply_condition_single_model,
    apply_condition_union,
    apply_condition_intersection,
    apply_condition_scalar_average,
    apply_condition_sl_fusion,
    apply_condition_sl_fusion_per_bin,
    build_opinion_per_bin,
    evaluate_entities,
    bootstrap_f1_difference_ci,
    cohens_h,
    spearman_conflict_error,
    optimize_threshold,
)
from jsonld_ex.confidence_algebra import (
    cumulative_fuse, conflict_metric,
)

SEED = 42
RESULTS_DIR = _SCRIPT_DIR / "results"
CHECKPOINT_DIR = _SCRIPT_DIR / "checkpoints"

ACCEPT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ABSTENTION_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# ─── Corrected type mappings ───
# The ibm-research/MedMentions-ZS test set contains ONLY these types:
#   T007 → Bacterium
#   T097 → Professional or Occupational Group
#   T168 → Food
#   T031 → Body Substance
#   T022 → Body System
#
# Map to 4 groups matching what's actually in the data:
CORRECTED_TYPE_TO_GROUP = {
    "T007": "Bacterium",
    "T031": "Body Substance",
    "T022": "Body System",
    "T097": "Occupation",
    "T168": "Food",
}

# GLiNER labels matched precisely to the actual gold types
CORRECTED_LABELS_GLINER = [
    "bacterium or bacterial organism",
    "body substance such as blood serum or fluid",
    "body system or organ system",
    "professional occupation or occupational group",
    "food or dietary substance",
]

CORRECTED_GLINER_TO_GROUP = {
    "bacterium or bacterial organism": "Bacterium",
    "body substance such as blood serum or fluid": "Body Substance",
    "body system or organ system": "Body System",
    "professional occupation or occupational group": "Occupation",
    "food or dietary substance": "Food",
}


def _tokens_to_text(tokens):
    offsets = []
    pos = 0
    for tok in tokens:
        offsets.append((pos, pos + len(tok)))
        pos += len(tok) + 1
    return " ".join(tokens), offsets


def load_mm_corrected(split: str) -> List[Dict]:
    """Load MedMentions-ZS with corrected type mapping."""
    from datasets import load_dataset

    print(f"  Loading MedMentions-ZS {split}...")
    ds = load_dataset("ibm-research/MedMentions-ZS", split=split)
    print(f"  Loaded {len(ds)} sentences")

    sentences = []
    for item in ds:
        tokens = item["tokens"]
        raw_tags = item["ner_tags"]
        text, offsets = _tokens_to_text(tokens)

        # Parse BIO tags with corrected mapping
        spans = []
        current_type = None
        current_group = None
        current_start = -1

        for i, tag in enumerate(raw_tags):
            tag_str = str(tag)
            if tag_str.startswith("B-"):
                if current_group is not None:
                    spans.append(EntitySpan(
                        start=current_start, end=offsets[i - 1][1],
                        entity_type=current_group, score=1.0, source="gold",
                    ))
                type_code = tag_str[2:]
                current_type = type_code
                current_group = CORRECTED_TYPE_TO_GROUP.get(type_code)
                current_start = offsets[i][0]
            elif tag_str.startswith("I-") and current_type == tag_str[2:]:
                pass
            else:
                if current_group is not None:
                    spans.append(EntitySpan(
                        start=current_start, end=offsets[i - 1][1],
                        entity_type=current_group, score=1.0, source="gold",
                    ))
                    current_type = None
                    current_group = None

        if current_group is not None and len(offsets) > 0:
            spans.append(EntitySpan(
                start=current_start, end=offsets[-1][1],
                entity_type=current_group, score=1.0, source="gold",
            ))

        sentences.append({
            "tokens": tokens, "text": text, "offsets": offsets,
            "gold_spans": spans,
        })

    n_ents = sum(len(s["gold_spans"]) for s in sentences)
    types = Counter(g.entity_type for s in sentences for g in s["gold_spans"])
    print(f"  {len(sentences)} sentences, {n_ents} gold entities")
    for t, c in types.most_common():
        print(f"    {t}: {c}")
    return sentences


def run_gliner2_corrected(sentences, threshold=0.3):
    """Run GLiNER2 with corrected labels."""
    from gliner2 import GLiNER2

    print(f"  Loading GLiNER2...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print(f"  Running on {len(sentences)} sentences...")

    all_preds = []
    t0 = time.time()
    for i, sent in enumerate(sentences):
        text = sent["text"]
        result = model.extract_entities(
            text, CORRECTED_LABELS_GLINER, threshold=threshold,
            include_confidence=True, include_spans=True,
        )
        spans = []
        entities = result.get("entities", {})
        for label, ent_list in entities.items():
            mapped = CORRECTED_GLINER_TO_GROUP.get(label)
            if mapped is None:
                continue
            for ent in ent_list:
                if isinstance(ent, dict):
                    spans.append(EntitySpan(
                        start=ent["start"], end=ent["end"],
                        entity_type=mapped, score=ent["confidence"],
                        source="gliner2", text=ent.get("text", ""),
                    ))
        all_preds.append(spans)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(sentences)} ({time.time()-t0:.0f}s)")

    print(f"  GLiNER2 complete ({time.time()-t0:.0f}s)")
    return all_preds


def run_biomed_corrected(sentences, threshold=0.3):
    """Run GLiNER-BioMed with corrected labels."""
    from gliner import GLiNER

    biomed_local = Path("E:/data/code/claudecode/jsonld/jsonld-ex/data/gliner-biomed-base-v1.0")
    model_path = str(biomed_local) if biomed_local.exists() else "Ihor/gliner-biomed-base-v1.0"
    print(f"  Loading GLiNER-BioMed from {model_path}...")
    model = GLiNER.from_pretrained(model_path)
    print(f"  Running on {len(sentences)} sentences...")

    all_preds = []
    t0 = time.time()
    for i, sent in enumerate(sentences):
        text = sent["text"]
        entities = model.predict_entities(text, CORRECTED_LABELS_GLINER, threshold=threshold)
        spans = []
        for ent in entities:
            mapped = CORRECTED_GLINER_TO_GROUP.get(ent.get("label", ""))
            if mapped is None:
                continue
            spans.append(EntitySpan(
                start=ent["start"], end=ent["end"],
                entity_type=mapped, score=ent["score"],
                source="biomed", text=ent.get("text", ""),
            ))
        all_preds.append(spans)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(sentences)} ({time.time()-t0:.0f}s)")

    print(f"  BioMed complete ({time.time()-t0:.0f}s)")
    return all_preds


def _save_checkpoint(data, name):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Checkpoint: {path.name}")


def _load_checkpoint(name):
    path = CHECKPOINT_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _preds_to_ser(preds):
    return [[{"start": s.start, "end": s.end, "entity_type": s.entity_type,
              "score": s.score, "source": s.source, "text": s.text or ""}
             for s in sp] for sp in preds]


def _ser_to_preds(data):
    return [[EntitySpan(d["start"], d["end"], d["entity_type"],
                        d["score"], d["source"], d.get("text", ""))
             for d in sd] for sd in data]


def _banner(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")


def _elapsed(t0):
    dt = time.time() - t0
    return f"{dt:.1f}s" if dt < 60 else f"{dt/60:.1f}min"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    set_global_seed(SEED)
    env = log_environment()

    _banner("MedMentions Corrected: 5 Types, Precise Labels")

    # ─── Load data ───
    dev_data = load_mm_corrected("validation")
    test_data = load_mm_corrected("test")

    # ─── Inference or load ───
    if args.skip_inference:
        print("\n  Loading corrected predictions from checkpoints...")
        g2_dev = _ser_to_preds(_load_checkpoint("en3_4_mm_corrected_dev_g2"))
        g2_test = _ser_to_preds(_load_checkpoint("en3_4_mm_corrected_test_g2"))
        bm_dev = _ser_to_preds(_load_checkpoint("en3_4_mm_corrected_dev_bm"))
        bm_test = _ser_to_preds(_load_checkpoint("en3_4_mm_corrected_test_bm"))
    else:
        g2_dev = run_gliner2_corrected(dev_data)
        _save_checkpoint(_preds_to_ser(g2_dev), "en3_4_mm_corrected_dev_g2")
        g2_test = run_gliner2_corrected(test_data)
        _save_checkpoint(_preds_to_ser(g2_test), "en3_4_mm_corrected_test_g2")
        bm_dev = run_biomed_corrected(dev_data)
        _save_checkpoint(_preds_to_ser(bm_dev), "en3_4_mm_corrected_dev_bm")
        bm_test = run_biomed_corrected(test_data)
        _save_checkpoint(_preds_to_ser(bm_test), "en3_4_mm_corrected_test_bm")

    # ─── Calibration on MM dev ───
    _banner("Calibration (MedMentions dev)")
    bins_data = {}
    for model_name, preds in [("gliner2", g2_dev), ("biomed", bm_dev)]:
        scores, correct = [], []
        for sent, sp in zip(dev_data, preds):
            for p in sp:
                scores.append(p.score)
                is_correct = any(
                    p.start == g.start and p.end == g.end
                    and p.entity_type == g.entity_type
                    for g in sent["gold_spans"]
                )
                correct.append(is_correct)
        if scores:
            ece = compute_ece(scores, correct, n_bins=10)
            bins = reliability_diagram_bins(scores, correct, n_bins=10)
            print(f"  {model_name}: ECE={ece:.4f}, N={len(scores)}")
            bins_data[model_name] = bins
        else:
            print(f"  {model_name}: No predictions!")
            bins_data[model_name] = []

    # ─── Threshold optimization ───
    _banner("Threshold Optimization (dev)")
    dev_golds = [s["gold_spans"] for s in dev_data]
    all_dev_gold = [g for sg in dev_golds for g in sg]
    all_dev_g2 = [p for sp in g2_dev for p in sp]
    all_dev_bm = [p for sp in bm_dev for p in sp]

    t_g2, _ = optimize_threshold(all_dev_g2, all_dev_gold, ACCEPT_THRESHOLDS)
    t_bm, _ = optimize_threshold(all_dev_bm, all_dev_gold, ACCEPT_THRESHOLDS)
    t_scalar, _ = optimize_threshold(all_dev_g2 + all_dev_bm, all_dev_gold, ACCEPT_THRESHOLDS)
    print(f"  GLiNER2: t={t_g2}, BioMed: t={t_bm}, Scalar: t={t_scalar}")

    # SL per-bin thresholds
    best_at, best_ct, best_f1_dev = 0.5, 0.5, -1.0
    for at in ACCEPT_THRESHOLDS:
        for ct in ABSTENTION_THRESHOLDS:
            acc = []
            for g2p, bmp in zip(g2_dev, bm_dev):
                groups = align_spans(g2p, bmp, iou_threshold=0.5)
                a, _ = apply_condition_sl_fusion_per_bin(
                    groups, bins_data["gliner2"], bins_data["biomed"], at, ct)
                acc.extend(a)
            m = evaluate_entities(acc, all_dev_gold)
            if m.f1 > best_f1_dev:
                best_f1_dev = m.f1
                best_at, best_ct = at, ct
    print(f"  SL per-bin: accept={best_at}, abstain={best_ct} (dev F1={best_f1_dev:.4f})")

    # ─── Test evaluation ───
    _banner("Test Evaluation (7 conditions)")
    test_golds = [s["gold_spans"] for s in test_data]
    all_test_gold = [g for sg in test_golds for g in sg]

    per_sent = {"B1": [], "B2": [], "B3": [], "B4": [], "B5": [],
                "SL_pb": []}
    all_abstained = []
    conflict_scores, conflict_errors = [], []

    for g2p, bmp, sg in zip(g2_test, bm_test, test_golds):
        per_sent["B1"].append(apply_condition_single_model(g2p, t_g2))
        per_sent["B2"].append(apply_condition_single_model(bmp, t_bm))
        groups = align_spans(g2p, bmp, iou_threshold=0.5)
        per_sent["B3"].append(apply_condition_union(groups, t_g2, t_bm))
        per_sent["B4"].append(apply_condition_intersection(groups, t_g2, t_bm))
        per_sent["B5"].append(apply_condition_scalar_average(groups, t_scalar))
        acc, abst = apply_condition_sl_fusion_per_bin(
            groups, bins_data["gliner2"], bins_data["biomed"], best_at, best_ct)
        per_sent["SL_pb"].append(acc)
        all_abstained.extend(abst)

        for g in groups:
            if g["span_a"] is not None and g["span_b"] is not None:
                op_a = build_opinion_per_bin(g["span_a"].score, bins_data["gliner2"])
                op_b = build_opinion_per_bin(g["span_b"].score, bins_data["biomed"])
                fused = cumulative_fuse(op_a, op_b)
                conf = conflict_metric(fused)
                winner = g["span_a"] if g["span_a"].score >= g["span_b"].score else g["span_b"]
                is_err = not any(
                    winner.start == gld.start and winner.end == gld.end
                    and winner.entity_type == gld.entity_type for gld in sg)
                conflict_scores.append(conf)
                conflict_errors.append(is_err)

    cond_keys = [("B1: GLiNER2", "B1"), ("B2: BioMed", "B2"),
                 ("B3: Union", "B3"), ("B4: Intersection", "B4"),
                 ("B5: Scalar Avg", "B5"), ("SL: Per-bin", "SL_pb")]
    conditions = []
    for name, key in cond_keys:
        flat = [s for sp in per_sent[key] for s in sp]
        m = evaluate_entities(flat, all_test_gold)
        conditions.append((name, m))

    print(f"\n  {'Condition':<20s} {'P':>8s} {'R':>8s} {'F1':>8s} "
          f"{'TP':>6s} {'FP':>6s} {'FN':>6s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")
    for name, m in conditions:
        print(f"  {name:<20s} {m.precision:8.4f} {m.recall:8.4f} {m.f1:8.4f} "
              f"{m.tp:6d} {m.fp:6d} {m.fn:6d}")
    print(f"  SL abstained: {len(all_abstained)}")

    # ─── Bootstrap CI ───
    best_bl = max(conditions[:5], key=lambda x: x[1].f1)
    best_name, best_m = best_bl
    best_key = [k for n, k in cond_keys if n == best_name][0]

    def _tp_fp_fn(key):
        result = []
        for sp, sg in zip(per_sent[key], test_golds):
            m = evaluate_entities(sp, sg)
            result.append((m.tp, m.fp, m.fn))
        return result

    boot = {k: _tp_fp_fn(k) for _, k in cond_keys}

    m_sl = conditions[5][1]
    ci_lo, ci_mean, ci_hi = bootstrap_f1_difference_ci(
        boot["SL_pb"], boot[best_key], n_bootstrap=2000, seed=SEED)
    f1_diff = m_sl.f1 - best_m.f1
    h = cohens_h(m_sl.f1, best_m.f1)
    verdict = "ACCEPTED" if ci_lo > 0 else ("REJECTED" if ci_hi < 0 else "INCONCLUSIVE")

    print(f"\n  Best baseline: {best_name} (F1={best_m.f1:.4f})")
    print(f"  SL Per-bin - Best = {f1_diff:+.4f}, CI [{ci_lo:+.4f}, {ci_hi:+.4f}], "
          f"h={h:.4f} → {verdict}")

    # SL vs scalar
    m_b5 = conditions[4][1]
    ci_lo_s, _, ci_hi_s = bootstrap_f1_difference_ci(
        boot["SL_pb"], boot["B5"], n_bootstrap=2000, seed=SEED)
    print(f"  SL Per-bin - Scalar = {m_sl.f1 - m_b5.f1:+.4f}, "
          f"CI [{ci_lo_s:+.4f}, {ci_hi_s:+.4f}]")

    # Conflict
    if len(conflict_scores) >= 10:
        rho, p = spearman_conflict_error(conflict_scores, conflict_errors)
        print(f"  Conflict-error rho={rho:.4f}, p={p:.6f}")
        try:
            from sklearn.metrics import roc_auc_score
            labels = [1 if e else 0 for e in conflict_errors]
            if len(set(labels)) == 2:
                auroc = roc_auc_score(labels, conflict_scores)
                print(f"  Conflict AUROC: {auroc:.4f}")
        except ImportError:
            pass

    # ─── Per-type breakdown ───
    _banner("Per-Type Breakdown")
    for etype in sorted(set(g.entity_type for g in all_test_gold)):
        tg = [g for g in all_test_gold if g.entity_type == etype]
        for cname, m_key in [("BioMed", "B2"), ("SL", "SL_pb")]:
            tp_flat = [p for sp in per_sent[m_key] for p in sp if p.entity_type == etype]
            m = evaluate_entities(tp_flat, tg)
            print(f"  {etype:15s} ({cname:6s}): Gold={len(tg):4d}, "
                  f"Pred={len(tp_flat):4d}, P={m.precision:.3f}, R={m.recall:.3f}, F1={m.f1:.3f}")

    # ─── Save ───
    result_metrics = {
        "dataset": "MedMentions-ZS-corrected",
        "n_gold_types": 5,
        "labels_used": CORRECTED_LABELS_GLINER,
        "conditions": {n: {"p": m.precision, "r": m.recall, "f1": m.f1,
                           "tp": m.tp, "fp": m.fp, "fn": m.fn}
                       for n, m in conditions},
        "best_baseline": best_name,
        "sl_f1_minus_best": f1_diff,
        "bootstrap_ci": {"lo": ci_lo, "mean": ci_mean, "hi": ci_hi},
        "verdict": verdict,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = ExperimentResult(
        experiment_id="EN3.4-MM-corrected",
        parameters={"seed": SEED, "n_types": 5},
        metrics=result_metrics,
        environment=env,
    )
    result.save_json(str(RESULTS_DIR / "EN3_4_mm_corrected.json"))
    print(f"\n  Saved: EN3_4_mm_corrected.json")


if __name__ == "__main__":
    main()
