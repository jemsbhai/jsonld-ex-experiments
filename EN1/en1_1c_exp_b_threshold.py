#!/usr/bin/env python
"""EN1.1c Experiment B -- GLiNER2 Threshold Sweep.

Experiment C showed that O-token confidence has zero effect on fusion.
The damage comes from GLiNER2's entity predictions: 55% are wrong,
and even with T=9.82 calibration, they tip fusion toward incorrect tags.

This experiment re-generates GLiNER2 predictions at multiple thresholds
(0.3 to 0.9) and runs the S4 (1:2) and S6 (3:2) subsets for each.
Higher thresholds produce fewer but more accurate entity predictions.

The question: does reducing GLiNER2's false entity rate recover SL
fusion performance?

Thresholds tested:
    0.3 (original), 0.5 (GLiNER2 default), 0.7, 0.8, 0.9

For each threshold:
    - Load raw GLiNER2 checkpoint (threshold=0.3)
    - Apply threshold: entity predictions below threshold become O
    - Re-calibrate (entity-only T + O-conf=0.95)
    - Run S4 and S6 subsets (B, D, E2, G2)

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN1/en1_1c_exp_b_threshold.py

Output:
    experiments/EN1/results/en1_1c_exp_b_results.json
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize as sp_optimize

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "packages" / "python" / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    trust_discount,
)

from experiments.infra.config import set_global_seed
from experiments.infra.results import ExperimentResult
from experiments.infra.env_log import log_environment

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
GLOBAL_SEED = 42
N_BOOTSTRAP = 1000
EVIDENCE_WEIGHT = 10

ENTITY_TYPES = ["PER", "LOC", "ORG", "MISC"]
ALL_TAGS = ["O"] + [f"{p}-{e}" for e in ENTITY_TYPES for p in ["B", "I"]]
ALL_MODELS = ["spacy", "flair", "stanza", "huggingface", "gliner2"]

GLINER2_O_CONFIDENCE = 0.95
THRESHOLDS = [0.3, 0.5, 0.7, 0.8, 0.9]

# Test on these subsets (the scientifically interesting ones)
SUBSETS = {
    "S4_2strong_1weak_gliner2": ["flair", "huggingface", "gliner2"],
    "S5_original_4model": ["spacy", "flair", "stanza", "huggingface"],
    "S6_all_5model": ["spacy", "flair", "stanza", "huggingface", "gliner2"],
}


# =====================================================================
# Data Loading
# =====================================================================

def load_checkpoint(name: str) -> Any:
    path = CHECKPOINT_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_conll2003(split: str) -> List[Dict]:
    from datasets import load_dataset
    tag_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
                 "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    sources = [
        ("eriktks/conll2003", {"revision": "refs/convert/parquet"}),
        ("conll2003", {"revision": "refs/convert/parquet"}),
        ("DFKI-SLT/conll2003", {}),
    ]
    for source, kwargs in sources:
        try:
            ds = load_dataset(source, split=split, **kwargs)
            break
        except Exception:
            continue
    else:
        raise RuntimeError("Could not load CoNLL-2003")
    sentences = []
    for item in ds:
        tokens = item["tokens"]
        raw_tags = item["ner_tags"]
        tags = [tag_names[t] for t in raw_tags] if raw_tags and isinstance(raw_tags[0], int) else list(raw_tags)
        sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences


# =====================================================================
# Threshold Application
# =====================================================================

def apply_threshold(preds, threshold, o_confidence=GLINER2_O_CONFIDENCE):
    """Apply a confidence threshold to entity predictions.

    Entity predictions with confidence < threshold become O predictions.
    O predictions get fixed confidence = o_confidence.

    This simulates running GLiNER2 at a higher threshold without
    re-running the model — since we saved predictions at threshold=0.3,
    all entities at higher thresholds are a subset.
    """
    filtered = []
    n_converted = 0
    n_kept = 0

    for sent in preds:
        new_sent = []
        for p in sent:
            if p["tag"] == "O":
                new_sent.append({"tag": "O", "confidence": o_confidence})
            elif p["confidence"] < threshold:
                # Below threshold: convert to O
                new_sent.append({"tag": "O", "confidence": o_confidence})
                n_converted += 1
            else:
                # Above threshold: keep entity prediction
                new_sent.append({"tag": p["tag"], "confidence": p["confidence"]})
                n_kept += 1
        filtered.append(new_sent)

    return filtered, n_kept, n_converted


def compute_entity_accuracy(preds, gold):
    """Compute accuracy on entity-predicted tokens only."""
    correct = 0
    total = 0
    for sent_preds, sent_gold in zip(preds, gold):
        for pred, gt in zip(sent_preds, sent_gold["ner_tags"]):
            if pred["tag"] != "O":
                total += 1
                if pred["tag"] == gt:
                    correct += 1
    return correct / total if total > 0 else 0.0, total


# =====================================================================
# Calibration
# =====================================================================

def calibrate_temperature_all(preds, gold):
    confidences, corrects = [], []
    for sent_preds, sent_gold in zip(preds, gold):
        for pred, gt in zip(sent_preds, sent_gold["ner_tags"]):
            confidences.append(pred["confidence"])
            corrects.append(1.0 if pred["tag"] == gt else 0.0)
    confidences = np.array(confidences)
    corrects = np.array(corrects)
    eps = 1e-10

    def nll(T):
        logits = np.log(np.clip(confidences, eps, 1 - eps) / np.clip(1 - confidences, eps, 1 - eps))
        scaled = 1 / (1 + np.exp(-logits / T))
        return -np.mean(corrects * np.log(np.clip(scaled, eps, None)) +
                        (1 - corrects) * np.log(np.clip(1 - scaled, eps, None)))

    result = sp_optimize.minimize_scalar(nll, bounds=(0.1, 20.0), method="bounded")
    return float(result.x)


def calibrate_temperature_entity_only(preds, gold):
    confidences, corrects = [], []
    for sent_preds, sent_gold in zip(preds, gold):
        for pred, gt in zip(sent_preds, sent_gold["ner_tags"]):
            if pred["tag"] != "O":
                confidences.append(pred["confidence"])
                corrects.append(1.0 if pred["tag"] == gt else 0.0)
    if len(confidences) < 50:
        return 1.0  # Not enough entity tokens; use identity
    confidences = np.array(confidences)
    corrects = np.array(corrects)
    eps = 1e-10

    def nll(T):
        logits = np.log(np.clip(confidences, eps, 1 - eps) / np.clip(1 - confidences, eps, 1 - eps))
        scaled = 1 / (1 + np.exp(-logits / T))
        return -np.mean(corrects * np.log(np.clip(scaled, eps, None)) +
                        (1 - corrects) * np.log(np.clip(1 - scaled, eps, None)))

    result = sp_optimize.minimize_scalar(nll, bounds=(0.1, 20.0), method="bounded")
    return float(result.x)


def apply_temperature(preds, T):
    eps = 1e-10
    calibrated = []
    for sent in preds:
        cal = []
        for p in sent:
            c = p["confidence"]
            logit = math.log(max(c, eps) / max(1 - c, eps))
            cal_c = 1.0 / (1.0 + math.exp(-logit / T))
            cal.append({"tag": p["tag"], "confidence": cal_c})
        calibrated.append(cal)
    return calibrated


def apply_gliner2_entity_calibration(preds, T_entity, o_confidence):
    eps = 1e-10
    calibrated = []
    for sent in preds:
        cal = []
        for p in sent:
            if p["tag"] == "O":
                cal.append({"tag": "O", "confidence": o_confidence})
            else:
                c = p["confidence"]
                logit = math.log(max(c, eps) / max(1 - c, eps))
                cal_c = 1.0 / (1.0 + math.exp(-logit / T_entity))
                cal.append({"tag": p["tag"], "confidence": cal_c})
        calibrated.append(cal)
    return calibrated


# =====================================================================
# Fusion (same as ablation)
# =====================================================================

def opinion_from_model_confidence(conf: float) -> Opinion:
    pos = conf * EVIDENCE_WEIGHT
    neg = (1.0 - conf) * EVIDENCE_WEIGHT
    return Opinion.from_evidence(pos, neg, prior_weight=2.0, base_rate=0.5)


def _fuse_per_tag(model_preds, sent_idx, tok_idx, model_names, trust_opinions=None):
    tag_opinions = defaultdict(list)
    for m in model_names:
        preds = model_preds[m][sent_idx]
        if tok_idx < len(preds):
            tag = preds[tok_idx]["tag"]
            conf = preds[tok_idx]["confidence"]
            op = opinion_from_model_confidence(conf)
            if trust_opinions is not None:
                op = trust_discount(trust_opinions[m], op)
            tag_opinions[tag].append(op)
    result = {}
    for tag, opinions in tag_opinions.items():
        fused = opinions[0] if len(opinions) == 1 else cumulative_fuse(*opinions)
        result[tag] = (fused, len(opinions))
    return result


def _select_best_tag(tag_fused, n_models):
    scored = []
    for tag, (opinion, n_support) in tag_fused.items():
        pp = opinion.projected_probability()
        score = pp * (n_support / n_models)
        scored.append((tag, score, opinion))
    scored.sort(key=lambda x: x[1], reverse=True)
    best_tag, best_score, best_opinion = scored[0]
    second_tag = scored[1][0] if len(scored) > 1 else None
    return best_tag, best_score, best_opinion, second_tag, 0.0


def run_sl_fuse(model_preds, sentences, model_names, trust_opinions=None):
    result = []
    for sent_idx in range(len(sentences)):
        n_tok = len(sentences[sent_idx]["tokens"])
        sent_tags = []
        for tok_idx in range(n_tok):
            tag_fused = _fuse_per_tag(model_preds, sent_idx, tok_idx, model_names, trust_opinions)
            best_tag, _, _, _, _ = _select_best_tag(tag_fused, len(model_names))
            sent_tags.append(best_tag)
        result.append(sent_tags)
    return result


def run_scalar_weighted(model_preds, sentences, model_names, model_weights):
    result = []
    for si in range(len(sentences)):
        nt = len(sentences[si]["tokens"])
        st = []
        for ti in range(nt):
            ts = defaultdict(float)
            for mn in model_names:
                ps = model_preds[mn][si]
                if ti < len(ps):
                    ts[ps[ti]["tag"]] += ps[ti]["confidence"] * model_weights[mn]
            st.append(max(ts, key=ts.get))
        result.append(st)
    return result


def run_stacking(dev_preds, test_preds, dev_data, test_data, model_names):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(ALL_TAGS)
    X_tr, y_tr = [], []
    for si in range(len(dev_data)):
        gt = dev_data[si]["ner_tags"]
        for ti in range(len(gt)):
            feat = []
            for mn in model_names:
                ps = dev_preds[mn][si]
                if ti < len(ps):
                    te = [0.0] * len(ALL_TAGS)
                    if ps[ti]["tag"] in ALL_TAGS:
                        te[ALL_TAGS.index(ps[ti]["tag"])] = 1.0
                    feat.extend(te)
                    feat.append(ps[ti]["confidence"])
                else:
                    feat.extend([0.0] * (len(ALL_TAGS) + 1))
            X_tr.append(feat)
            y_tr.append(gt[ti])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=GLOBAL_SEED)
        clf.fit(np.array(X_tr), le.transform(y_tr))
    pred_D = []
    for si in range(len(test_data)):
        feats = []
        for ti in range(len(test_data[si]["tokens"])):
            feat = []
            for mn in model_names:
                ps = test_preds[mn][si]
                if ti < len(ps):
                    te = [0.0] * len(ALL_TAGS)
                    if ps[ti]["tag"] in ALL_TAGS:
                        te[ALL_TAGS.index(ps[ti]["tag"])] = 1.0
                    feat.extend(te)
                    feat.append(ps[ti]["confidence"])
                else:
                    feat.extend([0.0] * (len(ALL_TAGS) + 1))
            feats.append(feat)
        pred_D.append(list(le.inverse_transform(clf.predict(np.array(feats)))))
    return pred_D


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_predictions(predictions, gold):
    from seqeval.metrics import f1_score, precision_score, recall_score
    gold_tags = [s["ner_tags"] for s in gold]
    pred_aligned, gold_aligned = [], []
    for p, g in zip(predictions, gold_tags):
        n = min(len(p), len(g))
        pred_aligned.append(p[:n])
        gold_aligned.append(g[:n])
    return {
        "entity_f1": f1_score(gold_aligned, pred_aligned),
        "entity_precision": precision_score(gold_aligned, pred_aligned),
        "entity_recall": recall_score(gold_aligned, pred_aligned),
    }


def bootstrap_entity_f1(predictions, gold, n_bootstrap=N_BOOTSTRAP, seed=GLOBAL_SEED):
    from seqeval.metrics import f1_score as seq_f1
    gold_tags = [s["ner_tags"] for s in gold]
    n = len(predictions)
    rng = np.random.RandomState(seed)
    f1s = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bp = [predictions[i][:len(gold_tags[i])] for i in idx]
        bg = [gold_tags[i] for i in idx]
        try:
            f1s.append(seq_f1(bg, bp))
        except Exception:
            pass
    if not f1s:
        return (0.0, 0.0, 0.0)
    return (float(np.percentile(f1s, 2.5)), float(np.mean(f1s)), float(np.percentile(f1s, 97.5)))


def compute_model_dev_f1(preds, gold):
    from seqeval.metrics import f1_score
    pred_tags = [[p["tag"] for p in sent] for sent in preds]
    gold_tags = [s["ner_tags"] for s in gold]
    ap, ag = [], []
    for p, g in zip(pred_tags, gold_tags):
        n = min(len(p), len(g))
        ap.append(p[:n])
        ag.append(g[:n])
    return f1_score(ag, ap)


# =====================================================================
# Run one subset
# =====================================================================

def run_subset(model_names, dev_preds_cal, test_preds_cal, dev_data, test_data, dev_f1s):
    subset_weights = {n: dev_f1s[n] for n in model_names}
    tw = sum(subset_weights.values())
    subset_weights = {n: w / tw for n, w in subset_weights.items()}

    subset_trust = {}
    for name in model_names:
        f1 = dev_f1s[name]
        subset_trust[name] = Opinion(
            belief=f1 * 0.9,
            disbelief=(1 - f1) * 0.5,
            uncertainty=1.0 - f1 * 0.9 - (1 - f1) * 0.5,
            base_rate=0.5,
        )

    dev_sub = {n: dev_preds_cal[n] for n in model_names}
    test_sub = {n: test_preds_cal[n] for n in model_names}

    results = {}

    # B
    pred_B = run_scalar_weighted(test_sub, test_data, model_names, subset_weights)
    m_B = evaluate_predictions(pred_B, test_data)
    ci_B = bootstrap_entity_f1(pred_B, test_data)
    m_B["entity_f1_ci"] = {"lower": ci_B[0], "mean": ci_B[1], "upper": ci_B[2]}
    results["B"] = m_B

    # D
    pred_D = run_stacking(dev_sub, test_sub, dev_data, test_data, model_names)
    m_D = evaluate_predictions(pred_D, test_data)
    ci_D = bootstrap_entity_f1(pred_D, test_data)
    m_D["entity_f1_ci"] = {"lower": ci_D[0], "mean": ci_D[1], "upper": ci_D[2]}
    results["D"] = m_D

    # E2
    pred_E2 = run_sl_fuse(test_sub, test_data, model_names)
    m_E2 = evaluate_predictions(pred_E2, test_data)
    ci_E2 = bootstrap_entity_f1(pred_E2, test_data)
    m_E2["entity_f1_ci"] = {"lower": ci_E2[0], "mean": ci_E2[1], "upper": ci_E2[2]}
    results["E2"] = m_E2

    # G2
    pred_G2 = run_sl_fuse(test_sub, test_data, model_names, trust_opinions=subset_trust)
    m_G2 = evaluate_predictions(pred_G2, test_data)
    ci_G2 = bootstrap_entity_f1(pred_G2, test_data)
    m_G2["entity_f1_ci"] = {"lower": ci_G2[0], "mean": ci_G2[1], "upper": ci_G2[2]}
    results["G2"] = m_G2

    results["delta_G2_vs_B"] = m_G2["entity_f1"] - m_B["entity_f1"]
    results["delta_E2_vs_B"] = m_E2["entity_f1"] - m_B["entity_f1"]

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("EN1.1c Experiment B -- GLiNER2 Threshold Sweep")
    print("=" * 70)

    t_start = time.time()
    env = log_environment()
    set_global_seed(GLOBAL_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("\n--- Loading data ---")
    dev_data = load_conll2003("validation")
    test_data = load_conll2003("test")
    print(f"  Dev: {len(dev_data)}, Test: {len(test_data)} sentences")

    # ── Load raw checkpoints ──
    print("\n--- Loading checkpoints ---")
    raw_dev_all, raw_test_all = {}, {}
    for name in ALL_MODELS:
        raw_dev_all[name] = load_checkpoint(f"dev_preds_{name}")
        raw_test_all[name] = load_checkpoint(f"test_preds_{name}")

    # ── Calibrate non-GLiNER2 models once ──
    print("\n--- Calibrating non-GLiNER2 models (once) ---")
    base_temps = {}
    cal_dev_base, cal_test_base = {}, {}
    for name in ["spacy", "flair", "stanza", "huggingface"]:
        T = calibrate_temperature_all(raw_dev_all[name], dev_data)
        base_temps[name] = T
        cal_dev_base[name] = apply_temperature(raw_dev_all[name], T)
        cal_test_base[name] = apply_temperature(raw_test_all[name], T)
        print(f"  {name}: T={T:.4f}")

    # Non-GLiNER2 dev F1s (constant across thresholds)
    base_dev_f1s = {}
    for name in ["spacy", "flair", "stanza", "huggingface"]:
        base_dev_f1s[name] = compute_model_dev_f1(cal_dev_base[name], dev_data)

    # ── Sweep thresholds ──
    all_threshold_results = {}

    for threshold in THRESHOLDS:
        print(f"\n{'=' * 70}")
        print(f"  THRESHOLD = {threshold}")
        print(f"{'=' * 70}")

        # Apply threshold to GLiNER2 raw predictions
        dev_gliner2_filtered, dev_kept, dev_converted = apply_threshold(
            copy.deepcopy(raw_dev_all["gliner2"]), threshold)
        test_gliner2_filtered, test_kept, test_converted = apply_threshold(
            copy.deepcopy(raw_test_all["gliner2"]), threshold)

        # Entity accuracy at this threshold
        dev_ent_acc, dev_ent_count = compute_entity_accuracy(dev_gliner2_filtered, dev_data)
        test_ent_acc, test_ent_count = compute_entity_accuracy(test_gliner2_filtered, test_data)

        print(f"  Dev:  {dev_kept} entities kept, {dev_converted} converted to O")
        print(f"  Test: {test_kept} entities kept, {test_converted} converted to O")
        print(f"  Dev entity accuracy:  {dev_ent_acc:.3f} ({dev_ent_count} tokens)")
        print(f"  Test entity accuracy: {test_ent_acc:.3f} ({test_ent_count} tokens)")

        # Calibrate GLiNER2 at this threshold (entity-only T + O-conf)
        T_gliner2 = calibrate_temperature_entity_only(dev_gliner2_filtered, dev_data)
        print(f"  GLiNER2 T (entity-only): {T_gliner2:.4f}")

        cal_dev_gliner2 = apply_gliner2_entity_calibration(
            dev_gliner2_filtered, T_gliner2, GLINER2_O_CONFIDENCE)
        cal_test_gliner2 = apply_gliner2_entity_calibration(
            test_gliner2_filtered, T_gliner2, GLINER2_O_CONFIDENCE)

        # GLiNER2 dev F1 at this threshold
        gliner2_dev_f1 = compute_model_dev_f1(cal_dev_gliner2, dev_data)
        print(f"  GLiNER2 dev F1: {gliner2_dev_f1:.4f}")

        # Build complete calibrated pred dicts
        cal_dev = dict(cal_dev_base)
        cal_dev["gliner2"] = cal_dev_gliner2
        cal_test = dict(cal_test_base)
        cal_test["gliner2"] = cal_test_gliner2

        dev_f1s = dict(base_dev_f1s)
        dev_f1s["gliner2"] = gliner2_dev_f1

        # Run subsets
        threshold_results = {
            "threshold": threshold,
            "gliner2_temperature": T_gliner2,
            "gliner2_dev_f1": gliner2_dev_f1,
            "dev_entities_kept": dev_kept,
            "dev_entities_converted": dev_converted,
            "dev_entity_accuracy": dev_ent_acc,
            "test_entities_kept": test_kept,
            "test_entities_converted": test_converted,
            "test_entity_accuracy": test_ent_acc,
            "subsets": {},
        }

        for subset_name, model_names in SUBSETS.items():
            print(f"\n  --- {subset_name} ---")
            t_sub = time.time()
            results = run_subset(
                model_names, cal_dev, cal_test, dev_data, test_data, dev_f1s)
            elapsed = time.time() - t_sub

            threshold_results["subsets"][subset_name] = results

            print(f"    B  f1={results['B']['entity_f1']:.4f}  "
                  f"prec={results['B']['entity_precision']:.4f}")
            print(f"    D  f1={results['D']['entity_f1']:.4f}  "
                  f"prec={results['D']['entity_precision']:.4f}")
            print(f"    E2 f1={results['E2']['entity_f1']:.4f}  "
                  f"prec={results['E2']['entity_precision']:.4f}  "
                  f"delta_vs_B: {results['delta_E2_vs_B']:+.4f}")
            print(f"    G2 f1={results['G2']['entity_f1']:.4f}  "
                  f"prec={results['G2']['entity_precision']:.4f}  "
                  f"delta_vs_B: {results['delta_G2_vs_B']:+.4f}")
            print(f"    Time: {elapsed:.1f}s")

        all_threshold_results[f"t={threshold}"] = threshold_results

    # ── Summary Tables ──
    total_time = time.time() - t_start

    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP SUMMARY")
    print("=" * 70)

    # GLiNER2 profile per threshold
    print(f"\n  GLiNER2 profile by threshold:")
    print(f"  {'Thresh':>7s}  {'Ent kept':>8s}  {'Ent acc':>7s}  {'T':>7s}  {'Dev F1':>7s}")
    print(f"  {'-'*42}")
    for threshold in THRESHOLDS:
        r = all_threshold_results[f"t={threshold}"]
        print(f"  {threshold:>7.1f}  {r['test_entities_kept']:>8d}  "
              f"{r['test_entity_accuracy']:>7.3f}  {r['gliner2_temperature']:>7.4f}  "
              f"{r['gliner2_dev_f1']:>7.4f}")

    # S4 results
    print(f"\n  S4 (Flair + HF + GLiNER2) -- 1 weak : 2 strong:")
    print(f"  {'Thresh':>7s}  {'B F1':>7s}  {'D F1':>7s}  {'E2 F1':>7s}  {'G2 F1':>7s}  {'G2-B':>7s}")
    print(f"  {'-'*50}")
    for threshold in THRESHOLDS:
        r = all_threshold_results[f"t={threshold}"]["subsets"]["S4_2strong_1weak_gliner2"]
        print(f"  {threshold:>7.1f}  {r['B']['entity_f1']:>7.4f}  "
              f"{r['D']['entity_f1']:>7.4f}  {r['E2']['entity_f1']:>7.4f}  "
              f"{r['G2']['entity_f1']:>7.4f}  {r['delta_G2_vs_B']:>+7.4f}")

    # S6 results
    print(f"\n  S6 (all 5 models) -- 3 weak : 2 strong:")
    print(f"  {'Thresh':>7s}  {'B F1':>7s}  {'D F1':>7s}  {'E2 F1':>7s}  {'G2 F1':>7s}  {'G2-B':>7s}")
    print(f"  {'-'*50}")
    for threshold in THRESHOLDS:
        r = all_threshold_results[f"t={threshold}"]["subsets"]["S6_all_5model"]
        print(f"  {threshold:>7.1f}  {r['B']['entity_f1']:>7.4f}  "
              f"{r['D']['entity_f1']:>7.4f}  {r['E2']['entity_f1']:>7.4f}  "
              f"{r['G2']['entity_f1']:>7.4f}  {r['delta_G2_vs_B']:>+7.4f}")

    # S5 reference (no GLiNER2, should be constant)
    print(f"\n  S5 reference (4-model, no GLiNER2) -- should be constant:")
    for threshold in THRESHOLDS:
        r = all_threshold_results[f"t={threshold}"]["subsets"]["S5_original_4model"]
        print(f"  t={threshold}: G2={r['G2']['entity_f1']:.4f}  B={r['B']['entity_f1']:.4f}")

    # Key question
    print(f"\n{'=' * 70}")
    print("KEY QUESTION: Does higher threshold recover SL performance?")
    print("=" * 70)
    best_s6_g2 = max(
        (all_threshold_results[f"t={t}"]["subsets"]["S6_all_5model"]["G2"]["entity_f1"], t)
        for t in THRESHOLDS
    )
    s5_g2 = all_threshold_results[f"t={THRESHOLDS[0]}"]["subsets"]["S5_original_4model"]["G2"]["entity_f1"]
    print(f"  Best S6 G2: {best_s6_g2[0]:.4f} at threshold={best_s6_g2[1]}")
    print(f"  S5 G2 (target): {s5_g2:.4f}")
    print(f"  Gap: {best_s6_g2[0] - s5_g2:+.4f}")

    print(f"\n  Total time: {total_time:.1f}s")

    # ── Save ──
    output_path = RESULTS_DIR / "en1_1c_exp_b_results.json"
    experiment_result = ExperimentResult(
        experiment_id="EN1.1c-ExpB",
        parameters={
            "global_seed": GLOBAL_SEED,
            "thresholds": THRESHOLDS,
            "subsets": {k: v for k, v in SUBSETS.items()},
            "evidence_weight": EVIDENCE_WEIGHT,
            "base_temperatures": base_temps,
            "gliner2_o_confidence": GLINER2_O_CONFIDENCE,
            "n_bootstrap": N_BOOTSTRAP,
        },
        metrics={
            "threshold_results": all_threshold_results,
            "total_wall_time_seconds": round(total_time, 2),
        },
        environment=env,
        notes=(
            "EN1.1c Experiment B: GLiNER2 threshold sweep. Tests whether "
            "reducing GLiNER2's false entity rate (by raising threshold from "
            "0.3 to 0.9) recovers SL fusion performance. Uses entity-only "
            f"calibration + O-conf={GLINER2_O_CONFIDENCE}. Runs S4 (1:2), "
            "S5 (control), and S6 (3:2) at each threshold."
        ),
    )
    experiment_result.save_json(str(output_path))
    print(f"\n  Results saved to: {output_path}")

    ts = experiment_result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    archive_path = RESULTS_DIR / f"en1_1c_exp_b_results_{ts}.json"
    experiment_result.save_json(str(archive_path))
    print(f"  Archived to:      {archive_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
