#!/usr/bin/env python
"""EN1.1c Experiment C -- Entity-Only Calibration for GLiNER2.

GLiNER2's span-matching architecture does not produce O-token scores.
Our runner assigned O-conf=0.50, which creates a two-population problem
for temperature scaling:
  - O tokens: 73% of tokens, conf=0.50, actual accuracy=98.9%
  - Entity tokens: 27% of tokens, conf=0.30-1.00, actual accuracy=45%

Temperature scaling (logit/T) cannot independently adjust these populations
because logit(0.50)=0 is invariant to T. Result: T=9.82 destroys entity
discrimination to minimize NLL across both populations.

This experiment uses entity-only calibration:
  - GLiNER2 O-tokens: fixed conf=0.95 (matching actual ~98.9% accuracy)
  - GLiNER2 entity tokens: T calibrated on entity tokens only
  - Other 4 models: unchanged (standard all-token temperature scaling)

All 6 subsets from the ablation are re-run for clean comparison.

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN1/en1_1c_exp_c_entity_cal.py

Output:
    experiments/EN1/results/en1_1c_exp_c_results.json
"""

from __future__ import annotations

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

SUBSETS = {
    "S1_top2_strong": ["flair", "huggingface"],
    "S2_2strong_1weak_spacy": ["flair", "huggingface", "spacy"],
    "S3_2strong_1weak_stanza": ["flair", "huggingface", "stanza"],
    "S4_2strong_1weak_gliner2": ["flair", "huggingface", "gliner2"],
    "S5_original_4model": ["spacy", "flair", "stanza", "huggingface"],
    "S6_all_5model": ["spacy", "flair", "stanza", "huggingface", "gliner2"],
}

# Fixed O-token confidence for GLiNER2 (Experiment C)
GLINER2_O_CONFIDENCE = 0.95


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
# Calibration
# =====================================================================

def calibrate_temperature_all(preds, gold):
    """Standard all-token temperature calibration (for non-GLiNER2 models)."""
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
    """Entity-only temperature calibration for GLiNER2.

    Only uses tokens where the model predicted an entity tag (not O).
    This avoids the two-population problem: O-tokens with conf=0.50
    have logit=0, which is invariant to T and contributes nothing to
    calibration but distorts the optimization landscape.
    """
    confidences, corrects = [], []
    for sent_preds, sent_gold in zip(preds, gold):
        for pred, gt in zip(sent_preds, sent_gold["ner_tags"]):
            if pred["tag"] != "O":  # Entity predictions only
                confidences.append(pred["confidence"])
                corrects.append(1.0 if pred["tag"] == gt else 0.0)

    if len(confidences) < 100:
        print("    WARNING: fewer than 100 entity tokens for calibration")

    confidences = np.array(confidences)
    corrects = np.array(corrects)
    eps = 1e-10

    def nll(T):
        logits = np.log(np.clip(confidences, eps, 1 - eps) / np.clip(1 - confidences, eps, 1 - eps))
        scaled = 1 / (1 + np.exp(-logits / T))
        return -np.mean(corrects * np.log(np.clip(scaled, eps, None)) +
                        (1 - corrects) * np.log(np.clip(1 - scaled, eps, None)))

    result = sp_optimize.minimize_scalar(nll, bounds=(0.1, 20.0), method="bounded")

    # Report calibration stats
    T = float(result.x)
    ent_acc = corrects.mean()
    ent_conf = confidences.mean()
    print(f"    Entity-only calibration: {len(confidences)} tokens, "
          f"acc={ent_acc:.3f}, mean_conf={ent_conf:.3f}, T={T:.4f}")

    return T


def apply_temperature(preds, T):
    """Apply temperature scaling to all tokens."""
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
    """Apply entity-only temperature to entity tokens, fixed conf to O tokens.

    This is the key Experiment C calibration:
    - Entity tokens: temperature scale with T_entity
    - O tokens: set to o_confidence (not temperature scaled)
    """
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
# Fusion (same as ablation script)
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
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    return best_tag, best_score, best_opinion, second_tag, second_score


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

def run_subset(subset_name, model_names, dev_preds_cal, test_preds_cal,
               dev_data, test_data, dev_f1s):
    n_models = len(model_names)

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

    results = {"models": model_names, "n_models": n_models}

    # B: Scalar weighted
    pred_B = run_scalar_weighted(test_sub, test_data, model_names, subset_weights)
    m_B = evaluate_predictions(pred_B, test_data)
    ci_B = bootstrap_entity_f1(pred_B, test_data)
    m_B["entity_f1_ci"] = {"lower": ci_B[0], "mean": ci_B[1], "upper": ci_B[2]}
    results["B_scalar_weighted"] = m_B

    # D: Stacking
    pred_D = run_stacking(dev_sub, test_sub, dev_data, test_data, model_names)
    m_D = evaluate_predictions(pred_D, test_data)
    ci_D = bootstrap_entity_f1(pred_D, test_data)
    m_D["entity_f1_ci"] = {"lower": ci_D[0], "mean": ci_D[1], "upper": ci_D[2]}
    results["D_stacking"] = m_D

    # E2: SL evidence fuse
    pred_E2 = run_sl_fuse(test_sub, test_data, model_names)
    m_E2 = evaluate_predictions(pred_E2, test_data)
    ci_E2 = bootstrap_entity_f1(pred_E2, test_data)
    m_E2["entity_f1_ci"] = {"lower": ci_E2[0], "mean": ci_E2[1], "upper": ci_E2[2]}
    results["E2_sl_evidence"] = m_E2

    # G2: SL evidence + trust
    pred_G2 = run_sl_fuse(test_sub, test_data, model_names, trust_opinions=subset_trust)
    m_G2 = evaluate_predictions(pred_G2, test_data)
    ci_G2 = bootstrap_entity_f1(pred_G2, test_data)
    m_G2["entity_f1_ci"] = {"lower": ci_G2[0], "mean": ci_G2[1], "upper": ci_G2[2]}
    results["G2_sl_trust"] = m_G2

    # Deltas
    results["delta_E2_vs_B"] = {
        "f1": m_E2["entity_f1"] - m_B["entity_f1"],
        "precision": m_E2["entity_precision"] - m_B["entity_precision"],
    }
    results["delta_G2_vs_B"] = {
        "f1": m_G2["entity_f1"] - m_B["entity_f1"],
        "precision": m_G2["entity_precision"] - m_B["entity_precision"],
    }
    results["trust_benefit"] = {
        "f1": m_G2["entity_f1"] - m_E2["entity_f1"],
        "precision": m_G2["entity_precision"] - m_E2["entity_precision"],
    }

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("EN1.1c Experiment C -- Entity-Only Calibration")
    print("=" * 70)
    print(f"  GLiNER2 calibration: entity-only temperature + O-conf={GLINER2_O_CONFIDENCE}")
    print(f"  Other models: standard all-token temperature scaling")

    t_start = time.time()
    env = log_environment()
    set_global_seed(GLOBAL_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("\n--- Loading data ---")
    dev_data = load_conll2003("validation")
    test_data = load_conll2003("test")
    print(f"  Dev: {len(dev_data)}, Test: {len(test_data)} sentences")

    # ── Load all checkpoints ──
    print("\n--- Loading all model checkpoints ---")
    raw_dev_preds, raw_test_preds = {}, {}
    for name in ALL_MODELS:
        raw_dev_preds[name] = load_checkpoint(f"dev_preds_{name}")
        raw_test_preds[name] = load_checkpoint(f"test_preds_{name}")

    # ── Step 1: Remap GLiNER2 O-token confidence BEFORE calibration ──
    print("\n--- Step 1: Remap GLiNER2 O-tokens ---")
    n_remapped_dev = 0
    for sent in raw_dev_preds["gliner2"]:
        for p in sent:
            if p["tag"] == "O":
                p["confidence"] = GLINER2_O_CONFIDENCE
                n_remapped_dev += 1

    n_remapped_test = 0
    for sent in raw_test_preds["gliner2"]:
        for p in sent:
            if p["tag"] == "O":
                p["confidence"] = GLINER2_O_CONFIDENCE
                n_remapped_test += 1

    print(f"  Remapped GLiNER2 O-tokens: dev={n_remapped_dev}, test={n_remapped_test}")
    print(f"  O-confidence: 0.50 -> {GLINER2_O_CONFIDENCE}")

    # ── Step 2: Calibrate ──
    print("\n--- Temperature Calibration ---")
    temperatures = {}
    cal_dev_preds, cal_test_preds = {}, {}

    for name in ALL_MODELS:
        if name == "gliner2":
            # Entity-only calibration for GLiNER2
            print(f"  {name}: entity-only calibration...")
            T = calibrate_temperature_entity_only(raw_dev_preds[name], dev_data)
            temperatures[name] = T

            # Apply entity-only temperature + fixed O-confidence
            cal_dev_preds[name] = apply_gliner2_entity_calibration(
                raw_dev_preds[name], T, GLINER2_O_CONFIDENCE)
            cal_test_preds[name] = apply_gliner2_entity_calibration(
                raw_test_preds[name], T, GLINER2_O_CONFIDENCE)
            print(f"  {name}: T={T:.4f} (entity-only), O-conf={GLINER2_O_CONFIDENCE}")
        else:
            # Standard all-token calibration
            T = calibrate_temperature_all(raw_dev_preds[name], dev_data)
            temperatures[name] = T
            cal_dev_preds[name] = apply_temperature(raw_dev_preds[name], T)
            cal_test_preds[name] = apply_temperature(raw_test_preds[name], T)
            print(f"  {name}: T={T:.4f} (all-token)")

    # ── Step 3: Dev F1 with new calibration ──
    print("\n--- Dev F1 (with entity-only calibration for GLiNER2) ---")
    dev_f1s = {}
    for name in ALL_MODELS:
        f1 = compute_model_dev_f1(cal_dev_preds[name], dev_data)
        dev_f1s[name] = f1
        print(f"  {name}: {f1:.4f}")

    # ── Step 4: Run all subsets ──
    all_results = {}
    for subset_name, model_names in SUBSETS.items():
        n_weak = sum(1 for m in model_names if dev_f1s[m] < 0.7)
        n_strong = sum(1 for m in model_names if dev_f1s[m] >= 0.7)

        print(f"\n{'=' * 70}")
        print(f"  {subset_name}: {model_names}")
        print(f"  Weak:Strong = {n_weak}:{n_strong}")
        print(f"{'=' * 70}")

        t_sub = time.time()
        results = run_subset(
            subset_name, model_names,
            cal_dev_preds, cal_test_preds,
            dev_data, test_data, dev_f1s,
        )
        results["weak_count"] = n_weak
        results["strong_count"] = n_strong
        results["wall_time_seconds"] = round(time.time() - t_sub, 1)
        all_results[subset_name] = results

        print(f"  B  f1={results['B_scalar_weighted']['entity_f1']:.4f}  "
              f"prec={results['B_scalar_weighted']['entity_precision']:.4f}")
        print(f"  D  f1={results['D_stacking']['entity_f1']:.4f}  "
              f"prec={results['D_stacking']['entity_precision']:.4f}")
        print(f"  E2 f1={results['E2_sl_evidence']['entity_f1']:.4f}  "
              f"prec={results['E2_sl_evidence']['entity_precision']:.4f}  "
              f"delta_vs_B: {results['delta_E2_vs_B']['f1']:+.4f}")
        print(f"  G2 f1={results['G2_sl_trust']['entity_f1']:.4f}  "
              f"prec={results['G2_sl_trust']['entity_precision']:.4f}  "
              f"delta_vs_B: {results['delta_G2_vs_B']['f1']:+.4f}")
        print(f"  Trust benefit (G2-E2): f1={results['trust_benefit']['f1']:+.4f}")
        print(f"  Time: {results['wall_time_seconds']}s")

    # ── Summary: Experiment C vs original ablation ──
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("COMPARISON: Experiment C (entity-cal) vs Original (all-token cal)")
    print("=" * 70)

    # Load original ablation results for comparison
    orig_path = RESULTS_DIR / "en1_1c_ablation_results.json"
    if orig_path.exists():
        orig = ExperimentResult.load_json(str(orig_path))
        orig_results = orig.metrics["subset_results"]

        print(f"\n  {'Subset':<30s} {'Orig G2':>8s} {'ExpC G2':>8s} {'Delta':>8s}  "
              f"{'Orig E2':>8s} {'ExpC E2':>8s} {'Delta':>8s}")
        print("  " + "-" * 85)

        for sn in SUBSETS:
            if sn in orig_results:
                o_g2 = orig_results[sn]["G2_sl_trust"]["entity_f1"]
                c_g2 = all_results[sn]["G2_sl_trust"]["entity_f1"]
                o_e2 = orig_results[sn]["E2_sl_evidence"]["entity_f1"]
                c_e2 = all_results[sn]["E2_sl_evidence"]["entity_f1"]
                print(f"  {sn:<30s} {o_g2:>8.4f} {c_g2:>8.4f} {c_g2-o_g2:>+8.4f}  "
                      f"{o_e2:>8.4f} {c_e2:>8.4f} {c_e2-o_e2:>+8.4f}")

        # Also compare B and D (should be different due to different calibrated confidences)
        print(f"\n  {'Subset':<30s} {'Orig B':>8s} {'ExpC B':>8s} {'Delta':>8s}  "
              f"{'Orig D':>8s} {'ExpC D':>8s} {'Delta':>8s}")
        print("  " + "-" * 85)

        for sn in SUBSETS:
            if sn in orig_results:
                o_b = orig_results[sn]["B_scalar_weighted"]["entity_f1"]
                c_b = all_results[sn]["B_scalar_weighted"]["entity_f1"]
                o_d = orig_results[sn]["D_stacking"]["entity_f1"]
                c_d = all_results[sn]["D_stacking"]["entity_f1"]
                print(f"  {sn:<30s} {o_b:>8.4f} {c_b:>8.4f} {c_b-o_b:>+8.4f}  "
                      f"{o_d:>8.4f} {c_d:>8.4f} {c_d-o_d:>+8.4f}")
    else:
        print("\n  Original ablation results not found -- cannot compare")

    # ── Key question: did entity-only calibration fix S6? ──
    print(f"\n" + "=" * 70)
    print("KEY QUESTION: Does entity-only calibration fix the S6 crash?")
    print("=" * 70)
    s6_e2 = all_results["S6_all_5model"]["E2_sl_evidence"]["entity_f1"]
    s6_g2 = all_results["S6_all_5model"]["G2_sl_trust"]["entity_f1"]
    s6_b = all_results["S6_all_5model"]["B_scalar_weighted"]["entity_f1"]
    print(f"  S6 E2 (entity-cal): {s6_e2:.4f}")
    print(f"  S6 G2 (entity-cal): {s6_g2:.4f}")
    print(f"  S6 B  (entity-cal): {s6_b:.4f}")
    if orig_path.exists():
        o_s6_e2 = orig_results["S6_all_5model"]["E2_sl_evidence"]["entity_f1"]
        o_s6_g2 = orig_results["S6_all_5model"]["G2_sl_trust"]["entity_f1"]
        print(f"  S6 E2 (original):   {o_s6_e2:.4f} -> {s6_e2:.4f} ({s6_e2-o_s6_e2:+.4f})")
        print(f"  S6 G2 (original):   {o_s6_g2:.4f} -> {s6_g2:.4f} ({s6_g2-o_s6_g2:+.4f})")
        if s6_g2 > 0.935:
            print(f"  -> YES: entity-only calibration substantially fixes S6")
        elif s6_g2 > s6_e2 and s6_g2 > o_s6_g2 + 0.01:
            print(f"  -> PARTIAL: improvement but gap remains")
        else:
            print(f"  -> NO: calibration is not the primary issue")

    print(f"\n  Total time: {total_time:.1f}s")

    # ── Save ──
    output_path = RESULTS_DIR / "en1_1c_exp_c_results.json"
    experiment_result = ExperimentResult(
        experiment_id="EN1.1c-ExpC",
        parameters={
            "global_seed": GLOBAL_SEED,
            "subsets": {k: v for k, v in SUBSETS.items()},
            "evidence_weight": EVIDENCE_WEIGHT,
            "temperatures": temperatures,
            "dev_f1s": dev_f1s,
            "n_bootstrap": N_BOOTSTRAP,
            "gliner2_calibration": "entity-only",
            "gliner2_o_confidence": GLINER2_O_CONFIDENCE,
        },
        metrics={
            "subset_results": all_results,
            "total_wall_time_seconds": round(total_time, 2),
        },
        environment=env,
        notes=(
            "EN1.1c Experiment C: entity-only temperature calibration for GLiNER2. "
            f"O-tokens set to fixed conf={GLINER2_O_CONFIDENCE} (matching ~98.9% "
            "actual accuracy). Entity tokens calibrated with entity-only temperature. "
            "Other 4 models use standard all-token temperature scaling. "
            "Tests whether the two-population calibration problem (T=9.82) was "
            "the primary driver of S6 degradation."
        ),
    )
    experiment_result.save_json(str(output_path))
    print(f"\n  Results saved to: {output_path}")

    ts = experiment_result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    archive_path = RESULTS_DIR / f"en1_1c_exp_c_results_{ts}.json"
    experiment_result.save_json(str(archive_path))
    print(f"  Archived to:      {archive_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
