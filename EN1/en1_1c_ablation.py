#!/usr/bin/env python
"""EN1.1c Ablation -- Model Subset Quality-Ratio Sensitivity Analysis.

Systematically varies the set of models included in fusion to characterize
how SL performance depends on the weak:strong source ratio.

6 subsets tested:
    S1: {Flair, HF}                         -- 0 weak : 2 strong
    S2: {Flair, HF, spaCy}                  -- 1 weak : 2 strong (softmax)
    S3: {Flair, HF, Stanza}                 -- 1 weak : 2 strong (softmax)
    S4: {Flair, HF, GLiNER2}                -- 1 weak : 2 strong (sigmoid)
    S5: {spaCy, Flair, Stanza, HF}          -- 2 weak : 2 strong (EN1.1b sanity)
    S6: {spaCy, Flair, Stanza, HF, GLiNER2} -- 3 weak : 2 strong (EN1.1c)

For each subset: B (scalar weighted), D (stacking), E2 (SL evidence),
G2 (SL+trust). Bootstrap 95% CIs on all metrics.

Key scientific questions:
    1. Does SL degrade monotonically with weak:strong ratio?
    2. Is sigmoid-based GLiNER2 more damaging than softmax weak models?
    3. Does SL still win at 1:2 weak:strong?
    4. Do we reproduce EN1.1b results exactly? (sanity check)

Prerequisites:
    All model checkpoints must exist (run en1_1_ner_fusion.py and
    en1_1c_gliner2_runner.py first).

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN1/en1_1c_ablation.py

Output:
    experiments/EN1/results/en1_1c_ablation_results.json
    experiments/EN1/results/en1_1c_ablation_results_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import json
import math
import sys
import time
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
from experiments.infra.stats import bootstrap_ci

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
GLOBAL_SEED = 42
N_BOOTSTRAP = 1000
EVIDENCE_WEIGHT = 10

ENTITY_TYPES = ["PER", "LOC", "ORG", "MISC"]
ALL_TAGS = ["O"] + [f"{p}-{e}" for e in ENTITY_TYPES for p in ["B", "I"]]

# All available models
ALL_MODELS = ["spacy", "flair", "stanza", "huggingface", "gliner2"]

# ── Model subsets to test ────────────────────────────────────────
SUBSETS = {
    "S1_top2_strong": ["flair", "huggingface"],
    "S2_2strong_1weak_spacy": ["flair", "huggingface", "spacy"],
    "S3_2strong_1weak_stanza": ["flair", "huggingface", "stanza"],
    "S4_2strong_1weak_gliner2": ["flair", "huggingface", "gliner2"],
    "S5_original_4model": ["spacy", "flair", "stanza", "huggingface"],
    "S6_all_5model": ["spacy", "flair", "stanza", "huggingface", "gliner2"],
}


# =====================================================================
# Data Loading
# =====================================================================

def load_checkpoint(name: str) -> Any:
    path = CHECKPOINT_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {name} not found at {path}")
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

def calibrate_temperature(preds, gold):
    confidences, corrects = [], []
    for sent_preds, sent_gold in zip(preds, gold):
        for pred, gt in zip(sent_preds, sent_gold["ner_tags"]):
            confidences.append(pred["confidence"])
            corrects.append(1.0 if pred["tag"] == gt else 0.0)
    confidences = np.array(confidences)
    corrects = np.array(corrects)
    eps = 1e-10

    def nll(T):
        logits = np.log(np.clip(confidences, eps, 1-eps) / np.clip(1-confidences, eps, 1-eps))
        scaled = 1 / (1 + np.exp(-logits / T))
        return -np.mean(corrects * np.log(np.clip(scaled, eps, None)) +
                        (1-corrects) * np.log(np.clip(1-scaled, eps, None)))

    result = sp_optimize.minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def apply_temperature(preds, T):
    eps = 1e-10
    calibrated = []
    for sent in preds:
        cal = []
        for p in sent:
            c = p["confidence"]
            logit = math.log(max(c, eps) / max(1-c, eps))
            cal_c = 1.0 / (1.0 + math.exp(-logit / T))
            cal.append({"tag": p["tag"], "confidence": cal_c})
        calibrated.append(cal)
    return calibrated


# =====================================================================
# Opinion Construction
# =====================================================================

def opinion_from_model_confidence(conf: float) -> Opinion:
    pos = conf * EVIDENCE_WEIGHT
    neg = (1.0 - conf) * EVIDENCE_WEIGHT
    return Opinion.from_evidence(pos, neg, prior_weight=2.0, base_rate=0.5)


# =====================================================================
# Fusion (parameterized by model_names)
# =====================================================================

def _fuse_per_tag(
    model_preds: Dict[str, List[List[Dict]]],
    sent_idx: int,
    tok_idx: int,
    model_names: List[str],
    trust_opinions: Optional[Dict[str, Opinion]] = None,
) -> Dict[str, Tuple[Opinion, int]]:
    tag_opinions: Dict[str, List[Opinion]] = defaultdict(list)
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
        if len(opinions) == 1:
            fused = opinions[0]
        else:
            fused = cumulative_fuse(*opinions)
        result[tag] = (fused, len(opinions))
    return result


def _select_best_tag(
    tag_fused: Dict[str, Tuple[Opinion, int]],
    n_models: int,
) -> Tuple[str, float, Opinion, Optional[str], Optional[float]]:
    scored = []
    for tag, (opinion, n_support) in tag_fused.items():
        pp = opinion.projected_probability()
        score = pp * (n_support / n_models)
        scored.append((tag, score, opinion))
    scored.sort(key=lambda x: x[1], reverse=True)
    best_tag, best_score, best_opinion = scored[0]
    if len(scored) > 1:
        second_tag, second_score, _ = scored[1]
    else:
        second_tag, second_score = None, 0.0
    return best_tag, best_score, best_opinion, second_tag, second_score


def run_sl_fuse(
    model_preds: Dict[str, List[List[Dict]]],
    sentences: List[Dict],
    model_names: List[str],
    trust_opinions: Optional[Dict[str, Opinion]] = None,
) -> List[List[str]]:
    result = []
    for sent_idx in range(len(sentences)):
        n_tok = len(sentences[sent_idx]["tokens"])
        sent_tags = []
        for tok_idx in range(n_tok):
            tag_fused = _fuse_per_tag(
                model_preds, sent_idx, tok_idx, model_names,
                trust_opinions=trust_opinions,
            )
            best_tag, _, _, _, _ = _select_best_tag(tag_fused, len(model_names))
            sent_tags.append(best_tag)
        result.append(sent_tags)
    return result


def run_scalar_weighted(
    model_preds: Dict[str, List[List[Dict]]],
    sentences: List[Dict],
    model_names: List[str],
    model_weights: Dict[str, float],
) -> List[List[str]]:
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


def run_stacking(
    dev_preds: Dict[str, List[List[Dict]]],
    test_preds: Dict[str, List[List[Dict]]],
    dev_data: List[Dict],
    test_data: List[Dict],
    model_names: List[str],
) -> List[List[str]]:
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

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        clf = LogisticRegression(
            max_iter=1000, solver="lbfgs",
            random_state=GLOBAL_SEED,
        )
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
    from sklearn.metrics import f1_score as sklearn_f1

    gold_tags = [s["ner_tags"] for s in gold]
    pred_aligned, gold_aligned = [], []
    for p, g in zip(predictions, gold_tags):
        n = min(len(p), len(g))
        pred_aligned.append(p[:n])
        gold_aligned.append(g[:n])

    entity_f1 = f1_score(gold_aligned, pred_aligned)
    entity_prec = precision_score(gold_aligned, pred_aligned)
    entity_rec = recall_score(gold_aligned, pred_aligned)

    all_pred = [t for s in pred_aligned for t in s]
    all_gold = [t for s in gold_aligned for t in s]
    n_total = len(all_pred)
    token_acc = sum(1 for p, g in zip(all_pred, all_gold) if p == g) / n_total if n_total else 0

    return {
        "entity_f1": entity_f1,
        "entity_precision": entity_prec,
        "entity_recall": entity_rec,
        "token_accuracy": token_acc,
        "n_tokens": n_total,
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

def run_subset(
    subset_name: str,
    model_names: List[str],
    dev_preds_cal: Dict[str, List[List[Dict]]],
    test_preds_cal: Dict[str, List[List[Dict]]],
    dev_data: List[Dict],
    test_data: List[Dict],
    dev_f1s: Dict[str, float],
) -> Dict[str, Any]:
    """Run all 4 strategies for a given model subset."""
    n_models = len(model_names)

    # Compute weights and trust for this subset
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

    # Filter preds to this subset
    dev_sub = {n: dev_preds_cal[n] for n in model_names}
    test_sub = {n: test_preds_cal[n] for n in model_names}

    results = {"models": model_names, "n_models": n_models}

    # B: Scalar weighted
    pred_B = run_scalar_weighted(test_sub, test_data, model_names, subset_weights)
    metrics_B = evaluate_predictions(pred_B, test_data)
    ci_B = bootstrap_entity_f1(pred_B, test_data)
    metrics_B["entity_f1_ci"] = {"lower": ci_B[0], "mean": ci_B[1], "upper": ci_B[2]}
    results["B_scalar_weighted"] = metrics_B

    # D: Stacking
    pred_D = run_stacking(dev_sub, test_sub, dev_data, test_data, model_names)
    metrics_D = evaluate_predictions(pred_D, test_data)
    ci_D = bootstrap_entity_f1(pred_D, test_data)
    metrics_D["entity_f1_ci"] = {"lower": ci_D[0], "mean": ci_D[1], "upper": ci_D[2]}
    results["D_stacking"] = metrics_D

    # E2: SL evidence fuse
    pred_E2 = run_sl_fuse(test_sub, test_data, model_names)
    metrics_E2 = evaluate_predictions(pred_E2, test_data)
    ci_E2 = bootstrap_entity_f1(pred_E2, test_data)
    metrics_E2["entity_f1_ci"] = {"lower": ci_E2[0], "mean": ci_E2[1], "upper": ci_E2[2]}
    results["E2_sl_evidence"] = metrics_E2

    # G2: SL evidence + trust
    pred_G2 = run_sl_fuse(test_sub, test_data, model_names, trust_opinions=subset_trust)
    metrics_G2 = evaluate_predictions(pred_G2, test_data)
    ci_G2 = bootstrap_entity_f1(pred_G2, test_data)
    metrics_G2["entity_f1_ci"] = {"lower": ci_G2[0], "mean": ci_G2[1], "upper": ci_G2[2]}
    results["G2_sl_trust"] = metrics_G2

    # Delta: SL vs scalar
    results["delta_E2_vs_B"] = {
        "f1": metrics_E2["entity_f1"] - metrics_B["entity_f1"],
        "precision": metrics_E2["entity_precision"] - metrics_B["entity_precision"],
    }
    results["delta_G2_vs_B"] = {
        "f1": metrics_G2["entity_f1"] - metrics_B["entity_f1"],
        "precision": metrics_G2["entity_precision"] - metrics_B["entity_precision"],
    }
    results["trust_benefit"] = {
        "f1": metrics_G2["entity_f1"] - metrics_E2["entity_f1"],
        "precision": metrics_G2["entity_precision"] - metrics_E2["entity_precision"],
    }

    # Model weights and trust for this subset
    results["model_weights"] = subset_weights
    results["model_trust"] = {
        n: {"b": subset_trust[n].belief, "u": subset_trust[n].uncertainty}
        for n in model_names
    }

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("EN1.1c Ablation -- Model Subset Quality-Ratio Sensitivity")
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

    # ── Load all checkpoints ──
    print("\n--- Loading all model checkpoints ---")
    raw_dev_preds, raw_test_preds = {}, {}
    for name in ALL_MODELS:
        raw_dev_preds[name] = load_checkpoint(f"dev_preds_{name}")
        raw_test_preds[name] = load_checkpoint(f"test_preds_{name}")
    print(f"  Loaded: {ALL_MODELS}")

    # ── Calibrate all models (once, shared across subsets) ──
    print("\n--- Temperature Calibration (all models) ---")
    temperatures = {}
    cal_dev_preds, cal_test_preds = {}, {}
    for name in ALL_MODELS:
        T = calibrate_temperature(raw_dev_preds[name], dev_data)
        temperatures[name] = T
        cal_dev_preds[name] = apply_temperature(raw_dev_preds[name], T)
        cal_test_preds[name] = apply_temperature(raw_test_preds[name], T)
        print(f"  {name}: T={T:.4f}")

    # ── Dev F1 for all models (for weights/trust) ──
    print("\n--- Dev F1 (all models) ---")
    dev_f1s = {}
    for name in ALL_MODELS:
        f1 = compute_model_dev_f1(cal_dev_preds[name], dev_data)
        dev_f1s[name] = f1
        print(f"  {name}: {f1:.4f}")

    # ── Run all subsets ──
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

        # Print summary
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
        print(f"  Trust benefit (G2-E2): f1={results['trust_benefit']['f1']:+.4f}  "
              f"prec={results['trust_benefit']['precision']:+.4f}")
        print(f"  Time: {results['wall_time_seconds']}s")

    # ── Summary matrix ──
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("QUALITY-RATIO SENSITIVITY MATRIX")
    print("=" * 70)

    print(f"\n  {'Subset':<30s} {'W:S':>5s}  {'B F1':>7s}  {'D F1':>7s}  "
          f"{'E2 F1':>7s}  {'G2 F1':>7s}  {'E2-B':>7s}  {'G2-B':>7s}  {'G2-E2':>7s}")
    print("  " + "-" * 100)

    for subset_name in SUBSETS:
        r = all_results[subset_name]
        ws = f"{r['weak_count']}:{r['strong_count']}"
        b_f1 = r["B_scalar_weighted"]["entity_f1"]
        d_f1 = r["D_stacking"]["entity_f1"]
        e2_f1 = r["E2_sl_evidence"]["entity_f1"]
        g2_f1 = r["G2_sl_trust"]["entity_f1"]
        d_e2 = r["delta_E2_vs_B"]["f1"]
        d_g2 = r["delta_G2_vs_B"]["f1"]
        tb = r["trust_benefit"]["f1"]
        print(f"  {subset_name:<30s} {ws:>5s}  {b_f1:>7.4f}  {d_f1:>7.4f}  "
              f"{e2_f1:>7.4f}  {g2_f1:>7.4f}  {d_e2:>+7.4f}  {d_g2:>+7.4f}  {tb:>+7.4f}")

    print(f"\n  {'Subset':<30s} {'W:S':>5s}  {'B Prec':>7s}  {'D Prec':>7s}  "
          f"{'E2 Prec':>7s}  {'G2 Prec':>7s}  {'E2-B':>7s}  {'G2-B':>7s}")
    print("  " + "-" * 90)

    for subset_name in SUBSETS:
        r = all_results[subset_name]
        ws = f"{r['weak_count']}:{r['strong_count']}"
        b_p = r["B_scalar_weighted"]["entity_precision"]
        d_p = r["D_stacking"]["entity_precision"]
        e2_p = r["E2_sl_evidence"]["entity_precision"]
        g2_p = r["G2_sl_trust"]["entity_precision"]
        d_e2 = r["delta_E2_vs_B"]["precision"]
        d_g2 = r["delta_G2_vs_B"]["precision"]
        print(f"  {subset_name:<30s} {ws:>5s}  {b_p:>7.4f}  {d_p:>7.4f}  "
              f"{e2_p:>7.4f}  {g2_p:>7.4f}  {d_e2:>+7.4f}  {d_g2:>+7.4f}")

    # ── Key analysis ──
    print("\n" + "=" * 70)
    print("KEY ANALYSIS")
    print("=" * 70)

    # S4 vs S2/S3: is sigmoid GLiNER2 more damaging than softmax weak?
    s2_delta = all_results["S2_2strong_1weak_spacy"]["delta_G2_vs_B"]["f1"]
    s3_delta = all_results["S3_2strong_1weak_stanza"]["delta_G2_vs_B"]["f1"]
    s4_delta = all_results["S4_2strong_1weak_gliner2"]["delta_G2_vs_B"]["f1"]
    print(f"\n  Q: Is sigmoid GLiNER2 more damaging than softmax weak models?")
    print(f"    S2 (+spaCy softmax):   G2-B delta = {s2_delta:+.4f}")
    print(f"    S3 (+Stanza softmax):  G2-B delta = {s3_delta:+.4f}")
    print(f"    S4 (+GLiNER2 sigmoid): G2-B delta = {s4_delta:+.4f}")
    if abs(s4_delta) > abs(s2_delta) and abs(s4_delta) > abs(s3_delta):
        print(f"    -> YES: sigmoid weak model is more damaging than softmax weak")
    elif abs(s4_delta) < abs(s2_delta) and abs(s4_delta) < abs(s3_delta):
        print(f"    -> NO: sigmoid weak model is LESS damaging than softmax weak")
    else:
        print(f"    -> MIXED: damage depends on specific model, not confidence mechanism")

    # S5 sanity check vs EN1.1b
    s5_e2 = all_results["S5_original_4model"]["E2_sl_evidence"]["entity_f1"]
    s5_g2 = all_results["S5_original_4model"]["G2_sl_trust"]["entity_f1"]
    print(f"\n  Q: Does S5 reproduce EN1.1b?")
    print(f"    S5 E2 F1: {s5_e2:.4f}  (EN1.1b E2: 0.9397)")
    print(f"    S5 G2 F1: {s5_g2:.4f}  (EN1.1b G2: 0.9405)")
    if abs(s5_e2 - 0.9397) < 0.001 and abs(s5_g2 - 0.9405) < 0.001:
        print(f"    -> YES: reproduced within 0.1pp (sanity check PASS)")
    else:
        print(f"    -> DEVIATION: investigate (may be due to sklearn version)")

    # Monotonicity check
    print(f"\n  Q: Does SL degrade monotonically with weak:strong ratio?")
    ratios = []
    for sn in SUBSETS:
        r = all_results[sn]
        ratio = r["weak_count"] / max(r["strong_count"], 1)
        g2_f1 = r["G2_sl_trust"]["entity_f1"]
        ratios.append((ratio, g2_f1, sn))
    ratios.sort()
    for ratio, f1, sn in ratios:
        print(f"    {ratio:.1f}  G2 F1={f1:.4f}  ({sn})")

    print(f"\n  Total time: {total_time:.1f}s")

    # ── Save ──
    output_path = RESULTS_DIR / "en1_1c_ablation_results.json"
    experiment_result = ExperimentResult(
        experiment_id="EN1.1c-ablation",
        parameters={
            "global_seed": GLOBAL_SEED,
            "subsets": {k: v for k, v in SUBSETS.items()},
            "evidence_weight": EVIDENCE_WEIGHT,
            "temperatures": temperatures,
            "dev_f1s": dev_f1s,
            "n_bootstrap": N_BOOTSTRAP,
        },
        metrics={
            "subset_results": all_results,
            "total_wall_time_seconds": round(total_time, 2),
        },
        environment=env,
        notes=(
            "EN1.1c ablation: systematic model subset analysis to characterize "
            "SL fusion sensitivity to weak:strong source quality ratio. "
            "6 subsets from 0:2 to 3:2 weak:strong. Same methodology as EN1.1b/c. "
            "Tests whether sigmoid confidence (GLiNER2) is more damaging than "
            "softmax confidence, and whether SL advantage holds at 1:2 ratio."
        ),
    )
    experiment_result.save_json(str(output_path))
    print(f"\n  Results saved to: {output_path}")

    ts = experiment_result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    archive_path = RESULTS_DIR / f"en1_1c_ablation_results_{ts}.json"
    experiment_result.save_json(str(archive_path))
    print(f"  Archived to:      {archive_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
