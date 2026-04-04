#!/usr/bin/env python
"""EN1.1c -- 5-Model Fusion: GLiNER2 + Original 4 Models.

Extends EN1.1b by adding GLiNER2 (fastino/gliner2-base-v1) as the 5th
NER model. Uses the exact same fusion methodology as EN1.1b — same
calibration, same fusion strategies, same evaluation — but with 5 models
instead of 4.

Scientific motivation:
    The original 4 models all use softmax-based token classification.
    GLiNER2 uses dot-product+sigmoid span matching — a fundamentally
    different confidence mechanism. This tests whether SL fusion
    generalizes across heterogeneous confidence scales.

    GLiNER2 is also zero-shot (no CoNLL-2003 training), while the other
    4 models are supervised. This creates a calibration asymmetry that
    SL's uncertainty representation should handle naturally.

Models (5):
    1. spaCy en_core_web_trf       -- RoBERTa pipeline (softmax)
    2. Flair ner-large              -- Stacked LSTM (softmax)
    3. Stanza en NER                -- BiLSTM-CRF (softmax)
    4. HuggingFace bert-base-NER   -- BERT fine-tuned (softmax)
    5. GLiNER2 gliner2-base-v1     -- DeBERTa span matching (sigmoid)

Preserves EN1.1b results exactly. This is a separate experiment
producing separate results files.

Prerequisites:
    Run en1_1_ner_fusion.py first (generates 4-model checkpoints)
    Run en1_1c_gliner2_runner.py first (generates GLiNER2 checkpoints)

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN1/en1_1c_5model_fusion.py

Output:
    experiments/EN1/results/en1_1c_results.json
    experiments/EN1/results/en1_1c_results_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
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
    averaging_fuse,
    conflict_metric,
    trust_discount,
    pairwise_conflict,
)

from experiments.infra.config import set_global_seed
from experiments.infra.results import ExperimentResult
from experiments.infra.env_log import log_environment
from experiments.infra.stats import bootstrap_ci

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
GLOBAL_SEED = 42
N_BOOTSTRAP = 1000

ENTITY_TYPES = ["PER", "LOC", "ORG", "MISC"]
ALL_TAGS = ["O"] + [f"{p}-{e}" for e in ENTITY_TYPES for p in ["B", "I"]]

# ── 5 models: original 4 + GLiNER2 ──────────────────────────────
MODEL_NAMES = ["spacy", "flair", "stanza", "huggingface", "gliner2"]

# Evidence weight: each model contributes this many "virtual observations"
# Higher = more dogmatic. 10 gives u = 2/(10+2) = 0.167 per model.
EVIDENCE_WEIGHT = 10


# =====================================================================
# Data Loading (from checkpoints + HuggingFace)
# =====================================================================

def load_checkpoint(name: str) -> Any:
    path = CHECKPOINT_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint {name} not found at {path}. "
            "Run en1_1_ner_fusion.py and en1_1c_gliner2_runner.py first."
        )
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
    """Create an opinion from a model's confidence using evidence mapping."""
    pos = conf * EVIDENCE_WEIGHT
    neg = (1.0 - conf) * EVIDENCE_WEIGHT
    return Opinion.from_evidence(pos, neg, prior_weight=2.0, base_rate=0.5)


# =====================================================================
# Fusion Strategies
# =====================================================================

def _fuse_per_tag(
    model_preds: Dict[str, List[List[Dict]]],
    sent_idx: int,
    tok_idx: int,
    model_names: List[str],
    use_evidence: bool = True,
    trust_opinions: Optional[Dict[str, Opinion]] = None,
) -> Dict[str, Tuple[Opinion, int]]:
    """Core fusion: for each tag, fuse opinions from supporting models."""
    tag_opinions: Dict[str, List[Opinion]] = defaultdict(list)
    for m in model_names:
        preds = model_preds[m][sent_idx]
        if tok_idx < len(preds):
            tag = preds[tok_idx]["tag"]
            conf = preds[tok_idx]["confidence"]

            if use_evidence:
                op = opinion_from_model_confidence(conf)
            else:
                op = Opinion.from_confidence(conf, uncertainty=1.0 - conf)

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
    """Select best tag by support-weighted projected probability."""
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


def strategy_sl_evidence_fuse(
    model_preds: Dict[str, List[List[Dict]]],
    sentences: List[Dict],
    trust_opinions: Optional[Dict[str, Opinion]] = None,
) -> List[List[str]]:
    """E2/G2: SL fusion with evidence-based opinions."""
    result = []
    for sent_idx in range(len(sentences)):
        n_tok = len(sentences[sent_idx]["tokens"])
        sent_tags = []
        for tok_idx in range(n_tok):
            tag_fused = _fuse_per_tag(
                model_preds, sent_idx, tok_idx, MODEL_NAMES,
                use_evidence=True, trust_opinions=trust_opinions,
            )
            best_tag, _, _, _, _ = _select_best_tag(tag_fused, len(MODEL_NAMES))
            sent_tags.append(best_tag)
        result.append(sent_tags)
    return result


def strategy_sl_abstain(
    model_preds: Dict[str, List[List[Dict]]],
    sentences: List[Dict],
    margin_threshold: float = 0.1,
    trust_opinions: Optional[Dict[str, Opinion]] = None,
) -> Tuple[List[List[str]], List[List[bool]]]:
    """F2: SL fusion with margin-based abstention."""
    result_tags = []
    result_abstain = []

    for sent_idx in range(len(sentences)):
        n_tok = len(sentences[sent_idx]["tokens"])
        sent_tags = []
        sent_abstain = []

        for tok_idx in range(n_tok):
            tag_fused = _fuse_per_tag(
                model_preds, sent_idx, tok_idx, MODEL_NAMES,
                use_evidence=True, trust_opinions=trust_opinions,
            )
            best_tag, best_score, best_op, second_tag, second_score = \
                _select_best_tag(tag_fused, len(MODEL_NAMES))

            margin = best_score - second_score

            should_abstain = (
                second_tag is not None and
                second_tag != "O" and
                margin < margin_threshold
            )

            if should_abstain:
                sent_tags.append("O")
                sent_abstain.append(True)
            else:
                sent_tags.append(best_tag)
                sent_abstain.append(False)

        result_tags.append(sent_tags)
        result_abstain.append(sent_abstain)

    return result_tags, result_abstain


def strategy_scalar_abstain(
    model_preds: Dict[str, List[List[Dict]]],
    model_weights: Dict[str, float],
    sentences: List[Dict],
    disagreement_threshold: float = 0.5,
) -> Tuple[List[List[str]], List[List[bool]]]:
    """H: Scalar weighted average with disagreement-based abstention."""
    result_tags = []
    result_abstain = []

    for sent_idx in range(len(sentences)):
        n_tok = len(sentences[sent_idx]["tokens"])
        sent_tags = []
        sent_abstain = []

        for tok_idx in range(n_tok):
            tag_scores = defaultdict(float)
            tags = []
            for m in MODEL_NAMES:
                preds = model_preds[m][sent_idx]
                if tok_idx < len(preds):
                    tag = preds[tok_idx]["tag"]
                    conf = preds[tok_idx]["confidence"]
                    tag_scores[tag] += conf * model_weights[m]
                    tags.append(tag)

            best_tag = max(tag_scores, key=tag_scores.get)

            if tags:
                best_count = sum(1 for t in tags if t == best_tag)
                disagreement = 1.0 - best_count / len(tags)
            else:
                disagreement = 0.0

            entity_tags = [t for t in set(tags) if t != "O" and t != best_tag]
            should_abstain = disagreement >= disagreement_threshold and len(entity_tags) > 0

            if should_abstain:
                sent_tags.append("O")
                sent_abstain.append(True)
            else:
                sent_tags.append(best_tag)
                sent_abstain.append(False)

        result_tags.append(sent_tags)
        result_abstain.append(sent_abstain)

    return result_tags, result_abstain


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_predictions(
    predictions: List[List[str]],
    gold: List[Dict],
    abstention_mask: Optional[List[List[bool]]] = None,
) -> Dict[str, Any]:
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
    token_f1 = sklearn_f1(all_gold, all_pred, average="micro",
                           labels=[t for t in ALL_TAGS if t != "O"], zero_division=0)

    result = {
        "entity_f1": entity_f1, "entity_precision": entity_prec,
        "entity_recall": entity_rec, "token_accuracy": token_acc,
        "token_f1_micro": token_f1, "n_tokens": n_total,
    }

    if abstention_mask is not None:
        all_abstain = [a for s in abstention_mask for a in s]
        n_abstained = sum(all_abstain)
        abs_rate = n_abstained / n_total if n_total else 0

        if n_total - n_abstained > 0:
            na_pred_s, na_gold_s = [], []
            for ps, gs, abss in zip(pred_aligned, gold_aligned, abstention_mask):
                p2 = [p for p, a in zip(ps, abss) if not a]
                g2 = [g for g, a in zip(gs, abss) if not a]
                if p2:
                    na_pred_s.append(p2)
                    na_gold_s.append(g2)
            if na_pred_s:
                na_f1 = f1_score(na_gold_s, na_pred_s)
                na_prec = precision_score(na_gold_s, na_pred_s)
                na_rec = recall_score(na_gold_s, na_pred_s)
            else:
                na_f1 = na_prec = na_rec = 0.0
        else:
            na_f1 = na_prec = na_rec = 0.0

        result.update({
            "abstention_rate": abs_rate, "n_abstained": n_abstained,
            "non_abstained_entity_f1": na_f1,
            "non_abstained_entity_precision": na_prec,
            "non_abstained_entity_recall": na_rec,
        })

    return result


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


def compute_model_dev_metrics(preds, gold):
    from seqeval.metrics import f1_score
    pred_tags = [[p["tag"] for p in sent] for sent in preds]
    gold_tags = [s["ner_tags"] for s in gold]
    ap, ag = [], []
    for p, g in zip(pred_tags, gold_tags):
        n = min(len(p), len(g))
        ap.append(p[:n])
        ag.append(g[:n])
    f1 = f1_score(ag, ap)
    all_p = [t for s in ap for t in s]
    all_g = [t for s in ag for t in s]
    acc = sum(1 for p, g in zip(all_p, all_g) if p == g) / len(all_p) if all_p else 0
    return {"entity_f1": f1, "token_accuracy": acc}


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("EN1.1c -- 5-Model Fusion (original 4 + GLiNER2)")
    print("=" * 70)

    t_start = time.time()
    env = log_environment()
    set_global_seed(GLOBAL_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data + checkpoints ──
    print("\n--- Loading data and model predictions from checkpoints ---")
    dev_data = load_conll2003("validation")
    test_data = load_conll2003("test")

    dev_preds, test_preds = {}, {}
    for name in MODEL_NAMES:
        dev_preds[name] = load_checkpoint(f"dev_preds_{name}")
        test_preds[name] = load_checkpoint(f"test_preds_{name}")

    print(f"  Loaded: dev={len(dev_data)} sents, test={len(test_data)} sents, "
          f"models={MODEL_NAMES}")

    # ── Calibrate ──
    print("\n--- Temperature Calibration ---")
    temperatures = {}
    for name in MODEL_NAMES:
        T = calibrate_temperature(dev_preds[name], dev_data)
        temperatures[name] = T
        dev_preds[name] = apply_temperature(dev_preds[name], T)
        test_preds[name] = apply_temperature(test_preds[name], T)
        print(f"  {name}: T={T:.4f}")

    # ── Dev metrics + weights + trust ──
    print("\n--- Model Metrics & Weights ---")
    dev_metrics = {}
    for name in MODEL_NAMES:
        dev_metrics[name] = compute_model_dev_metrics(dev_preds[name], dev_data)
        print(f"  {name} (dev): f1={dev_metrics[name]['entity_f1']:.4f}")

    model_weights = {n: m["entity_f1"] for n, m in dev_metrics.items()}
    tw = sum(model_weights.values())
    model_weights = {n: w / tw for n, w in model_weights.items()}

    model_trust = {}
    for name, m in dev_metrics.items():
        f1 = m["entity_f1"]
        model_trust[name] = Opinion(
            belief=f1 * 0.9,
            disbelief=(1 - f1) * 0.5,
            uncertainty=1.0 - f1 * 0.9 - (1 - f1) * 0.5,
            base_rate=0.5,
        )
        print(f"  Trust {name}: b={model_trust[name].belief:.3f} "
              f"u={model_trust[name].uncertainty:.3f}")

    # ── Individual models (test) ──
    print("\n--- Individual Model Results (test) ---")
    test_individual = {}
    for name in MODEL_NAMES:
        pt = [[p["tag"] for p in s] for s in test_preds[name]]
        m = evaluate_predictions(pt, test_data)
        ci = bootstrap_entity_f1(pt, test_data)
        m["entity_f1_ci"] = {"lower": ci[0], "mean": ci[1], "upper": ci[2]}
        test_individual[name] = m
        print(f"  {name}: f1={m['entity_f1']:.4f} [{ci[0]:.4f}, {ci[2]:.4f}] "
              f"prec={m['entity_precision']:.4f} rec={m['entity_recall']:.4f}")

    # ── Fusion strategies ──
    print("\n" + "=" * 70)
    print("FUSION STRATEGIES (5 models)")
    print("=" * 70)
    fusion = {}
    tp = {n: test_preds[n] for n in MODEL_NAMES}

    def _eval_and_print(label, preds, abstain=None):
        m = evaluate_predictions(preds, test_data, abstain)
        ci = bootstrap_entity_f1(preds, test_data)
        m["entity_f1_ci"] = {"lower": ci[0], "mean": ci[1], "upper": ci[2]}
        line = f"  {label:<35s} f1={m['entity_f1']:.4f} [{ci[0]:.4f},{ci[2]:.4f}]"
        line += f"  prec={m['entity_precision']:.4f}  rec={m['entity_recall']:.4f}"
        if abstain:
            line += f"  abs={m.get('abstention_rate',0):.2%}"
            line += f"  na_f1={m.get('non_abstained_entity_f1',0):.4f}"
            line += f"  na_prec={m.get('non_abstained_entity_precision',0):.4f}"
        print(line)
        return m

    # B: Scalar weighted (reference)
    print("\n--- Scalar Baselines ---")
    pred_B = []
    for si in range(len(test_data)):
        nt = len(test_data[si]["tokens"])
        st = []
        for ti in range(nt):
            ts2 = defaultdict(float)
            for mn in MODEL_NAMES:
                ps = tp[mn][si]
                if ti < len(ps):
                    ts2[ps[ti]["tag"]] += ps[ti]["confidence"] * model_weights[mn]
            st.append(max(ts2, key=ts2.get))
        pred_B.append(st)
    fusion["B_scalar_weighted"] = _eval_and_print("B: Scalar weighted", pred_B)

    # D: Stacking
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(ALL_TAGS)
    X_tr, y_tr = [], []
    for si in range(len(dev_data)):
        gt = dev_data[si]["ner_tags"]
        for ti in range(len(gt)):
            feat = []
            for mn in MODEL_NAMES:
                ps = dev_preds[mn][si]
                if ti < len(ps):
                    te = [0.0]*len(ALL_TAGS)
                    if ps[ti]["tag"] in ALL_TAGS:
                        te[ALL_TAGS.index(ps[ti]["tag"])] = 1.0
                    feat.extend(te)
                    feat.append(ps[ti]["confidence"])
                else:
                    feat.extend([0.0]*(len(ALL_TAGS)+1))
            X_tr.append(feat)
            y_tr.append(gt[ti])
    clf = LogisticRegression(max_iter=1000,
                              solver="lbfgs", random_state=GLOBAL_SEED, n_jobs=-1)
    clf.fit(np.array(X_tr), le.transform(y_tr))
    pred_D = []
    for si in range(len(test_data)):
        feats = []
        for ti in range(len(test_data[si]["tokens"])):
            feat = []
            for mn in MODEL_NAMES:
                ps = tp[mn][si]
                if ti < len(ps):
                    te = [0.0]*len(ALL_TAGS)
                    if ps[ti]["tag"] in ALL_TAGS:
                        te[ALL_TAGS.index(ps[ti]["tag"])] = 1.0
                    feat.extend(te)
                    feat.append(ps[ti]["confidence"])
                else:
                    feat.extend([0.0]*(len(ALL_TAGS)+1))
            feats.append(feat)
        pred_D.append(list(le.inverse_transform(clf.predict(np.array(feats)))))
    fusion["D_stacking"] = _eval_and_print("D: Stacking meta-learner", pred_D)

    # H: Scalar abstention baseline (sweep)
    print("\n--- H: Scalar Abstention Baseline ---")
    h_thresholds = [0.25, 0.40, 0.50, 0.60, 0.75]
    fusion["H_scalar_abstain"] = {}
    for dt in h_thresholds:
        pred_H, abs_H = strategy_scalar_abstain(tp, model_weights, test_data, dt)
        fusion["H_scalar_abstain"][f"dis={dt}"] = _eval_and_print(
            f"H: Scalar abstain (dis>={dt})", pred_H, abs_H)

    # ── SL Strategies ──
    print("\n--- SL Strategies (evidence-based opinions) ---")

    # E2: SL evidence fuse (no trust)
    pred_E2 = strategy_sl_evidence_fuse(tp, test_data)
    fusion["E2_sl_evidence_fuse"] = _eval_and_print("E2: SL evidence fuse", pred_E2)

    # G2: SL evidence fuse + trust discount
    pred_G2 = strategy_sl_evidence_fuse(tp, test_data, trust_opinions=model_trust)
    fusion["G2_sl_evidence_trust"] = _eval_and_print("G2: SL evidence+trust", pred_G2)

    # F2: SL abstention (sweep margin thresholds)
    print("\n--- F2: SL Margin-Based Abstention ---")
    margin_thresholds = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    fusion["F2_sl_abstain"] = {}
    for mt in margin_thresholds:
        pred_F2, abs_F2 = strategy_sl_abstain(tp, test_data, margin_threshold=mt)
        fusion["F2_sl_abstain"][f"margin={mt}"] = _eval_and_print(
            f"F2: SL abstain (margin<{mt})", pred_F2, abs_F2)

    # F2+trust: SL abstention with trust discount
    print("\n--- F2+trust: SL Abstention + Trust Discount ---")
    fusion["F2_sl_abstain_trust"] = {}
    for mt in margin_thresholds:
        pred_F2t, abs_F2t = strategy_sl_abstain(
            tp, test_data, margin_threshold=mt, trust_opinions=model_trust)
        fusion["F2_sl_abstain_trust"][f"margin={mt}"] = _eval_and_print(
            f"F2+trust: SL abstain+trust (m<{mt})", pred_F2t, abs_F2t)

    # ── Comparison with EN1.1b (4-model) ──
    print("\n" + "=" * 70)
    print("COMPARISON: 5-MODEL vs 4-MODEL (EN1.1b)")
    print("=" * 70)
    en1_1b_path = RESULTS_DIR / "en1_1b_results.json"
    if en1_1b_path.exists():
        en1_1b = ExperimentResult.load_json(str(en1_1b_path))
        b4 = en1_1b.metrics["fusion_strategies"]

        print("\n  Strategy              4-model F1    5-model F1    Delta")
        print("  " + "-" * 65)
        comparisons = [
            ("B: Scalar weighted", "B_scalar_weighted"),
            ("D: Stacking",        "D_stacking"),
            ("E2: SL evidence",    "E2_sl_evidence_fuse"),
            ("G2: SL evidence+trust", "G2_sl_evidence_trust"),
        ]
        comparison_deltas = {}
        for label, key in comparisons:
            f1_4 = b4[key]["entity_f1"]
            f1_5 = fusion[key]["entity_f1"]
            prec_4 = b4[key]["entity_precision"]
            prec_5 = fusion[key]["entity_precision"]
            delta = f1_5 - f1_4
            delta_p = prec_5 - prec_4
            comparison_deltas[key] = {
                "f1_4model": f1_4, "f1_5model": f1_5, "delta_f1": delta,
                "prec_4model": prec_4, "prec_5model": prec_5, "delta_prec": delta_p,
            }
            print(f"  {label:<25s} {f1_4:.4f}        {f1_5:.4f}        {delta:+.4f}"
                  f"  (prec: {prec_4:.4f} -> {prec_5:.4f}, {delta_p:+.4f})")

        # Individual model comparison
        print("\n  Individual Models (test):")
        for name in MODEL_NAMES:
            f1 = test_individual[name]["entity_f1"]
            if name in en1_1b.metrics.get("individual_models_test", {}):
                f1_old = en1_1b.metrics["individual_models_test"][name]["entity_f1"]
                print(f"    {name:<15s} 4-model: {f1_old:.4f}  5-model: {f1:.4f}")
            else:
                print(f"    {name:<15s} NEW:     {f1:.4f}")
    else:
        print("\n  EN1.1b results not found -- cannot compare")
        comparison_deltas = {}

    # ── Summary ──
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("HEADLINE COMPARISON (5 models)")
    print("=" * 70)
    print(f"\n  Best scalar (no abstain):  B  f1={fusion['B_scalar_weighted']['entity_f1']:.4f}  "
          f"prec={fusion['B_scalar_weighted']['entity_precision']:.4f}")
    print(f"  Trained meta-learner:      D  f1={fusion['D_stacking']['entity_f1']:.4f}  "
          f"prec={fusion['D_stacking']['entity_precision']:.4f}")
    print(f"  SL evidence fuse:          E2 f1={fusion['E2_sl_evidence_fuse']['entity_f1']:.4f}  "
          f"prec={fusion['E2_sl_evidence_fuse']['entity_precision']:.4f}")
    print(f"  SL evidence+trust:         G2 f1={fusion['G2_sl_evidence_trust']['entity_f1']:.4f}  "
          f"prec={fusion['G2_sl_evidence_trust']['entity_precision']:.4f}")

    best_f2 = max(fusion["F2_sl_abstain"].values(),
                   key=lambda x: x.get("non_abstained_entity_precision", 0))
    best_h = max(fusion["H_scalar_abstain"].values(),
                  key=lambda x: x.get("non_abstained_entity_precision", 0))

    print(f"\n  Best SL abstain:           F2 na_prec={best_f2.get('non_abstained_entity_precision',0):.4f}  "
          f"abs={best_f2.get('abstention_rate',0):.2%}  "
          f"na_f1={best_f2.get('non_abstained_entity_f1',0):.4f}")
    print(f"  Best scalar abstain:       H  na_prec={best_h.get('non_abstained_entity_precision',0):.4f}  "
          f"abs={best_h.get('abstention_rate',0):.2%}  "
          f"na_f1={best_h.get('non_abstained_entity_f1',0):.4f}")

    # GLiNER2-specific: report its temperature and calibration
    print(f"\n  GLiNER2 calibration:")
    print(f"    Temperature: {temperatures['gliner2']:.4f}")
    print(f"    Dev F1: {dev_metrics['gliner2']['entity_f1']:.4f}")
    print(f"    Test F1: {test_individual['gliner2']['entity_f1']:.4f}")

    print(f"\n  Total time: {total_time:.1f}s")

    # ── Save ──
    output_path = RESULTS_DIR / "en1_1c_results.json"
    experiment_result = ExperimentResult(
        experiment_id="EN1.1c",
        parameters={
            "global_seed": GLOBAL_SEED,
            "models": MODEL_NAMES,
            "evidence_weight": EVIDENCE_WEIGHT,
            "margin_thresholds_F2": margin_thresholds,
            "disagreement_thresholds_H": h_thresholds,
            "temperatures": temperatures,
            "model_weights": model_weights,
            "n_bootstrap": N_BOOTSTRAP,
        },
        metrics={
            "individual_models_test": test_individual,
            "fusion_strategies": fusion,
            "dev_metrics": dev_metrics,
            "comparison_4model_vs_5model": comparison_deltas,
            "total_wall_time_seconds": round(total_time, 2),
        },
        raw_data={
            "model_trust_opinions": {
                n: {"b": t.belief, "d": t.disbelief, "u": t.uncertainty, "a": t.base_rate}
                for n, t in model_trust.items()
            },
        },
        environment=env,
        notes=(
            "EN1.1c: 5-model NER fusion extending EN1.1b with GLiNER2 "
            "(fastino/gliner2-base-v1, zero-shot span matching with "
            "dot-product+sigmoid confidence). Tests whether SL fusion "
            "generalizes across heterogeneous confidence mechanisms "
            "(softmax vs sigmoid) and training paradigms (supervised vs "
            f"zero-shot). Same methodology as EN1.1b with W={EVIDENCE_WEIGHT}. "
            "Original 4-model results preserved in en1_1b_results.json."
        ),
    )
    experiment_result.save_json(str(output_path))
    print(f"\n  Results saved to: {output_path}")

    ts = experiment_result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    archive_path = RESULTS_DIR / f"en1_1c_results_{ts}.json"
    experiment_result.save_json(str(archive_path))
    print(f"  Archived to:      {archive_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
