#!/usr/bin/env python
"""EN1.1c Diagnostic -- Investigate GLiNER2 Temperature Calibration.

Why does GLiNER2 require T=9.82 while all other models need T~1.0?
This script tests several hypotheses by examining the calibration
landscape under different conditions.

Hypotheses:
    H1: The O-token confidence of 0.5 is corrupting calibration.
        Test: calibrate entity-only tokens, calibrate O-only tokens,
        compare temperatures.
    H2: GLiNER2 is genuinely overconfident on entities.
        Test: compute ECE/reliability for entity tokens only.
    H3: The bimodal confidence distribution (0.5 for O, ~1.0 for entities)
        creates a calibration landscape where no single T works.
        Test: Platt scaling (logit + intercept) vs temperature scaling.

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN1/en1_1c_calibration_diagnostic.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from scipy import optimize as sp_optimize

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "packages" / "python" / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

# Same data loader as the main scripts
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


def load_checkpoint(name: str):
    path = CHECKPOINT_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================================
# Calibration Functions
# =====================================================================

def temperature_nll(T, confidences, corrects):
    """Negative log-likelihood for temperature T."""
    eps = 1e-10
    logits = np.log(np.clip(confidences, eps, 1-eps) / np.clip(1-confidences, eps, 1-eps))
    scaled = 1 / (1 + np.exp(-logits / T))
    return -np.mean(corrects * np.log(np.clip(scaled, eps, None)) +
                    (1-corrects) * np.log(np.clip(1-scaled, eps, None)))


def find_temperature(confidences, corrects, bounds=(0.1, 20.0)):
    """Find optimal temperature."""
    result = sp_optimize.minimize_scalar(
        lambda T: temperature_nll(T, confidences, corrects),
        bounds=bounds, method="bounded"
    )
    return float(result.x), float(result.fun)


def platt_scaling(confidences, corrects):
    """Platt scaling: sigmoid(a * logit + b). More flexible than temperature."""
    eps = 1e-10
    logits = np.log(np.clip(confidences, eps, 1-eps) / np.clip(1-confidences, eps, 1-eps))

    def nll(params):
        a, b = params
        scaled = 1 / (1 + np.exp(-(a * logits + b)))
        return -np.mean(corrects * np.log(np.clip(scaled, eps, None)) +
                        (1-corrects) * np.log(np.clip(1-scaled, eps, None)))

    result = sp_optimize.minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return result.x[0], result.x[1], result.fun


def compute_ece(confidences, corrects, n_bins=15):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = corrects[mask].mean()
        ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)
    return ece


def apply_temperature(conf, T):
    eps = 1e-10
    logit = math.log(max(conf, eps) / max(1-conf, eps))
    return 1.0 / (1.0 + math.exp(-logit / T))


# =====================================================================
# Extract token-level data
# =====================================================================

def extract_token_data(preds, gold):
    """Extract per-token (confidence, correct, is_entity) tuples."""
    data = []
    for sent_preds, sent_gold in zip(preds, gold):
        for pred, gt in zip(sent_preds, sent_gold["ner_tags"]):
            is_entity_pred = pred["tag"] != "O"
            is_entity_gt = gt != "O"
            correct = 1.0 if pred["tag"] == gt else 0.0
            data.append({
                "confidence": pred["confidence"],
                "correct": correct,
                "is_entity_pred": is_entity_pred,
                "is_entity_gt": is_entity_gt,
                "pred_tag": pred["tag"],
                "gt_tag": gt,
            })
    return data


# =====================================================================
# Main diagnostic
# =====================================================================

def main():
    print("=" * 70)
    print("EN1.1c Calibration Diagnostic")
    print("=" * 70)

    dev_data = load_conll2003("validation")

    # Load all models' dev predictions
    all_models = ["spacy", "flair", "stanza", "huggingface", "gliner2"]

    print("\n" + "=" * 70)
    print("PART 1: Temperature comparison across all models")
    print("=" * 70)

    for model_name in all_models:
        preds = load_checkpoint(f"dev_preds_{model_name}")
        token_data = extract_token_data(preds, dev_data)

        all_confs = np.array([d["confidence"] for d in token_data])
        all_corrects = np.array([d["correct"] for d in token_data])

        ent_mask = np.array([d["is_entity_pred"] for d in token_data])
        o_mask = ~ent_mask

        T_all, nll_all = find_temperature(all_confs, all_corrects)

        print(f"\n  {model_name}:")
        print(f"    Total tokens: {len(token_data)}")
        print(f"    Entity tokens: {ent_mask.sum()} ({100*ent_mask.mean():.1f}%)")
        print(f"    O tokens: {o_mask.sum()} ({100*o_mask.mean():.1f}%)")
        print(f"    Temperature (all tokens): T={T_all:.4f}")

        # Entity-only calibration
        if ent_mask.sum() > 100:
            ent_confs = all_confs[ent_mask]
            ent_corrects = all_corrects[ent_mask]
            T_ent, _ = find_temperature(ent_confs, ent_corrects)
            ece_ent_raw = compute_ece(ent_confs, ent_corrects)
            ent_acc = ent_corrects.mean()
            ent_conf_mean = ent_confs.mean()
            print(f"    Entity-only: T={T_ent:.4f}, ECE={ece_ent_raw:.4f}, "
                  f"acc={ent_acc:.3f}, mean_conf={ent_conf_mean:.3f}")
        else:
            T_ent = None
            print(f"    Entity-only: too few tokens")

        # O-only calibration
        if o_mask.sum() > 100:
            o_confs = all_confs[o_mask]
            o_corrects = all_corrects[o_mask]
            T_o, _ = find_temperature(o_confs, o_corrects)
            ece_o_raw = compute_ece(o_confs, o_corrects)
            o_acc = o_corrects.mean()
            o_conf_mean = o_confs.mean()
            print(f"    O-only: T={T_o:.4f}, ECE={ece_o_raw:.4f}, "
                  f"acc={o_acc:.3f}, mean_conf={o_conf_mean:.3f}")
        else:
            T_o = None
            print(f"    O-only: too few tokens")

        # Platt scaling (a*logit + b)
        a, b, nll_platt = platt_scaling(all_confs, all_corrects)
        print(f"    Platt scaling: a={a:.4f}, b={b:.4f}, NLL={nll_platt:.6f} "
              f"(vs temp NLL={nll_all:.6f})")

    # ── Part 2: GLiNER2 deep dive ──
    print("\n" + "=" * 70)
    print("PART 2: GLiNER2 Deep Dive")
    print("=" * 70)

    g_preds = load_checkpoint("dev_preds_gliner2")
    g_data = extract_token_data(g_preds, dev_data)

    all_confs = np.array([d["confidence"] for d in g_data])
    all_corrects = np.array([d["correct"] for d in g_data])
    ent_pred_mask = np.array([d["is_entity_pred"] for d in g_data])

    # 2a: Entity token accuracy by confidence bucket
    print("\n  2a: Entity token accuracy by confidence bucket")
    ent_confs = all_confs[ent_pred_mask]
    ent_corrects = all_corrects[ent_pred_mask]

    bins = [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.95),
            (0.95, 0.99), (0.99, 0.999), (0.999, 1.001)]
    print(f"    {'Confidence Range':<20s} {'Count':>6s} {'Accuracy':>8s} {'Gap':>8s}")
    print(f"    {'-'*45}")
    for lo, hi in bins:
        mask = (ent_confs >= lo) & (ent_confs < hi)
        if mask.sum() == 0:
            continue
        acc = ent_corrects[mask].mean()
        mean_conf = ent_confs[mask].mean()
        gap = mean_conf - acc
        print(f"    [{lo:.3f}, {hi:.3f})     {mask.sum():>6d} {acc:>8.3f} {gap:>+8.3f}")

    # 2b: O-token analysis
    print("\n  2b: O-token analysis")
    o_mask = ~ent_pred_mask
    o_confs = all_confs[o_mask]
    o_corrects = all_corrects[o_mask]
    print(f"    O tokens: {o_mask.sum()}")
    print(f"    O confidence: all = {o_confs[0]:.2f} (constant)")
    print(f"    O accuracy: {o_corrects.mean():.4f}")
    print(f"    O tokens correct: {int(o_corrects.sum())} / {len(o_corrects)}")

    # What fraction of O predictions are actually entities in GT?
    o_is_entity_gt = np.array([d["is_entity_gt"] for d in g_data])[o_mask]
    print(f"    O preds that are actually entities (missed): "
          f"{o_is_entity_gt.sum()} ({100*o_is_entity_gt.mean():.1f}%)")

    # 2c: Temperature sensitivity
    print("\n  2c: Temperature sensitivity for GLiNER2")
    print(f"    {'T':>6s}  {'NLL (all)':>10s}  {'NLL (ent)':>10s}  "
          f"{'ECE (all)':>10s}  {'ECE (ent)':>10s}")
    print(f"    {'-'*55}")

    for T in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 9.8, 15.0, 20.0]:
        nll_a = temperature_nll(T, all_confs, all_corrects)
        nll_e = temperature_nll(T, ent_confs, ent_corrects) if len(ent_confs) > 0 else 0

        # Compute ECE after applying temperature
        scaled_all = np.array([apply_temperature(c, T) for c in all_confs])
        scaled_ent = np.array([apply_temperature(c, T) for c in ent_confs])
        ece_a = compute_ece(scaled_all, all_corrects)
        ece_e = compute_ece(scaled_ent, ent_corrects) if len(ent_confs) > 0 else 0

        marker = " <-- optimal (all)" if abs(T - 9.8) < 0.1 else ""
        print(f"    {T:>6.1f}  {nll_a:>10.6f}  {nll_e:>10.6f}  "
              f"{ece_a:>10.4f}  {ece_e:>10.4f}{marker}")

    # Find entity-only optimal T
    T_ent_opt, _ = find_temperature(ent_confs, ent_corrects)
    print(f"\n    Optimal T (entity-only): {T_ent_opt:.4f}")
    print(f"    Optimal T (all tokens):  9.8182")
    print(f"    Ratio: {9.8182 / T_ent_opt:.1f}x")

    # 2d: What if O-confidence were different?
    print("\n  2d: Temperature sensitivity to O-token confidence")
    print(f"    {'O conf':>7s}  {'Optimal T':>10s}  {'NLL':>10s}  {'ECE':>10s}")
    print(f"    {'-'*42}")

    for o_conf in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        # Replace O-token confidences
        modified_confs = all_confs.copy()
        modified_confs[o_mask] = o_conf

        T_mod, nll_mod = find_temperature(modified_confs, all_corrects)
        scaled_mod = np.array([apply_temperature(c, T_mod) for c in modified_confs])
        ece_mod = compute_ece(scaled_mod, all_corrects)

        marker = " <-- current" if abs(o_conf - 0.5) < 0.01 else ""
        print(f"    {o_conf:>7.2f}  {T_mod:>10.4f}  {nll_mod:>10.6f}  "
              f"{ece_mod:>10.4f}{marker}")

    # 2e: Platt scaling vs temperature for GLiNER2
    print("\n  2e: Platt scaling (a*logit + b) vs temperature for GLiNER2")
    a, b, nll_platt = platt_scaling(all_confs, all_corrects)
    T_opt, nll_temp = find_temperature(all_confs, all_corrects)

    # Also try entity-only Platt
    a_ent, b_ent, nll_platt_ent = platt_scaling(ent_confs, ent_corrects)

    print(f"    All tokens:")
    print(f"      Temperature: T={T_opt:.4f}, NLL={nll_temp:.6f}")
    print(f"      Platt:       a={a:.4f}, b={b:.4f}, NLL={nll_platt:.6f}")
    print(f"      Platt NLL improvement: {nll_temp - nll_platt:.6f} "
          f"({100*(nll_temp - nll_platt)/nll_temp:.2f}%)")
    print(f"    Entity-only:")
    print(f"      Temperature: T={T_ent_opt:.4f}")
    print(f"      Platt:       a={a_ent:.4f}, b={b_ent:.4f}")

    # 2f: What the calibration landscape looks like
    print("\n  2f: The fundamental problem")
    print(f"    O tokens: {o_mask.sum()} tokens, conf=0.50, accuracy={o_corrects.mean():.4f}")
    print(f"      -> These are {100*o_corrects.mean():.1f}% correct but scored at 50%")
    print(f"      -> Calibrator wants to INCREASE their confidence")
    print(f"    Entity tokens: {ent_pred_mask.sum()} tokens, "
          f"mean conf={ent_confs.mean():.4f}, accuracy={ent_corrects.mean():.4f}")
    print(f"      -> These are {100*ent_corrects.mean():.1f}% correct but scored at "
          f"{100*ent_confs.mean():.1f}%")
    print(f"      -> Calibrator wants to DECREASE their confidence")
    print(f"    Temperature scaling can only do ONE thing: push everything toward 0.5")
    print(f"    or push everything toward 0/1. It cannot push O UP and entities DOWN.")
    print(f"    High T (9.82) compresses entities toward 0.5 — best NLL compromise")
    print(f"    but destroys all discrimination between entity confidence levels.")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
  The O-token confidence of 0.5 is the PRIMARY driver of T=9.82.

  GLiNER2 doesn't produce O predictions — we assigned conf=0.5 to
  all non-entity tokens. These O predictions are ~{o_acc:.0%} correct
  but scored at 50%. Temperature scaling is a single-parameter model:
  it can compress (T>1) or expand (T<1) the logit scale, but it cannot
  independently adjust two populations on different scales.

  The optimizer compromises by choosing T=9.82, which:
  - Compresses entity scores from [0.3, 1.0] to [0.48, 0.67]
  - Leaves O scores at 0.50 (logit=0, invariant to T)
  - Minimizes NLL by accepting poor entity discrimination
    to avoid penalizing the large O-token population

  RECOMMENDED FIX: Either:
  1. Calibrate entity tokens only (separate from O tokens)
  2. Use a higher O-token confidence reflecting actual O accuracy
  3. Use Platt scaling (a*logit + b) which can shift + scale
  4. Use two-population calibration (separate T for O and entity)
""".format(o_acc=o_corrects.mean()))


if __name__ == "__main__":
    main()
