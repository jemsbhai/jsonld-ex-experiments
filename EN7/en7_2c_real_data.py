#!/usr/bin/env python
"""EN7.2c -- Real-World Data Validation of Information-Theoretic Capacity.

NeurIPS 2026 D&B, Suite EN7, Experiment 2c.

Validates the information loss finding from EN7.2/7.2b on THREE real-world
data sources, eliminating the "synthetic distribution artifact" critique.

Real data sources:
    R1 -- Intel Lab sensor data (54 sensors, 2.3M readings, canonical IoT)
          Readings grouped by sensor+hour -> from_evidence() opinions
    R2 -- scikit-learn classifiers on canonical ML benchmarks
          LogisticRegression + RandomForest on Breast Cancer Wisconsin
          and Digits datasets -> from_confidence() opinions
    R3 -- CoNLL-2003 NER per-token confidences (cached from EN1.1)
          5 real NER models (spaCy, Flair, Stanza, HuggingFace, GLiNER-2)
          -> from_confidence() opinions with calibration-derived uncertainty

Pre-registered hypotheses:
    H7 -- Information loss >15% on real sensor evidence-based opinions
    H8 -- Information loss >15% on real ML classifier outputs
    H9 -- Information loss >15% on real NER model per-token confidences

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN7/en7_2c_real_data.py

References:
    Intel Lab data: Bodik et al. (2004), 54 Mica2Dot sensors, MIT license.
    CoNLL-2003: Sang & De Meulder (2003), NER shared task.
    Breast Cancer Wisconsin: Wolberg et al. (1995), UCI ML Repository.
    Digits: Alpaydin & Kaynak (1998), optical handwritten digits.
    Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# -- Path setup ----------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "packages" / "python" / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from jsonld_ex.confidence_algebra import Opinion

from experiments.infra.config import set_global_seed
from experiments.infra.results import ExperimentResult
from experiments.infra.env_log import log_environment

# Import analysis functions from EN7.2
from experiments.EN7.en7_2_info_theoretic import (
    analysis_collisions,
    analysis_entropy,
    analysis_decision_loss,
)

SEED = 42
CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "EN1" / "checkpoints"
INTEL_DATA_PATH = _REPO_ROOT / "data" / "intel_lab" / "data.txt"


# ========================================================================
# R1: Intel Lab Sensor Data
# ========================================================================


def load_intel_lab_opinions(
    max_readings: int = 500_000,
) -> tuple[list[Opinion], dict[str, Any]]:
    """Load Intel Lab sensor data and convert to multi-modal evidence opinions.

    Strategy: Use MULTIPLE physical quantities (temperature, humidity, light)
    with PER-SENSOR reliability and VARIANCE-DERIVED uncertainty to produce
    genuinely heterogeneous opinions.

    For each sensor-window combination:
      1. Compute the empirical fraction above a modality-specific threshold
      2. Derive uncertainty from two independent sources:
         a) Evidence count: fewer readings = higher uncertainty (standard SL)
         b) Reading variance: high variance within window = unreliable reading
      3. Use per-sensor historical noise level to adjust the prior weight

    This mirrors real IoT deployments where multiple sensor channels have
    different noise characteristics and reliability profiles.

    Modalities:
      - Temperature: threshold 25C (comfortable vs warm)
      - Humidity: threshold 40% (dry vs humid)
      - Light: threshold 200 lux (dim vs bright)
    """
    print("  Loading Intel Lab sensor data (multi-modal)...")
    if not INTEL_DATA_PATH.exists():
        print(f"    WARNING: {INTEL_DATA_PATH} not found, skipping R1")
        return [], {"status": "SKIPPED", "reason": "data file not found"}

    # Parse the data file
    # Format: date time epoch moteid temperature humidity light voltage
    raw_readings: list[dict] = []
    total_read = 0
    parse_errors = 0

    with open(INTEL_DATA_PATH, "r") as f:
        for line in f:
            if total_read >= max_readings:
                break
            parts = line.strip().split()
            if len(parts) < 8:
                parse_errors += 1
                continue
            try:
                epoch = int(parts[2])
                mote_id = int(parts[3])
                temperature = float(parts[4])
                humidity = float(parts[5])
                light = float(parts[6])
                voltage = float(parts[7])
                # Skip clearly invalid readings
                if temperature < -10 or temperature > 60:
                    parse_errors += 1
                    continue
                if humidity < 0 or humidity > 100:
                    parse_errors += 1
                    continue
                raw_readings.append({
                    "epoch": epoch, "mote": mote_id,
                    "temp": temperature, "humidity": humidity,
                    "light": light, "voltage": voltage,
                })
                total_read += 1
            except (ValueError, IndexError):
                parse_errors += 1
                continue

    print(f"    Parsed {total_read:,} readings, {parse_errors:,} errors")

    # Step 1: Compute per-sensor historical reliability (noise level)
    # Sensors with high reading variance are less reliable -> higher prior_weight
    sensor_temps: dict[int, list[float]] = defaultdict(list)
    for r in raw_readings:
        sensor_temps[r["mote"]].append(r["temp"])

    sensor_noise: dict[int, float] = {}
    for mote, temps in sensor_temps.items():
        if len(temps) >= 10:
            mean_t = sum(temps) / len(temps)
            var_t = sum((t - mean_t) ** 2 for t in temps) / (len(temps) - 1)
            sensor_noise[mote] = var_t ** 0.5  # std dev
        else:
            sensor_noise[mote] = 5.0  # default high noise for sparse sensors

    # Normalize noise to prior_weight: low noise -> pw=2, high noise -> pw=10
    noise_vals = list(sensor_noise.values())
    noise_min = min(noise_vals) if noise_vals else 1.0
    noise_max = max(noise_vals) if noise_vals else 5.0
    noise_range = max(noise_max - noise_min, 0.01)

    sensor_prior_weight: dict[int, float] = {}
    for mote, noise in sensor_noise.items():
        normalized = (noise - noise_min) / noise_range  # 0 = cleanest, 1 = noisiest
        sensor_prior_weight[mote] = 2.0 + normalized * 8.0  # pw in [2, 10]

    print(f"    Sensor noise range: {noise_min:.2f} - {noise_max:.2f} (std dev)")
    print(f"    Prior weight range: {min(sensor_prior_weight.values()):.1f} - "
          f"{max(sensor_prior_weight.values()):.1f}")

    # Step 2: Group readings into 10-minute windows
    windows: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in raw_readings:
        w = r["epoch"] // 600  # 10-minute windows
        windows[(r["mote"], w)].append(r)

    # Step 3: Generate opinions from multiple modalities per window
    MODALITIES = [
        ("temp", 25.0, "temperature"),
        ("humidity", 40.0, "humidity"),
        ("light", 200.0, "light"),
    ]

    opinions = []
    modality_counts = defaultdict(int)

    for (mote, w), readings in windows.items():
        if len(readings) < 2:
            continue

        pw = sensor_prior_weight.get(mote, 5.0)

        for field, threshold, mod_name in MODALITIES:
            values = [r[field] for r in readings]

            # Skip if all values are identical (no information)
            if len(set(round(v, 1) for v in values)) < 2 and len(values) < 5:
                continue

            # Evidence counts
            pos = sum(1 for v in values if v >= threshold)
            neg = len(values) - pos

            # Variance-based uncertainty boost:
            # High within-window variance = less trustworthy = inflate prior
            mean_val = sum(values) / len(values)
            if len(values) >= 2:
                var_val = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
                std_val = var_val ** 0.5
                # Coefficient of variation (relative to threshold)
                cv = std_val / max(abs(threshold), 1.0)
                # Boost prior weight by CV: high variance -> more uncertainty
                effective_pw = pw * (1.0 + cv * 3.0)
            else:
                effective_pw = pw * 2.0

            op = Opinion.from_evidence(
                positive=pos,
                negative=neg,
                prior_weight=effective_pw,
            )
            opinions.append(op)
            modality_counts[mod_name] += 1

    # Compute stats
    uncertainties = [op.uncertainty for op in opinions]
    beliefs = [op.belief for op in opinions]
    projected = [op.projected_probability() for op in opinions]

    metadata = {
        "status": "OK",
        "total_readings_parsed": total_read,
        "parse_errors": parse_errors,
        "n_sensors": len(sensor_noise),
        "n_windows": len(windows),
        "n_opinions_generated": len(opinions),
        "modality_counts": dict(modality_counts),
        "mean_uncertainty": round(sum(uncertainties) / max(len(uncertainties), 1), 4),
        "std_uncertainty": round(
            (sum((u - sum(uncertainties)/len(uncertainties))**2
                 for u in uncertainties) / max(len(uncertainties)-1, 1))**0.5, 4
        ) if len(uncertainties) > 1 else 0,
        "mean_belief": round(sum(beliefs) / max(len(beliefs), 1), 4),
        "mean_P": round(sum(projected) / max(len(projected), 1), 4),
    }
    print(f"    Modalities: {dict(modality_counts)}")
    print(f"    Total: {len(opinions):,} opinions, "
          f"mean_u={metadata['mean_uncertainty']:.4f}, "
          f"std_u={metadata['std_uncertainty']:.4f}")

    return opinions, metadata


# ========================================================================
# R2: scikit-learn Classifiers on Canonical Datasets
# ========================================================================


def load_sklearn_opinions() -> tuple[list[Opinion], dict[str, Any]]:
    """Train classifiers on canonical datasets, extract real predicted probabilities.

    Datasets:
        - Breast Cancer Wisconsin (binary, 569 samples)
        - Digits (10-class, 1797 samples)

    Models:
        - LogisticRegression (well-calibrated by default)
        - RandomForestClassifier (typically overconfident)

    For each test sample, the model's predicted probability for the
    true class is converted to an SL opinion:
      - from_confidence(p_true_class, uncertainty=entropy_of_distribution)

    The uncertainty is derived from the full softmax/probability vector
    using normalized entropy: u = H(p) / log2(n_classes). This gives
    a principled, data-derived uncertainty that reflects how spread
    the model's belief is across classes.
    """
    from sklearn.datasets import load_breast_cancer, load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict

    print("  Training scikit-learn classifiers on canonical datasets...")

    opinions = []
    metadata_parts = {}

    for ds_name, loader in [("breast_cancer", load_breast_cancer),
                            ("digits", load_digits)]:
        data = loader()
        X, y = data.data, data.target
        n_classes = len(set(y))

        for model_name, model in [
            ("logistic_regression", LogisticRegression(max_iter=5000, random_state=SEED)),
            ("random_forest", RandomForestClassifier(n_estimators=100, random_state=SEED)),
        ]:
            print(f"    {ds_name} / {model_name} (n={len(y)}, classes={n_classes})...")

            # Cross-validated predictions for unbiased probability estimates
            proba = cross_val_predict(model, X, y, cv=5, method="predict_proba")

            ds_opinions = []
            for i in range(len(y)):
                true_class = y[i]
                p_true = float(proba[i, true_class])
                p_true = max(0.001, min(0.999, p_true))

                # Uncertainty from normalized entropy of probability vector
                p_vec = proba[i]
                entropy = -sum(
                    p * math.log2(p) if p > 1e-12 else 0.0
                    for p in p_vec
                )
                max_entropy = math.log2(n_classes) if n_classes > 1 else 1.0
                u = min(0.99, entropy / max_entropy)
                u = max(0.01, u)

                op = Opinion.from_confidence(p_true, uncertainty=u)
                ds_opinions.append(op)
                opinions.append(op)

            # Per-dataset/model stats
            confs = [proba[i, y[i]] for i in range(len(y))]
            uncertainties = [op.uncertainty for op in ds_opinions]
            metadata_parts[f"{ds_name}_{model_name}"] = {
                "n_samples": len(y),
                "n_classes": n_classes,
                "mean_confidence": round(sum(confs) / len(confs), 4),
                "mean_uncertainty": round(sum(uncertainties) / len(uncertainties), 4),
                "n_opinions": len(ds_opinions),
            }
            print(f"      mean_conf={sum(confs)/len(confs):.3f}, "
                  f"mean_u={sum(uncertainties)/len(uncertainties):.3f}")

    metadata = {
        "status": "OK",
        "total_opinions": len(opinions),
        "datasets": metadata_parts,
    }

    return opinions, metadata


# ========================================================================
# R3: CoNLL-2003 NER Per-Token Confidences
# ========================================================================


def load_ner_opinions(
    uncertainty_mode: str = "disagreement",
) -> tuple[list[Opinion], dict[str, Any]]:
    """Load cached NER predictions and derive per-token SL opinions.

    KEY INSIGHT: The scientifically correct way to derive uncertainty
    from multiple NER models is CROSS-MODEL DISAGREEMENT. If 5/5 models
    agree on a token's tag, uncertainty is low. If only 2/5 agree,
    uncertainty is high.

    This produces per-token uncertainty that varies based on actual
    model behavior, not a fixed constant.

    Modes:
        "disagreement" -- u = 1 - (max_agreement / n_models)
                          Confidence = mean confidence of agreeing models.
                          This is the primary, scientifically defensible mode.
        "calibration"  -- u = 1 - model_accuracy (fixed per model, baseline)
        "mixed"        -- Weighted combination of disagreement + calibration
    """
    print("  Loading CoNLL-2003 NER cached predictions...")

    # Known model accuracies from EN1.1 (entity-level F1)
    model_accuracy = {
        "spacy": 0.463,
        "flair": 0.925,
        "stanza": 0.524,
        "huggingface": 0.913,
        "gliner2": 0.900,
    }

    # Load all available model predictions
    all_model_preds: dict[str, list[list[dict]]] = {}
    for model_name in ["spacy", "flair", "stanza", "huggingface", "gliner2"]:
        pred_path = CHECKPOINTS_DIR / f"test_preds_{model_name}.json"
        if not pred_path.exists():
            print(f"    {model_name}: checkpoint not found, skipping")
            continue
        with open(pred_path, "r") as f:
            all_model_preds[model_name] = json.load(f)
        print(f"    {model_name}: loaded {len(all_model_preds[model_name])} sentences")

    if len(all_model_preds) < 2:
        return [], {"status": "SKIPPED", "reason": f"need >=2 models, found {len(all_model_preds)}"}

    model_names = list(all_model_preds.keys())
    n_models = len(model_names)
    print(f"    Using {n_models} models: {model_names}")

    # Find common sentence count (models may differ slightly in tokenization)
    n_sents = min(len(preds) for preds in all_model_preds.values())

    opinions = []
    agreement_scores = []
    per_model_counts: dict[str, int] = defaultdict(int)

    for sent_idx in range(n_sents):
        # Get predictions from each model for this sentence
        model_sent_preds = {}
        min_tokens = float('inf')
        for model_name in model_names:
            preds = all_model_preds[model_name]
            if sent_idx < len(preds):
                model_sent_preds[model_name] = preds[sent_idx]
                min_tokens = min(min_tokens, len(preds[sent_idx]))

        if len(model_sent_preds) < 2 or min_tokens == float('inf'):
            continue

        # For each token position, compute cross-model agreement
        for tok_idx in range(int(min_tokens)):
            tags = []
            confs = []
            for model_name in model_names:
                if model_name in model_sent_preds:
                    tok = model_sent_preds[model_name]
                    if tok_idx < len(tok):
                        tags.append(tok[tok_idx]["tag"])
                        confs.append(float(tok[tok_idx]["confidence"]))
                        per_model_counts[model_name] += 1

            if len(tags) < 2:
                continue

            # Compute agreement
            tag_counts = defaultdict(int)
            tag_conf_sums = defaultdict(float)
            for tag, conf in zip(tags, confs):
                tag_counts[tag] += 1
                tag_conf_sums[tag] += conf

            # Majority tag and its agreement fraction
            majority_tag = max(tag_counts, key=tag_counts.get)
            agreement_frac = tag_counts[majority_tag] / len(tags)
            agreement_scores.append(agreement_frac)

            # Mean confidence of agreeing models for the majority tag
            mean_majority_conf = tag_conf_sums[majority_tag] / tag_counts[majority_tag]
            mean_majority_conf = max(0.001, min(0.999, mean_majority_conf))

            if uncertainty_mode == "disagreement":
                # Pure disagreement-based uncertainty
                # All agree (5/5): u = 0.0 -> clamp to 0.02
                # 3/5 agree: u = 0.4
                # 2/5 agree: u = 0.6
                u = max(0.02, 1.0 - agreement_frac)
            elif uncertainty_mode == "calibration":
                # Accuracy-weighted mean uncertainty across models
                model_us = [1.0 - model_accuracy.get(m, 0.5) for m in model_names
                            if m in model_sent_preds]
                u = sum(model_us) / len(model_us)
                u = max(0.02, min(0.98, u))
            elif uncertainty_mode == "mixed":
                # 50/50 blend of disagreement and calibration
                u_disagree = max(0.02, 1.0 - agreement_frac)
                model_us = [1.0 - model_accuracy.get(m, 0.5) for m in model_names
                            if m in model_sent_preds]
                u_cal = sum(model_us) / len(model_us)
                u = 0.5 * u_disagree + 0.5 * u_cal
                u = max(0.02, min(0.98, u))
            else:
                u = max(0.02, 1.0 - agreement_frac)

            op = Opinion.from_confidence(mean_majority_conf, uncertainty=u)
            opinions.append(op)

    # Stats
    uncertainties = [op.uncertainty for op in opinions]
    metadata = {
        "status": "OK",
        "uncertainty_mode": uncertainty_mode,
        "n_models": n_models,
        "model_names": model_names,
        "n_sentences": n_sents,
        "n_opinions_generated": len(opinions),
        "mean_uncertainty": round(sum(uncertainties) / max(len(uncertainties), 1), 4),
        "std_uncertainty": round(
            (sum((u - sum(uncertainties)/len(uncertainties))**2
                 for u in uncertainties) / max(len(uncertainties)-1, 1))**0.5, 4
        ) if len(uncertainties) > 1 else 0,
        "mean_agreement": round(
            sum(agreement_scores) / max(len(agreement_scores), 1), 4
        ),
        "per_model_token_counts": dict(per_model_counts),
    }
    print(f"    Total: {len(opinions):,} cross-model opinions, "
          f"mean_u={metadata['mean_uncertainty']:.4f}, "
          f"std_u={metadata['std_uncertainty']:.4f}, "
          f"mean_agreement={metadata['mean_agreement']:.4f}")

    return opinions, metadata


# ========================================================================
# Run Analysis on a Real Dataset
# ========================================================================


def analyze_real_dataset(
    name: str,
    opinions: list[Opinion],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Run the full EN7.2 analysis suite on a real-world opinion set."""
    if not opinions:
        return {"status": "SKIPPED", "metadata": metadata}

    # Cap at 50K for analysis efficiency (random subsample if larger)
    if len(opinions) > 50_000:
        rng = random.Random(SEED)
        opinions = rng.sample(opinions, 50_000)
        print(f"    (subsampled to 50,000 for analysis)")

    # Descriptive stats
    beliefs = [op.belief for op in opinions]
    disbeliefs = [op.disbelief for op in opinions]
    uncertainties = [op.uncertainty for op in opinions]
    projected = [op.projected_probability() for op in opinions]

    desc = {
        "n_opinions": len(opinions),
        "mean_belief": round(sum(beliefs) / len(beliefs), 4),
        "mean_disbelief": round(sum(disbeliefs) / len(disbeliefs), 4),
        "mean_uncertainty": round(sum(uncertainties) / len(uncertainties), 4),
        "std_uncertainty": round(
            (sum((u - sum(uncertainties)/len(uncertainties))**2
                 for u in uncertainties) / max(len(uncertainties)-1, 1))**0.5, 4
        ),
        "mean_P": round(sum(projected) / len(projected), 4),
    }

    # Run analyses
    collisions = analysis_collisions(opinions)
    entropy = analysis_entropy(opinions)
    decision = analysis_decision_loss(opinions)

    return {
        "status": "OK",
        "metadata": metadata,
        "descriptive_stats": desc,
        "entropy": {
            "H_opinion_bits": entropy["H_opinion_bits"],
            "H_scalar_bits": entropy["H_scalar_bits"],
            "H_opinion_given_scalar_bits": entropy["H_opinion_given_scalar_bits"],
            "mutual_information_bits": entropy["mutual_information_bits"],
            "information_preserved_pct": entropy["information_preserved_pct"],
            "information_lost_pct": entropy["information_lost_pct"],
        },
        "collisions": {
            "mean_per_bin": collisions["mean_per_bin"],
            "max_per_bin": collisions["max_per_bin"],
            "mean_uncertainty_range": collisions["mean_uncertainty_range"],
        },
        "decision": {
            "conflicting_pairs": decision["conflicting_pairs_found"],
            "fraction_conflicting": decision["fraction_conflicting"],
            "total_pairs_checked": decision["total_pairs_checked"],
        },
    }


# ========================================================================
# NER Uncertainty Mode Sweep
# ========================================================================


def run_ner_uncertainty_sweep() -> dict[str, Any]:
    """Test NER info loss under all three uncertainty assignment modes.

    Shows that the finding is robust to the choice of how
    uncertainty is derived from the NER model predictions.
    """
    results = {}
    for mode in ["disagreement", "calibration", "mixed"]:
        print(f"\n    NER uncertainty mode: {mode}")
        opinions, meta = load_ner_opinions(uncertainty_mode=mode)
        if not opinions:
            results[mode] = {"status": "SKIPPED"}
            continue

        # Subsample for speed
        if len(opinions) > 30_000:
            rng = random.Random(SEED)
            opinions = rng.sample(opinions, 30_000)

        entropy = analysis_entropy(opinions)
        results[mode] = {
            "n_opinions": len(opinions),
            "H_opinion_bits": entropy["H_opinion_bits"],
            "H_lost_bits": entropy["H_opinion_given_scalar_bits"],
            "pct_lost": entropy["information_lost_pct"],
            "uncertainty_mode": mode,
            "mean_uncertainty": meta.get("mean_uncertainty", 0),
            "std_uncertainty": meta.get("std_uncertainty", 0),
        }
        print(f"      H_lost={entropy['H_opinion_given_scalar_bits']:.2f} bits "
              f"({entropy['information_lost_pct']:.1f}% lost)")

    return results


# ========================================================================
# Main Runner
# ========================================================================


def run_en7_2c() -> ExperimentResult:
    """Run all EN7.2c real-data analyses."""
    set_global_seed(SEED)

    print("=" * 60)
    print("  EN7.2c: Real-World Data Validation")
    print("=" * 60)

    t_start = time.perf_counter()
    all_results = {}

    # R1: Intel Lab sensors
    print(f"\n--- R1: Intel Lab Sensor Data ---")
    t0 = time.perf_counter()
    sensor_opinions, sensor_meta = load_intel_lab_opinions()
    r1 = analyze_real_dataset("Intel Lab Sensors", sensor_opinions, sensor_meta)
    all_results["R1_intel_lab"] = r1
    if r1["status"] == "OK":
        print(f"    H_lost={r1['entropy']['H_opinion_given_scalar_bits']:.2f} bits "
              f"({r1['entropy']['information_lost_pct']:.1f}% lost)")
    print(f"  [R1 complete in {time.perf_counter() - t0:.1f}s]")

    # R2: scikit-learn classifiers
    print(f"\n--- R2: scikit-learn Classifiers ---")
    t0 = time.perf_counter()
    sklearn_opinions, sklearn_meta = load_sklearn_opinions()
    r2 = analyze_real_dataset("scikit-learn Classifiers", sklearn_opinions, sklearn_meta)
    all_results["R2_sklearn"] = r2
    if r2["status"] == "OK":
        print(f"    H_lost={r2['entropy']['H_opinion_given_scalar_bits']:.2f} bits "
              f"({r2['entropy']['information_lost_pct']:.1f}% lost)")
    print(f"  [R2 complete in {time.perf_counter() - t0:.1f}s]")

    # R3: CoNLL-2003 NER (default: disagreement uncertainty)
    print(f"\n--- R3: CoNLL-2003 NER (cross-model disagreement uncertainty) ---")
    t0 = time.perf_counter()
    ner_opinions, ner_meta = load_ner_opinions(uncertainty_mode="disagreement")
    r3 = analyze_real_dataset("CoNLL-2003 NER", ner_opinions, ner_meta)
    all_results["R3_conll2003"] = r3
    if r3["status"] == "OK":
        print(f"    H_lost={r3['entropy']['H_opinion_given_scalar_bits']:.2f} bits "
              f"({r3['entropy']['information_lost_pct']:.1f}% lost)")
    print(f"  [R3 complete in {time.perf_counter() - t0:.1f}s]")

    # R3b: NER uncertainty mode sweep
    print(f"\n--- R3b: NER Uncertainty Mode Sweep ---")
    t0 = time.perf_counter()
    ner_sweep = run_ner_uncertainty_sweep()
    all_results["R3b_ner_uncertainty_sweep"] = ner_sweep
    print(f"  [R3b complete in {time.perf_counter() - t0:.1f}s]")

    total_time = time.perf_counter() - t_start

    # Summary table
    print(f"\n{'='*60}")
    print(f"  CROSS-DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<30} {'H_lost':>7} {'%lost':>6} {'u_range':>8} {'conflict':>9}")
    print(f"  {'-'*30} {'-'*7} {'-'*6} {'-'*8} {'-'*9}")

    summary_rows = []
    for name, key in [("Intel Lab Sensors", "R1_intel_lab"),
                      ("scikit-learn Classifiers", "R2_sklearn"),
                      ("CoNLL-2003 NER", "R3_conll2003")]:
        r = all_results.get(key, {})
        if r.get("status") != "OK":
            print(f"  {name:<30} SKIPPED")
            continue
        h_lost = r["entropy"]["H_opinion_given_scalar_bits"]
        pct = r["entropy"]["information_lost_pct"]
        u_range = r["collisions"]["mean_uncertainty_range"]
        conflict = r["decision"]["fraction_conflicting"]
        print(f"  {name:<30} {h_lost:>7.2f} {pct:>5.1f}% {u_range:>8.3f} {conflict:>8.3f}")
        summary_rows.append({
            "dataset": name,
            "H_lost": h_lost,
            "pct_lost": pct,
            "u_range": u_range,
            "conflict_rate": conflict,
        })

    # Hypothesis outcomes
    h7 = (all_results.get("R1_intel_lab", {}).get("status") == "OK" and
           all_results["R1_intel_lab"]["entropy"]["information_lost_pct"] > 15)
    h8 = (all_results.get("R2_sklearn", {}).get("status") == "OK" and
           all_results["R2_sklearn"]["entropy"]["information_lost_pct"] > 15)
    h9 = (all_results.get("R3_conll2003", {}).get("status") == "OK" and
           all_results["R3_conll2003"]["entropy"]["information_lost_pct"] > 15)

    metrics = {
        "H7_sensor_loss_gt_15pct": h7,
        "H8_sklearn_loss_gt_15pct": h8,
        "H9_ner_loss_gt_15pct": h9,
        "all_confirmed": h7 and h8 and h9,
        "summary": summary_rows,
        "total_time_sec": round(total_time, 1),
    }

    result = ExperimentResult(
        experiment_id="EN7.2c",
        parameters={
            "seed": SEED,
            "intel_lab_max_readings": 500_000,
            "intel_lab_modalities": ["temperature (>25C)", "humidity (>40%)", "light (>200 lux)"],
            "intel_lab_window_minutes": 10,
            "intel_lab_uncertainty": "per-sensor noise + within-window variance",
            "sklearn_datasets": ["breast_cancer", "digits"],
            "sklearn_models": ["logistic_regression", "random_forest"],
            "ner_models": ["spacy", "flair", "stanza", "huggingface", "gliner2"],
            "ner_uncertainty_mode": "disagreement",
        },
        metrics=metrics,
        raw_data=all_results,
        environment=log_environment(),
        notes=(
            f"H7={'CONFIRMED' if h7 else 'REJECTED'}, "
            f"H8={'CONFIRMED' if h8 else 'REJECTED'}, "
            f"H9={'CONFIRMED' if h9 else 'REJECTED'}. "
            f"Total time: {total_time:.1f}s."
        ),
    )

    # Save
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "en7_2c_results.json"
    result.save_json(str(out_path))
    print(f"\nResults saved: {out_path}")

    print(f"\n{'='*60}")
    print(f"  HYPOTHESIS OUTCOMES (total: {total_time:.1f}s)")
    print(f"{'='*60}")
    print(f"  H7 (sensor data >15% loss):  {'CONFIRMED' if h7 else 'REJECTED'}")
    print(f"  H8 (sklearn ML >15% loss):   {'CONFIRMED' if h8 else 'REJECTED'}")
    print(f"  H9 (NER tokens >15% loss):   {'CONFIRMED' if h9 else 'REJECTED'}")
    print(f"  All confirmed: {h7 and h8 and h9}")
    print(f"{'='*60}")

    return result


if __name__ == "__main__":
    run_en7_2c()
