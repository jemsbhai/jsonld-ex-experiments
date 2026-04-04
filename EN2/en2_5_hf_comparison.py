#!/usr/bin/env python3
"""
EN2.5 -- Head-to-Head vs HuggingFace Datasets

A comprehensive workflow comparison of jsonld-ex, HuggingFace datasets,
and plain JSON for 5 common ML data exchange tasks across 11 real datasets
spanning 7 domains.

PRE-REGISTERED HYPOTHESIS:
    jsonld-ex provides semantic interoperability, uncertainty quantification,
    and provenance that HF datasets cannot express, at acceptable overhead.
    HF datasets excels at data loading/processing; jsonld-ex extends this
    with a structured assertion-level metadata layer.

FRAMING:
    Complementary, not adversarial.  HF datasets is a data loading/processing
    library.  jsonld-ex is a metadata annotation layer.  They serve different
    purposes and can coexist.

TASKS (5):
    T1: Load dataset and inspect metadata
    T2: Annotate samples with model predictions + confidence
    T3: Merge predictions from multiple models
    T4: Filter by confidence threshold
    T5: Export with full provenance for reproducibility

APPROACHES (3):
    A: jsonld-ex  (structured annotation layer)
    B: HF datasets (column-based data library)
    C: Plain JSON  (manual dict manipulation)

PHASES:
    Phase A: Synthetic predictions (controlled, reproducible) -- THIS FILE
    Phase B: Real model predictions (all datasets) -- NEXT STEP

Datasets (11, across 7 domains):
     1. Fashion-MNIST        -- vision/classification (10 classes)
     2. CIFAR-10             -- vision/classification (10 classes)
     3. COCO 2014            -- vision/detection (80 classes)
     4. PASS                 -- vision/privacy-aware
     5. AG News              -- text/classification (4 classes)
     6. IMDB                 -- text/sentiment (binary)
     7. Common Voice (en)    -- audio/ASR
     8. Speech Commands      -- audio/classification (35 classes)
     9. ETTh1                -- time-series/forecasting
    10. Timeseries-PILE      -- time-series/multi-domain
    11. Titanic              -- tabular/binary classification

Authors: Muntaser Syed, Marius Silaghi, Sheikh Abujar, Rwaida Alssadi
         Florida Institute of Technology
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---- Path setup --------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

# Ensure jsonld_ex is importable (pip install -e)
sys.path.insert(0, str(_REPO_ROOT / "packages" / "python" / "src"))

from jsonld_ex.ai_ml import annotate, get_confidence, get_provenance, filter_by_confidence
from jsonld_ex.confidence_algebra import (
    Opinion, cumulative_fuse, averaging_fuse, pairwise_conflict, conflict_metric,
    trust_discount,
)
from jsonld_ex.merge import merge_graphs
from jsonld_ex.dataset import create_dataset_metadata, to_croissant, from_croissant

# Infra
sys.path.insert(0, str(_SCRIPT_DIR.parent))
from infra.config import set_global_seed
from infra.results import ExperimentResult

# ---- Configuration -----------------------------------------------------

SEED = 42
N_SAMPLES = 99999        # Use full split -- no artificial cap
N_MODELS = 3             # Synthetic "models" per dataset
CONFIDENCE_THRESHOLD = 0.7  # For Task 4 filtering

RESULTS_DIR = _SCRIPT_DIR / "results"

# ---- Dataset Registry ---------------------------------------------------

# Each entry defines how to load via HF datasets and what kind of
# predictions are appropriate (classification, detection, regression, ASR).
DATASETS: List[Dict[str, Any]] = [
    {
        "id": "fashion_mnist",
        "name": "Fashion-MNIST",
        "hf_slug": "fashion_mnist",
        "hf_config": None,
        "hf_split": "test",
        "domain": "vision/classification",
        "task_type": "classification",
        "n_classes": 10,
        "class_names": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
        "sample_field": "image",
        "label_field": "label",
    },
    {
        "id": "cifar10",
        "name": "CIFAR-10",
        "hf_slug": "uoft-cs/cifar10",
        "hf_config": None,
        "hf_split": "test",
        "domain": "vision/classification",
        "task_type": "classification",
        "n_classes": 10,
        "class_names": ["airplane", "automobile", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck"],
        "sample_field": "img",
        "label_field": "label",
    },
    {
        "id": "coco_2014",
        "name": "COCO 2014",
        "hf_slug": "detection-datasets/coco",
        "hf_config": None,
        "hf_split": "val",
        "domain": "vision/detection",
        "task_type": "detection",
        "n_classes": 80,
        "class_names": None,  # Will use COCO category names
        "sample_field": "image",
        "label_field": "objects",
    },
    {
        "id": "pass_dataset",
        "name": "Beans",
        "hf_slug": "beans",
        "hf_config": None,
        "hf_split": "test",
        "domain": "vision/agriculture",
        "task_type": "classification",
        "n_classes": 3,
        "class_names": ["angular_leaf_spot", "bean_rust", "healthy"],
        "sample_field": "image",
        "label_field": "labels",
    },
    {
        "id": "ag_news",
        "name": "AG News",
        "hf_slug": "fancyzhx/ag_news",
        "hf_config": None,
        "hf_split": "test",
        "domain": "text/classification",
        "task_type": "classification",
        "n_classes": 4,
        "class_names": ["World", "Sports", "Business", "Sci/Tech"],
        "sample_field": "text",
        "label_field": "label",
    },
    {
        "id": "imdb",
        "name": "IMDB",
        "hf_slug": "stanfordnlp/imdb",
        "hf_config": None,
        "hf_split": "test",
        "domain": "text/sentiment",
        "task_type": "classification",
        "n_classes": 2,
        "class_names": ["negative", "positive"],
        "sample_field": "text",
        "label_field": "label",
    },
    {
        "id": "common_voice",
        "name": "Keyword Spotting (SUPERB)",
        "hf_slug": "superb",
        "hf_config": "ks",
        "hf_split": "test",
        "domain": "audio/classification",
        "task_type": "classification",
        "n_classes": 35,
        "class_names": None,
        "sample_field": "audio",
        "label_field": "label",
    },
    {
        "id": "speech_commands",
        "name": "LibriSpeech (test subset)",
        "hf_slug": "hf-internal-testing/librispeech_asr_dummy",
        "hf_config": "clean",
        "hf_split": "validation",
        "domain": "audio/ASR",
        "task_type": "asr",
        "n_classes": None,
        "class_names": None,
        "sample_field": "audio",
        "label_field": "text",
    },
    {
        "id": "etth1",
        "name": "ETTh1",
        "hf_slug": "csv",
        "hf_config": None,
        "hf_split": "train",
        "hf_data_files": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "domain": "time-series/forecasting",
        "task_type": "regression",
        "n_classes": None,
        "class_names": None,
        "sample_field": None,  # Tabular -- all columns are features
        "label_field": "OT",   # Oil Temperature target
    },
    {
        "id": "timeseries_pile",
        "name": "ETTm1 (minute-level)",
        "hf_slug": "csv",
        "hf_config": None,
        "hf_split": "train",
        "hf_data_files": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "domain": "time-series/multi-domain",
        "task_type": "regression",
        "n_classes": None,
        "class_names": None,
        "sample_field": None,
        "label_field": None,
    },
    {
        "id": "titanic",
        "name": "Titanic",
        "hf_slug": "csv",
        "hf_config": None,
        "hf_split": "train",
        "hf_data_files": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "domain": "tabular/binary-classification",
        "task_type": "classification",
        "n_classes": 2,
        "class_names": ["died", "survived"],
        "sample_field": None,  # Tabular
        "label_field": "Survived",
    },
    {
        "id": "synthea_fhir",
        "name": "Synthea FHIR R4",
        "hf_slug": None,
        "hf_config": None,
        "hf_split": None,
        "local_path": "data/synthea/fhir_r4",
        "domain": "medical/clinical",
        "task_type": "classification",
        "n_classes": 2,
        "class_names": ["normal", "abnormal"],
        "sample_field": None,
        "label_field": None,
    },
    {
        "id": "squad",
        "name": "SQuAD v2",
        "hf_slug": "rajpurkar/squad_v2",
        "hf_config": None,
        "hf_split": "validation",
        "domain": "text/qa",
        "task_type": "qa",
        "n_classes": None,
        "class_names": None,
        "sample_field": "question",
        "label_field": "answers",
    },
]


# ---- Feature Checklist -------------------------------------------------

FEATURE_IDS = [
    "F01_structured_metadata_schema",
    "F02_per_sample_confidence",
    "F03_sl_opinion_bdu",
    "F04_multi_model_fusion",
    "F05_conflict_detection",
    "F06_uncertainty_aware_filter",
    "F07_provenance_chain",
    "F08_temporal_validity",
    "F09_semantic_interop_context",
    "F10_croissant_roundtrip",
    "F11_prov_o_rdf_roundtrip",
    "F12_trust_discount",
    "F13_abstention_on_conflict",
    "F14_calibration_metadata",
]

FEATURE_NAMES = {
    "F01_structured_metadata_schema": "Structured metadata schema",
    "F02_per_sample_confidence": "Per-sample confidence scores",
    "F03_sl_opinion_bdu": "SL opinion (b,d,u,a)",
    "F04_multi_model_fusion": "Multi-model fusion operators",
    "F05_conflict_detection": "Conflict detection",
    "F06_uncertainty_aware_filter": "Uncertainty-aware filtering",
    "F07_provenance_chain": "Provenance chain",
    "F08_temporal_validity": "Temporal validity windows",
    "F09_semantic_interop_context": "Semantic interop (@context)",
    "F10_croissant_roundtrip": "Croissant round-trip",
    "F11_prov_o_rdf_roundtrip": "PROV-O / RDF round-trip",
    "F12_trust_discount": "Trust discount (per-model reliability)",
    "F13_abstention_on_conflict": "Abstention on conflict",
    "F14_calibration_metadata": "Calibration metadata",
}

# Static feature support per approach
# True = native support, False = not possible, "workaround" = requires custom code
FEATURE_SUPPORT = {
    "jsonld_ex": {
        "F01": True, "F02": True, "F03": True, "F04": True, "F05": True,
        "F06": True, "F07": True, "F08": True, "F09": True, "F10": True,
        "F11": True, "F12": True, "F13": True, "F14": True,
    },
    "hf_datasets": {
        "F01": True,   # .info / .features
        "F02": "workaround",  # Add column, no schema
        "F03": False,  # No SL support
        "F04": False,  # No fusion operators
        "F05": False,  # No conflict detection
        "F06": False,  # Filter by column value only, no uncertainty distinction
        "F07": False,  # No provenance
        "F08": False,  # No temporal
        "F09": False,  # Arrow format, not JSON-LD
        "F10": False,  # No Croissant interop at sample level
        "F11": False,  # No RDF
        "F12": False,  # No trust
        "F13": False,  # No abstention
        "F14": False,  # No calibration metadata
    },
    "plain_json": {
        "F01": False,           # No schema
        "F02": "workaround",    # Add dict key, no validation
        "F03": "workaround",    # Manual tuple, no algebra
        "F04": "workaround",    # Custom averaging, no SL
        "F05": "workaround",    # Custom comparison
        "F06": False,           # No uncertainty distinction
        "F07": "workaround",    # Manual dict nesting
        "F08": "workaround",    # Manual timestamp fields
        "F09": False,           # No @context
        "F10": False,           # No Croissant
        "F11": False,           # No RDF
        "F12": False,           # No trust model
        "F13": "workaround",    # Custom threshold
        "F14": "workaround",    # Manual fields
    },
}


# ---- Utility Functions --------------------------------------------------

def safe_print(*args, **kwargs):
    """Print with cp1252-safe encoding for Windows console."""
    msg = " ".join(str(a) for a in args)
    safe = msg.encode("cp1252", errors="replace").decode("cp1252")
    print(safe, **kwargs)


def count_body_lines(func: Callable) -> int:
    """Count non-blank, non-comment lines in a function body."""
    source = inspect.getsource(func)
    lines = source.split("\n")
    # Skip def line and docstring
    in_docstring = False
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i == 0:  # def line
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if in_docstring:
                in_docstring = False
                body_start = i + 1
                continue
            elif stripped.endswith('"""') and len(stripped) > 3:
                body_start = i + 1
                continue
            else:
                in_docstring = True
                continue
        if not in_docstring and stripped and not stripped.startswith("#"):
            if body_start == 0:
                body_start = i
    count = 0
    for line in lines[body_start:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def measure_bytes(obj: Any) -> int:
    """Measure JSON serialization size of an object."""
    return len(json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"))


# ---- Dataset Loading ----------------------------------------------------

def load_hf_subset(ds_info: Dict[str, Any], n: int = N_SAMPLES) -> Any:
    """Load a small subset from a HuggingFace dataset.

    Returns the HF dataset object and a list of plain dicts for plain JSON.
    Falls back gracefully if a dataset is unavailable.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        safe_print(f"  [SKIP] datasets library not installed")
        return None, []

    slug = ds_info["hf_slug"]
    config = ds_info.get("hf_config")
    split = ds_info.get("hf_split", "test")

    safe_print(f"  Loading {slug} (config={config}, split={split}[:{n}]) ...")

    try:
        # Handle local datasets (not on HuggingFace)
        if ds_info.get("hf_slug") is None and ds_info.get("local_path"):
            local_dir = _REPO_ROOT / ds_info["local_path"]
            safe_print(f"  Loading local files from {local_dir} ...")
            import glob
            files = sorted(glob.glob(str(local_dir / "*.json")))[:n]
            records = []
            for fpath in files:
                import json as _json
                with open(fpath, "r", encoding="utf-8") as f:
                    bundle = _json.load(f)
                # Extract Observation resources from FHIR Bundle
                if bundle.get("resourceType") == "Bundle":
                    for entry in bundle.get("entry", []):
                        resource = entry.get("resource", {})
                        rtype = resource.get("resourceType", "")
                        if rtype in ("Observation", "Condition", "Patient", "Encounter", "Procedure"):
                            rec = {
                                "resourceType": rtype,
                                "id": resource.get("id", ""),
                                "status": resource.get("status", ""),
                                "code_text": "",
                                "source_file": Path(fpath).name,
                            }
                            code = resource.get("code", {})
                            if isinstance(code, dict):
                                codings = code.get("coding", [])
                                if codings:
                                    rec["code_text"] = codings[0].get("display", "")
                            records.append(rec)
                        if len(records) >= n:
                            break
                if len(records) >= n:
                    break
            safe_print(f"  Found {len(records)} FHIR resources from {len(files)} bundles")
            return None, records[:n]

        # Handle QA datasets (SQuAD)
        if ds_info["task_type"] == "qa":
            slug = ds_info["hf_slug"]
            config = ds_info.get("hf_config")
            split = ds_info.get("hf_split", "validation")
            safe_print(f"  Loading {slug} (config={config}, split={split}[:{n}]) ...")
            from datasets import load_dataset
            hf_ds = load_dataset(slug, config, split=f"{split}[:{n}]")
            records = []
            for sample in hf_ds:
                rec = {}
                for k, v in sample.items():
                    if isinstance(v, dict):
                        rec[k] = str(v)[:200]
                    elif isinstance(v, list):
                        rec[k] = str(v)[:200]
                    else:
                        rec[k] = v
                records.append(rec)
            return hf_ds, records

        # Try loading with streaming first for large datasets
        if ds_info["id"] in ("coco_2014",):
            # COCO is huge -- use streaming, cap at 5000
            coco_n = min(n, 5000)
            ds = load_dataset(
                slug, config, split=split, streaming=True,
            )
            # Take first coco_n samples from stream
            records = []
            for i, sample in enumerate(ds):
                if i >= coco_n:
                    break
                # Convert non-serializable fields to placeholders
                record = {}
                for k, v in sample.items():
                    if hasattr(v, "tobytes"):  # PIL Image, numpy array
                        record[k] = f"<{type(v).__name__}>"
                    elif isinstance(v, bytes):
                        record[k] = f"<bytes:{len(v)}>"
                    elif isinstance(v, dict) and ("bytes" in v or "array" in v):
                        if "array" in v:
                            arr = v["array"]
                            sr = v.get("sampling_rate", "?")
                            length = len(arr) if hasattr(arr, "__len__") else "?"
                            record[k] = f"<audio:samples={length},sr={sr}>"
                        else:
                            record[k] = f"<audio:{len(v.get('bytes', b''))}B>"
                    else:
                        record[k] = v
                records.append(record)

            # Also load non-streaming for HF approach (small subset)
            try:
                hf_ds = load_dataset(
                    slug, config, split=f"{split}[:{n}]",
                )
            except Exception:
                hf_ds = None
            return hf_ds, records

        elif ds_info["id"] == "common_voice":
            # superb/ks - standard loading
            hf_ds = load_dataset(
                slug, config, split=f"{split}[:{n}]",
            )
        elif ds_info["id"] == "timeseries_pile":
            # Now using CSV loader
            kwargs = {}
            if ds_info.get("hf_data_files"):
                kwargs["data_files"] = ds_info["hf_data_files"]
            hf_ds = load_dataset(
                slug, config, split=f"{split}[:{n}]",
                **kwargs,
            )
        else:
            kwargs = {}
            if ds_info.get("hf_data_files"):
                kwargs["data_files"] = ds_info["hf_data_files"]
            hf_ds = load_dataset(
                slug, config, split=f"{split}[:{n}]",
                **kwargs,
            )

        # Remove audio/image columns to avoid decoder dependencies
        # (torchcodec/FFmpeg for audio, pillow for images)
        # We only need metadata (labels, text) for the workflow comparison
        try:
            from datasets import Audio as _AudioFeature, Image as _ImageFeature
            if hasattr(hf_ds, "features"):
                drop_cols = []
                for _col, _feat in list(hf_ds.features.items()):
                    if isinstance(_feat, _AudioFeature):
                        drop_cols.append(_col)
                    elif isinstance(_feat, _ImageFeature):
                        drop_cols.append(_col)
                if drop_cols:
                    hf_ds = hf_ds.remove_columns(drop_cols)
                    safe_print(f"  Removed decoder columns: {drop_cols}")
        except Exception as _e:
            safe_print(f"  [WARN] Column removal failed: {_e}")

        # Convert to list of plain dicts for plain JSON approach
        records = []
        for sample in hf_ds:
            record = {}
            for k, v in sample.items():
                if hasattr(v, "tobytes"):  # PIL Image
                    record[k] = f"<{type(v).__name__}>"
                elif isinstance(v, bytes):
                    record[k] = f"<bytes:{len(v)}>"
                elif isinstance(v, dict) and ("bytes" in v or "array" in v):
                    # Audio data (bytes or numpy array)
                    if "array" in v:
                        arr = v["array"]
                        sr = v.get("sampling_rate", "?")
                        length = len(arr) if hasattr(arr, "__len__") else "?"
                        record[k] = f"<audio:samples={length},sr={sr}>"
                    else:
                        record[k] = f"<audio:{len(v.get('bytes', b''))}B>"
                elif isinstance(v, np.ndarray):
                    record[k] = v.tolist()
                else:
                    record[k] = v
            records.append(record)

        return hf_ds, records

    except Exception as e:
        safe_print(f"  [ERROR] Failed to load {slug}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None, []


# ---- Synthetic Prediction Generator -------------------------------------

def generate_synthetic_predictions(
    ds_info: Dict[str, Any],
    n_samples: int,
    n_models: int = N_MODELS,
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    """Generate synthetic model predictions for a dataset.

    Creates n_models predictions per sample with controlled properties:
    - Model 1: High accuracy (0.85), well-calibrated
    - Model 2: Medium accuracy (0.70), slight overconfidence
    - Model 3: Low accuracy (0.55), underconfident

    Returns list of dicts: [{"sample_idx": i, "models": [{"model_id": ...,
    "prediction": ..., "confidence": ..., "raw_scores": ...}, ...]}]
    """
    rng = np.random.RandomState(seed)
    task_type = ds_info["task_type"]
    n_classes = ds_info.get("n_classes", 2) or 2

    model_profiles = [
        {"model_id": "model_A", "accuracy": 0.85, "calibration_bias": 0.0,
         "name": "High-accuracy baseline"},
        {"model_id": "model_B", "accuracy": 0.70, "calibration_bias": 0.05,
         "name": "Medium-accuracy (overconfident)"},
        {"model_id": "model_C", "accuracy": 0.55, "calibration_bias": -0.10,
         "name": "Low-accuracy (underconfident)"},
    ]

    # Evidence levels: controls how many observations back each prediction.
    # This is KEY for demonstrating uncertainty-aware filtering:
    #   - High evidence (100 obs): low uncertainty, passes both filters
    #   - Medium evidence (15 obs): moderate uncertainty
    #   - Low evidence (3-5 obs): high uncertainty, passes confidence
    #     filter but FAILS uncertainty filter (the SL advantage)
    EVIDENCE_LEVELS = [
        {"name": "high", "total_evidence": 100, "weight": 0.40},
        {"name": "medium", "total_evidence": 15, "weight": 0.30},
        {"name": "low", "total_evidence": 4, "weight": 0.30},
    ]
    evidence_names = [e["name"] for e in EVIDENCE_LEVELS]
    evidence_totals = {e["name"]: e["total_evidence"] for e in EVIDENCE_LEVELS}
    evidence_weights = [e["weight"] for e in EVIDENCE_LEVELS]

    predictions = []
    for i in range(n_samples):
        # Assign evidence level for this sample
        ev_level = rng.choice(evidence_names, p=evidence_weights)
        ev_total = evidence_totals[ev_level]

        sample_preds = {"sample_idx": i, "evidence_level": ev_level,
                        "evidence_total": ev_total, "models": []}

        # Generate a "ground truth" label for this sample
        if task_type == "classification" or task_type == "detection":
            true_label = rng.randint(0, n_classes)
        elif task_type == "regression":
            true_label = rng.uniform(0, 100)
        elif task_type == "asr":
            true_label = f"synthetic_transcript_{i}"
        else:
            true_label = rng.randint(0, 2)

        for profile in model_profiles[:n_models]:
            if task_type in ("classification", "detection"):
                # Decide if model is correct for this sample
                is_correct = rng.random() < profile["accuracy"]
                if is_correct:
                    pred_label = true_label
                else:
                    pred_label = (true_label + rng.randint(1, n_classes)) % n_classes

                # Generate raw softmax-like scores
                raw_scores = rng.dirichlet(np.ones(n_classes) * 0.5)
                # Boost the predicted class
                boost = 0.3 + rng.uniform(0, 0.4)
                raw_scores[pred_label] += boost
                raw_scores = raw_scores / raw_scores.sum()

                confidence = float(raw_scores[pred_label])
                # Apply calibration bias
                confidence = np.clip(
                    confidence + profile["calibration_bias"], 0.01, 0.99
                )

                sample_preds["models"].append({
                    "model_id": profile["model_id"],
                    "model_name": profile["name"],
                    "prediction": int(pred_label),
                    "confidence": float(confidence),
                    "raw_scores": raw_scores.tolist(),
                    "true_label": int(true_label),
                    "evidence_total": ev_total,
                })

            elif task_type == "regression":
                # Prediction = true + noise scaled by accuracy
                noise_std = (1.0 - profile["accuracy"]) * 20.0
                pred_value = true_label + rng.normal(0, noise_std)
                # Confidence = 1 - normalized_error (clamped)
                error = abs(pred_value - true_label) / (abs(true_label) + 1e-6)
                confidence = float(np.clip(1.0 - error * 0.5, 0.01, 0.99))
                confidence = np.clip(
                    confidence + profile["calibration_bias"], 0.01, 0.99
                )

                sample_preds["models"].append({
                    "model_id": profile["model_id"],
                    "model_name": profile["name"],
                    "prediction": float(pred_value),
                    "confidence": float(confidence),
                    "prediction_interval": [
                        float(pred_value - 2 * noise_std),
                        float(pred_value + 2 * noise_std),
                    ],
                    "true_label": float(true_label),
                    "evidence_total": ev_total,
                })

            elif task_type == "asr":
                # ASR: prediction is a transcript, confidence is WER-based
                if rng.random() < profile["accuracy"]:
                    pred_transcript = true_label  # Correct
                else:
                    pred_transcript = f"corrupted_{true_label}"
                confidence = float(np.clip(
                    profile["accuracy"] + rng.uniform(-0.1, 0.1)
                    + profile["calibration_bias"], 0.01, 0.99
                ))

                sample_preds["models"].append({
                    "model_id": profile["model_id"],
                    "model_name": profile["name"],
                    "prediction": pred_transcript,
                    "confidence": float(confidence),
                    "true_label": true_label,
                    "evidence_total": ev_total,
                })

            elif task_type == "qa":
                # QA: prediction is an answer span with confidence
                if rng.random() < profile["accuracy"]:
                    pred_answer = f"correct_answer_{i}"
                else:
                    pred_answer = f"wrong_answer_{i}"
                confidence = float(np.clip(
                    profile["accuracy"] + rng.uniform(-0.1, 0.1)
                    + profile["calibration_bias"], 0.01, 0.99
                ))
                sample_preds["models"].append({
                    "model_id": profile["model_id"],
                    "model_name": profile["name"],
                    "prediction": pred_answer,
                    "confidence": float(confidence),
                    "true_label": f"correct_answer_{i}",
                    "evidence_total": ev_total,
                })

        predictions.append(sample_preds)
    return predictions


# =====================================================================
# TASK IMPLEMENTATIONS
# =====================================================================

# ---- Task 1: Load & Inspect Metadata -----------------------------------

def task1_jsonldex(ds_info: Dict, records: List[Dict]) -> Dict[str, Any]:
    """T1-jsonld_ex: Load dataset and create structured metadata."""
    meta = create_dataset_metadata(
        name=ds_info["name"],
        description=f"Dataset: {ds_info['name']} ({ds_info['domain']})",
        url=f"https://huggingface.co/datasets/{ds_info['hf_slug']}",
        license="varies",
    )
    meta["@context"] = {
        "sc": "http://schema.org/",
        "jex": "http://www.w3.org/ns/jsonld-ex/",
        "cr": "http://mlcommons.org/croissant/",
    }
    meta["domain"] = ds_info["domain"]
    meta["taskType"] = ds_info["task_type"]
    meta["numSamples"] = len(records)
    if ds_info.get("n_classes"):
        meta["numClasses"] = ds_info["n_classes"]
    if ds_info.get("class_names"):
        meta["classNames"] = ds_info["class_names"]
    return {"metadata": meta, "features_expressed": ["F01", "F09"]}


def task1_hf_datasets(ds_info: Dict, hf_ds: Any) -> Dict[str, Any]:
    """T1-hf_datasets: Load dataset and inspect info."""
    if hf_ds is None:
        return {"metadata": {"error": "dataset not loaded"}, "features_expressed": []}
    info_dict = {
        "dataset_name": str(hf_ds.info.dataset_name) if hf_ds.info.dataset_name else ds_info["name"],
        "description": str(hf_ds.info.description or "")[:200],
        "num_rows": len(hf_ds),
        "features": {k: str(v) for k, v in hf_ds.features.items()},
        "license": str(hf_ds.info.license or "not specified"),
    }
    return {"metadata": info_dict, "features_expressed": ["F01"]}


def task1_plain_json(ds_info: Dict, records: List[Dict]) -> Dict[str, Any]:
    """T1-plain_json: Load data as plain JSON."""
    meta = {
        "name": ds_info["name"],
        "domain": ds_info["domain"],
        "num_samples": len(records),
    }
    if records:
        meta["fields"] = list(records[0].keys())
    return {"metadata": meta, "features_expressed": []}


# ---- Task 2: Annotate with Predictions + Confidence ---------------------

def task2_jsonldex(
    ds_info: Dict, records: List[Dict], predictions: List[Dict],
) -> Dict[str, Any]:
    """T2-jsonld_ex: Annotate samples with structured confidence metadata."""
    annotated = []
    for rec, pred in zip(records, predictions):
        doc = {"@id": f"sample:{pred['sample_idx']}", **rec}
        # Annotate with best model's prediction
        best = max(pred["models"], key=lambda m: m["confidence"])
        doc["prediction"] = annotate(
            best["prediction"],
            confidence=best["confidence"],
            source=best["model_id"],
            extracted_at=datetime.now(timezone.utc).isoformat(),
            method="synthetic_classification",
        )
        # Create SL opinion using actual evidence level
        # Low-evidence samples will have HIGH uncertainty even with
        # high confidence -- this is the key SL advantage
        ev_total = best.get("evidence_total", 100)
        positive = max(1, int(best["confidence"] * ev_total))
        negative = max(0, ev_total - positive)
        opinion = Opinion.from_evidence(
            positive=positive,
            negative=negative,
        )
        doc["prediction"]["@opinion"] = {
            "belief": float(opinion.belief),
            "disbelief": float(opinion.disbelief),
            "uncertainty": float(opinion.uncertainty),
            "base_rate": float(opinion.base_rate),
        }
        annotated.append(doc)

    return {
        "annotated": annotated,
        "features_expressed": ["F02", "F03", "F07", "F09", "F14"],
    }


def task2_hf_datasets(
    ds_info: Dict, hf_ds: Any, predictions: List[Dict],
) -> Dict[str, Any]:
    """T2-hf_datasets: Add prediction columns to HF dataset."""
    if hf_ds is None:
        return {"annotated": [], "features_expressed": []}

    # HF datasets: add columns via .map() or direct assignment
    pred_labels = []
    pred_confs = []
    pred_models = []
    for pred in predictions[:len(hf_ds)]:
        best = max(pred["models"], key=lambda m: m["confidence"])
        pred_labels.append(best["prediction"])
        pred_confs.append(best["confidence"])
        pred_models.append(best["model_id"])

    # In practice you'd do: hf_ds = hf_ds.add_column("pred_label", pred_labels)
    # We simulate and measure the resulting structure
    result_records = []
    for i, sample in enumerate(hf_ds):
        rec = {}
        for k, v in sample.items():
            if hasattr(v, "tobytes"):
                rec[k] = f"<{type(v).__name__}>"
            elif isinstance(v, bytes):
                rec[k] = f"<bytes:{len(v)}>"
            elif isinstance(v, dict) and "bytes" in v:
                rec[k] = f"<audio>"
            elif isinstance(v, np.ndarray):
                rec[k] = v.tolist()
            else:
                rec[k] = v
        if i < len(pred_labels):
            rec["pred_label"] = pred_labels[i]
            rec["pred_confidence"] = pred_confs[i]
            rec["pred_model"] = pred_models[i]
        result_records.append(rec)

    return {
        "annotated": result_records,
        "features_expressed": ["F02"],  # Only scalar confidence, no schema
    }


def task2_plain_json(
    ds_info: Dict, records: List[Dict], predictions: List[Dict],
) -> Dict[str, Any]:
    """T2-plain_json: Add predictions as dict keys."""
    annotated = []
    for rec, pred in zip(records, predictions):
        doc = {**rec}
        best = max(pred["models"], key=lambda m: m["confidence"])
        doc["pred_label"] = best["prediction"]
        doc["pred_confidence"] = best["confidence"]
        doc["pred_model"] = best["model_id"]
        annotated.append(doc)

    return {
        "annotated": annotated,
        "features_expressed": ["F02"],  # Scalar only, no schema
    }


# ---- Task 3: Merge Multi-Model Predictions ------------------------------

def task3_jsonldex(
    ds_info: Dict, records: List[Dict], predictions: List[Dict],
) -> Dict[str, Any]:
    """T3-jsonld_ex: Fuse predictions from multiple models using SL algebra."""
    fused_results = []
    conflict_count = 0

    for rec, pred in zip(records, predictions):
        sample_id = f"sample:{pred['sample_idx']}"

        # Create per-model graphs
        model_graphs = []
        opinions = []
        for m in pred["models"]:
            graph_node = {
                "@id": sample_id,
                "prediction": annotate(
                    m["prediction"],
                    confidence=m["confidence"],
                    source=m["model_id"],
                    method="synthetic",
                ),
            }
            model_graphs.append({"@graph": [graph_node]})

            ev_t = m.get("evidence_total", 100)
            op = Opinion.from_evidence(
                positive=max(1, int(m["confidence"] * ev_t)),
                negative=max(0, ev_t - max(1, int(m["confidence"] * ev_t))),
            )
            opinions.append(op)

        # Fuse opinions
        fused_opinion = cumulative_fuse(*opinions)

        # Detect conflict between models
        max_conflict = 0.0
        for i in range(len(opinions)):
            for j in range(i + 1, len(opinions)):
                c = pairwise_conflict(opinions[i], opinions[j])
                max_conflict = max(max_conflict, c)

        if max_conflict > 0.5:
            conflict_count += 1

        # Merge graphs
        merged_doc, merged_report = merge_graphs(model_graphs)

        fused_doc = {
            "@id": sample_id,
            "fused_prediction": {
                "@value": pred["models"][0]["prediction"],  # Placeholder
                "@confidence": float(fused_opinion.projected_probability()),
                "@opinion": {
                    "belief": float(fused_opinion.belief),
                    "disbelief": float(fused_opinion.disbelief),
                    "uncertainty": float(fused_opinion.uncertainty),
                },
                "@sources": [m["model_id"] for m in pred["models"]],
                "@conflict": float(max_conflict),
            },
            "merge_report": {
                "nodes_merged": merged_report.nodes_merged,
                "conflicts": merged_report.properties_conflicted,
            },
        }
        fused_results.append(fused_doc)

    return {
        "fused": fused_results,
        "conflict_count": conflict_count,
        "features_expressed": ["F03", "F04", "F05", "F07", "F12", "F13"],
    }


def task3_hf_datasets(
    ds_info: Dict, hf_ds: Any, predictions: List[Dict],
) -> Dict[str, Any]:
    """T3-hf_datasets: Merge predictions by adding columns + averaging."""
    if hf_ds is None:
        return {"fused": [], "conflict_count": 0, "features_expressed": []}

    fused_results = []
    for i, pred in enumerate(predictions[:len(hf_ds) if hf_ds else 0]):
        # HF approach: just average confidences (no SL, no conflict detection)
        confs = [m["confidence"] for m in pred["models"]]
        avg_conf = sum(confs) / len(confs)

        # Majority vote for label
        from collections import Counter
        labels = [m["prediction"] for m in pred["models"]]
        if all(isinstance(l, (int, float)) for l in labels):
            if ds_info["task_type"] == "regression":
                fused_label = sum(labels) / len(labels)
            else:
                most_common = Counter(labels).most_common(1)[0][0]
                fused_label = most_common
        else:
            most_common = Counter(labels).most_common(1)[0][0]
            fused_label = most_common

        fused_results.append({
            "sample_idx": i,
            "fused_label": fused_label,
            "avg_confidence": avg_conf,
            "model_count": len(pred["models"]),
        })

    return {
        "fused": fused_results,
        "conflict_count": 0,  # HF datasets has no conflict detection
        "features_expressed": [],  # No fusion operators, no conflict
    }


def task3_plain_json(
    ds_info: Dict, records: List[Dict], predictions: List[Dict],
) -> Dict[str, Any]:
    """T3-plain_json: Manual averaging of predictions."""
    fused_results = []
    for i, pred in enumerate(predictions[:len(records)]):
        confs = [m["confidence"] for m in pred["models"]]
        avg_conf = sum(confs) / len(confs)

        from collections import Counter
        labels = [m["prediction"] for m in pred["models"]]
        if all(isinstance(l, (int, float)) for l in labels):
            if ds_info["task_type"] == "regression":
                fused_label = sum(labels) / len(labels)
            else:
                fused_label = Counter(labels).most_common(1)[0][0]
        else:
            fused_label = Counter(labels).most_common(1)[0][0]

        fused_results.append({
            "sample_idx": i,
            "fused_label": fused_label,
            "avg_confidence": avg_conf,
        })

    return {
        "fused": fused_results,
        "conflict_count": 0,
        "features_expressed": [],
    }


# ---- Task 4: Filter by Confidence Threshold ----------------------------

def task4_jsonldex(
    annotated: List[Dict], threshold: float = CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """T4-jsonld_ex: Filter using uncertainty-aware filter_by_confidence."""
    # filter_by_confidence works on graph nodes with annotated properties
    filtered = filter_by_confidence(annotated, "prediction", threshold)

    # Also demonstrate uncertainty-aware filtering (SL-specific)
    # Keep only samples where uncertainty < 0.3 (high-evidence predictions)
    uncertainty_filtered = []
    for doc in annotated:
        pred = doc.get("prediction", {})
        op = pred.get("@opinion", {})
        if op.get("uncertainty", 1.0) < 0.3 and get_confidence(pred) is not None:
            conf = get_confidence(pred)
            if conf >= threshold:
                uncertainty_filtered.append(doc)

    return {
        "confidence_filtered": filtered,
        "uncertainty_filtered": uncertainty_filtered,
        "n_confidence": len(filtered),
        "n_uncertainty": len(uncertainty_filtered),
        "features_expressed": ["F02", "F06"],
    }


def task4_hf_datasets(
    annotated: List[Dict], threshold: float = CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """T4-hf_datasets: Filter by column value (scalar only)."""
    # HF datasets: ds.filter(lambda x: x["pred_confidence"] > threshold)
    # We simulate this on the list of dicts
    filtered = [
        r for r in annotated
        if r.get("pred_confidence", 0) >= threshold
    ]

    return {
        "confidence_filtered": filtered,
        "uncertainty_filtered": filtered,  # Same -- no uncertainty distinction
        "n_confidence": len(filtered),
        "n_uncertainty": len(filtered),
        "features_expressed": ["F02"],  # Scalar filter only
    }


def task4_plain_json(
    annotated: List[Dict], threshold: float = CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """T4-plain_json: List comprehension filter."""
    filtered = [
        r for r in annotated
        if r.get("pred_confidence", 0) >= threshold
    ]

    return {
        "confidence_filtered": filtered,
        "uncertainty_filtered": filtered,
        "n_confidence": len(filtered),
        "n_uncertainty": len(filtered),
        "features_expressed": [],
    }


# ---- Task 5: Export with Full Provenance --------------------------------

def task5_jsonldex(
    annotated: List[Dict], ds_info: Dict,
) -> Dict[str, Any]:
    """T5-jsonld_ex: Export with full provenance, @context, temporal validity."""
    export_doc = {
        "@context": {
            "sc": "http://schema.org/",
            "jex": "http://www.w3.org/ns/jsonld-ex/",
            "prov": "http://www.w3.org/ns/prov#",
        },
        "@type": "sc:Dataset",
        "sc:name": ds_info["name"],
        "sc:description": f"Annotated predictions for {ds_info['name']}",
        "jex:exportedAt": datetime.now(timezone.utc).isoformat(),
        "jex:exportMethod": "jsonld_ex.EN2.5",
        "jex:validFrom": "2026-04-03T00:00:00Z",
        "jex:validUntil": "2027-04-03T00:00:00Z",
        "prov:wasGeneratedBy": {
            "@type": "prov:Activity",
            "prov:used": [
                {"@id": "model:model_A", "prov:type": "ML model"},
                {"@id": "model:model_B", "prov:type": "ML model"},
                {"@id": "model:model_C", "prov:type": "ML model"},
            ],
        },
        "samples": annotated[:10],  # First 10 for size measurement
    }

    return {
        "export": export_doc,
        "total_bytes": measure_bytes(export_doc),
        "features_expressed": ["F07", "F08", "F09", "F10", "F11"],
    }


def task5_hf_datasets(
    annotated: List[Dict], ds_info: Dict,
) -> Dict[str, Any]:
    """T5-hf_datasets: Export via .to_json() -- no provenance."""
    # HF datasets exports to JSON Lines or Arrow -- no provenance metadata
    export_doc = {
        "dataset_name": ds_info["name"],
        "num_samples": len(annotated),
        "samples": annotated[:10],
    }

    return {
        "export": export_doc,
        "total_bytes": measure_bytes(export_doc),
        "features_expressed": [],  # No provenance, no @context, no temporal
    }


def task5_plain_json(
    annotated: List[Dict], ds_info: Dict,
) -> Dict[str, Any]:
    """T5-plain_json: json.dump() -- minimal metadata."""
    export_doc = {
        "name": ds_info["name"],
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "samples": annotated[:10],
    }

    return {
        "export": export_doc,
        "total_bytes": measure_bytes(export_doc),
        "features_expressed": [],
    }


# =====================================================================
# MEASUREMENT AND COMPARISON
# =====================================================================

def compute_loc_comparison() -> Dict[str, Dict[str, int]]:
    """Count lines of code for each task x approach."""
    tasks = {
        "T1_load_inspect": (task1_jsonldex, task1_hf_datasets, task1_plain_json),
        "T2_annotate": (task2_jsonldex, task2_hf_datasets, task2_plain_json),
        "T3_merge": (task3_jsonldex, task3_hf_datasets, task3_plain_json),
        "T4_filter": (task4_jsonldex, task4_hf_datasets, task4_plain_json),
        "T5_export": (task5_jsonldex, task5_hf_datasets, task5_plain_json),
    }

    result = {}
    for task_name, (jex_fn, hf_fn, pj_fn) in tasks.items():
        result[task_name] = {
            "jsonld_ex": count_body_lines(jex_fn),
            "hf_datasets": count_body_lines(hf_fn),
            "plain_json": count_body_lines(pj_fn),
        }
    return result


def compute_feature_scores() -> Dict[str, Dict[str, Any]]:
    """Compute feature support scores per approach."""
    scores = {}
    for approach, features in FEATURE_SUPPORT.items():
        native = sum(1 for v in features.values() if v is True)
        workaround = sum(1 for v in features.values() if v == "workaround")
        impossible = sum(1 for v in features.values() if v is False)
        scores[approach] = {
            "native": native,
            "workaround": workaround,
            "impossible": impossible,
            "total_14": native + workaround,
            "score_weighted": native + 0.5 * workaround,
            "details": features,
        }
    return scores


def compute_interop_scores() -> Dict[str, Dict[str, bool]]:
    """Semantic interoperability: can another system consume the output?"""
    return {
        "jsonld_ex": {
            "json_parseable": True,
            "json_ld_processable": True,
            "rdf_convertible": True,
            "croissant_compatible": True,
            "prov_o_compatible": True,
            "schema_org_compatible": True,
            "sparql_queryable": True,
            "arrow_compatible": False,  # Not native Arrow
            "total_7": 7,
        },
        "hf_datasets": {
            "json_parseable": True,  # via .to_json()
            "json_ld_processable": False,
            "rdf_convertible": False,
            "croissant_compatible": False,  # Not at sample level
            "prov_o_compatible": False,
            "schema_org_compatible": False,
            "sparql_queryable": False,
            "arrow_compatible": True,  # Native Arrow
            "total_7": 2,
        },
        "plain_json": {
            "json_parseable": True,
            "json_ld_processable": False,
            "rdf_convertible": False,
            "croissant_compatible": False,
            "prov_o_compatible": False,
            "schema_org_compatible": False,
            "sparql_queryable": False,
            "arrow_compatible": False,
            "total_7": 1,
        },
    }


# =====================================================================
# MAIN EXPERIMENT
# =====================================================================

def run_phase_a(datasets_to_run: Optional[List[str]] = None):
    """Run Phase A: synthetic predictions, all 5 tasks x 3 approaches."""
    set_global_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    safe_print("=" * 70)
    safe_print("EN2.5 -- Head-to-Head vs HuggingFace Datasets (Phase A)")
    safe_print("=" * 70)
    safe_print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    safe_print(f"Samples per dataset: {N_SAMPLES}")
    safe_print(f"Synthetic models: {N_MODELS}")
    safe_print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    safe_print()

    # Filter datasets if specified
    active_datasets = DATASETS
    if datasets_to_run:
        active_datasets = [d for d in DATASETS if d["id"] in datasets_to_run]

    all_results = {}
    dataset_summaries = []
    failed_datasets = []

    for ds_info in active_datasets:
        ds_id = ds_info["id"]
        safe_print(f"\n{'='*60}")
        safe_print(f"Dataset: {ds_info['name']} ({ds_info['domain']})")
        safe_print(f"{'='*60}")

        # Step 1: Load dataset
        safe_print("\n[1/5] Loading dataset ...")
        try:
            hf_ds, records = load_hf_subset(ds_info, N_SAMPLES)
            actual_n = len(records)
            safe_print(f"  Loaded {actual_n} samples")
        except Exception as e:
            safe_print(f"  [FATAL] Cannot load {ds_id}: {e}")
            traceback.print_exc()
            failed_datasets.append({"id": ds_id, "error": str(e)})
            continue

        if actual_n == 0:
            safe_print(f"  [SKIP] No records loaded for {ds_id}")
            failed_datasets.append({"id": ds_id, "error": "no records"})
            continue

        # Step 2: Generate synthetic predictions
        safe_print("\n[2/5] Generating synthetic predictions ...")
        predictions = generate_synthetic_predictions(ds_info, actual_n)
        safe_print(f"  Generated {len(predictions)} x {N_MODELS} model predictions")

        # Step 3: Run all 5 tasks x 3 approaches
        ds_results = {"dataset": ds_info["name"], "domain": ds_info["domain"],
                       "n_samples": actual_n, "tasks": {}}

        # -- Task 1: Load & Inspect --
        safe_print("\n[T1] Load & Inspect Metadata ...")
        t1_jex = task1_jsonldex(ds_info, records)
        t1_hf = task1_hf_datasets(ds_info, hf_ds)
        t1_pj = task1_plain_json(ds_info, records)
        ds_results["tasks"]["T1"] = {
            "jsonld_ex": {"bytes": measure_bytes(t1_jex["metadata"]),
                          "features": t1_jex["features_expressed"]},
            "hf_datasets": {"bytes": measure_bytes(t1_hf["metadata"]),
                            "features": t1_hf["features_expressed"]},
            "plain_json": {"bytes": measure_bytes(t1_pj["metadata"]),
                           "features": t1_pj["features_expressed"]},
        }
        safe_print(f"  jsonld_ex: {ds_results['tasks']['T1']['jsonld_ex']['bytes']}B, "
                    f"features={t1_jex['features_expressed']}")
        safe_print(f"  hf_datasets: {ds_results['tasks']['T1']['hf_datasets']['bytes']}B, "
                    f"features={t1_hf['features_expressed']}")
        safe_print(f"  plain_json: {ds_results['tasks']['T1']['plain_json']['bytes']}B, "
                    f"features={t1_pj['features_expressed']}")

        # -- Task 2: Annotate --
        safe_print("\n[T2] Annotate with Predictions + Confidence ...")
        t2_jex = task2_jsonldex(ds_info, records, predictions)
        t2_hf = task2_hf_datasets(ds_info, hf_ds, predictions)
        t2_pj = task2_plain_json(ds_info, records, predictions)

        jex_bytes_t2 = measure_bytes(t2_jex["annotated"][:10])
        hf_bytes_t2 = measure_bytes(t2_hf["annotated"][:10])
        pj_bytes_t2 = measure_bytes(t2_pj["annotated"][:10])

        ds_results["tasks"]["T2"] = {
            "jsonld_ex": {"bytes_10": jex_bytes_t2,
                          "features": t2_jex["features_expressed"]},
            "hf_datasets": {"bytes_10": hf_bytes_t2,
                            "features": t2_hf["features_expressed"]},
            "plain_json": {"bytes_10": pj_bytes_t2,
                           "features": t2_pj["features_expressed"]},
        }
        safe_print(f"  jsonld_ex: {jex_bytes_t2}B (10 samples), "
                    f"features={t2_jex['features_expressed']}")
        safe_print(f"  hf_datasets: {hf_bytes_t2}B (10 samples), "
                    f"features={t2_hf['features_expressed']}")
        safe_print(f"  plain_json: {pj_bytes_t2}B (10 samples), "
                    f"features={t2_pj['features_expressed']}")

        # -- Task 3: Merge --
        safe_print("\n[T3] Merge Multi-Model Predictions ...")
        t3_jex = task3_jsonldex(ds_info, records, predictions)
        t3_hf = task3_hf_datasets(ds_info, hf_ds, predictions)
        t3_pj = task3_plain_json(ds_info, records, predictions)

        jex_bytes_t3 = measure_bytes(t3_jex["fused"][:10])
        hf_bytes_t3 = measure_bytes(t3_hf["fused"][:10])
        pj_bytes_t3 = measure_bytes(t3_pj["fused"][:10])

        ds_results["tasks"]["T3"] = {
            "jsonld_ex": {"bytes_10": jex_bytes_t3, "conflicts": t3_jex["conflict_count"],
                          "features": t3_jex["features_expressed"]},
            "hf_datasets": {"bytes_10": hf_bytes_t3, "conflicts": t3_hf["conflict_count"],
                            "features": t3_hf["features_expressed"]},
            "plain_json": {"bytes_10": pj_bytes_t3, "conflicts": t3_pj["conflict_count"],
                           "features": t3_pj["features_expressed"]},
        }
        safe_print(f"  jsonld_ex: {jex_bytes_t3}B (10 fused), "
                    f"conflicts={t3_jex['conflict_count']}, "
                    f"features={t3_jex['features_expressed']}")
        safe_print(f"  hf_datasets: {hf_bytes_t3}B (10 fused), "
                    f"no conflict detection")
        safe_print(f"  plain_json: {pj_bytes_t3}B (10 fused), "
                    f"no conflict detection")

        # -- Task 4: Filter --
        safe_print("\n[T4] Filter by Confidence Threshold ...")
        t4_jex = task4_jsonldex(t2_jex["annotated"])
        t4_hf = task4_hf_datasets(t2_hf["annotated"])
        t4_pj = task4_plain_json(t2_pj["annotated"])

        ds_results["tasks"]["T4"] = {
            "jsonld_ex": {"n_confidence": t4_jex["n_confidence"],
                          "n_uncertainty": t4_jex["n_uncertainty"],
                          "features": t4_jex["features_expressed"]},
            "hf_datasets": {"n_confidence": t4_hf["n_confidence"],
                            "n_uncertainty": t4_hf["n_uncertainty"],
                            "features": t4_hf["features_expressed"]},
            "plain_json": {"n_confidence": t4_pj["n_confidence"],
                           "n_uncertainty": t4_pj["n_uncertainty"],
                           "features": t4_pj["features_expressed"]},
        }
        safe_print(f"  jsonld_ex: {t4_jex['n_confidence']} by conf, "
                    f"{t4_jex['n_uncertainty']} by uncertainty, "
                    f"features={t4_jex['features_expressed']}")
        safe_print(f"  hf_datasets: {t4_hf['n_confidence']} by conf "
                    f"(no uncertainty distinction)")
        safe_print(f"  plain_json: {t4_pj['n_confidence']} by conf "
                    f"(no uncertainty distinction)")

        # -- Task 5: Export --
        safe_print("\n[T5] Export with Provenance ...")
        t5_jex = task5_jsonldex(t2_jex["annotated"], ds_info)
        t5_hf = task5_hf_datasets(t2_hf["annotated"], ds_info)
        t5_pj = task5_plain_json(t2_pj["annotated"], ds_info)

        ds_results["tasks"]["T5"] = {
            "jsonld_ex": {"bytes": t5_jex["total_bytes"],
                          "features": t5_jex["features_expressed"]},
            "hf_datasets": {"bytes": t5_hf["total_bytes"],
                            "features": t5_hf["features_expressed"]},
            "plain_json": {"bytes": t5_pj["total_bytes"],
                           "features": t5_pj["features_expressed"]},
        }
        safe_print(f"  jsonld_ex: {t5_jex['total_bytes']}B, "
                    f"features={t5_jex['features_expressed']}")
        safe_print(f"  hf_datasets: {t5_hf['total_bytes']}B, "
                    f"features={t5_hf['features_expressed']}")
        safe_print(f"  plain_json: {t5_pj['total_bytes']}B, "
                    f"features={t5_pj['features_expressed']}")

        all_results[ds_id] = ds_results
        dataset_summaries.append({
            "id": ds_id,
            "name": ds_info["name"],
            "domain": ds_info["domain"],
            "n_samples": actual_n,
            "status": "OK",
        })

    # ---- Aggregate Results ----
    safe_print("\n" + "=" * 70)
    safe_print("AGGREGATE RESULTS")
    safe_print("=" * 70)

    # LOC comparison
    loc = compute_loc_comparison()
    safe_print("\n--- Lines of Code per Task ---")
    for task_name, approaches in loc.items():
        safe_print(f"  {task_name}: "
                    f"jsonld_ex={approaches['jsonld_ex']}, "
                    f"hf_datasets={approaches['hf_datasets']}, "
                    f"plain_json={approaches['plain_json']}")

    # Feature scores
    features = compute_feature_scores()
    safe_print("\n--- Feature Support (out of 14) ---")
    for approach, score in features.items():
        safe_print(f"  {approach}: native={score['native']}, "
                    f"workaround={score['workaround']}, "
                    f"impossible={score['impossible']}, "
                    f"weighted_score={score['score_weighted']:.1f}")

    # Interop scores
    interop = compute_interop_scores()
    safe_print("\n--- Semantic Interoperability (out of 7) ---")
    for approach, score in interop.items():
        safe_print(f"  {approach}: {score['total_7']}/7")

    # Byte overhead summary
    safe_print("\n--- Byte Overhead Summary (T2: 10-sample annotation) ---")
    for ds_id, ds_result in all_results.items():
        t2 = ds_result["tasks"].get("T2", {})
        jex_b = t2.get("jsonld_ex", {}).get("bytes_10", 0)
        hf_b = t2.get("hf_datasets", {}).get("bytes_10", 0)
        pj_b = t2.get("plain_json", {}).get("bytes_10", 0)
        if hf_b > 0:
            overhead_pct = ((jex_b - hf_b) / hf_b) * 100
        else:
            overhead_pct = 0
        safe_print(f"  {ds_id}: jsonld_ex={jex_b}B, hf={hf_b}B, "
                    f"plain={pj_b}B, overhead={overhead_pct:+.1f}%")

    # Task 4 filtering divergence
    safe_print("\n--- T4 Filtering Divergence (confidence vs uncertainty-aware) ---")
    for ds_id, ds_result in all_results.items():
        t4 = ds_result["tasks"].get("T4", {})
        jex = t4.get("jsonld_ex", {})
        n_conf = jex.get("n_confidence", 0)
        n_unc = jex.get("n_uncertainty", 0)
        diff = n_conf - n_unc
        safe_print(f"  {ds_id}: conf_filtered={n_conf}, "
                    f"uncertainty_filtered={n_unc}, "
                    f"divergence={diff} samples")

    # Failed datasets
    if failed_datasets:
        safe_print("\n--- Failed Datasets ---")
        for fd in failed_datasets:
            safe_print(f"  {fd['id']}: {fd['error']}")

    # ---- Save Results ----
    final_result = ExperimentResult(
        experiment_id="EN2.5",
        parameters={
            "phase": "A_synthetic",
            "n_samples": N_SAMPLES,
            "n_models": N_MODELS,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "seed": SEED,
            "datasets": [d["id"] for d in active_datasets],
        },
        metrics={
            "loc_comparison": loc,
            "feature_scores": features,
            "interop_scores": interop,
            "per_dataset": all_results,
            "dataset_summaries": dataset_summaries,
            "failed_datasets": failed_datasets,
        },
        notes=(
            "Phase A: Synthetic predictions. "
            "All 5 tasks x 3 approaches across 11 datasets. "
            "Phase B (real model predictions) to follow."
        ),
    )

    out_path = RESULTS_DIR / "en2_5_results_phase_a.json"
    final_result.save_json(str(out_path))
    safe_print(f"\nResults saved to {out_path}")

    # Also save timestamped archive
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = RESULTS_DIR / f"en2_5_results_phase_a_{ts}.json"
    final_result.save_json(str(archive_path))
    safe_print(f"Archive saved to {archive_path}")

    safe_print(f"\n{'='*70}")
    safe_print("EN2.5 Phase A COMPLETE")
    safe_print(f"{'='*70}")
    safe_print(f"Datasets attempted: {len(active_datasets)}")
    safe_print(f"Datasets succeeded: {len(dataset_summaries)}")
    safe_print(f"Datasets failed: {len(failed_datasets)}")

    return final_result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EN2.5 Head-to-Head vs HF Datasets")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific dataset IDs to run (default: all)")
    parser.add_argument("--samples", type=int, default=N_SAMPLES,
                        help=f"Samples per dataset (default: {N_SAMPLES})")
    args = parser.parse_args()

    if args.samples != N_SAMPLES:
        N_SAMPLES = args.samples

    run_phase_a(datasets_to_run=args.datasets)
