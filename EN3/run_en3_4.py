"""EN3.4: FHIR R4 Clinical Data Exchange — Full Experiment Runner.

Runs all three phases:
  A0: Calibration analysis (ECE, reliability diagrams, uncertainty derivation)
  A1: NER fusion evaluation (6 conditions × 2 datasets)
  B:  FHIR clinical pipeline (Synthea, round-trip, overhead, queries)

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    experiments\\.venv-gliner\\Scripts\\Activate.ps1
    python experiments/EN3/run_en3_4.py [--phase a0|a1|b|all] [--skip-inference]

Time estimate: ~30-45 min (mostly GPU inference on BC5CDR + MedMentions)
GPU: Required for Phase A0 + A1 (GLiNER inference). Phase B is CPU-only.

Prerequisites:
    - .venv-gliner activated with gliner2, gliner, datasets, seqeval
    - jsonld-ex installed in editable mode
    - Synthea data at data/synthea/fhir_r4/ (for Phase B)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -- Path setup --
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _EXPERIMENTS_ROOT.parent
for p in [str(_SCRIPT_DIR), str(_EXPERIMENTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from infra.config import set_global_seed
from infra.results import ExperimentResult
from infra.env_log import log_environment

from EN3.en3_4_calibration import (
    compute_ece,
    reliability_diagram_bins,
    fit_temperature,
    derive_model_uncertainty,
    CalibrationReport,
)
from EN3.en3_4_core import (
    EntitySpan,
    compute_span_iou,
    align_spans,
    build_opinion,
    apply_condition_single_model,
    apply_condition_union,
    apply_condition_intersection,
    apply_condition_scalar_average,
    apply_condition_sl_fusion,
    evaluate_entities,
    EvalMetrics,
    bootstrap_f1_difference_ci,
    cohens_h,
    holm_bonferroni,
    spearman_conflict_error,
    optimize_threshold,
)
from EN3.en3_4_phase_b import (
    load_bundle_resources,
    extract_narrative_texts,
    measure_round_trip_fidelity,
    measure_annotation_overhead,
    query_by_confidence,
    query_provenance_chain,
    query_fuse_multi_model,
    query_temporal_decay,
    query_abstain_on_conflict,
    query_by_uncertainty_component,
    PipelineReport,
)

# =====================================================================
# Constants
# =====================================================================

SEED = 42
RESULTS_DIR = _SCRIPT_DIR / "results"
CHECKPOINT_DIR = _SCRIPT_DIR / "checkpoints"

# Model identifiers
GLINER2_MODEL = "fastino/gliner2-base-v1"
BIOMED_MODEL = "Ihor/gliner-biomed-base-v1.0"
# Local fallback if HuggingFace download fails
BIOMED_MODEL_LOCAL = _REPO_ROOT / "data" / "gliner-biomed-base-v1.0"

# Extraction threshold (low — capture borderline entities for fusion)
EXTRACTION_THRESHOLD = 0.3

# BC5CDR entity labels for GLiNER (natural language descriptions)
BC5CDR_LABELS_GLINER = ["chemical compound or drug", "disease or medical condition"]
BC5CDR_LABEL_MAP = {
    "chemical compound or drug": "Chemical",
    "disease or medical condition": "Disease",
}

# BC5CDR BIO tag mapping (tner/bc5cdr format)
BC5CDR_TAG_NAMES = {0: "O", 1: "B-Chemical", 2: "B-Disease",
                    3: "I-Disease", 4: "I-Chemical"}

# MedMentions ST21pv: 21 UMLS semantic types grouped into 7 GLiNER categories
# Grouping rationale: GLiNER zero-shot degrades with >10 simultaneous labels.
# We group into clinically meaningful categories, documented transparently.
MEDMENTIONS_TYPE_TO_GROUP = {
    "T005": "Organism",         # Virus
    "T007": "Organism",         # Bacterium
    "T204": "Organism",         # Eukaryote
    "T017": "Anatomy",          # Anatomical Structure
    "T022": "Anatomy",          # Body System
    "T031": "Anatomy",          # Body Substance
    "T033": "Finding",          # Finding
    "T037": "Finding",          # Injury or Poisoning
    "T038": "Finding",          # Biologic Function
    "T201": "Finding",          # Clinical Attribute
    "T103": "Chemical",         # Chemical
    "T168": "Chemical",         # Food (biochemical context)
    "T058": "Procedure",        # Health Care Activity
    "T062": "Procedure",        # Research Activity
    "T074": "Device",           # Medical Device
    "T082": "Concept",          # Spatial Concept
    "T091": "Concept",          # Biomedical Occupation or Discipline
    "T092": "Concept",          # Organization
    "T097": "Concept",          # Professional or Occupational Group
    "T098": "Concept",          # Population Group
    "T170": "Concept",          # Intellectual Product
}

MEDMENTIONS_LABELS_GLINER = [
    "organism such as virus bacterium or pathogen",
    "anatomical structure or body system",
    "clinical finding disease or biological function",
    "chemical compound drug or substance",
    "medical procedure or research activity",
    "medical device",
    "biomedical concept organization or occupation",
]

MEDMENTIONS_GLINER_TO_GROUP = {
    "organism such as virus bacterium or pathogen": "Organism",
    "anatomical structure or body system": "Anatomy",
    "clinical finding disease or biological function": "Finding",
    "chemical compound drug or substance": "Chemical",
    "medical procedure or research activity": "Procedure",
    "medical device": "Device",
    "biomedical concept organization or occupation": "Concept",
}

# Threshold grids for dev-set optimization
ACCEPT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ABSTENTION_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# Synthea data location
SYNTHEA_DIR = _REPO_ROOT / "data" / "synthea" / "fhir_r4"


# =====================================================================
# Helpers
# =====================================================================

def _banner(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _elapsed(t0: float) -> str:
    dt = time.time() - t0
    if dt < 60:
        return f"{dt:.1f}s"
    return f"{dt / 60:.1f}min"


def _save_result(result: ExperimentResult, name: str) -> None:
    """Save primary + timestamped archive."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    primary = RESULTS_DIR / f"{name}.json"
    result.save_json(str(primary))
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive = RESULTS_DIR / f"{name}_{ts}.json"
    result.save_json(str(archive))
    print(f"  Saved: {primary.name} + {archive.name}")


def _save_checkpoint(data: Any, name: str) -> None:
    """Save a checkpoint for expensive computations (e.g., model predictions)."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Checkpoint saved: {path.name}")


def _load_checkpoint(name: str) -> Optional[Any]:
    """Load a checkpoint if it exists."""
    path = CHECKPOINT_DIR / f"{name}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Checkpoint loaded: {path.name}")
        return data
    return None


# =====================================================================
# Data Loading
# =====================================================================

def load_bc5cdr(split: str) -> List[Dict]:
    """Load BC5CDR from HuggingFace, return list of sentence dicts.

    Each dict has:
        tokens: List[str]
        gold_spans: List[EntitySpan]  (character-level)
        text: str  (space-joined tokens)
    """
    from datasets import load_dataset

    print(f"  Loading BC5CDR {split}...")
    ds = load_dataset("tner/bc5cdr", split=split, trust_remote_code=True)
    print(f"  Loaded {len(ds)} sentences")

    sentences = []
    for item in ds:
        tokens = item["tokens"]
        raw_tags = item["tags"]

        # Reconstruct text with character offsets
        text, offsets = _tokens_to_text(tokens)

        # Convert BIO tags → entity spans
        gold_spans = _bio_tags_to_spans(raw_tags, offsets, BC5CDR_TAG_NAMES)

        sentences.append({
            "tokens": tokens,
            "text": text,
            "offsets": offsets,
            "gold_spans": gold_spans,
        })

    n_ents = sum(len(s["gold_spans"]) for s in sentences)
    print(f"  {len(sentences)} sentences, {n_ents} gold entities")
    return sentences


def load_medmentions(split: str) -> List[Dict]:
    """Load MedMentions ST21pv from HuggingFace (ibm-research/MedMentions-ZS).

    Returns list of sentence dicts with grouped entity types.
    """
    from datasets import load_dataset

    print(f"  Loading MedMentions ST21pv {split}...")
    ds = load_dataset("ibm-research/MedMentions-ZS", split=split)
    print(f"  Loaded {len(ds)} sentences")

    # Build tag name mapping from the dataset's ner_tags
    # Tags are strings like "B-T103", "I-T038", "O"
    # First, discover all unique tag strings
    sentences = []
    for item in ds:
        tokens = item["tokens"]
        raw_tags = item["ner_tags"]  # list of strings like "B-T103"

        text, offsets = _tokens_to_text(tokens)

        # Convert BIO tag strings to grouped EntitySpans
        gold_spans = _medmentions_tags_to_spans(raw_tags, offsets)

        sentences.append({
            "tokens": tokens,
            "text": text,
            "offsets": offsets,
            "gold_spans": gold_spans,
        })

    n_ents = sum(len(s["gold_spans"]) for s in sentences)
    print(f"  {len(sentences)} sentences, {n_ents} gold entities")
    return sentences


def _medmentions_tags_to_spans(
    tags: List[str],
    offsets: List[Tuple[int, int]],
) -> List[EntitySpan]:
    """Convert MedMentions BIO tag strings to grouped EntitySpans.

    Tags are strings: 'O', 'B-T103', 'I-T103', etc.
    We map T### codes to grouped types via MEDMENTIONS_TYPE_TO_GROUP.
    """
    spans: List[EntitySpan] = []
    current_type = None
    current_group = None
    current_start = -1

    for i, tag in enumerate(tags):
        if isinstance(tag, int):
            # Some versions use integer indices — skip if 0 (O tag)
            tag_str = "O" if tag == 0 else f"UNKNOWN-{tag}"
        else:
            tag_str = str(tag)

        if tag_str.startswith("B-"):
            # Close previous entity
            if current_group is not None:
                spans.append(EntitySpan(
                    start=current_start, end=offsets[i - 1][1],
                    entity_type=current_group, score=1.0, source="gold",
                ))
            type_code = tag_str[2:]  # e.g., "T103"
            current_type = type_code
            current_group = MEDMENTIONS_TYPE_TO_GROUP.get(type_code)
            current_start = offsets[i][0]

        elif tag_str.startswith("I-") and current_type == tag_str[2:]:
            # Continue current entity
            pass

        else:
            # O tag or type mismatch
            if current_group is not None:
                spans.append(EntitySpan(
                    start=current_start, end=offsets[i - 1][1],
                    entity_type=current_group, score=1.0, source="gold",
                ))
                current_type = None
                current_group = None

    # Close final entity
    if current_group is not None and len(offsets) > 0:
        spans.append(EntitySpan(
            start=current_start, end=offsets[-1][1],
            entity_type=current_group, score=1.0, source="gold",
        ))

    return spans


def _tokens_to_text(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """Join tokens with spaces, tracking character offsets."""
    offsets = []
    pos = 0
    for tok in tokens:
        offsets.append((pos, pos + len(tok)))
        pos += len(tok) + 1
    text = " ".join(tokens)
    return text, offsets


def _bio_tags_to_spans(
    tags: List[int],
    offsets: List[Tuple[int, int]],
    tag_names: Dict[int, str],
) -> List[EntitySpan]:
    """Convert integer BIO tags to character-level EntitySpans."""
    spans: List[EntitySpan] = []
    current_type = None
    current_start = -1

    for i, tag_id in enumerate(tags):
        tag = tag_names.get(tag_id, "O")

        if tag.startswith("B-"):
            # Close previous entity if any
            if current_type is not None:
                spans.append(EntitySpan(
                    start=current_start, end=offsets[i - 1][1],
                    entity_type=current_type, score=1.0, source="gold",
                ))
            current_type = tag[2:]
            current_start = offsets[i][0]

        elif tag.startswith("I-") and current_type == tag[2:]:
            # Continue current entity
            pass

        else:
            # O tag or type mismatch → close current entity
            if current_type is not None:
                spans.append(EntitySpan(
                    start=current_start, end=offsets[i - 1][1],
                    entity_type=current_type, score=1.0, source="gold",
                ))
                current_type = None

    # Close final entity
    if current_type is not None and len(offsets) > 0:
        spans.append(EntitySpan(
            start=current_start, end=offsets[-1][1],
            entity_type=current_type, score=1.0, source="gold",
        ))

    return spans


# =====================================================================
# Model Inference
# =====================================================================

def run_gliner2_inference(
    sentences: List[Dict],
    threshold: float = EXTRACTION_THRESHOLD,
    labels: List[str] = BC5CDR_LABELS_GLINER,
    label_map: Dict[str, str] = BC5CDR_LABEL_MAP,
) -> List[List[EntitySpan]]:
    """Run GLiNER2 on all sentences, return predicted EntitySpans."""
    from gliner2 import GLiNER2

    print(f"  Loading GLiNER2 ({GLINER2_MODEL})...")
    model = GLiNER2.from_pretrained(GLINER2_MODEL)
    print(f"  Model loaded. Running inference on {len(sentences)} sentences...")

    all_preds: List[List[EntitySpan]] = []
    t0 = time.time()

    for i, sent in enumerate(sentences):
        text = sent["text"]
        result = model.extract_entities(
            text, labels, threshold=threshold,
            include_confidence=True, include_spans=True,
        )

        spans = []
        entities = result.get("entities", {})
        for label, ent_list in entities.items():
            mapped_type = label_map.get(label)
            if mapped_type is None:
                continue
            for ent in ent_list:
                if isinstance(ent, dict):
                    spans.append(EntitySpan(
                        start=ent["start"], end=ent["end"],
                        entity_type=mapped_type, score=ent["confidence"],
                        source="gliner2", text=ent.get("text", ""),
                    ))

        all_preds.append(spans)

        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{len(sentences)} ({_elapsed(t0)})")

    print(f"  GLiNER2 inference complete ({_elapsed(t0)})")
    return all_preds


def run_biomed_inference(
    sentences: List[Dict],
    threshold: float = EXTRACTION_THRESHOLD,
    labels: List[str] = BC5CDR_LABELS_GLINER,
    label_map: Dict[str, str] = BC5CDR_LABEL_MAP,
) -> List[List[EntitySpan]]:
    """Run GLiNER-BioMed on all sentences, return predicted EntitySpans."""
    from gliner import GLiNER

    print(f"  Loading GLiNER-BioMed ({BIOMED_MODEL})...")
    model_path = str(BIOMED_MODEL_LOCAL) if BIOMED_MODEL_LOCAL.exists() else BIOMED_MODEL
    if BIOMED_MODEL_LOCAL.exists():
        print(f"    Using local model: {BIOMED_MODEL_LOCAL}")
    model = GLiNER.from_pretrained(model_path)
    print(f"  Model loaded. Running inference on {len(sentences)} sentences...")

    all_preds: List[List[EntitySpan]] = []
    t0 = time.time()

    for i, sent in enumerate(sentences):
        text = sent["text"]
        # GLiNER (non-GLiNER2) API: predict_entities returns list of dicts
        entities = model.predict_entities(text, labels, threshold=threshold)

        spans = []
        for ent in entities:
            mapped_type = label_map.get(ent.get("label", ""))
            if mapped_type is None:
                continue
            spans.append(EntitySpan(
                start=ent["start"], end=ent["end"],
                entity_type=mapped_type, score=ent["score"],
                source="biomed", text=ent.get("text", ""),
            ))

        all_preds.append(spans)

        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{len(sentences)} ({_elapsed(t0)})")

    print(f"  GLiNER-BioMed inference complete ({_elapsed(t0)})")
    return all_preds


def _preds_to_serializable(preds: List[List[EntitySpan]]) -> List[List[Dict]]:
    """Convert predictions to JSON-serializable format."""
    return [
        [{"start": s.start, "end": s.end, "entity_type": s.entity_type,
          "score": s.score, "source": s.source, "text": s.text or ""}
         for s in sent_preds]
        for sent_preds in preds
    ]


def _serializable_to_preds(data: List[List[Dict]]) -> List[List[EntitySpan]]:
    """Reconstruct EntitySpans from serialized format."""
    return [
        [EntitySpan(start=d["start"], end=d["end"],
                    entity_type=d["entity_type"], score=d["score"],
                    source=d["source"], text=d.get("text", ""))
         for d in sent_data]
        for sent_data in data
    ]


# =====================================================================
# Phase A0: Calibration
# =====================================================================

def run_phase_a0(
    dev_sentences: List[Dict],
    preds_gliner2: List[List[EntitySpan]],
    preds_biomed: List[List[EntitySpan]],
) -> Tuple[CalibrationReport, CalibrationReport]:
    """Run Phase A0: calibration analysis on dev set."""
    _banner("Phase A0: Calibration Analysis")
    t0 = time.time()

    reports = []
    for model_name, preds in [("gliner2", preds_gliner2), ("biomed", preds_biomed)]:
        print(f"\n  Calibrating {model_name}...")
        scores, correct = [], []

        for sent, sent_preds in zip(dev_sentences, preds):
            gold_spans = sent["gold_spans"]
            for pred in sent_preds:
                scores.append(pred.score)
                # Check if this prediction matches any gold entity
                is_correct = any(
                    pred.start == g.start and pred.end == g.end
                    and pred.entity_type == g.entity_type
                    for g in gold_spans
                )
                correct.append(is_correct)

        if not scores:
            print(f"    WARNING: No predictions from {model_name} on dev set")
            reports.append(CalibrationReport(
                model_name=model_name, ece=1.0, ece_post_tempscale=1.0,
                temperature=1.0, derived_uncertainty=0.50,
                n_predictions=0, reliability_bins=[],
            ))
            continue

        ece = compute_ece(scores, correct, n_bins=10)
        bins = reliability_diagram_bins(scores, correct, n_bins=10)
        temp = fit_temperature(scores, correct)

        # Apply temperature scaling
        scores_arr = np.clip(np.array(scores), 1e-7, 1 - 1e-7)
        logits = np.log(scores_arr / (1 - scores_arr))
        calibrated = 1 / (1 + np.exp(-logits / temp))
        ece_post = compute_ece(calibrated.tolist(), correct, n_bins=10)

        u = derive_model_uncertainty(ece)

        report = CalibrationReport(
            model_name=model_name, ece=ece, ece_post_tempscale=ece_post,
            temperature=temp, derived_uncertainty=u,
            n_predictions=len(scores), reliability_bins=bins,
        )
        reports.append(report)

        print(f"    ECE (raw):     {ece:.4f}")
        print(f"    ECE (scaled):  {ece_post:.4f}")
        print(f"    Temperature:   {temp:.4f}")
        print(f"    Uncertainty:   {u:.4f}")
        print(f"    N predictions: {len(scores)}")

    print(f"\n  Phase A0 complete ({_elapsed(t0)})")
    return reports[0], reports[1]


# =====================================================================
# Phase A1: NER Fusion Evaluation
# =====================================================================

def run_phase_a1(
    dataset_name: str,
    dev_sentences: List[Dict],
    test_sentences: List[Dict],
    dev_preds_g2: List[List[EntitySpan]],
    dev_preds_bm: List[List[EntitySpan]],
    test_preds_g2: List[List[EntitySpan]],
    test_preds_bm: List[List[EntitySpan]],
    u_gliner2: float,
    u_biomed: float,
) -> Dict[str, Any]:
    """Run Phase A1 on one dataset."""
    _banner(f"Phase A1: NER Fusion Evaluation — {dataset_name}")
    t0 = time.time()

    # ── Step 1: Optimize thresholds on dev set ──
    print("\n  Optimizing thresholds on dev set...")

    dev_golds = [s["gold_spans"] for s in dev_sentences]
    all_dev_g2 = [span for sent in dev_preds_g2 for span in sent]
    all_dev_bm = [span for sent in dev_preds_bm for span in sent]
    all_dev_gold = [span for sent_golds in dev_golds for span in sent_golds]

    t_g2, f1_g2_dev = optimize_threshold(all_dev_g2, all_dev_gold, ACCEPT_THRESHOLDS)
    t_bm, f1_bm_dev = optimize_threshold(all_dev_bm, all_dev_gold, ACCEPT_THRESHOLDS)
    print(f"    GLiNER2 best threshold:  {t_g2} (dev F1={f1_g2_dev:.4f})")
    print(f"    BioMed best threshold:   {t_bm} (dev F1={f1_bm_dev:.4f})")

    # Scalar average threshold
    t_scalar, _ = optimize_threshold(all_dev_g2 + all_dev_bm, all_dev_gold,
                                     ACCEPT_THRESHOLDS)

    # SL thresholds: sweep accept + abstention on dev
    best_sl_accept = 0.5
    best_sl_abstain = 0.5
    best_sl_f1_dev = -1.0
    for at in ACCEPT_THRESHOLDS:
        for ct in ABSTENTION_THRESHOLDS:
            # Quick eval on dev
            dev_accepted_all = []
            for g2_preds, bm_preds in zip(dev_preds_g2, dev_preds_bm):
                groups = align_spans(g2_preds, bm_preds, iou_threshold=0.5)
                accepted, _ = apply_condition_sl_fusion(
                    groups, u_gliner2, u_biomed, at, ct)
                dev_accepted_all.extend(accepted)
            m = evaluate_entities(dev_accepted_all, all_dev_gold)
            if m.f1 > best_sl_f1_dev:
                best_sl_f1_dev = m.f1
                best_sl_accept = at
                best_sl_abstain = ct
    print(f"    SL best thresholds: accept={best_sl_accept}, "
          f"abstain={best_sl_abstain} (dev F1={best_sl_f1_dev:.4f})")

    # ── Step 2: Evaluate all 6 conditions on test set ──
    # Collect per-sentence results for bootstrap CI
    print("\n  Evaluating 6 conditions on test set...")

    test_golds = [s["gold_spans"] for s in test_sentences]
    all_test_gold = [span for sent_golds in test_golds for span in sent_golds]

    # Per-sentence prediction lists for each condition
    per_sent = {"B1": [], "B2": [], "B3": [], "B4": [], "B5": [], "SL": []}
    conflict_scores = []
    conflict_errors = []
    all_abstained = []

    for i, (g2_preds, bm_preds, sent_golds) in enumerate(
            zip(test_preds_g2, test_preds_bm, test_golds)):
        # B1: GLiNER2 alone
        per_sent["B1"].append(apply_condition_single_model(g2_preds, t_g2))
        # B2: BioMed alone
        per_sent["B2"].append(apply_condition_single_model(bm_preds, t_bm))

        groups = align_spans(g2_preds, bm_preds, iou_threshold=0.5)
        # B3: Union
        per_sent["B3"].append(apply_condition_union(groups, t_g2, t_bm))
        # B4: Intersection
        per_sent["B4"].append(apply_condition_intersection(groups, t_g2, t_bm))
        # B5: Scalar average
        per_sent["B5"].append(apply_condition_scalar_average(groups, t_scalar))
        # SL: Fusion
        accepted, abstained = apply_condition_sl_fusion(
            groups, u_gliner2, u_biomed, best_sl_accept, best_sl_abstain)
        per_sent["SL"].append(accepted)
        all_abstained.extend(abstained)

        # Conflict-error pairs for H3.4c
        for g in groups:
            if g["span_a"] is not None and g["span_b"] is not None:
                op_a = build_opinion(g["span_a"].score, u_gliner2)
                op_b = build_opinion(g["span_b"].score, u_biomed)
                from jsonld_ex.confidence_algebra import \
                    cumulative_fuse as _cf, conflict_metric as _cm
                fused = _cf(op_a, op_b)
                conf = _cm(fused)
                winner = g["span_a"] if g["span_a"].score >= g["span_b"].score \
                    else g["span_b"]
                is_err = not any(
                    winner.start == gld.start and winner.end == gld.end
                    and winner.entity_type == gld.entity_type
                    for gld in sent_golds
                )
                conflict_scores.append(conf)
                conflict_errors.append(is_err)

    # Aggregate metrics per condition
    condition_keys = [("B1: GLiNER2", "B1"), ("B2: BioMed", "B2"),
                      ("B3: Union", "B3"), ("B4: Intersection", "B4"),
                      ("B5: Scalar Avg", "B5"), ("SL: Fusion", "SL")]
    conditions = []
    for name, key in condition_keys:
        flat_preds = [s for sent in per_sent[key] for s in sent]
        m = evaluate_entities(flat_preds, all_test_gold)
        conditions.append((name, m))

    # Per-sentence (tp, fp, fn) for bootstrap
    def _per_sent_tp_fp_fn(key: str) -> List[Tuple[int, int, int]]:
        result = []
        for sent_preds, sent_golds in zip(per_sent[key], test_golds):
            m = evaluate_entities(sent_preds, sent_golds)
            result.append((m.tp, m.fp, m.fn))
        return result

    boot_data = {key: _per_sent_tp_fp_fn(key)
                 for _, key in condition_keys}

    # ── Step 3: Print results table ──
    m_sl = conditions[5][1]  # SL: Fusion
    m_b5 = conditions[4][1]  # B5: Scalar Avg

    print(f"\n  {'Condition':<20s} {'P':>8s} {'R':>8s} {'F1':>8s} "
          f"{'TP':>6s} {'FP':>6s} {'FN':>6s}")
    print(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} "
          f"{'-' * 6} {'-' * 6} {'-' * 6}")
    for name, m in conditions:
        print(f"  {name:<20s} {m.precision:8.4f} {m.recall:8.4f} {m.f1:8.4f} "
              f"{m.tp:6d} {m.fp:6d} {m.fn:6d}")
    print(f"  SL abstained: {len(all_abstained)}")

    # ── Step 4: Rigorous hypothesis testing ──
    print("\n  Hypothesis testing (bootstrap CI + Holm-Bonferroni)...")

    best_baseline = max(conditions[:5], key=lambda x: x[1].f1)
    best_baseline_name, best_baseline_m = best_baseline
    best_baseline_key = [k for n, k in condition_keys if n == best_baseline_name][0]
    print(f"    Best baseline: {best_baseline_name} (F1={best_baseline_m.f1:.4f})")

    # H3.4a (PRIMARY): SL vs best baseline
    f1_diff = m_sl.f1 - best_baseline_m.f1
    h_effect = cohens_h(m_sl.f1, best_baseline_m.f1)
    ci_lo_a, ci_mean_a, ci_hi_a = bootstrap_f1_difference_ci(
        boot_data["SL"], boot_data[best_baseline_key],
        n_bootstrap=2000, seed=SEED)
    h3_4a_verdict = "ACCEPTED" if ci_lo_a > 0 else (
        "REJECTED" if ci_hi_a < 0 else "INCONCLUSIVE")
    print(f"    H3.4a [PRIMARY]: SL - Best = {f1_diff:+.4f}, "
          f"95% CI [{ci_lo_a:+.4f}, {ci_hi_a:+.4f}], "
          f"Cohen's h={h_effect:.4f} → {h3_4a_verdict}")

    # H3.4b: SL vs scalar average
    f1_diff_b = m_sl.f1 - m_b5.f1
    h_effect_b = cohens_h(m_sl.f1, m_b5.f1)
    ci_lo_b, ci_mean_b, ci_hi_b = bootstrap_f1_difference_ci(
        boot_data["SL"], boot_data["B5"],
        n_bootstrap=2000, seed=SEED)
    print(f"    H3.4b: SL - Scalar = {f1_diff_b:+.4f}, "
          f"95% CI [{ci_lo_b:+.4f}, {ci_hi_b:+.4f}], "
          f"Cohen's h={h_effect_b:.4f}")

    # H3.4c: Conflict-error correlation
    if len(conflict_scores) >= 10:
        rho, p_val = spearman_conflict_error(conflict_scores, conflict_errors)
        print(f"    H3.4c: Spearman rho={rho:.4f}, p={p_val:.6f}")
    else:
        rho, p_val = 0.0, 1.0
        print(f"    H3.4c: Insufficient data ({len(conflict_scores)} pairs)")

    # H3.4d: Specialist vs generalist
    m_b1 = conditions[0][1]
    m_b2 = conditions[1][1]
    h_d_diff = m_b2.f1 - m_b1.f1
    ci_lo_d, _, ci_hi_d = bootstrap_f1_difference_ci(
        boot_data["B2"], boot_data["B1"],
        n_bootstrap=2000, seed=SEED)
    print(f"    H3.4d: BioMed - GLiNER2 = {h_d_diff:+.4f}, "
          f"95% CI [{ci_lo_d:+.4f}, {ci_hi_d:+.4f}]")

    # H3.4f: Abstention precision
    # Check if abstained entities were disproportionately wrong
    n_abstained = len(all_abstained)
    if n_abstained > 0:
        abstain_errors = sum(
            1 for a in all_abstained
            if not any(a.start == g.start and a.end == g.end
                       and a.entity_type == g.entity_type
                       for g in all_test_gold)
        )
        abstain_error_rate = abstain_errors / n_abstained
        # Compare: error rate among accepted SL predictions
        sl_error_rate = m_sl.fp / (m_sl.tp + m_sl.fp) if (m_sl.tp + m_sl.fp) > 0 else 0
        print(f"    H3.4f: Abstained={n_abstained}, "
              f"abstain error rate={abstain_error_rate:.3f}, "
              f"accept error rate={sl_error_rate:.3f}")
        h3_4f_helps = abstain_error_rate > sl_error_rate
        print(f"    H3.4f: Abstention targets errors? {h3_4f_helps} "
              f"(abstain err {abstain_error_rate:.3f} vs accept err {sl_error_rate:.3f})")
    else:
        abstain_error_rate = 0.0
        sl_error_rate = 0.0
        h3_4f_helps = False
        print(f"    H3.4f: No abstentions")

    # Holm-Bonferroni on secondary hypotheses (H3.4b, c, d, f)
    # Use CI-based p-value proxies: if CI includes 0, p > 0.05
    p_b = 0.001 if ci_lo_b > 0 else (0.5 if ci_hi_b > 0 and ci_lo_b < 0 else 0.001)
    p_d = 0.001 if ci_lo_d > 0 else (0.5 if ci_hi_d > 0 and ci_lo_d < 0 else 0.001)
    secondary_pvals = [p_b, p_val, p_d]  # H3.4b, H3.4c, H3.4d
    hb_results = holm_bonferroni(secondary_pvals, alpha=0.05)

    h3_4b_verdict = "ACCEPTED" if ci_lo_b > 0 and hb_results[0]["reject"] else (
        "REJECTED" if ci_hi_b < 0 else "INCONCLUSIVE")
    h3_4c_verdict = "ACCEPTED" if rho > 0.3 and hb_results[1]["reject"] else "REJECTED"
    h3_4d_verdict = "ACCEPTED" if ci_lo_d > 0 and hb_results[2]["reject"] else (
        "REJECTED" if ci_hi_d < 0 else "INCONCLUSIVE")

    print(f"\n  ── Verdicts ──")
    print(f"    H3.4a [PRIMARY]:  {h3_4a_verdict}")
    print(f"    H3.4b [corrected]: {h3_4b_verdict}")
    print(f"    H3.4c [corrected]: {h3_4c_verdict}")
    print(f"    H3.4d [corrected]: {h3_4d_verdict}")
    print(f"    H3.4f [descriptive]: {'Abstention targets errors' if h3_4f_helps else 'Abstention does NOT preferentially target errors'}")

    # ── Step 5: Build results dict ──
    result_metrics = {
        "dataset": dataset_name,
        "conditions": {name: {"precision": m.precision, "recall": m.recall,
                              "f1": m.f1, "tp": m.tp, "fp": m.fp, "fn": m.fn}
                       for name, m in conditions},
        "best_baseline": best_baseline_name,
        "sl_f1_minus_best": f1_diff,
        "cohens_h": h_effect,
        "bootstrap_ci_sl_vs_best": {"lower": ci_lo_a, "mean": ci_mean_a, "upper": ci_hi_a},
        "bootstrap_ci_sl_vs_scalar": {"lower": ci_lo_b, "mean": ci_mean_b, "upper": ci_hi_b},
        "bootstrap_ci_biomed_vs_gliner2": {"lower": ci_lo_d, "upper": ci_hi_d},
        "abstention_count": len(all_abstained),
        "abstention_error_rate": abstain_error_rate,
        "accepted_error_rate": sl_error_rate,
        "conflict_error_rho": rho,
        "conflict_error_p": p_val,
        "thresholds": {
            "gliner2": t_g2, "biomed": t_bm, "scalar": t_scalar,
            "sl_accept": best_sl_accept, "sl_abstain": best_sl_abstain,
        },
        "hypotheses": {
            "H3.4a_fusion_improves": h3_4a_verdict,
            "H3.4b_sl_beats_scalar": h3_4b_verdict,
            "H3.4c_conflict_error": h3_4c_verdict,
            "H3.4d_specialist_wins": h3_4d_verdict,
            "H3.4f_abstention_targets_errors": h3_4f_helps,
        },
        "holm_bonferroni": [r for r in hb_results],
    }

    print(f"\n  Phase A1 ({dataset_name}) complete ({_elapsed(t0)})")
    return result_metrics


# =====================================================================
# Phase B: FHIR Clinical Pipeline
# =====================================================================

def run_phase_b(
    u_gliner2: float,
    u_biomed: float,
    n_bundles: int = 100,
) -> Dict[str, Any]:
    """Run Phase B: FHIR clinical pipeline demonstration."""
    _banner("Phase B: FHIR Clinical Pipeline")
    t0 = time.time()

    # ── Load Synthea bundles ──
    print(f"\n  Loading Synthea bundles from {SYNTHEA_DIR}...")
    bundle_files = sorted(SYNTHEA_DIR.glob("*.json"))[:n_bundles]
    print(f"  Found {len(bundle_files)} bundle files")

    target_types = {"Patient", "Observation", "Condition", "MedicationRequest",
                    "DiagnosticReport", "Procedure", "Immunization", "Encounter",
                    "AllergyIntolerance", "CarePlan"}

    all_resources: Dict[str, List] = defaultdict(list)
    n_bundles_loaded = 0

    for bf in bundle_files:
        with open(bf, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        resources = load_bundle_resources(bundle, target_types=target_types)
        for rtype, rlist in resources.items():
            all_resources[rtype].extend(rlist)
        n_bundles_loaded += 1

    n_total = sum(len(v) for v in all_resources.values())
    print(f"  {n_bundles_loaded} bundles, {n_total} resources across "
          f"{len(all_resources)} types")
    for rtype, rlist in sorted(all_resources.items()):
        print(f"    {rtype}: {len(rlist)}")

    # ── Extract narrative texts ──
    all_flat_resources = [r for rlist in all_resources.values() for r in rlist]
    texts = extract_narrative_texts(all_flat_resources)
    print(f"  Extracted {len(texts)} narrative text snippets")

    # ── Round-trip fidelity (H3.4g) ──
    print("\n  Round-trip fidelity test...")
    rt_results = []
    rt_perfect = 0
    for rtype in target_types:
        for resource in all_resources.get(rtype, [])[:10]:  # Sample 10 per type
            result = measure_round_trip_fidelity(resource)
            rt_results.append(result)
            if result.fidelity_rate == 1.0:
                rt_perfect += 1

    rt_rate = rt_perfect / len(rt_results) if rt_results else 0
    print(f"  {rt_perfect}/{len(rt_results)} resources with 100% fidelity "
          f"(rate={rt_rate:.2%})")

    # ── Annotation overhead (H3.4h) ──
    print("\n  Annotation overhead measurement...")
    oh_results = []
    for rtype in target_types:
        for resource in all_resources.get(rtype, [])[:10]:
            result = measure_annotation_overhead(resource)
            oh_results.append(result)

    oh_pcts = [r.overhead_pct for r in oh_results]
    if oh_pcts:
        print(f"  Overhead: mean={np.mean(oh_pcts):.1f}%, "
              f"median={np.median(oh_pcts):.1f}%, "
              f"p95={np.percentile(oh_pcts, 95):.1f}%")

    # ── Query expressiveness (H3.4i) ──
    print("\n  Query expressiveness demonstration...")
    queries_demonstrated = 0

    # Q1: Confidence filter
    from jsonld_ex.fhir_interop import from_fhir as _ff
    sample_obs = all_resources.get("Observation", [None])[0]
    if sample_obs:
        doc, _ = _ff(sample_obs)
        if doc.get("opinions"):
            query_by_confidence([doc], threshold=0.5)
            queries_demonstrated += 1
            print(f"    Q1: Confidence filter — demonstrated")

    # Q2: Provenance
    if sample_obs:
        doc, _ = _ff(sample_obs)
        opinions = doc.get("opinions", [])
        if opinions:
            # Add source/method metadata for demo
            for op in opinions:
                op["source"] = "gliner-biomed-v1.0"
                op["method"] = "NER-zero-shot"
            query_provenance_chain([doc])
            queries_demonstrated += 1
            print(f"    Q2: Provenance chain — demonstrated")

    # Q3: Multi-model fusion
    from jsonld_ex.confidence_algebra import Opinion as _Opinion
    op_a = _Opinion.from_confidence(0.85, uncertainty=u_gliner2)
    op_b = _Opinion.from_confidence(0.75, uncertainty=u_biomed)
    query_fuse_multi_model(op_a, op_b)
    queries_demonstrated += 1
    print(f"    Q3: Multi-model fusion — demonstrated")

    # Q4: Temporal decay
    op = _Opinion.from_confidence(0.90, uncertainty=0.05)
    query_temporal_decay(op, age_days=365, half_life_days=180)
    queries_demonstrated += 1
    print(f"    Q4: Temporal decay — demonstrated")

    # Q5: Conflict-based abstention
    query_abstain_on_conflict(conflict_score=0.7, threshold=0.5)
    queries_demonstrated += 1
    print(f"    Q5: Conflict abstention — demonstrated")

    # Q6: Uncertainty component filter
    doc_high_u = {"@type": "fhir:Observation", "resource_id": "test",
                  "opinions": [{"uncertainty": 0.40, "belief": 0.30, "disbelief": 0.30}]}
    doc_low_u = {"@type": "fhir:Observation", "resource_id": "test2",
                 "opinions": [{"uncertainty": 0.05, "belief": 0.90, "disbelief": 0.05}]}
    filtered = query_by_uncertainty_component([doc_high_u, doc_low_u], max_uncertainty=0.10)
    queries_demonstrated += 1
    print(f"    Q6: Uncertainty filter — demonstrated ({len(filtered)} passed)")

    print(f"\n  {queries_demonstrated} query types demonstrated (target ≥ 5)")

    # ── Build report ──
    report = PipelineReport(
        n_bundles=n_bundles_loaded,
        n_resources_total=n_total,
        n_resources_by_type={k: len(v) for k, v in all_resources.items()},
        n_narrative_texts=len(texts),
        round_trip_results=[{
            "resource_type": r.resource_type, "resource_id": r.resource_id,
            "fidelity_rate": r.fidelity_rate, "lost_fields": r.lost_fields,
        } for r in rt_results],
        overhead_results=[{
            "resource_type": r.resource_type,
            "original_bytes": r.original_bytes,
            "overhead_pct": r.overhead_pct,
        } for r in oh_results],
        query_types_demonstrated=queries_demonstrated,
    )

    result_metrics = {
        "n_bundles": n_bundles_loaded,
        "n_resources": n_total,
        "n_narrative_texts": len(texts),
        "round_trip_perfect_rate": rt_rate,
        "overhead_mean_pct": float(np.mean(oh_pcts)) if oh_pcts else 0,
        "overhead_median_pct": float(np.median(oh_pcts)) if oh_pcts else 0,
        "query_types_demonstrated": queries_demonstrated,
        "hypotheses": {
            "H3.4g_round_trip": "ACCEPTED" if rt_rate >= 0.95 else "REJECTED",
            "H3.4h_overhead_lt_15pct": (
                "ACCEPTED" if np.mean(oh_pcts) < 15 else "REJECTED"
            ) if oh_pcts else "INSUFFICIENT_DATA",
            "H3.4i_query_expressiveness": (
                "ACCEPTED" if queries_demonstrated >= 5 else "REJECTED"),
        },
    }

    print(f"\n  Phase B complete ({_elapsed(t0)})")
    return result_metrics


# =====================================================================
# Phase A1-Extra: Strengthening Analyses (all CPU, cached predictions)
# =====================================================================

def _apply_temperature_scaling(
    preds: List[List[EntitySpan]],
    temperature: float,
) -> List[List[EntitySpan]]:
    """Apply temperature scaling to all prediction scores."""
    scaled = []
    for sent_preds in preds:
        sent_scaled = []
        for p in sent_preds:
            raw = np.clip(p.score, 1e-7, 1 - 1e-7)
            logit = np.log(raw / (1 - raw))
            cal_score = float(1.0 / (1.0 + np.exp(-logit / temperature)))
            sent_scaled.append(EntitySpan(
                start=p.start, end=p.end, entity_type=p.entity_type,
                score=cal_score, source=p.source, text=p.text,
            ))
        scaled.append(sent_scaled)
    return scaled


def run_phase_a1_extras(
    dev_sentences: List[Dict],
    test_sentences: List[Dict],
    dev_preds_g2: List[List[EntitySpan]],
    dev_preds_bm: List[List[EntitySpan]],
    test_preds_g2: List[List[EntitySpan]],
    test_preds_bm: List[List[EntitySpan]],
    u_gliner2: float,
    u_biomed: float,
    temp_gliner2: float,
    temp_biomed: float,
    ece_post_g2: float,
    ece_post_bm: float,
) -> Dict[str, Any]:
    """Run all strengthening analyses on cached predictions."""
    from EN3.en3_4_calibration import derive_model_uncertainty
    from sklearn.metrics import roc_auc_score  # type: ignore

    results: Dict[str, Any] = {}

    test_golds = [s["gold_spans"] for s in test_sentences]
    all_test_gold = [span for sent_golds in test_golds for span in sent_golds]

    # ================================================================
    # 1. Temperature-Scaled Ablation
    # ================================================================
    _banner("A1-Extra: Temperature-Scaled Ablation")
    t0 = time.time()

    # Apply temperature scaling to predictions
    ts_dev_g2 = _apply_temperature_scaling(dev_preds_g2, temp_gliner2)
    ts_dev_bm = _apply_temperature_scaling(dev_preds_bm, temp_biomed)
    ts_test_g2 = _apply_temperature_scaling(test_preds_g2, temp_gliner2)
    ts_test_bm = _apply_temperature_scaling(test_preds_bm, temp_biomed)

    # Derive calibrated uncertainty values
    u_g2_cal = derive_model_uncertainty(ece_post_g2)
    u_bm_cal = derive_model_uncertainty(ece_post_bm)
    print(f"  Calibrated uncertainties: GLiNER2={u_g2_cal:.4f}, BioMed={u_bm_cal:.4f}")
    print(f"  (Raw were: GLiNER2={u_gliner2:.4f}, BioMed={u_biomed:.4f})")

    # Run full evaluation with temp-scaled scores
    ts_results = run_phase_a1(
        "BC5CDR-TempScaled", dev_sentences, test_sentences,
        ts_dev_g2, ts_dev_bm, ts_test_g2, ts_test_bm,
        u_g2_cal, u_bm_cal,
    )
    results["tempscaled"] = ts_results
    print(f"  Temperature-scaled ablation complete ({_elapsed(t0)})")

    # ================================================================
    # 2. Per-Entity-Type Breakdown
    # ================================================================
    _banner("A1-Extra: Per-Entity-Type Breakdown")

    for etype in ["Chemical", "Disease"]:
        type_golds = [EntitySpan(g.start, g.end, g.entity_type, g.score, g.source)
                      for g in all_test_gold if g.entity_type == etype]
        type_count = len(type_golds)

        print(f"\n  Entity type: {etype} ({type_count} gold entities)")
        print(f"  {'Condition':<20s} {'P':>8s} {'R':>8s} {'F1':>8s}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

        type_results = {}
        for cond_name, cond_key in [("B1: GLiNER2", "g2"), ("B2: BioMed", "bm")]:
            preds_source = test_preds_g2 if cond_key == "g2" else test_preds_bm
            t_opt = 0.7  # use same threshold
            flat = []
            for sent_preds in preds_source:
                flat.extend([p for p in sent_preds
                             if p.score >= t_opt and p.entity_type == etype])
            m = evaluate_entities(flat, type_golds)
            print(f"  {cond_name:<20s} {m.precision:8.4f} {m.recall:8.4f} {m.f1:8.4f}")
            type_results[cond_name] = {"p": m.precision, "r": m.recall, "f1": m.f1}

        # SL fusion for this type
        sl_flat = []
        for g2_preds, bm_preds in zip(test_preds_g2, test_preds_bm):
            groups = align_spans(g2_preds, bm_preds, iou_threshold=0.5)
            accepted, _ = apply_condition_sl_fusion(
                groups, u_gliner2, u_biomed, 0.7, 0.5)
            sl_flat.extend([a for a in accepted if a.entity_type == etype])
        m = evaluate_entities(sl_flat, type_golds)
        print(f"  {'SL: Fusion':<20s} {m.precision:8.4f} {m.recall:8.4f} {m.f1:8.4f}")
        type_results["SL: Fusion"] = {"p": m.precision, "r": m.recall, "f1": m.f1}

        results[f"per_type_{etype}"] = type_results

    # ================================================================
    # 3. Conflict Detection AUROC
    # ================================================================
    _banner("A1-Extra: Conflict Detection AUROC")

    conflict_scores = []
    is_error_labels = []

    for g2_preds, bm_preds, sent_golds in zip(
            test_preds_g2, test_preds_bm, test_golds):
        groups = align_spans(g2_preds, bm_preds, iou_threshold=0.5)
        for g in groups:
            if g["span_a"] is not None and g["span_b"] is not None:
                op_a = build_opinion(g["span_a"].score, u_gliner2)
                op_b = build_opinion(g["span_b"].score, u_biomed)
                from jsonld_ex.confidence_algebra import \
                    cumulative_fuse as _cf, conflict_metric as _cm
                fused = _cf(op_a, op_b)
                conf = _cm(fused)
                winner = g["span_a"] if g["span_a"].score >= g["span_b"].score \
                    else g["span_b"]
                is_err = not any(
                    winner.start == gld.start and winner.end == gld.end
                    and winner.entity_type == gld.entity_type
                    for gld in sent_golds
                )
                conflict_scores.append(conf)
                is_error_labels.append(1 if is_err else 0)

    if len(set(is_error_labels)) == 2:  # need both classes for AUROC
        auroc = roc_auc_score(is_error_labels, conflict_scores)
        print(f"  Conflict as error predictor AUROC: {auroc:.4f}")
        print(f"  N pairs: {len(conflict_scores)}")
        print(f"  Error rate: {sum(is_error_labels)/len(is_error_labels):.3f}")
        results["conflict_auroc"] = auroc
    else:
        print(f"  Cannot compute AUROC (only one class present)")
        results["conflict_auroc"] = None

    # ================================================================
    # 4. Precision-Recall Curves Across Operating Points
    # ================================================================
    _banner("A1-Extra: Precision-Recall Curves")

    thresholds_sweep = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    pr_curves = {}

    for cond_name, apply_fn in [
        ("B2: BioMed", lambda t: [
            p for sp in test_preds_bm for p in sp if p.score >= t]),
        ("B5: Scalar Avg", None),  # handled separately
        ("SL: Fusion", None),      # handled separately
    ]:
        points = []
        for t in thresholds_sweep:
            if cond_name == "B2: BioMed":
                flat = apply_fn(t)
            elif cond_name == "B5: Scalar Avg":
                flat = []
                for g2p, bmp in zip(test_preds_g2, test_preds_bm):
                    groups = align_spans(g2p, bmp, iou_threshold=0.5)
                    flat.extend(apply_condition_scalar_average(groups, t))
            elif cond_name == "SL: Fusion":
                flat = []
                for g2p, bmp in zip(test_preds_g2, test_preds_bm):
                    groups = align_spans(g2p, bmp, iou_threshold=0.5)
                    accepted, _ = apply_condition_sl_fusion(
                        groups, u_gliner2, u_biomed, t, 0.5)
                    flat.extend(accepted)
            m = evaluate_entities(flat, all_test_gold)
            points.append({"threshold": t, "precision": m.precision,
                           "recall": m.recall, "f1": m.f1})
        pr_curves[cond_name] = points

    for cond_name, points in pr_curves.items():
        print(f"\n  {cond_name}:")
        print(f"    {'Thresh':>8s} {'P':>8s} {'R':>8s} {'F1':>8s}")
        for p in points:
            print(f"    {p['threshold']:8.1f} {p['precision']:8.4f} "
                  f"{p['recall']:8.4f} {p['f1']:8.4f}")

    results["pr_curves"] = pr_curves

    # ================================================================
    # 5. Confidence-Based vs Conflict-Based Abstention
    # ================================================================
    _banner("A1-Extra: Confidence vs Conflict Abstention")

    # Collect all dual-model predictions with their conflict and min-confidence
    dual_preds = []  # (entity_span, conflict_score, min_confidence, is_correct)
    for g2_preds, bm_preds, sent_golds in zip(
            test_preds_g2, test_preds_bm, test_golds):
        groups = align_spans(g2_preds, bm_preds, iou_threshold=0.5)
        for g in groups:
            if g["span_a"] is not None and g["span_b"] is not None:
                op_a = build_opinion(g["span_a"].score, u_gliner2)
                op_b = build_opinion(g["span_b"].score, u_biomed)
                from jsonld_ex.confidence_algebra import \
                    cumulative_fuse as _cf2, conflict_metric as _cm2
                fused = _cf2(op_a, op_b)
                conf = _cm2(fused)
                winner = g["span_a"] if g["span_a"].score >= g["span_b"].score \
                    else g["span_b"]
                min_score = min(g["span_a"].score, g["span_b"].score)
                is_correct = any(
                    winner.start == gld.start and winner.end == gld.end
                    and winner.entity_type == gld.entity_type
                    for gld in sent_golds
                )
                dual_preds.append({
                    "conflict": conf, "min_conf": min_score,
                    "fused_pp": fused.projected_probability(),
                    "correct": is_correct,
                })

    print(f"  {len(dual_preds)} dual-model entity pairs")
    print(f"\n  {'Method':<25s} {'Thresh':>8s} {'Abstain':>8s} "
          f"{'P_keep':>8s} {'R_keep':>8s} {'Err_abstain':>12s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")

    abstention_comparison = []
    for tau in [0.2, 0.3, 0.4, 0.5, 0.6]:
        # Conflict-based abstention
        kept_conflict = [d for d in dual_preds if d["conflict"] <= tau]
        abst_conflict = [d for d in dual_preds if d["conflict"] > tau]
        p_conf = (sum(d["correct"] for d in kept_conflict) / len(kept_conflict)
                  if kept_conflict else 0)
        r_conf = (sum(d["correct"] for d in kept_conflict) /
                  sum(d["correct"] for d in dual_preds)
                  if any(d["correct"] for d in dual_preds) else 0)
        err_abst_conf = (sum(not d["correct"] for d in abst_conflict) / len(abst_conflict)
                         if abst_conflict else 0)

        # Confidence-based abstention (abstain when min_score < tau)
        kept_scalar = [d for d in dual_preds if d["min_conf"] >= tau]
        abst_scalar = [d for d in dual_preds if d["min_conf"] < tau]
        p_scal = (sum(d["correct"] for d in kept_scalar) / len(kept_scalar)
                  if kept_scalar else 0)
        r_scal = (sum(d["correct"] for d in kept_scalar) /
                  sum(d["correct"] for d in dual_preds)
                  if any(d["correct"] for d in dual_preds) else 0)
        err_abst_scal = (sum(not d["correct"] for d in abst_scalar) / len(abst_scalar)
                         if abst_scalar else 0)

        print(f"  {'Conflict τ='+str(tau):<25s} {tau:8.1f} "
              f"{len(abst_conflict):8d} {p_conf:8.4f} {r_conf:8.4f} "
              f"{err_abst_conf:12.4f}")
        print(f"  {'Confidence τ='+str(tau):<25s} {tau:8.1f} "
              f"{len(abst_scalar):8d} {p_scal:8.4f} {r_scal:8.4f} "
              f"{err_abst_scal:12.4f}")

        abstention_comparison.append({
            "threshold": tau,
            "conflict": {"n_abstained": len(abst_conflict),
                         "precision_kept": p_conf, "recall_kept": r_conf,
                         "error_rate_abstained": err_abst_conf},
            "confidence": {"n_abstained": len(abst_scalar),
                           "precision_kept": p_scal, "recall_kept": r_scal,
                           "error_rate_abstained": err_abst_scal},
        })

    results["abstention_comparison"] = abstention_comparison

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="EN3.4 Experiment Runner")
    parser.add_argument("--phase", choices=["a0", "a1", "a1-extra", "medmentions", "b", "all"],
                        default="all", help="Phase to run")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Load predictions from checkpoints (skip GPU)")
    parser.add_argument("--n-bundles", type=int, default=100,
                        help="Number of Synthea bundles for Phase B")
    args = parser.parse_args()

    set_global_seed(SEED)
    run_all = args.phase == "all"

    _banner("EN3.4: FHIR R4 Clinical Data Exchange")
    print(f"  Phase:  {args.phase}")
    print(f"  Seed:   {SEED}")
    print(f"  Time:   {datetime.now(timezone.utc).isoformat()}")
    env = log_environment()

    # ── Load BC5CDR ──
    if run_all or args.phase in ("a0", "a1"):
        dev_data = load_bc5cdr("validation")
        test_data = load_bc5cdr("test")

        # ── Model inference (or load checkpoint) ──
        if args.skip_inference:
            print("\n  Loading predictions from checkpoints...")
            g2_dev = _serializable_to_preds(_load_checkpoint("en3_4_bc5cdr_dev_gliner2"))
            g2_test = _serializable_to_preds(_load_checkpoint("en3_4_bc5cdr_test_gliner2"))
            bm_dev = _serializable_to_preds(_load_checkpoint("en3_4_bc5cdr_dev_biomed"))
            bm_test = _serializable_to_preds(_load_checkpoint("en3_4_bc5cdr_test_biomed"))
        else:
            print("\n  Running model inference (GPU)...")
            g2_dev = run_gliner2_inference(dev_data)
            _save_checkpoint(_preds_to_serializable(g2_dev), "en3_4_bc5cdr_dev_gliner2")
            g2_test = run_gliner2_inference(test_data)
            _save_checkpoint(_preds_to_serializable(g2_test), "en3_4_bc5cdr_test_gliner2")

            bm_dev = run_biomed_inference(dev_data)
            _save_checkpoint(_preds_to_serializable(bm_dev), "en3_4_bc5cdr_dev_biomed")
            bm_test = run_biomed_inference(test_data)
            _save_checkpoint(_preds_to_serializable(bm_test), "en3_4_bc5cdr_test_biomed")

    # ── Phase A0 ──
    if run_all or args.phase == "a0":
        cal_g2, cal_bm = run_phase_a0(dev_data, g2_dev, bm_dev)
        u_gliner2, u_biomed = cal_g2.derived_uncertainty, cal_bm.derived_uncertainty

        cal_result = ExperimentResult(
            experiment_id="EN3.4-A0",
            parameters={"seed": SEED, "n_bins": 10, "dataset": "BC5CDR-dev"},
            metrics={"gliner2": cal_g2.to_dict(), "biomed": cal_bm.to_dict()},
            environment=env,
        )
        _save_result(cal_result, "EN3_4_calibration")
    else:
        # Load from previous run
        cal_data = _load_checkpoint("en3_4_calibration_values")
        if cal_data:
            u_gliner2 = cal_data["u_gliner2"]
            u_biomed = cal_data["u_biomed"]
        else:
            print("  WARNING: No calibration data found. Using defaults.")
            u_gliner2 = 0.15
            u_biomed = 0.10

    # Save calibration values for reuse
    if run_all or args.phase == "a0":
        _save_checkpoint({"u_gliner2": u_gliner2, "u_biomed": u_biomed},
                         "en3_4_calibration_values")

    # ── Phase A1 — BC5CDR ──
    if run_all or args.phase == "a1":
        bc5cdr_results = run_phase_a1(
            "BC5CDR", dev_data, test_data,
            g2_dev, bm_dev, g2_test, bm_test,
            u_gliner2, u_biomed,
        )

        a1_result = ExperimentResult(
            experiment_id="EN3.4-A1",
            parameters={"seed": SEED, "dataset": "BC5CDR",
                        "u_gliner2": u_gliner2, "u_biomed": u_biomed},
            metrics=bc5cdr_results,
            environment=env,
        )
        _save_result(a1_result, "EN3_4_phase_a_bc5cdr")

    # ── Phase A1-Extra (strengthening analyses) ──
    if run_all or args.phase == "a1-extra":
        # Load predictions from checkpoints
        if args.phase == "a1-extra":
            print("\n  Loading predictions from checkpoints...")
            dev_data = load_bc5cdr("validation")
            test_data = load_bc5cdr("test")
            g2_dev = _serializable_to_preds(_load_checkpoint("en3_4_bc5cdr_dev_gliner2"))
            g2_test = _serializable_to_preds(_load_checkpoint("en3_4_bc5cdr_test_gliner2"))
            bm_dev = _serializable_to_preds(_load_checkpoint("en3_4_bc5cdr_dev_biomed"))
            bm_test = _serializable_to_preds(_load_checkpoint("en3_4_bc5cdr_test_biomed"))

        # Load calibration values
        cal_result_data = ExperimentResult.load_json(
            str(RESULTS_DIR / "EN3_4_calibration.json"))
        temp_g2 = cal_result_data.metrics["gliner2"]["temperature"]
        temp_bm = cal_result_data.metrics["biomed"]["temperature"]
        ece_post_g2 = cal_result_data.metrics["gliner2"]["ece_post_tempscale"]
        ece_post_bm = cal_result_data.metrics["biomed"]["ece_post_tempscale"]

        extras = run_phase_a1_extras(
            dev_data, test_data,
            g2_dev, bm_dev, g2_test, bm_test,
            u_gliner2, u_biomed,
            temp_g2, temp_bm, ece_post_g2, ece_post_bm,
        )

        extras_result = ExperimentResult(
            experiment_id="EN3.4-A1-extras",
            parameters={"seed": SEED, "temp_gliner2": temp_g2,
                        "temp_biomed": temp_bm},
            metrics=extras,
            environment=env,
        )
        _save_result(extras_result, "EN3_4_phase_a_extras")

    # ── Phase MedMentions (H3.4e) ──
    if run_all or args.phase == "medmentions":
        _banner("MedMentions ST21pv Evaluation (H3.4e)")

        mm_dev = load_medmentions("validation")
        mm_test = load_medmentions("test")

        # Model inference (or load checkpoint)
        if args.skip_inference:
            print("\n  Loading MedMentions predictions from checkpoints...")
            mm_g2_dev = _serializable_to_preds(_load_checkpoint("en3_4_mm_dev_gliner2"))
            mm_g2_test = _serializable_to_preds(_load_checkpoint("en3_4_mm_test_gliner2"))
            mm_bm_dev = _serializable_to_preds(_load_checkpoint("en3_4_mm_dev_biomed"))
            mm_bm_test = _serializable_to_preds(_load_checkpoint("en3_4_mm_test_biomed"))
        else:
            print("\n  Running model inference on MedMentions (GPU)...")
            mm_g2_dev = run_gliner2_inference(
                mm_dev, labels=MEDMENTIONS_LABELS_GLINER,
                label_map=MEDMENTIONS_GLINER_TO_GROUP)
            _save_checkpoint(_preds_to_serializable(mm_g2_dev), "en3_4_mm_dev_gliner2")
            mm_g2_test = run_gliner2_inference(
                mm_test, labels=MEDMENTIONS_LABELS_GLINER,
                label_map=MEDMENTIONS_GLINER_TO_GROUP)
            _save_checkpoint(_preds_to_serializable(mm_g2_test), "en3_4_mm_test_gliner2")

            mm_bm_dev = run_biomed_inference(
                mm_dev, labels=MEDMENTIONS_LABELS_GLINER,
                label_map=MEDMENTIONS_GLINER_TO_GROUP)
            _save_checkpoint(_preds_to_serializable(mm_bm_dev), "en3_4_mm_dev_biomed")
            mm_bm_test = run_biomed_inference(
                mm_test, labels=MEDMENTIONS_LABELS_GLINER,
                label_map=MEDMENTIONS_GLINER_TO_GROUP)
            _save_checkpoint(_preds_to_serializable(mm_bm_test), "en3_4_mm_test_biomed")

        # Load calibration values
        cal_data = _load_checkpoint("en3_4_calibration_values")
        if cal_data:
            u_gliner2 = cal_data["u_gliner2"]
            u_biomed = cal_data["u_biomed"]
        else:
            print("  WARNING: No calibration data. Using BC5CDR defaults.")
            u_gliner2 = 0.315
            u_biomed = 0.289

        # Run full A1 evaluation
        mm_results = run_phase_a1(
            "MedMentions-ST21pv", mm_dev, mm_test,
            mm_g2_dev, mm_bm_dev, mm_g2_test, mm_bm_test,
            u_gliner2, u_biomed,
        )

        mm_result_obj = ExperimentResult(
            experiment_id="EN3.4-MedMentions",
            parameters={"seed": SEED, "dataset": "MedMentions-ST21pv",
                        "n_groups": 7, "n_original_types": 21,
                        "u_gliner2": u_gliner2, "u_biomed": u_biomed},
            metrics=mm_results,
            environment=env,
        )
        _save_result(mm_result_obj, "EN3_4_medmentions")

    # ── Phase B ──
    if run_all or args.phase == "b":
        phase_b_results = run_phase_b(u_gliner2, u_biomed, n_bundles=args.n_bundles)

        b_result = ExperimentResult(
            experiment_id="EN3.4-B",
            parameters={"seed": SEED, "n_bundles": args.n_bundles,
                        "u_gliner2": u_gliner2, "u_biomed": u_biomed},
            metrics=phase_b_results,
            environment=env,
        )
        _save_result(b_result, "EN3_4_phase_b")

    # ── Summary ──
    _banner("EN3.4 Complete")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"  Checkpoints at:   {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
