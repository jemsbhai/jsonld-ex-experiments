#!/usr/bin/env python
"""EN1.1c — GLiNER2 Runner: Generate predictions for CoNLL-2003.

Runs GLiNER2 (fastino/gliner2-base-v1) on CoNLL-2003 dev and test sets,
producing per-token IOB2 predictions with confidence scores in the same
format as the existing 4 models (spaCy, Flair, Stanza, HuggingFace).

Saves checkpoints to:
    experiments/EN1/checkpoints/dev_preds_gliner2.json
    experiments/EN1/checkpoints/test_preds_gliner2.json

These checkpoints are then consumed by en1_1c_5model_fusion.py to run
the full 5-model fusion experiment.

Key design notes:
    - GLiNER2 is a span-matching model (dot-product + sigmoid), not a
      token classifier (softmax). Its confidence scores have fundamentally
      different calibration properties from the other 4 models.
    - We use include_confidence=True and include_spans=True to get
      per-entity scores and character offsets.
    - We use a LOW threshold (0.3) to get more candidate entities.
      Temperature scaling calibration happens during fusion, not here.
      Using the default threshold (0.5) would pre-filter entities that
      the fusion step should evaluate.
    - GLiNER2 uses natural language entity labels, not CoNLL codes.
      We map: person->PER, location->LOC, organization->ORG,
      miscellaneous->MISC.

Prerequisites:
    pip install gliner2
    Run check_en1_1c_gliner2.py first to verify alignment.

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN1/en1_1c_gliner2_runner.py

Output:
    experiments/EN1/checkpoints/dev_preds_gliner2.json
    experiments/EN1/checkpoints/test_preds_gliner2.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Path setup ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_PKG_SRC = _REPO_ROOT / "packages" / "python" / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

CHECKPOINT_DIR = _SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────

# CoNLL-2003 entity types
ENTITY_TYPES = ["PER", "LOC", "ORG", "MISC"]

# GLiNER2 natural language labels -> CoNLL codes
GLINER2_LABEL_MAP = {
    "person": "PER",
    "location": "LOC",
    "organization": "ORG",
    "miscellaneous": "MISC",
}
GLINER2_ENTITY_LABELS = list(GLINER2_LABEL_MAP.keys())

# Model configuration
GLINER2_MODEL_NAME = "fastino/gliner2-base-v1"
GLINER2_THRESHOLD = 0.3  # Low threshold; calibration happens at fusion time
GLINER2_DEFAULT_O_CONF = 0.5  # Conservative default for non-entity tokens

# CoNLL-2003 tag names for data loading
TAG_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
             "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


# =====================================================================
# Data Loading (same logic as en1_1_ner_fusion.py)
# =====================================================================

def load_conll2003(split: str = "test") -> List[Dict]:
    """Load CoNLL-2003 dataset from HuggingFace.

    Returns list of sentences, each with:
        tokens: List[str]
        ner_tags: List[str]   (IOB2 format: O, B-PER, I-PER, ...)
    """
    from datasets import load_dataset

    sources = [
        ("eriktks/conll2003", {"revision": "refs/convert/parquet"}),
        ("conll2003", {"revision": "refs/convert/parquet"}),
        ("tner/conll2003", {"revision": "refs/convert/parquet"}),
        ("DFKI-SLT/conll2003", {}),
    ]

    ds = None
    for source, kwargs in sources:
        try:
            print(f"    Trying {source} ...")
            ds = load_dataset(source, split=split, **kwargs)
            print(f"    Loaded from {source}: {len(ds)} sentences")
            break
        except Exception as e:
            print(f"    {source} failed: {type(e).__name__}: {str(e)[:120]}")
            continue

    if ds is None:
        raise RuntimeError(
            "Could not load CoNLL-2003 from any source. "
            "Try: pip install datasets && huggingface-cli login"
        )

    sentences = []
    for item in ds:
        tokens = item["tokens"]
        raw_tags = item["ner_tags"]
        if raw_tags and isinstance(raw_tags[0], int):
            tags = [TAG_NAMES[t] for t in raw_tags]
        else:
            tags = list(raw_tags)
        sentences.append({"tokens": tokens, "ner_tags": tags})

    return sentences


# =====================================================================
# Span-to-Token Alignment
# =====================================================================

def tokens_to_text_with_offsets(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """Join tokens with spaces, tracking character offsets per token.

    Returns:
        text: The space-joined string.
        offsets: List of (start_char, end_char) for each token.
    """
    offsets = []
    pos = 0
    for tok in tokens:
        offsets.append((pos, pos + len(tok)))
        pos += len(tok) + 1  # +1 for the space separator
    text = " ".join(tokens)
    return text, offsets


def char_span_to_token_indices(
    span_start: int,
    span_end: int,
    token_offsets: List[Tuple[int, int]],
) -> List[int]:
    """Map a character-level span to token indices via overlap."""
    indices = []
    for idx, (tok_start, tok_end) in enumerate(token_offsets):
        if tok_start < span_end and tok_end > span_start:
            indices.append(idx)
    return indices


def gliner2_result_to_iob2(
    result: Dict[str, Any],
    tokens: List[str],
    token_offsets: List[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """Convert GLiNER2 output (with confidence+spans) to token-level IOB2.

    Args:
        result: GLiNER2 output from extract_entities with
                include_confidence=True, include_spans=True.
                Format: {"entities": {"person": [{"text": ..., "confidence": ...,
                         "start": ..., "end": ...}], ...}}
        tokens: Original token list.
        token_offsets: Character offsets from tokens_to_text_with_offsets.

    Returns:
        Per-token predictions: [{"tag": "B-PER", "confidence": 0.95}, ...]
    """
    n_tokens = len(tokens)

    # Initialize all tokens as O with conservative confidence
    token_preds = [{"tag": "O", "confidence": GLINER2_DEFAULT_O_CONF}
                   for _ in range(n_tokens)]

    # Collect all entity spans with confidence
    all_spans = []  # (token_indices, conll_tag, confidence)

    entities = result.get("entities", {})
    for label, ent_list in entities.items():
        conll_tag = GLINER2_LABEL_MAP.get(label)
        if conll_tag is None:
            continue

        for ent in ent_list:
            # With include_confidence=True, include_spans=True,
            # each entity is a dict with text, confidence, start, end
            if isinstance(ent, dict):
                conf = ent["confidence"]
                span_start = ent["start"]
                span_end = ent["end"]
            else:
                # Fallback for unexpected format: string-only entity
                # This should not happen with include_confidence=True
                print(f"  WARNING: entity is not a dict: {ent} (label={label})")
                continue

            tok_indices = char_span_to_token_indices(
                span_start, span_end, token_offsets
            )

            if tok_indices:
                all_spans.append((tok_indices, conll_tag, conf))

    # Sort by confidence descending — highest confidence wins overlaps
    all_spans.sort(key=lambda x: -x[2])

    # Assign IOB2 tags (no-overlap: first assigned = highest confidence wins)
    assigned = set()
    for tok_indices, conll_tag, conf in all_spans:
        if any(idx in assigned for idx in tok_indices):
            continue

        for i, idx in enumerate(tok_indices):
            prefix = "B" if i == 0 else "I"
            token_preds[idx] = {
                "tag": f"{prefix}-{conll_tag}",
                "confidence": float(conf),
            }
            assigned.add(idx)

    return token_preds


# =====================================================================
# GLiNER2 Runner
# =====================================================================

def run_gliner2(
    sentences: List[Dict],
    model_name: str = GLINER2_MODEL_NAME,
    threshold: float = GLINER2_THRESHOLD,
) -> List[List[Dict]]:
    """Run GLiNER2 on all sentences, returning per-token IOB2 predictions.

    Args:
        sentences: List of {"tokens": [...], "ner_tags": [...]} dicts.
        model_name: HuggingFace model name for GLiNER2.
        threshold: Minimum matching score to consider an entity.

    Returns:
        List of per-sentence predictions, each a list of
        {"tag": "B-PER", "confidence": 0.95} dicts.
    """
    from gliner2 import GLiNER2

    print(f"  Loading {model_name} ...")
    t0 = time.time()
    model = GLiNER2.from_pretrained(model_name)
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    all_preds = []
    n_total = len(sentences)
    n_entities_found = 0
    t_start = time.time()

    print(f"  Running GLiNER2 on {n_total} sentences "
          f"(threshold={threshold}) ...")

    for sent_idx, sent in enumerate(sentences):
        tokens = sent["tokens"]
        text, offsets = tokens_to_text_with_offsets(tokens)

        # Skip empty sentences
        if not text.strip():
            all_preds.append([{"tag": "O", "confidence": GLINER2_DEFAULT_O_CONF}
                              for _ in tokens])
            continue

        try:
            result = model.extract_entities(
                text,
                GLINER2_ENTITY_LABELS,
                threshold=threshold,
                include_confidence=True,
                include_spans=True,
            )
        except Exception as e:
            # Log error and produce all-O predictions for this sentence
            print(f"  ERROR on sentence {sent_idx}: {e}")
            all_preds.append([{"tag": "O", "confidence": GLINER2_DEFAULT_O_CONF}
                              for _ in tokens])
            continue

        preds = gliner2_result_to_iob2(result, tokens, offsets)
        all_preds.append(preds)

        # Count entities for progress reporting
        for label_ents in result.get("entities", {}).values():
            n_entities_found += len(label_ents)

        # Progress reporting every 500 sentences
        if (sent_idx + 1) % 500 == 0:
            elapsed = time.time() - t_start
            rate = (sent_idx + 1) / elapsed
            eta = (n_total - sent_idx - 1) / rate if rate > 0 else 0
            print(f"    {sent_idx + 1}/{n_total} "
                  f"({rate:.1f} sent/s, ETA {eta:.0f}s, "
                  f"{n_entities_found} entities found)")

    elapsed = time.time() - t_start
    print(f"  GLiNER2 complete: {n_total} sentences in {elapsed:.1f}s "
          f"({n_total/elapsed:.1f} sent/s), "
          f"{n_entities_found} entities found")

    return all_preds


# =====================================================================
# Checkpoint Save/Load
# =====================================================================

def save_checkpoint(preds: List[List[Dict]], path: Path) -> None:
    """Save predictions to JSON checkpoint."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Saved checkpoint: {path.name} ({size_mb:.1f} MB)")


def load_checkpoint(path: Path) -> List[List[Dict]]:
    """Load predictions from JSON checkpoint."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================================
# Validation
# =====================================================================

def validate_predictions(
    preds: List[List[Dict]],
    sentences: List[Dict],
    model_name: str = "gliner2",
) -> bool:
    """Validate prediction format and alignment with ground truth.

    Checks:
        1. Same number of sentences
        2. Same number of tokens per sentence
        3. Each prediction has exactly {"tag", "confidence"} keys
        4. Tags are valid IOB2 tags
        5. Confidence is a float in [0, 1]
    """
    valid_tags = {"O"} | {f"{p}-{e}" for e in ENTITY_TYPES for p in ["B", "I"]}

    if len(preds) != len(sentences):
        print(f"  FAIL: {model_name} has {len(preds)} sentences, "
              f"expected {len(sentences)}")
        return False

    errors = 0
    for i, (pred_sent, gt_sent) in enumerate(zip(preds, sentences)):
        if len(pred_sent) != len(gt_sent["tokens"]):
            print(f"  FAIL: sentence {i}: {len(pred_sent)} predictions, "
                  f"{len(gt_sent['tokens'])} tokens")
            errors += 1
            if errors > 10:
                print("  ... (truncated)")
                break
            continue

        for j, pred in enumerate(pred_sent):
            keys = set(pred.keys())
            if keys != {"tag", "confidence"}:
                print(f"  FAIL: sentence {i}, token {j}: "
                      f"keys={keys}, expected {{'tag', 'confidence'}}")
                errors += 1
                break

            if pred["tag"] not in valid_tags:
                print(f"  FAIL: sentence {i}, token {j}: "
                      f"invalid tag '{pred['tag']}'")
                errors += 1
                break

            if not (0.0 <= pred["confidence"] <= 1.0):
                print(f"  FAIL: sentence {i}, token {j}: "
                      f"confidence={pred['confidence']} out of [0, 1]")
                errors += 1
                break

    if errors == 0:
        print(f"  PASS: {model_name} predictions validated "
              f"({len(preds)} sentences)")
        return True
    else:
        print(f"  FAIL: {errors} validation errors")
        return False


# =====================================================================
# Quick Stats
# =====================================================================

def print_prediction_stats(
    preds: List[List[Dict]],
    sentences: List[Dict],
    split_name: str,
) -> Dict[str, Any]:
    """Print summary statistics for GLiNER2 predictions."""
    from collections import Counter

    tag_counts = Counter()
    confs = []
    entity_confs = []
    n_tokens = 0

    for pred_sent in preds:
        for pred in pred_sent:
            tag_counts[pred["tag"]] += 1
            confs.append(pred["confidence"])
            if pred["tag"] != "O":
                entity_confs.append(pred["confidence"])
            n_tokens += 1

    # Compare with ground truth
    gt_entity_count = 0
    pred_entity_count = 0
    for gt_sent, pred_sent in zip(sentences, preds):
        for tag in gt_sent["ner_tags"]:
            if tag.startswith("B-"):
                gt_entity_count += 1
        for pred in pred_sent:
            if pred["tag"].startswith("B-"):
                pred_entity_count += 1

    import numpy as np
    confs_arr = np.array(confs)
    ent_confs_arr = np.array(entity_confs) if entity_confs else np.array([0.0])

    stats = {
        "split": split_name,
        "n_sentences": len(preds),
        "n_tokens": n_tokens,
        "tag_distribution": dict(tag_counts),
        "gt_entity_count": gt_entity_count,
        "pred_entity_count": pred_entity_count,
        "entity_ratio": pred_entity_count / gt_entity_count if gt_entity_count > 0 else 0,
        "confidence_mean": float(np.mean(confs_arr)),
        "confidence_std": float(np.std(confs_arr)),
        "entity_confidence_mean": float(np.mean(ent_confs_arr)),
        "entity_confidence_min": float(np.min(ent_confs_arr)),
        "entity_confidence_max": float(np.max(ent_confs_arr)),
        "entity_confidence_std": float(np.std(ent_confs_arr)),
    }

    print(f"\n  === GLiNER2 Prediction Stats ({split_name}) ===")
    print(f"    Sentences: {stats['n_sentences']}")
    print(f"    Tokens:    {stats['n_tokens']}")
    print(f"    GT entities (B- tags):   {stats['gt_entity_count']}")
    print(f"    Pred entities (B- tags): {stats['pred_entity_count']} "
          f"(ratio: {stats['entity_ratio']:.2f})")
    print(f"    Tag distribution:")
    for tag in sorted(tag_counts.keys()):
        print(f"      {tag:8s}: {tag_counts[tag]:6d}")
    print(f"    Entity confidence: "
          f"mean={stats['entity_confidence_mean']:.6f}, "
          f"std={stats['entity_confidence_std']:.6f}, "
          f"min={stats['entity_confidence_min']:.6f}, "
          f"max={stats['entity_confidence_max']:.6f}")

    return stats


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 60)
    print("EN1.1c — GLiNER2 Runner")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # ── Phase 1: Load Data ──────────────────────────────────────
    print("\nPhase 1: Loading CoNLL-2003 ...")
    dev_sentences = load_conll2003("validation")
    test_sentences = load_conll2003("test")

    print(f"  Dev:  {len(dev_sentences)} sentences")
    print(f"  Test: {len(test_sentences)} sentences")

    # ── Phase 2: Run GLiNER2 on Dev ─────────────────────────────
    print("\nPhase 2a: Running GLiNER2 on dev set ...")
    dev_preds = run_gliner2(dev_sentences)

    print("\n  Validating dev predictions ...")
    if not validate_predictions(dev_preds, dev_sentences):
        print("FATAL: Dev prediction validation failed.")
        sys.exit(1)

    dev_stats = print_prediction_stats(dev_preds, dev_sentences, "dev")

    # Save dev checkpoint
    dev_ckpt_path = CHECKPOINT_DIR / "dev_preds_gliner2.json"
    save_checkpoint(dev_preds, dev_ckpt_path)

    # ── Phase 3: Run GLiNER2 on Test ────────────────────────────
    print("\nPhase 2b: Running GLiNER2 on test set ...")
    test_preds = run_gliner2(test_sentences)

    print("\n  Validating test predictions ...")
    if not validate_predictions(test_preds, test_sentences):
        print("FATAL: Test prediction validation failed.")
        sys.exit(1)

    test_stats = print_prediction_stats(test_preds, test_sentences, "test")

    # Save test checkpoint
    test_ckpt_path = CHECKPOINT_DIR / "test_preds_gliner2.json"
    save_checkpoint(test_preds, test_ckpt_path)

    # ── Phase 4: Summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GLiNER2 Runner Complete")
    print(f"  Model:     {GLINER2_MODEL_NAME}")
    print(f"  Threshold: {GLINER2_THRESHOLD}")
    print(f"  Dev:       {len(dev_sentences)} sentences -> {dev_ckpt_path.name}")
    print(f"  Test:      {len(test_sentences)} sentences -> {test_ckpt_path.name}")
    print(f"  Dev entity ratio:  {dev_stats['entity_ratio']:.2f} "
          f"(pred/gt B- tags)")
    print(f"  Test entity ratio: {test_stats['entity_ratio']:.2f} "
          f"(pred/gt B- tags)")
    print("\nCheckpoints ready for en1_1c_5model_fusion.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
