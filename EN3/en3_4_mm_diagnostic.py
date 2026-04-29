"""EN3.4 MedMentions Diagnostic — understand why F1 is so low.

Run: python experiments/EN3/en3_4_mm_diagnostic.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import Counter

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _SCRIPT_DIR.parent
for p in [str(_SCRIPT_DIR), str(_EXPERIMENTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from EN3.run_en3_4 import (
    load_medmentions, _serializable_to_preds, _load_checkpoint,
    MEDMENTIONS_TYPE_TO_GROUP,
)
from EN3.en3_4_core import EntitySpan, evaluate_entities, align_spans

CHECKPOINT_DIR = _SCRIPT_DIR / "checkpoints"


def main():
    print("=" * 70)
    print("  MedMentions Diagnostic")
    print("=" * 70)

    # ── 1. Check raw tag format ──
    print("\n── 1. Raw Tag Format ──")
    from datasets import load_dataset
    ds = load_dataset("ibm-research/MedMentions-ZS", split="test")
    sample = ds[0]
    tags = sample["ner_tags"]
    print(f"  Type of ner_tags[0]: {type(tags[0])}")
    print(f"  First 20 tags: {tags[:20]}")
    print(f"  Tokens: {sample['tokens'][:20]}")

    # If integer, check the feature mapping
    features = ds.features
    print(f"  Features: {features}")
    if hasattr(features["ner_tags"], "feature"):
        feat = features["ner_tags"].feature
        print(f"  Tag feature type: {type(feat)}")
        if hasattr(feat, "names"):
            print(f"  Tag names: {feat.names[:30]}")
            print(f"  N tag classes: {len(feat.names)}")

    # ── 2. Count gold entities with our parser ──
    print("\n── 2. Gold Entity Counts ──")
    test_data = load_medmentions("test")
    dev_data = load_medmentions("validation")

    test_golds = [s["gold_spans"] for s in test_data]
    all_test_gold = [g for sent in test_golds for g in sent]
    print(f"  Test sentences: {len(test_data)}")
    print(f"  Test gold entities (our parser): {len(all_test_gold)}")
    print(f"  Entities/sentence: {len(all_test_gold)/len(test_data):.1f}")

    # Count by type
    type_counts = Counter(g.entity_type for g in all_test_gold)
    print(f"  By type:")
    for t, c in type_counts.most_common():
        print(f"    {t}: {c}")

    # ── 3. What do raw tags actually contain? ──
    print("\n── 3. Raw Tag Distribution ──")
    all_tags = []
    for item in ds:
        all_tags.extend(item["ner_tags"])
    tag_dist = Counter(all_tags)
    print(f"  Total tokens: {len(all_tags)}")
    print(f"  Unique tags: {len(tag_dist)}")
    for tag, cnt in tag_dist.most_common(25):
        print(f"    {tag!r}: {cnt}")

    # ── 4. Recount gold entities directly from raw data ──
    print("\n── 4. Direct Gold Entity Count (from raw BIO) ──")
    n_entities_raw = 0
    for item in ds:
        prev_was_entity = False
        for tag in item["ner_tags"]:
            tag_str = str(tag) if not isinstance(tag, int) else ""
            if isinstance(tag, int):
                # Check if the feature has names
                if hasattr(features["ner_tags"], "feature") and hasattr(features["ner_tags"].feature, "int2str"):
                    tag_str = features["ner_tags"].feature.int2str(tag)
                else:
                    tag_str = str(tag)
            if tag_str.startswith("B-"):
                n_entities_raw += 1
    print(f"  Raw B- tag count (entities): {n_entities_raw}")

    # ── 5. Load predictions and analyze FP types ──
    print("\n── 5. Prediction Analysis ──")
    bm_test = _serializable_to_preds(_load_checkpoint("en3_4_mm_test_biomed"))
    g2_test = _serializable_to_preds(_load_checkpoint("en3_4_mm_test_gliner2"))

    # BioMed predictions
    bm_flat = [p for sent in bm_test for p in sent]
    print(f"  BioMed predictions: {len(bm_flat)}")
    bm_types = Counter(p.entity_type for p in bm_flat)
    print(f"  BioMed by type:")
    for t, c in bm_types.most_common():
        print(f"    {t}: {c}")

    # GLiNER2 predictions
    g2_flat = [p for sent in g2_test for p in sent]
    print(f"  GLiNER2 predictions: {len(g2_flat)}")
    g2_types = Counter(p.entity_type for p in g2_flat)
    print(f"  GLiNER2 by type:")
    for t, c in g2_types.most_common():
        print(f"    {t}: {c}")

    # ── 6. Per-type evaluation ──
    print("\n── 6. Per-Type Evaluation (BioMed, threshold=0.5) ──")
    for etype in sorted(type_counts.keys()):
        type_golds = [g for g in all_test_gold if g.entity_type == etype]
        type_preds = [p for p in bm_flat if p.entity_type == etype and p.score >= 0.5]
        m = evaluate_entities(type_preds, type_golds)
        print(f"  {etype:12s}: Gold={len(type_golds):5d}, Pred={len(type_preds):5d}, "
              f"TP={m.tp:4d}, FP={m.fp:4d}, FN={m.fn:4d}, "
              f"P={m.precision:.3f}, R={m.recall:.3f}, F1={m.f1:.3f}")

    # ── 7. Span boundary analysis ──
    print("\n── 7. Span Boundary Analysis (first 100 sentences) ──")
    n_exact = 0
    n_partial = 0
    n_miss = 0
    for i in range(min(100, len(test_data))):
        sent_golds = test_data[i]["gold_spans"]
        sent_preds = [p for p in bm_test[i] if p.score >= 0.5]
        for gold in sent_golds:
            found_exact = False
            found_partial = False
            for pred in sent_preds:
                if (pred.start == gold.start and pred.end == gold.end
                        and pred.entity_type == gold.entity_type):
                    found_exact = True
                    break
                # Check partial overlap (same type, overlapping span)
                overlap = max(0, min(pred.end, gold.end) - max(pred.start, gold.start))
                if overlap > 0 and pred.entity_type == gold.entity_type:
                    found_partial = True
            if found_exact:
                n_exact += 1
            elif found_partial:
                n_partial += 1
            else:
                n_miss += 1
    total = n_exact + n_partial + n_miss
    print(f"  Exact matches: {n_exact} ({100*n_exact/total:.1f}%)")
    print(f"  Partial overlap (right type, wrong boundary): {n_partial} ({100*n_partial/total:.1f}%)")
    print(f"  Complete miss: {n_miss} ({100*n_miss/total:.1f}%)")

    # ── 8. Sample mismatches ──
    print("\n── 8. Sample Mismatches (first 5 partial overlaps) ──")
    count = 0
    for i in range(min(200, len(test_data))):
        if count >= 5:
            break
        sent_golds = test_data[i]["gold_spans"]
        sent_preds = [p for p in bm_test[i] if p.score >= 0.5]
        text = test_data[i]["text"]
        for gold in sent_golds:
            for pred in sent_preds:
                overlap = max(0, min(pred.end, gold.end) - max(pred.start, gold.start))
                if (overlap > 0 and pred.entity_type == gold.entity_type
                        and not (pred.start == gold.start and pred.end == gold.end)):
                    gold_text = text[gold.start:gold.end]
                    pred_text = text[pred.start:pred.end]
                    print(f"  Gold: [{gold.start}:{gold.end}] '{gold_text}' ({gold.entity_type})")
                    print(f"  Pred: [{pred.start}:{pred.end}] '{pred_text}' ({pred.entity_type}, score={pred.score:.3f})")
                    print()
                    count += 1
                    if count >= 5:
                        break
            if count >= 5:
                break

    # ── 9. Relaxed evaluation (type-agnostic) ──
    print("\n── 9. Type-Agnostic Evaluation (BioMed, t=0.5) ──")
    # What if we ignore entity types entirely?
    type_agnostic_golds = [EntitySpan(g.start, g.end, "ENTITY", g.score, g.source)
                           for g in all_test_gold]
    type_agnostic_preds = [EntitySpan(p.start, p.end, "ENTITY", p.score, p.source)
                           for p in bm_flat if p.score >= 0.5]
    m_agnostic = evaluate_entities(type_agnostic_preds, type_agnostic_golds)
    print(f"  Type-agnostic: P={m_agnostic.precision:.3f}, R={m_agnostic.recall:.3f}, "
          f"F1={m_agnostic.f1:.3f}")
    print(f"  TP={m_agnostic.tp}, FP={m_agnostic.fp}, FN={m_agnostic.fn}")

    # Compare to typed
    m_typed = evaluate_entities(
        [p for p in bm_flat if p.score >= 0.5], all_test_gold)
    print(f"  Typed:         P={m_typed.precision:.3f}, R={m_typed.recall:.3f}, "
          f"F1={m_typed.f1:.3f}")
    print(f"  TP gained by ignoring types: {m_agnostic.tp - m_typed.tp}")


if __name__ == "__main__":
    main()
