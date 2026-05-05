"""EN8.6 Phase 2 -- Evaluation using LLM-annotated ground truth.

Uses the conflict annotations from en8_6_annotate.py to:
    1. Report conflict taxonomy (granularity/factual/temporal/synonym/format)
    2. Compute merge accuracy on factual disagreements
    3. Test with correct vs incorrect confidence assignment
    4. Compare with Phase 1 calibration sensitivity predictions
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_EN8_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _EN8_DIR.parent
_PKG_SRC = _EXPERIMENTS_ROOT.parent / "packages" / "python" / "src"
for _p in [str(_EN8_DIR), str(_EXPERIMENTS_ROOT), str(_PKG_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from en8_6_core import (
    baseline_rdflib_union,
    baseline_random_choice,
    baseline_majority_vote,
    baseline_most_recent,
    baseline_rdflib_confidence_argmax,
    compute_ece,
)
from jsonld_ex.merge import merge_graphs
from infra.stats import bootstrap_ci

CACHE_DIR = _EN8_DIR / "data" / "phase2_cache"
RESULTS_DIR = _EN8_DIR / "results"


# ═══════════════════════════════════════════════════════════════════
# LOAD ANNOTATIONS
# ═══════════════════════════════════════════════════════════════════


def load_annotations() -> list[dict[str, Any]]:
    """Load LLM-annotated conflicts."""
    path = CACHE_DIR / "conflict_annotations.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# BUILD GRAPHS WITH CONFIGURABLE CONFIDENCE
# ═══════════════════════════════════════════════════════════════════


def build_factual_graphs(
    factual_conflicts: list[dict[str, Any]],
    dbpedia_conf: float,
    wikidata_conf: float,
) -> tuple[list[dict], dict[str, dict[str, str]], set[tuple[str, str]]]:
    """Build JSON-LD graphs from factual conflicts with given confidence.

    Returns:
        (graphs, ground_truth, conflict_set)
    """
    dbp_nodes: dict[str, dict] = {}
    wd_nodes: dict[str, dict] = {}
    gt: dict[str, dict[str, str]] = {}
    conflict_set: set[tuple[str, str]] = set()

    for c in factual_conflicts:
        correct = c.get("correct_source")
        if correct not in ("dbpedia", "wikidata"):
            continue  # skip unclear

        entity_id = f"urn:entity:{c['entity_qid']}"
        prop = c["property"]

        # DBpedia node
        if entity_id not in dbp_nodes:
            dbp_nodes[entity_id] = {"@id": entity_id, "@type": "Entity"}
        dbp_nodes[entity_id][prop] = {
            "@value": c["dbpedia"],
            "@confidence": dbpedia_conf,
            "@source": "dbpedia",
            "@extractedAt": "2025-01-01T00:00:00Z",
        }

        # Wikidata node
        if entity_id not in wd_nodes:
            wd_nodes[entity_id] = {"@id": entity_id, "@type": "Entity"}
        wd_nodes[entity_id][prop] = {
            "@value": c["wikidata"],
            "@confidence": wikidata_conf,
            "@source": "wikidata",
            "@extractedAt": "2025-06-01T00:00:00Z",
        }

        # Ground truth
        if entity_id not in gt:
            gt[entity_id] = {}
        gt[entity_id][prop] = c["dbpedia"] if correct == "dbpedia" else c["wikidata"]
        conflict_set.add((entity_id, prop))

    dbp_graph = {"@graph": list(dbp_nodes.values())}
    wd_graph = {"@graph": list(wd_nodes.values())}

    return [dbp_graph, wd_graph], gt, conflict_set


def evaluate_accuracy(
    merged: dict[str, Any],
    gt: dict[str, dict[str, str]],
    conflict_set: set[tuple[str, str]],
) -> tuple[float, int, int]:
    """Compute accuracy on conflicted properties.

    Returns (accuracy, n_correct, n_evaluated).
    """
    nodes = {n["@id"]: n for n in merged.get("@graph", [])}
    n_correct = 0
    n_evaluated = 0

    for eid, props in gt.items():
        node = nodes.get(eid)
        if node is None:
            continue
        for prop, gt_val in props.items():
            if (eid, prop) not in conflict_set:
                continue
            merged_val = node.get(prop)
            if merged_val is None:
                continue
            bare = merged_val.get("@value", merged_val) if isinstance(merged_val, dict) else merged_val
            n_evaluated += 1
            if bare == gt_val:
                n_correct += 1

    accuracy = n_correct / n_evaluated if n_evaluated > 0 else 0.0
    return accuracy, n_correct, n_evaluated


# ═══════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════


def run_evaluation() -> dict[str, Any]:
    """Run the full Phase 2 evaluation."""
    annotations = load_annotations()

    # ── Taxonomy ───────────────────────────────────────────────────
    type_counts = Counter(a["conflict_type"] for a in annotations)
    source_counts = Counter(a["correct_source"] for a in annotations)

    print("=" * 60)
    print("EN8.6 PHASE 2 EVALUATION")
    print("=" * 60)
    print(f"\nConflict Taxonomy ({len(annotations)} total):")
    for t, c in type_counts.most_common():
        print(f"  {t:15s}: {c:4d} ({c/len(annotations)*100:.1f}%)")

    # ── Factual conflicts only ─────────────────────────────────────
    factual = [a for a in annotations if a["conflict_type"] == "factual"]
    evaluable = [a for a in factual if a["correct_source"] in ("dbpedia", "wikidata")]
    n_dbp_correct = sum(1 for a in evaluable if a["correct_source"] == "dbpedia")
    n_wd_correct = sum(1 for a in evaluable if a["correct_source"] == "wikidata")

    print(f"\nFactual Disagreements: {len(factual)}")
    print(f"  Evaluable (clear ground truth): {len(evaluable)}")
    print(f"  DBpedia correct: {n_dbp_correct} ({n_dbp_correct/max(1,len(evaluable))*100:.1f}%)")
    print(f"  Wikidata correct: {n_wd_correct} ({n_wd_correct/max(1,len(evaluable))*100:.1f}%)")

    if len(evaluable) == 0:
        print("No evaluable factual conflicts. Aborting.")
        return {}

    # ── Scenario A: Naive confidence (Wikidata > DBpedia) ──────────
    print(f"\n--- Scenario A: Naive confidence (WD=0.75, DBp=0.65) ---")
    print(f"    (Assumes Wikidata is more reliable)")
    graphs_a, gt_a, cs_a = build_factual_graphs(evaluable, dbpedia_conf=0.65, wikidata_conf=0.75)

    merged_h_a, _ = merge_graphs(graphs_a, conflict_strategy="highest")
    merged_w_a, _ = merge_graphs(graphs_a, conflict_strategy="weighted_vote")
    b1_a = baseline_rdflib_union(graphs_a)
    b2_a = baseline_random_choice(graphs_a, seed=42)
    b3_a = baseline_majority_vote(graphs_a, seed=42)
    b4_a = baseline_most_recent(graphs_a)
    b5_a = baseline_rdflib_confidence_argmax(graphs_a)

    acc_h_a = evaluate_accuracy(merged_h_a, gt_a, cs_a)
    acc_w_a = evaluate_accuracy(merged_w_a, gt_a, cs_a)
    acc_b1_a = evaluate_accuracy(b1_a, gt_a, cs_a)
    acc_b2_a = evaluate_accuracy(b2_a, gt_a, cs_a)
    acc_b3_a = evaluate_accuracy(b3_a, gt_a, cs_a)
    acc_b4_a = evaluate_accuracy(b4_a, gt_a, cs_a)
    acc_b5_a = evaluate_accuracy(b5_a, gt_a, cs_a)

    print(f"    jsonld-ex highest:  {acc_h_a[0]:.3f} ({acc_h_a[1]}/{acc_h_a[2]})")
    print(f"    jsonld-ex weighted: {acc_w_a[0]:.3f}")
    print(f"    B5 rdflib+conf:     {acc_b5_a[0]:.3f}")
    print(f"    Majority vote:      {acc_b3_a[0]:.3f}")
    print(f"    Most recent:        {acc_b4_a[0]:.3f}")
    print(f"    rdflib union:       {acc_b1_a[0]:.3f}")
    print(f"    Random:             {acc_b2_a[0]:.3f}")

    # ── Scenario B: Correct confidence (DBpedia > Wikidata) ────────
    print(f"\n--- Scenario B: Correct confidence (DBp=0.75, WD=0.65) ---")
    print(f"    (Reflects empirical finding: DBpedia more often correct)")
    graphs_b, gt_b, cs_b = build_factual_graphs(evaluable, dbpedia_conf=0.75, wikidata_conf=0.65)

    merged_h_b, _ = merge_graphs(graphs_b, conflict_strategy="highest")
    merged_w_b, _ = merge_graphs(graphs_b, conflict_strategy="weighted_vote")
    b5_b = baseline_rdflib_confidence_argmax(graphs_b)

    acc_h_b = evaluate_accuracy(merged_h_b, gt_b, cs_b)
    acc_w_b = evaluate_accuracy(merged_w_b, gt_b, cs_b)
    acc_b5_b = evaluate_accuracy(b5_b, gt_b, cs_b)

    print(f"    jsonld-ex highest:  {acc_h_b[0]:.3f} ({acc_h_b[1]}/{acc_h_b[2]})")
    print(f"    jsonld-ex weighted: {acc_w_b[0]:.3f}")
    print(f"    B5 rdflib+conf:     {acc_b5_b[0]:.3f}")

    # ── Scenario C: Equal confidence (no prior) ────────────────────
    print(f"\n--- Scenario C: Equal confidence (both=0.70) ---")
    print(f"    (No assumption about which source is better)")
    graphs_c, gt_c, cs_c = build_factual_graphs(evaluable, dbpedia_conf=0.70, wikidata_conf=0.70)

    merged_h_c, _ = merge_graphs(graphs_c, conflict_strategy="highest")
    b5_c = baseline_rdflib_confidence_argmax(graphs_c)

    acc_h_c = evaluate_accuracy(merged_h_c, gt_c, cs_c)
    acc_b5_c = evaluate_accuracy(b5_c, gt_c, cs_c)

    print(f"    jsonld-ex highest:  {acc_h_c[0]:.3f} ({acc_h_c[1]}/{acc_h_c[2]})")
    print(f"    B5 rdflib+conf:     {acc_b5_c[0]:.3f}")
    print(f"    (With equal confidence, result depends on internal tiebreaking)")

    # ── Phase 1 transfer analysis ──────────────────────────────────
    print(f"\n--- Phase 1 -> Phase 2 Transfer ---")
    # DBpedia correct 59.5% of the time -> confidence-correctness correlation
    # If we assign Wikidata higher conf, correlation is NEGATIVE (adversarial)
    # If we assign DBpedia higher conf, correlation is POSITIVE
    dbp_correct_rate = n_dbp_correct / len(evaluable)
    print(f"    DBpedia correct rate: {dbp_correct_rate:.1%}")
    print(f"    Scenario A (naive WD>DBp): confidence anti-correlated with correctness")
    print(f"      -> Phase 1 predicts: NEGATIVE delta (adversarial regime)")
    print(f"      -> Actual delta (highest - majority): {acc_h_a[0] - acc_b3_a[0]:+.3f}")
    print(f"    Scenario B (correct DBp>WD): confidence correlated with correctness")
    print(f"      -> Phase 1 predicts: POSITIVE delta (noisy regime)")
    print(f"      -> Actual delta (highest - majority): {acc_h_b[0] - acc_b3_a[0]:+.3f}")

    # ── Save results ───────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = {
        "experiment_id": "EN8.6",
        "phase": "real_world_phase2_evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conflict_taxonomy": {t: c for t, c in type_counts.most_common()},
        "n_total_conflicts": len(annotations),
        "n_factual": len(factual),
        "n_evaluable": len(evaluable),
        "n_dbpedia_correct": n_dbp_correct,
        "n_wikidata_correct": n_wd_correct,
        "dbpedia_correct_rate": dbp_correct_rate,
        "scenario_a_naive": {
            "description": "Wikidata conf=0.75, DBpedia conf=0.65",
            "highest_accuracy": acc_h_a[0],
            "weighted_vote_accuracy": acc_w_a[0],
            "b5_accuracy": acc_b5_a[0],
            "majority_vote_accuracy": acc_b3_a[0],
            "most_recent_accuracy": acc_b4_a[0],
            "rdflib_union_accuracy": acc_b1_a[0],
            "random_accuracy": acc_b2_a[0],
            "delta_highest_vs_majority": acc_h_a[0] - acc_b3_a[0],
        },
        "scenario_b_correct": {
            "description": "DBpedia conf=0.75, Wikidata conf=0.65",
            "highest_accuracy": acc_h_b[0],
            "weighted_vote_accuracy": acc_w_b[0],
            "b5_accuracy": acc_b5_b[0],
            "delta_highest_vs_majority": acc_h_b[0] - acc_b3_a[0],
        },
        "scenario_c_equal": {
            "description": "Both conf=0.70",
            "highest_accuracy": acc_h_c[0],
            "b5_accuracy": acc_b5_c[0],
        },
        "phase1_transfer": {
            "naive_delta_predicted": "negative (adversarial regime)",
            "naive_delta_actual": acc_h_a[0] - acc_b3_a[0],
            "correct_delta_predicted": "positive (noisy regime)",
            "correct_delta_actual": acc_h_b[0] - acc_b3_a[0],
        },
    }

    primary = RESULTS_DIR / "en8_6_phase2_eval.json"
    with open(primary, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {primary}")

    archive = RESULTS_DIR / f"en8_6_phase2_eval_{timestamp}.json"
    with open(archive, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved: {archive}")

    return output


if __name__ == "__main__":
    run_evaluation()
