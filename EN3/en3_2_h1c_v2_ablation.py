"""EN3.2-H1c v2 Ablation — Per-Difficulty, Model Subset, Param Sweep, Precision-at-Coverage.

Ablation analyses for the per-passage multi-model fusion experiment.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex

    python experiments/EN3/en3_2_h1c_v2_ablation.py
    python experiments/EN3/en3_2_h1c_v2_ablation.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

RESULTS_DIR = SCRIPT_DIR / "results"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from infra.config import set_global_seed
from infra.env_log import log_environment
from infra.results import ExperimentResult
from infra.stats import bootstrap_ci

from en3_2_h1c_v2_core import (
    evaluate_all_strategies,
    STRATEGY_NAMES,
)
from en3_2_h1c_v2_ablation_core import (
    classify_question_difficulty_h1c,
    evaluate_model_subset,
    evaluate_param_combo,
    compute_precision_at_coverage,
    compute_per_difficulty_breakdown,
    compute_mcnemar_contingency,
    MODEL_SUBSETS,
    _SL_STRATEGY_NAMES,
)

GLOBAL_SEED = 42
N_BOOTSTRAP = 1000
POISON_RATES = [0.05, 0.10, 0.20, 0.30]

MODEL_TAGS = {
    "distilbert": "v1b_qa_extraction",
    "roberta": "multimodel_qa_roberta",
    "electra": "multimodel_qa_electra",
    "bert_tiny": "multimodel_qa_bert_tiny",
}

MODEL_TRUST = {
    "distilbert": 0.78,
    "roberta": 0.85,
    "electra": 0.81,
    "bert_tiny": 0.24,
}

EVIDENCE_WEIGHTS = [5, 10, 20, 50]
PRIOR_WEIGHTS = [1, 2, 5, 10]


def _pr_tag(pr: float) -> str:
    return f"pr{int(pr * 100):02d}"


def normalize_answer(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def exact_match(pred: str, golds: List[str]) -> bool:
    np_ = normalize_answer(pred)
    return any(np_ == normalize_answer(g) for g in golds)


def load_all_data(pr_tag: str):
    """Load questions, retrieval (with is_poison flags), and model extractions."""
    with open(str(CHECKPOINT_DIR / f"v1b_questions_{pr_tag}.json")) as f:
        questions = json.load(f)
    with open(str(CHECKPOINT_DIR / f"v1b_retrieval_{pr_tag}.json")) as f:
        retrieval = json.load(f)

    model_extractions = {}
    for model_tag, prefix in MODEL_TAGS.items():
        path = CHECKPOINT_DIR / f"{prefix}_{pr_tag}.json"
        with open(str(path)) as f:
            model_extractions[model_tag] = json.load(f)

    return questions, retrieval, model_extractions


def build_question_passages(
    qid: str,
    retrieved: List[Dict],
    model_extractions: Dict[str, Dict],
) -> List[Dict[str, Any]]:
    """Build the question_passages structure for evaluate_all_strategies."""
    passages = []
    for p in retrieved:
        pid = p["passage_id"]
        cosine = p["score"]
        extractions = {}
        for model_tag, ext in model_extractions.items():
            if qid in ext and pid in ext[qid]:
                extractions[model_tag] = ext[qid][pid]
        passages.append({
            "cosine": cosine,
            "passage_id": pid,
            "extractions": extractions,
        })
    return passages


def run_ablation(
    poison_rates: List[float],
    max_questions: Optional[int] = None,
    n_bootstrap: int = 1000,
    verbose: bool = True,
) -> Dict[str, Any]:
    all_results = {}

    for pr in poison_rates:
        tag = _pr_tag(pr)
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Poison rate: {pr} ({tag})")
            print(f"{'=' * 60}")

        t0 = time.time()
        questions, retrieval, model_ext = load_all_data(tag)
        r_lookup = {r["question_id"]: r["retrieved"] for r in retrieval}

        # Build question lookup
        q_lookup = {q["id"]: q for q in questions}

        if max_questions:
            questions = questions[:max_questions]

        if verbose:
            print(f"  Questions: {len(questions)}")

        # ── Phase 1: Per-question evaluation ──
        per_question_records = []

        for q in questions:
            qid = q["id"]
            retrieved = r_lookup.get(qid, [])
            if not retrieved:
                continue

            # Classify difficulty
            gold_pid = q["gold_passage_id"]
            poison_pids = {
                p["passage_id"] for p in retrieved if p.get("is_poison", False)
            }
            difficulty = classify_question_difficulty_h1c(
                gold_pid, retrieved, poison_pids,
            )

            # Build passages
            q_passages = build_question_passages(qid, retrieved, model_ext)

            # Evaluate all strategies (default params)
            results = evaluate_all_strategies(
                q_passages, model_trust=MODEL_TRUST,
            )

            # Correctness per strategy
            strategy_correct = {}
            strategy_confidence = {}
            for name, res in results.items():
                strategy_correct[name] = exact_match(res["answer"], q["answers"])
                strategy_confidence[name] = res.get("confidence", 0.0)

            # Model subset evaluation
            subset_correct = {}
            for subset_name, models in MODEL_SUBSETS.items():
                if subset_name == "all_4":
                    continue  # already computed above
                sub_results = evaluate_model_subset(
                    q_passages, models, MODEL_TRUST,
                )
                subset_correct[subset_name] = {}
                for name, res in sub_results.items():
                    subset_correct[subset_name][name] = exact_match(
                        res["answer"], q["answers"],
                    )

            per_question_records.append({
                "question_id": qid,
                "difficulty": difficulty,
                "strategies": strategy_correct,
                "confidences": strategy_confidence,
                "subsets": subset_correct,
            })

        if verbose:
            diffs = [r["difficulty"] for r in per_question_records]
            print(f"  Difficulty: easy={diffs.count('easy')}, "
                  f"medium={diffs.count('medium')}, hard={diffs.count('hard')}")

        # ── Phase 2: Per-difficulty breakdown ──
        difficulty_breakdown = compute_per_difficulty_breakdown(per_question_records)

        if verbose:
            print(f"\n  Per-difficulty EM (key strategies):")
            for diff in ["easy", "medium", "hard"]:
                n = difficulty_breakdown[diff].get("sl_trust_discount", {}).get("n", 0)
                for s in ["sl_trust_discount", "scalar_qa_weighted", "single_roberta"]:
                    info = difficulty_breakdown[diff].get(s, {})
                    acc = info.get("accuracy", 0.0)
                    print(f"    {diff:>8} {s:<25} {acc:.3f} (n={info.get('n', 0)})")

        # ── Phase 3: Model subset ablation ──
        subset_breakdown = {}
        for subset_name in MODEL_SUBSETS:
            if subset_name == "all_4":
                # Use default strategies
                strat_correct_lists = {}
                for s in STRATEGY_NAMES:
                    strat_correct_lists[s] = [
                        r["strategies"][s] for r in per_question_records
                        if s in r["strategies"]
                    ]
            else:
                strat_correct_lists = {}
                for s in STRATEGY_NAMES:
                    strat_correct_lists[s] = [
                        r["subsets"].get(subset_name, {}).get(s, False)
                        for r in per_question_records
                        if subset_name in r.get("subsets", {})
                    ]

            subset_breakdown[subset_name] = {}
            for s, correct_list in strat_correct_lists.items():
                if not correct_list:
                    continue
                n_correct = sum(correct_list)
                acc = n_correct / len(correct_list)
                ci_lo, _, ci_hi = bootstrap_ci(
                    [float(c) for c in correct_list],
                    n_bootstrap=n_bootstrap, seed=GLOBAL_SEED,
                )
                subset_breakdown[subset_name][s] = {
                    "accuracy": round(acc, 6),
                    "n": len(correct_list),
                    "ci_lower": round(ci_lo, 6),
                    "ci_upper": round(ci_hi, 6),
                }

        if verbose:
            print(f"\n  Model subset ablation (sl_trust_discount EM):")
            for subset_name in MODEL_SUBSETS:
                info = subset_breakdown.get(subset_name, {}).get(
                    "sl_trust_discount", {})
                acc = info.get("accuracy", 0.0)
                models = MODEL_SUBSETS[subset_name]
                print(f"    {subset_name:<20} {acc:.3f} models={models}")

        # ── Phase 4: Parameter sweep ──
        param_sweep = {}
        for ew in EVIDENCE_WEIGHTS:
            for pw in PRIOR_WEIGHTS:
                combo_key = f"ew{ew}_pw{pw}"
                combo_results = {}

                for q in questions:
                    qid = q["id"]
                    retrieved_q = r_lookup.get(qid, [])
                    if not retrieved_q:
                        continue
                    q_passages = build_question_passages(
                        qid, retrieved_q, model_ext,
                    )
                    sl_results = evaluate_param_combo(
                        q_passages, MODEL_TRUST,
                        evidence_weight=float(ew), prior_weight=float(pw),
                    )
                    for s, res in sl_results.items():
                        if s not in combo_results:
                            combo_results[s] = []
                        combo_results[s].append(
                            exact_match(res["answer"], q["answers"])
                        )

                param_sweep[combo_key] = {}
                for s, correct_list in combo_results.items():
                    if not correct_list:
                        continue
                    acc = sum(correct_list) / len(correct_list)
                    param_sweep[combo_key][s] = round(acc, 6)

        if verbose:
            print(f"\n  Param sweep (sl_trust_discount EM, top 5):")
            td_scores = [
                (k, v.get("sl_trust_discount", 0.0))
                for k, v in param_sweep.items()
            ]
            td_scores.sort(key=lambda x: -x[1])
            for k, acc in td_scores[:5]:
                print(f"    {k:<15} {acc:.3f}")

        # ── Phase 5: McNemar contingency tables ──
        mcnemar_pairs = [
            ("sl_trust_discount", "scalar_qa_weighted"),
            ("sl_trust_discount", "sl_fusion"),
            ("sl_trust_discount", "single_roberta"),
            ("sl_fusion", "scalar_majority"),
            ("sl_trust_discount", "scalar_majority_x_qa"),
        ]
        mcnemar_results = {}
        for a, b in mcnemar_pairs:
            mcnemar_data = [
                {a: r["strategies"].get(a, False),
                 b: r["strategies"].get(b, False)}
                for r in per_question_records
            ]
            mcnemar_results[f"{a}_vs_{b}"] = compute_mcnemar_contingency(
                mcnemar_data, a, b,
            )

        if verbose:
            print(f"\n  McNemar contingency tables:")
            for pair_name, result in mcnemar_results.items():
                print(f"    {pair_name}: a_only={result['a_only']}, "
                      f"b_only={result['b_only']}, p={result['p_value']:.4f}")

        # ── Phase 6: Precision-at-coverage curves ──
        pac_curves = {}
        for s in STRATEGY_NAMES:
            per_q_data = [
                {
                    "confidence": r["confidences"].get(s, 0.0),
                    "correct": r["strategies"].get(s, False),
                }
                for r in per_question_records
                if s in r["strategies"]
            ]
            pac_curves[s] = compute_precision_at_coverage(per_q_data)

        # ── Assemble results ──
        all_results[tag] = {
            "n_questions": len(per_question_records),
            "difficulty_counts": {
                diff: sum(1 for r in per_question_records if r["difficulty"] == diff)
                for diff in ["easy", "medium", "hard"]
            },
            "difficulty_breakdown": difficulty_breakdown,
            "subset_breakdown": subset_breakdown,
            "param_sweep": param_sweep,
            "mcnemar": mcnemar_results,
            "precision_at_coverage": pac_curves,
            "per_question": [
                {
                    "question_id": r["question_id"],
                    "difficulty": r["difficulty"],
                    **{s: r["strategies"][s] for s in STRATEGY_NAMES if s in r["strategies"]},
                }
                for r in per_question_records
            ],
        }

        if verbose:
            print(f"\n  Elapsed: {time.time() - t0:.1f}s")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="EN3.2-H1c v2 Ablation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poison-rate", type=float, default=None)
    args = parser.parse_args()

    set_global_seed(GLOBAL_SEED)

    if args.dry_run:
        prs = [0.10]
        max_q = 10
        n_boot = 100
        print("=== DRY RUN ===\n")
    else:
        prs = [args.poison_rate] if args.poison_rate else POISON_RATES
        max_q = None
        n_boot = N_BOOTSTRAP

    t_start = time.time()
    env = log_environment()

    results = run_ablation(
        poison_rates=prs, max_questions=max_q,
        n_bootstrap=n_boot, verbose=True,
    )

    elapsed = time.time() - t_start
    mode = "dry" if args.dry_run else "full"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_name = f"en3_2_h1c_v2_ablation_{mode}_results"

    experiment_result = ExperimentResult(
        experiment_id="EN3.2-H1c-v2-ablation",
        parameters={
            "poison_rates": prs,
            "models": list(MODEL_TAGS.keys()),
            "model_trust": MODEL_TRUST,
            "model_subsets": {k: v for k, v in MODEL_SUBSETS.items()},
            "evidence_weights": EVIDENCE_WEIGHTS,
            "prior_weights": PRIOR_WEIGHTS,
            "n_bootstrap": n_boot,
            "max_questions": max_q,
            "global_seed": GLOBAL_SEED,
            "mode": mode,
        },
        metrics=results,
        environment=env,
        notes=(
            f"H1c v2 ablation: per-difficulty, model subsets ({len(MODEL_SUBSETS)}), "
            f"param sweep ({len(EVIDENCE_WEIGHTS)}x{len(PRIOR_WEIGHTS)}), "
            f"precision-at-coverage, McNemar. {elapsed:.1f}s."
        ),
    )

    primary = RESULTS_DIR / f"{result_name}.json"
    archive = RESULTS_DIR / f"{result_name}_{timestamp}.json"
    experiment_result.save_json(str(primary))
    experiment_result.save_json(str(archive))

    print(f"\n{'=' * 60}")
    print(f"DONE — {elapsed:.1f}s")
    print(f"Results saved to:")
    print(f"  {primary}")
    print(f"  {archive}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
