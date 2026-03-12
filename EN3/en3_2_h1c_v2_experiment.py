"""EN3.2-H1c v2 — Per-Passage Multi-Model Fusion Experiment.

Two-level architecture: fuse 4 QA models within each passage,
then rank passages to select the final answer.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex

    python experiments/EN3/en3_2_h1c_v2_experiment.py
    python experiments/EN3/en3_2_h1c_v2_experiment.py --dry-run
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
from typing import Any, Dict, List

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
    """Load questions, retrieval, and all model extractions."""
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


def run_experiment(
    poison_rates: List[float],
    max_questions: int | None = None,
    n_bootstrap: int = 1000,
    verbose: bool = True,
) -> Dict[str, Any]:
    all_results = {}

    for pr in poison_rates:
        tag = _pr_tag(pr)
        if verbose:
            print(f"\n{'='*60}")
            print(f"Poison rate: {pr} ({tag})")
            print(f"{'='*60}")

        t0 = time.time()
        questions, retrieval, model_ext = load_all_data(tag)
        r_lookup = {r["question_id"]: r["retrieved"] for r in retrieval}

        if max_questions:
            questions = questions[:max_questions]

        if verbose:
            print(f"  Questions: {len(questions)}, Models: {list(MODEL_TAGS.keys())}")

        # Evaluate all questions
        per_question = []
        for q in questions:
            qid = q["id"]
            retrieved = r_lookup.get(qid, [])
            if not retrieved:
                continue

            q_passages = build_question_passages(qid, retrieved, model_ext)
            results = evaluate_all_strategies(
                q_passages, model_trust=MODEL_TRUST,
            )

            entry = {"question_id": qid}
            for name, res in results.items():
                entry[name] = {
                    "answer": res["answer"],
                    "correct": exact_match(res["answer"], q["answers"]),
                    "confidence": res.get("confidence", 0.0),
                }
            per_question.append(entry)

        # Aggregate
        pr_results = {
            "n_questions": len(per_question),
            "strategies": {},
        }

        for strategy in STRATEGY_NAMES:
            correct_list = [
                pq[strategy]["correct"]
                for pq in per_question
                if strategy in pq
            ]
            if not correct_list:
                continue

            n_correct = sum(correct_list)
            accuracy = n_correct / len(correct_list)
            ci_lo, _, ci_hi = bootstrap_ci(
                [float(c) for c in correct_list],
                n_bootstrap=n_bootstrap, seed=GLOBAL_SEED,
            )

            pr_results["strategies"][strategy] = {
                "n_evaluated": len(correct_list),
                "n_correct": n_correct,
                "accuracy": round(accuracy, 6),
                "ci_lower": round(ci_lo, 6),
                "ci_upper": round(ci_hi, 6),
            }

        # Per-question data for ablation
        pr_results["per_question"] = [
            {
                "question_id": pq["question_id"],
                **{
                    s: pq[s]["correct"]
                    for s in STRATEGY_NAMES
                    if s in pq
                },
            }
            for pq in per_question
        ]

        # Summary
        if verbose:
            print(f"\n  {'─'*55}")
            print(f"  {'Strategy':<25} {'EM':>7} {'95% CI':>18}")
            print(f"  {'─'*55}")
            sorted_strats = sorted(
                pr_results["strategies"].items(),
                key=lambda x: -x[1]["accuracy"],
            )
            for name, info in sorted_strats:
                print(f"  {name:<25} {info['accuracy']:>7.3f} "
                      f"[{info['ci_lower']:.3f}, {info['ci_upper']:.3f}]")
            print(f"  {'─'*55}")
            print(f"  (GPT-4o-mini PLAIN: ~0.628-0.644)")
            print(f"  Elapsed: {time.time()-t0:.1f}s")

        all_results[tag] = pr_results

    return all_results


def main():
    parser = argparse.ArgumentParser(description="EN3.2-H1c v2 Per-Passage Fusion")
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

    results = run_experiment(
        poison_rates=prs, max_questions=max_q,
        n_bootstrap=n_boot, verbose=True,
    )

    elapsed = time.time() - t_start
    mode = "dry" if args.dry_run else "full"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_name = f"en3_2_h1c_v2_{mode}_results"

    experiment_result = ExperimentResult(
        experiment_id="EN3.2-H1c-v2",
        parameters={
            "poison_rates": prs,
            "models": list(MODEL_TAGS.keys()),
            "model_trust": MODEL_TRUST,
            "n_bootstrap": n_boot,
            "max_questions": max_q,
            "global_seed": GLOBAL_SEED,
            "mode": mode,
        },
        metrics=results,
        environment=env,
        notes=(
            f"Per-passage multi-model fusion (two-level architecture). "
            f"4 QA models, 5 scalar + 4 SL strategies. {elapsed:.1f}s."
        ),
    )

    primary = RESULTS_DIR / f"{result_name}.json"
    archive = RESULTS_DIR / f"{result_name}_{timestamp}.json"
    experiment_result.save_json(str(primary))
    experiment_result.save_json(str(archive))

    print(f"\n{'='*60}")
    print(f"DONE — {elapsed:.1f}s")
    print(f"Results saved to:")
    print(f"  {primary}")
    print(f"  {archive}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
