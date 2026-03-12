"""EN3.2-H1c — Multi-Extractor RAG Fusion Experiment.

Evaluates whether SL fusion of multiple QA extractor answers outperforms
scalar fusion baselines for RAG answer selection.

Uses 4 QA models (distilbert, roberta, electra, bert_tiny) as independent
sources — directly mirroring EN1.1's proven paradigm.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex

    python experiments/EN3/en3_2_h1c_experiment.py
    python experiments/EN3/en3_2_h1c_experiment.py --dry-run
    python experiments/EN3/en3_2_h1c_experiment.py --poison-rate 0.10
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from en3_2_h1c_core import (
    group_answers_fuzzy,
    select_answer_majority,
    select_answer_weighted_qa,
    select_answer_model_agreement,
    select_answer_model_agree_x_qa,
    select_answer_sl_fusion,
    select_answer_sl_trust_discount,
    select_answer_sl_conflict_aware,
    compute_abstention_signal,
    SCALAR_STRATEGY_NAMES,
    SL_STRATEGY_NAMES,
)

GLOBAL_SEED = 42
N_BOOTSTRAP = 1000
POISON_RATES = [0.05, 0.10, 0.20, 0.30]

MODEL_TAGS = {
    "distilbert": "v1b_qa_extraction",      # existing v1b checkpoint
    "roberta": "multimodel_qa_roberta",
    "electra": "multimodel_qa_electra",
    "bert_tiny": "multimodel_qa_bert_tiny",
}

# Model trust levels derived from gold-passage accuracy (pr10):
# distilbert=78.4%, roberta=84.7%, electra=80.9%, bert_tiny=24.2%
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


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_all_extractions(
    pr_tag: str,
    model_subset: List[str] | None = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load QA extractions for all models.

    Returns: {model_tag: {qid: {pid: {answer, qa_score}}}}
    """
    models = model_subset or list(MODEL_TAGS.keys())
    result = {}
    for model_tag in models:
        prefix = MODEL_TAGS[model_tag]
        path = CHECKPOINT_DIR / f"{prefix}_{pr_tag}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        with open(str(path), "r") as f:
            result[model_tag] = json.load(f)
    return result


def load_questions(pr_tag: str) -> List[Dict]:
    path = CHECKPOINT_DIR / f"v1b_questions_{pr_tag}.json"
    with open(str(path), "r") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# Per-question evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_question(
    qid: str,
    gold_answers: List[str],
    all_model_extractions: Dict[str, Dict[str, Dict[str, Any]]],
    evidence_weight: float = 10.0,
    prior_weight: float = 2.0,
) -> Dict[str, Any]:
    """Evaluate all strategies on one question.

    Returns: {strategy_name: {answer, correct}} + abstention_signal
    """
    # Build multi-model extraction for this question
    multi_ext: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for model_tag, model_ext in all_model_extractions.items():
        if qid in model_ext:
            multi_ext[model_tag] = model_ext[qid]

    if not multi_ext:
        return {"error": "no extractions"}

    groups = group_answers_fuzzy(multi_ext)
    if not groups:
        return {"error": "no answer groups"}

    results: Dict[str, Any] = {}

    # Scalar strategies
    strategies = {
        "majority": select_answer_majority,
        "weighted_qa": select_answer_weighted_qa,
        "model_agreement": select_answer_model_agreement,
        "model_agree_x_qa": select_answer_model_agree_x_qa,
    }

    for name, fn in strategies.items():
        answer = fn(groups)
        results[name] = {
            "answer": answer,
            "correct": exact_match(answer, gold_answers),
        }

    # SL strategies
    answer_sl = select_answer_sl_fusion(groups, evidence_weight, prior_weight)
    results["sl_fusion"] = {
        "answer": answer_sl,
        "correct": exact_match(answer_sl, gold_answers),
    }

    answer_td = select_answer_sl_trust_discount(
        groups, model_trust=MODEL_TRUST,
        evidence_weight=evidence_weight, prior_weight=prior_weight,
    )
    results["sl_trust_discount"] = {
        "answer": answer_td,
        "correct": exact_match(answer_td, gold_answers),
    }

    answer_ca = select_answer_sl_conflict_aware(groups, evidence_weight, prior_weight)
    results["sl_conflict_aware"] = {
        "answer": answer_ca,
        "correct": exact_match(answer_ca, gold_answers),
    }

    # Per-model baselines (best passage by qa_score for each model)
    for model_tag, model_ext in all_model_extractions.items():
        if qid in model_ext and model_ext[qid]:
            best = max(model_ext[qid].values(), key=lambda x: x["qa_score"])
            results[f"single_{model_tag}"] = {
                "answer": best["answer"],
                "correct": exact_match(best["answer"], gold_answers),
            }

    # Abstention signal
    results["abstention_signal"] = compute_abstention_signal(
        groups, evidence_weight, prior_weight,
    )

    results["n_groups"] = len(groups)

    return results


# ═══════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════

def run_experiment(
    poison_rates: List[float],
    max_questions: int | None = None,
    model_subset: List[str] | None = None,
    n_bootstrap: int = 1000,
    verbose: bool = True,
) -> Dict[str, Any]:
    all_results: Dict[str, Any] = {}

    all_strategies = (
        SCALAR_STRATEGY_NAMES + SL_STRATEGY_NAMES
        + [f"single_{m}" for m in (model_subset or list(MODEL_TAGS.keys()))]
    )

    for pr in poison_rates:
        tag = _pr_tag(pr)
        if verbose:
            print(f"\n{'='*60}")
            print(f"Poison rate: {pr} ({tag})")
            print(f"{'='*60}")

        t0 = time.time()
        questions = load_questions(tag)
        all_ext = load_all_extractions(tag, model_subset)

        if max_questions:
            questions = questions[:max_questions]

        if verbose:
            print(f"  Questions: {len(questions)}, Models: {list(all_ext.keys())}")

        # Evaluate all questions
        per_question: List[Dict[str, Any]] = []
        for q in questions:
            qid = q["id"]
            result = evaluate_question(
                qid, q["answers"], all_ext,
            )
            result["question_id"] = qid
            per_question.append(result)

        # Aggregate metrics
        pr_results: Dict[str, Any] = {
            "n_questions": len(questions),
            "strategies": {},
        }

        for strategy in all_strategies:
            correct_list = []
            for pq in per_question:
                if strategy in pq and isinstance(pq[strategy], dict):
                    correct_list.append(pq[strategy]["correct"])

            if not correct_list:
                continue

            n_correct = sum(correct_list)
            accuracy = n_correct / len(correct_list)
            ci_lo, ci_mean, ci_hi = bootstrap_ci(
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

        # Summary table
        if verbose:
            print(f"\n  {'─'*55}")
            print(f"  {'Strategy':<25} {'EM':>7} {'95% CI':>18} {'N':>5}")
            print(f"  {'─'*55}")
            for strategy in all_strategies:
                if strategy in pr_results["strategies"]:
                    info = pr_results["strategies"][strategy]
                    print(f"  {strategy:<25} {info['accuracy']:>7.3f} "
                          f"[{info['ci_lower']:.3f}, {info['ci_upper']:.3f}] "
                          f"{info['n_evaluated']:>5}")
            print(f"  {'─'*55}")
            print(f"  (GPT-4o-mini PLAIN reference: ~0.628-0.644)")
            print(f"  Elapsed: {time.time()-t0:.1f}s")

        # Store per-question for ablation use
        pr_results["per_question"] = [
            {
                "question_id": pq["question_id"],
                "n_groups": pq.get("n_groups", 0),
                "abstention_signal": pq.get("abstention_signal", 1.0),
                **{
                    s: pq[s]["correct"] if s in pq and isinstance(pq[s], dict) else None
                    for s in all_strategies
                },
            }
            for pq in per_question
        ]

        all_results[tag] = pr_results

    return all_results


def main():
    parser = argparse.ArgumentParser(description="EN3.2-H1c Multi-Extractor Fusion")
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
        poison_rates=prs,
        max_questions=max_q,
        n_bootstrap=n_boot,
        verbose=True,
    )

    elapsed = time.time() - t_start
    mode = "dry" if args.dry_run else "full"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_name = f"en3_2_h1c_{mode}_results"

    experiment_result = ExperimentResult(
        experiment_id="EN3.2-H1c",
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
            f"Multi-extractor RAG fusion. 4 QA models, "
            f"4 scalar + 3 SL strategies. {elapsed:.1f}s."
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
