"""EN3.2-H3 — Metadata-Enriched Prompting Experiment.

Tests whether giving SL metadata IN the prompt improves LLM answer
quality, especially on hard questions where sources conflict.

Three conditions (within-subject, paired stats):
  PLAIN:     passages only (standard RAG)
  SCALAR:    passages + cosine similarity scores
  JSONLD-EX: passages + full SL metadata (b, d, u, conflict, agreement)

Uses v1b SQuAD checkpoints. API-based (GPT-4o-mini, temp=0).

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex

    # Full run (500 questions × 4 poison rates × 3 conditions ≈ $3-4):
    set OPENAI_API_KEY=sk-...
    python experiments/EN3/en3_2_h3_experiment.py

    # Dry run (5 questions, 1 poison rate, validate checkpoint loading):
    python experiments/EN3/en3_2_h3_experiment.py --dry-run

    # Limited run (50 questions, 2 poison rates):
    python experiments/EN3/en3_2_h3_experiment.py --limited
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

# ── Project paths ──
# SCRIPT_DIR:       experiments/EN3/     (for en3_2_h3_core sibling import)
# EXPERIMENTS_ROOT:  experiments/         (for infra.* imports)
# jsonld_ex is pip-installed; no path manipulation needed for it.
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

from en3_2_h3_core import (
    build_prompt_plain,
    build_prompt_scalar,
    build_prompt_jsonldex,
    classify_question_difficulty,
    mcnemars_test,
)

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

GLOBAL_SEED = 42
N_QUESTIONS = 500
TOP_K = 10
EVIDENCE_WEIGHT = 10
PRIOR_WEIGHT = 2
POISON_RATES = [0.05, 0.10, 0.20, 0.30]
N_BOOTSTRAP = 1000

# API
OPENAI_MODEL = "gpt-4o-mini"
API_DELAY_SECONDS = 0.1
MAX_RETRIES = 5

# Conditions
CONDITIONS = ["PLAIN", "SCALAR", "JSONLD-EX"]


# ═══════════════════════════════════════════════════════════════════
# SQuAD evaluation helpers (same as en3_1b_tier2_api.py)
# ═══════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(ground_truth).split()
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(pred_toks) if pred_toks else 0.0
    rec = num_same / len(gold_toks) if gold_toks else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def metric_over_answers(prediction: str, gold_answers: List[str], metric_fn) -> float:
    return max(metric_fn(prediction, a) for a in gold_answers) if gold_answers else 0.0


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_v1b_checkpoints(pr_tag: str):
    """Load all v1b checkpoints for a given poison rate.

    Returns (questions, pid_to_text, retrieval_results, extractions).
    """
    required = {
        "questions": CHECKPOINT_DIR / f"v1b_questions_{pr_tag}.json",
        "corpus_texts": CHECKPOINT_DIR / f"v1b_corpus_texts_{pr_tag}.json",
        "retrieval": CHECKPOINT_DIR / f"v1b_retrieval_{pr_tag}.json",
        "qa_extraction": CHECKPOINT_DIR / f"v1b_qa_extraction_{pr_tag}.json",
    }

    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing v1b checkpoints for {pr_tag}: {missing}. "
            f"Run 'python experiments/EN3/en3_1b_rag_pipeline.py --tier1' first."
        )

    with open(str(required["questions"]), "r") as f:
        questions = json.load(f)
    with open(str(required["corpus_texts"]), "r") as f:
        pid_to_text = json.load(f)
    with open(str(required["retrieval"]), "r") as f:
        retrieval_results = json.load(f)
    with open(str(required["qa_extraction"]), "r") as f:
        extractions = json.load(f)

    return questions, pid_to_text, retrieval_results, extractions


# ═══════════════════════════════════════════════════════════════════
# API generation
# ═══════════════════════════════════════════════════════════════════

def generate_answer_openai(client, prompt: str) -> str:
    """Generate answer via OpenAI API with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=64,
                temperature=0.0,
                seed=GLOBAL_SEED,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a question-answering assistant. "
                                   "Give short, direct answers.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            # Take first line only (avoid multi-line answers)
            return raw.split("\n")[0].strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"      API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"      API failed after {MAX_RETRIES} attempts: {e}")
                return ""


# ═══════════════════════════════════════════════════════════════════
# Prompt construction dispatch
# ═══════════════════════════════════════════════════════════════════

def build_prompt_for_condition(
    condition: str,
    question: str,
    passages: List[Dict[str, Any]],
    pid_to_text: Dict[str, str],
    extractions: Dict[str, Dict[str, Any]],
) -> str:
    """Dispatch to the correct prompt builder for a condition."""
    if condition == "PLAIN":
        return build_prompt_plain(question, passages, pid_to_text)
    elif condition == "SCALAR":
        return build_prompt_scalar(question, passages, pid_to_text)
    elif condition == "JSONLD-EX":
        return build_prompt_jsonldex(
            question, passages, pid_to_text, extractions,
            evidence_weight=EVIDENCE_WEIGHT, prior_weight=PRIOR_WEIGHT,
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ═══════════════════════════════════════════════════════════════════
# Difficulty + poison analysis helpers
# ═══════════════════════════════════════════════════════════════════

def get_poison_pids(retrieved: List[Dict[str, Any]]) -> Set[str]:
    """Extract set of poison passage IDs from retrieved list."""
    return {p["passage_id"] for p in retrieved if p.get("is_poison", False)}


# ═══════════════════════════════════════════════════════════════════
# Per-condition evaluation for a single poison rate
# ═══════════════════════════════════════════════════════════════════

def evaluate_condition(
    condition: str,
    questions: List[Dict[str, Any]],
    retrieval_results: List[Dict[str, Any]],
    pid_to_text: Dict[str, str],
    extractions: Dict[str, Dict[str, Any]],
    generate_fn,
    delay: float,
    n_questions: int,
) -> List[Dict[str, Any]]:
    """Evaluate a single condition across all questions.

    Returns per-question results list.
    """
    qid_to_question = {q["id"]: q for q in questions}
    per_question: List[Dict[str, Any]] = []

    for i, rr in enumerate(retrieval_results[:n_questions]):
        qid = rr["question_id"]
        q = qid_to_question[qid]
        ex = extractions.get(qid, {})
        passages = rr["retrieved"]

        # Build prompt
        prompt = build_prompt_for_condition(
            condition, q["question"], passages, pid_to_text, ex,
        )

        # Generate answer
        answer = generate_fn(prompt)

        # Evaluate
        em = metric_over_answers(answer, q["answers"], exact_match_score)
        f1 = metric_over_answers(answer, q["answers"], token_f1_score)

        # Classify difficulty
        poison_pids = get_poison_pids(passages)
        difficulty = classify_question_difficulty(
            gold_passage_id=q["gold_passage_id"],
            retrieved=passages,
            poison_pids=poison_pids,
        )

        per_question.append({
            "question_id": qid,
            "answer": answer,
            "exact_match": em,
            "token_f1": f1,
            "difficulty": difficulty,
            "n_poison_in_retrieved": len(poison_pids),
            "prompt_length": len(prompt),
        })

        if (i + 1) % 50 == 0:
            em_so_far = np.mean([pq["exact_match"] for pq in per_question])
            f1_so_far = np.mean([pq["token_f1"] for pq in per_question])
            print(f"    [{i+1}/{n_questions}] EM={em_so_far:.3f} F1={f1_so_far:.3f}")

        time.sleep(delay)

    return per_question


def compute_metrics(
    per_question: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute aggregate metrics from per-question results."""
    em_scores = [pq["exact_match"] for pq in per_question]
    f1_scores = [pq["token_f1"] for pq in per_question]

    em_lo, em_mean, em_hi = bootstrap_ci(em_scores, N_BOOTSTRAP, seed=GLOBAL_SEED)
    f1_lo, f1_mean, f1_hi = bootstrap_ci(f1_scores, N_BOOTSTRAP, seed=GLOBAL_SEED)

    # Per-difficulty breakdown
    difficulty_metrics = {}
    for diff in ["easy", "medium", "hard"]:
        diff_pqs = [pq for pq in per_question if pq["difficulty"] == diff]
        if not diff_pqs:
            difficulty_metrics[diff] = {
                "n": 0, "exact_match": None, "token_f1": None,
            }
            continue
        diff_em = [pq["exact_match"] for pq in diff_pqs]
        diff_f1 = [pq["token_f1"] for pq in diff_pqs]
        d_em_lo, d_em_mean, d_em_hi = bootstrap_ci(diff_em, N_BOOTSTRAP, seed=GLOBAL_SEED)
        d_f1_lo, d_f1_mean, d_f1_hi = bootstrap_ci(diff_f1, N_BOOTSTRAP, seed=GLOBAL_SEED)
        difficulty_metrics[diff] = {
            "n": len(diff_pqs),
            "exact_match": round(d_em_mean, 4),
            "exact_match_ci": [round(d_em_lo, 4), round(d_em_hi, 4)],
            "token_f1": round(d_f1_mean, 4),
            "token_f1_ci": [round(d_f1_lo, 4), round(d_f1_hi, 4)],
        }

    avg_prompt_len = np.mean([pq["prompt_length"] for pq in per_question])

    return {
        "exact_match": round(em_mean, 4),
        "exact_match_ci": [round(em_lo, 4), round(em_hi, 4)],
        "token_f1": round(f1_mean, 4),
        "token_f1_ci": [round(f1_lo, 4), round(f1_hi, 4)],
        "n_questions": len(per_question),
        "avg_prompt_length_chars": round(avg_prompt_len, 1),
        "by_difficulty": difficulty_metrics,
    }


# ═══════════════════════════════════════════════════════════════════
# Paired statistical comparisons
# ═══════════════════════════════════════════════════════════════════

def run_pairwise_mcnemar(
    all_per_question: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Run McNemar's tests between all condition pairs.

    Also runs per-difficulty breakdowns.
    """
    comparisons = {}
    pairs = [
        ("PLAIN", "SCALAR"),
        ("PLAIN", "JSONLD-EX"),
        ("SCALAR", "JSONLD-EX"),
    ]

    for cond_a, cond_b in pairs:
        pq_a = all_per_question[cond_a]
        pq_b = all_per_question[cond_b]

        # Overall
        em_a = [int(pq["exact_match"]) for pq in pq_a]
        em_b = [int(pq["exact_match"]) for pq in pq_b]
        overall = mcnemars_test(em_a, em_b)

        # Per-difficulty
        by_diff = {}
        for diff in ["easy", "medium", "hard"]:
            indices = [
                i for i, pq in enumerate(pq_a)
                if pq["difficulty"] == diff
            ]
            if len(indices) < 5:
                by_diff[diff] = {"n": len(indices), "skipped": True}
                continue
            d_em_a = [em_a[i] for i in indices]
            d_em_b = [em_b[i] for i in indices]
            by_diff[diff] = mcnemars_test(d_em_a, d_em_b)
            by_diff[diff]["n"] = len(indices)

        comparisons[f"{cond_a}_vs_{cond_b}"] = {
            "overall": overall,
            "by_difficulty": by_diff,
        }

    return comparisons


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="EN3.2-H3 — Metadata-Enriched Prompting Experiment"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 5 questions, 1 poison rate — validate pipeline only",
    )
    parser.add_argument(
        "--limited", action="store_true",
        help="Run 50 questions, 2 poison rates",
    )
    parser.add_argument(
        "--poison-rates", type=float, nargs="+", default=None,
        help="Override poison rates to evaluate",
    )
    args = parser.parse_args()

    # Determine run parameters
    if args.dry_run:
        n_questions = 5
        poison_rates = [0.10]
        run_tag = "dry"
    elif args.limited:
        n_questions = 50
        poison_rates = [0.10, 0.30]
        run_tag = "limited"
    else:
        n_questions = N_QUESTIONS
        poison_rates = POISON_RATES
        run_tag = "full"

    if args.poison_rates:
        poison_rates = args.poison_rates

    set_global_seed(GLOBAL_SEED)
    env = log_environment()

    # --- Initialize API client ---
    import openai
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    client = openai.OpenAI(api_key=api_key)

    def generate_fn(prompt: str) -> str:
        return generate_answer_openai(client, prompt)

    print("=" * 70)
    print("EN3.2-H3 — Metadata-Enriched Prompting Experiment")
    print("=" * 70)
    print(f"  Model:        {OPENAI_MODEL}")
    print(f"  Conditions:   {CONDITIONS}")
    print(f"  Poison rates: {poison_rates}")
    print(f"  Questions:    {n_questions}")
    print(f"  Run tag:      {run_tag}")
    print(f"  Seed:         {GLOBAL_SEED}")
    print()

    t_start = time.time()
    all_results = {}

    for poison_rate in poison_rates:
        pr_tag = f"pr{int(poison_rate * 100):02d}"

        # Check for existing checkpoint
        ckpt_path = CHECKPOINT_DIR / f"en3_2_h3_{run_tag}_{pr_tag}.json"
        if ckpt_path.exists():
            print(f"  Loading checkpoint: {ckpt_path.name}")
            with open(str(ckpt_path), "r") as f:
                all_results[pr_tag] = json.load(f)
            continue

        print("=" * 70)
        print(f"  POISON RATE: {poison_rate:.0%}")
        print("=" * 70)

        # Load checkpoints
        print(f"  Loading v1b checkpoints for {pr_tag}...")
        questions, pid_to_text, retrieval_results, extractions = (
            load_v1b_checkpoints(pr_tag)
        )
        qid_to_question = {q["id"]: q for q in questions}

        print(
            f"  Loaded {len(retrieval_results)} retrieval results, "
            f"{len(pid_to_text)} corpus passages"
        )

        # --- Difficulty distribution (diagnostic) ---
        diff_counts = Counter()
        for rr in retrieval_results[:n_questions]:
            qid = rr["question_id"]
            q = qid_to_question[qid]
            poison_pids = get_poison_pids(rr["retrieved"])
            diff = classify_question_difficulty(
                gold_passage_id=q["gold_passage_id"],
                retrieved=rr["retrieved"],
                poison_pids=poison_pids,
            )
            diff_counts[diff] += 1
        print(f"  Difficulty distribution: {dict(sorted(diff_counts.items()))}")

        # --- Evaluate each condition ---
        pr_results: Dict[str, Any] = {"conditions": {}, "comparisons": {}}
        all_per_question: Dict[str, List[Dict[str, Any]]] = {}

        for condition in CONDITIONS:
            print(f"\n  ── Condition: {condition} ──")
            per_question = evaluate_condition(
                condition=condition,
                questions=questions,
                retrieval_results=retrieval_results,
                pid_to_text=pid_to_text,
                extractions=extractions,
                generate_fn=generate_fn,
                delay=API_DELAY_SECONDS,
                n_questions=n_questions,
            )

            metrics = compute_metrics(per_question)
            pr_results["conditions"][condition] = metrics
            all_per_question[condition] = per_question

            print(
                f"    RESULT: EM={metrics['exact_match']:.4f} "
                f"[{metrics['exact_match_ci'][0]:.4f}, "
                f"{metrics['exact_match_ci'][1]:.4f}]  "
                f"F1={metrics['token_f1']:.4f}"
            )
            for diff in ["easy", "medium", "hard"]:
                dm = metrics["by_difficulty"][diff]
                if dm["n"] == 0:
                    continue
                print(
                    f"      {diff:>6s} (n={dm['n']:>3d}): "
                    f"EM={dm['exact_match']:.4f} F1={dm['token_f1']:.4f}"
                )

        # --- Pairwise McNemar's tests ---
        print("\n  ── Pairwise McNemar's Tests ──")
        comparisons = run_pairwise_mcnemar(all_per_question)
        pr_results["comparisons"] = comparisons

        for pair_name, pair_data in comparisons.items():
            ov = pair_data["overall"]
            sig_str = "***" if ov["significant"] else "n.s."
            print(
                f"    {pair_name}: p={ov['p_value']:.4f} {sig_str} "
                f"(n_01={ov['n_01']}, n_10={ov['n_10']})"
            )
            for diff, dd in pair_data["by_difficulty"].items():
                if dd.get("skipped"):
                    continue
                d_sig = "***" if dd["significant"] else "n.s."
                print(
                    f"      {diff:>6s}: p={dd['p_value']:.4f} {d_sig} "
                    f"(n_01={dd['n_01']}, n_10={dd['n_10']})"
                )

        # --- Difficulty distribution in results ---
        pr_results["difficulty_distribution"] = dict(sorted(diff_counts.items()))

        # --- Save per-question raw data for reproducibility ---
        pr_results["per_question_answers"] = {
            cond: [
                {
                    "question_id": pq["question_id"],
                    "answer": pq["answer"],
                    "exact_match": pq["exact_match"],
                    "difficulty": pq["difficulty"],
                }
                for pq in pqs
            ]
            for cond, pqs in all_per_question.items()
        }

        # Save checkpoint
        with open(str(ckpt_path), "w") as f:
            json.dump(pr_results, f, indent=2)
        print(f"\n  Saved checkpoint: {ckpt_path.name}")

        all_results[pr_tag] = pr_results

    # ── Final Summary ──
    total_time = time.time() - t_start

    print("\n" + "=" * 70)
    print("SUMMARY — EN3.2-H3 Metadata-Enriched Prompting")
    print("=" * 70)

    for pr_tag, pr_data in all_results.items():
        print(f"\n  {pr_tag}:")
        print(f"    {'Condition':<12s} {'EM':>8s} {'EM CI':>22s} {'F1':>8s}   "
              f"{'Easy EM':>8s} {'Med EM':>8s} {'Hard EM':>8s}")
        print(f"    {'-'*12} {'-'*8} {'-'*22} {'-'*8}   {'-'*8} {'-'*8} {'-'*8}")
        for cond, m in pr_data["conditions"].items():
            em_ci = f"[{m['exact_match_ci'][0]:.4f}, {m['exact_match_ci'][1]:.4f}]"
            easy_em = m["by_difficulty"].get("easy", {}).get("exact_match")
            med_em = m["by_difficulty"].get("medium", {}).get("exact_match")
            hard_em = m["by_difficulty"].get("hard", {}).get("exact_match")
            print(
                f"    {cond:<12s} {m['exact_match']:>8.4f} {em_ci:>22s} "
                f"{m['token_f1']:>8.4f}   "
                f"{easy_em if easy_em is not None else 'N/A':>8} "
                f"{med_em if med_em is not None else 'N/A':>8} "
                f"{hard_em if hard_em is not None else 'N/A':>8}"
            )

    print(f"\n  Total wall time: {total_time:.1f}s")

    # ── Save final results ──
    output_path = RESULTS_DIR / f"en3_2_h3_{run_tag}_results.json"
    total_api_calls = len(CONDITIONS) * n_questions * len(poison_rates)
    experiment_result = ExperimentResult(
        experiment_id="EN3.2-H3",
        parameters={
            "global_seed": GLOBAL_SEED,
            "n_questions": n_questions,
            "top_k": TOP_K,
            "evidence_weight": EVIDENCE_WEIGHT,
            "prior_weight": PRIOR_WEIGHT,
            "poison_rates": poison_rates,
            "model": OPENAI_MODEL,
            "temperature": 0.0,
            "n_bootstrap": N_BOOTSTRAP,
            "conditions": CONDITIONS,
            "run_tag": run_tag,
        },
        metrics={
            "total_wall_time_seconds": round(total_time, 4),
            "total_api_calls": total_api_calls,
            "results_by_poison_rate": all_results,
        },
        raw_data={
            "n_questions_per_condition_per_poison_rate": n_questions,
        },
        environment=env,
        notes=(
            "EN3.2-H3: Metadata-enriched prompting experiment. Tests whether "
            "SL epistemic metadata in the prompt improves LLM answer quality. "
            "Three conditions: PLAIN (passages only), SCALAR (+ cosine scores), "
            "JSONLD-EX (+ full SL b,d,u + conflict + agreement + fused assessment). "
            "Within-subject design with paired McNemar's test. "
            "Stratified by question difficulty (easy/medium/hard). "
            f"Total API calls: {total_api_calls}."
        ),
    )
    experiment_result.save_json(str(output_path))
    print(f"\n  Results saved to: {output_path}")

    ts = experiment_result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    archive_path = RESULTS_DIR / f"en3_2_h3_{run_tag}_results_{ts}.json"
    experiment_result.save_json(str(archive_path))
    print(f"  Archived to:      {archive_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
