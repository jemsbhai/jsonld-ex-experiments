"""EN3.2-H3 Ablation — ANSWERS-ONLY condition.

Runs the ANSWERS-ONLY ablation condition and compares against the
existing full-run results (PLAIN, SCALAR, JSONLD-EX) to disentangle
the extracted-answer confound.

Loads per-question answers from en3_2_h3_full_pr{XX}.json checkpoints
for the 3 existing conditions, runs ANSWERS-ONLY via API, then performs
all 6 pairwise McNemar's tests.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex

    set OPENAI_API_KEY=sk-...

    # Dry run (5 questions, 1 poison rate):
    python experiments/EN3/en3_2_h3_ablation.py --dry-run

    # Full ablation (500 questions, 4 poison rates, ~$0.65):
    python experiments/EN3/en3_2_h3_ablation.py
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
from typing import Any, Dict, List, Set

import numpy as np

# ── Project paths ──
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
    build_prompt_answers_only,
    classify_question_difficulty,
    mcnemars_test,
)

# ═══════════════════════════════════════════════════════════════════
# Configuration (must match en3_2_h3_experiment.py)
# ═══════════════════════════════════════════════════════════════════

GLOBAL_SEED = 42
N_QUESTIONS = 500
POISON_RATES = [0.05, 0.10, 0.20, 0.30]
N_BOOTSTRAP = 1000

OPENAI_MODEL = "gpt-4o-mini"
API_DELAY_SECONDS = 0.1
MAX_RETRIES = 5

ALL_CONDITIONS = ["PLAIN", "SCALAR", "ANSWERS-ONLY", "JSONLD-EX"]


# ═══════════════════════════════════════════════════════════════════
# SQuAD evaluation helpers (same as en3_2_h3_experiment.py)
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
    """Load v1b checkpoints for a given poison rate."""
    required = {
        "questions": CHECKPOINT_DIR / f"v1b_questions_{pr_tag}.json",
        "corpus_texts": CHECKPOINT_DIR / f"v1b_corpus_texts_{pr_tag}.json",
        "retrieval": CHECKPOINT_DIR / f"v1b_retrieval_{pr_tag}.json",
        "qa_extraction": CHECKPOINT_DIR / f"v1b_qa_extraction_{pr_tag}.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing v1b checkpoints for {pr_tag}: {missing}")

    data = {}
    for name, path in required.items():
        with open(str(path), "r") as f:
            data[name] = json.load(f)
    return data["questions"], data["corpus_texts"], data["retrieval"], data["qa_extraction"]


def load_full_run_checkpoint(pr_tag: str) -> Dict[str, Any]:
    """Load per-question answers from the full H3 run."""
    ckpt_path = CHECKPOINT_DIR / f"en3_2_h3_full_{pr_tag}.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing full-run checkpoint: {ckpt_path.name}. "
            f"Run 'python experiments/EN3/en3_2_h3_experiment.py' first."
        )
    with open(str(ckpt_path), "r") as f:
        return json.load(f)


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
# Helpers
# ═══════════════════════════════════════════════════════════════════

def get_poison_pids(retrieved: List[Dict[str, Any]]) -> Set[str]:
    return {p["passage_id"] for p in retrieved if p.get("is_poison", False)}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="EN3.2-H3 Ablation — ANSWERS-ONLY condition"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="5 questions, 1 poison rate")
    parser.add_argument("--poison-rates", type=float, nargs="+", default=None)
    args = parser.parse_args()

    if args.dry_run:
        n_questions = 5
        poison_rates = [0.10]
        run_tag = "ablation_dry"
    else:
        n_questions = N_QUESTIONS
        poison_rates = POISON_RATES
        run_tag = "ablation"

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
    print("EN3.2-H3 Ablation — ANSWERS-ONLY Condition")
    print("=" * 70)
    print(f"  Model:        {OPENAI_MODEL}")
    print(f"  Condition:    ANSWERS-ONLY")
    print(f"  Poison rates: {poison_rates}")
    print(f"  Questions:    {n_questions}")
    print(f"  Run tag:      {run_tag}")
    print()

    t_start = time.time()
    all_results = {}

    for poison_rate in poison_rates:
        pr_tag = f"pr{int(poison_rate * 100):02d}"

        # Check for existing ablation checkpoint
        ckpt_path = CHECKPOINT_DIR / f"en3_2_h3_{run_tag}_{pr_tag}.json"
        if ckpt_path.exists():
            print(f"  Loading ablation checkpoint: {ckpt_path.name}")
            with open(str(ckpt_path), "r") as f:
                all_results[pr_tag] = json.load(f)
            continue

        print("=" * 70)
        print(f"  POISON RATE: {poison_rate:.0%}")
        print("=" * 70)

        # Load full-run checkpoint for existing conditions
        full_run = load_full_run_checkpoint(pr_tag)

        # Load v1b checkpoints for ANSWERS-ONLY generation
        questions, pid_to_text, retrieval_results, extractions = (
            load_v1b_checkpoints(pr_tag)
        )
        qid_to_question = {q["id"]: q for q in questions}

        print(f"  Loaded full-run checkpoint + v1b data for {pr_tag}")

        # --- Run ANSWERS-ONLY condition ---
        print(f"\n  ── Generating ANSWERS-ONLY answers ──")
        answers_only_pq: List[Dict[str, Any]] = []

        for i, rr in enumerate(retrieval_results[:n_questions]):
            qid = rr["question_id"]
            q = qid_to_question[qid]
            ex = extractions.get(qid, {})
            passages = rr["retrieved"]

            prompt = build_prompt_answers_only(
                q["question"], passages, pid_to_text, ex,
            )
            answer = generate_fn(prompt)

            em = metric_over_answers(answer, q["answers"], exact_match_score)
            f1 = metric_over_answers(answer, q["answers"], token_f1_score)

            poison_pids = get_poison_pids(passages)
            difficulty = classify_question_difficulty(
                gold_passage_id=q["gold_passage_id"],
                retrieved=passages,
                poison_pids=poison_pids,
            )

            answers_only_pq.append({
                "question_id": qid,
                "answer": answer,
                "exact_match": em,
                "token_f1": f1,
                "difficulty": difficulty,
                "prompt_length": len(prompt),
            })

            if (i + 1) % 50 == 0:
                em_so_far = np.mean([pq["exact_match"] for pq in answers_only_pq])
                f1_so_far = np.mean([pq["token_f1"] for pq in answers_only_pq])
                print(f"    [{i+1}/{n_questions}] EM={em_so_far:.3f} F1={f1_so_far:.3f}")

            time.sleep(API_DELAY_SECONDS)

        # --- Compute ANSWERS-ONLY metrics ---
        ao_em = [pq["exact_match"] for pq in answers_only_pq]
        ao_f1 = [pq["token_f1"] for pq in answers_only_pq]
        ao_em_lo, ao_em_mean, ao_em_hi = bootstrap_ci(ao_em, N_BOOTSTRAP, seed=GLOBAL_SEED)
        ao_f1_lo, ao_f1_mean, ao_f1_hi = bootstrap_ci(ao_f1, N_BOOTSTRAP, seed=GLOBAL_SEED)

        ao_by_diff = {}
        for diff in ["easy", "medium", "hard"]:
            dpqs = [pq for pq in answers_only_pq if pq["difficulty"] == diff]
            if not dpqs:
                ao_by_diff[diff] = {"n": 0, "exact_match": None, "token_f1": None}
                continue
            dem = [pq["exact_match"] for pq in dpqs]
            df1 = [pq["token_f1"] for pq in dpqs]
            d_em_lo, d_em_mean, d_em_hi = bootstrap_ci(dem, N_BOOTSTRAP, seed=GLOBAL_SEED)
            d_f1_lo, d_f1_mean, d_f1_hi = bootstrap_ci(df1, N_BOOTSTRAP, seed=GLOBAL_SEED)
            ao_by_diff[diff] = {
                "n": len(dpqs),
                "exact_match": round(d_em_mean, 4),
                "exact_match_ci": [round(d_em_lo, 4), round(d_em_hi, 4)],
                "token_f1": round(d_f1_mean, 4),
                "token_f1_ci": [round(d_f1_lo, 4), round(d_f1_hi, 4)],
            }

        ao_metrics = {
            "exact_match": round(ao_em_mean, 4),
            "exact_match_ci": [round(ao_em_lo, 4), round(ao_em_hi, 4)],
            "token_f1": round(ao_f1_mean, 4),
            "token_f1_ci": [round(ao_f1_lo, 4), round(ao_f1_hi, 4)],
            "n_questions": len(answers_only_pq),
            "avg_prompt_length_chars": round(np.mean([pq["prompt_length"] for pq in answers_only_pq]), 1),
            "by_difficulty": ao_by_diff,
        }

        print(
            f"    RESULT: EM={ao_em_mean:.4f} [{ao_em_lo:.4f}, {ao_em_hi:.4f}]  "
            f"F1={ao_f1_mean:.4f}"
        )
        for diff in ["easy", "medium", "hard"]:
            dm = ao_by_diff[diff]
            if dm["n"] == 0:
                continue
            print(f"      {diff:>6s} (n={dm['n']:>3d}): EM={dm['exact_match']:.4f} F1={dm['token_f1']:.4f}")

        # --- Build per-question EM vectors for all 4 conditions ---
        # Existing conditions from full run (truncated to n_questions)
        full_pq = full_run["per_question_answers"]
        em_vectors = {}
        for cond in ["PLAIN", "SCALAR", "JSONLD-EX"]:
            em_vectors[cond] = [int(pq["exact_match"]) for pq in full_pq[cond][:n_questions]]
        em_vectors["ANSWERS-ONLY"] = [int(pq["exact_match"]) for pq in answers_only_pq]

        # Verify question alignment
        ao_qids = [pq["question_id"] for pq in answers_only_pq]
        for cond in ["PLAIN", "SCALAR", "JSONLD-EX"]:
            cond_qids = [pq["question_id"] for pq in full_pq[cond][:n_questions]]
            assert ao_qids == cond_qids, (
                f"Question ID mismatch between ANSWERS-ONLY and {cond} at {pr_tag}"
            )

        # --- Pairwise McNemar's tests (all 6 pairs) ---
        print(f"\n  ── Pairwise McNemar's Tests ──")
        pairs = [
            ("PLAIN", "SCALAR"),
            ("PLAIN", "ANSWERS-ONLY"),
            ("PLAIN", "JSONLD-EX"),
            ("SCALAR", "ANSWERS-ONLY"),
            ("ANSWERS-ONLY", "JSONLD-EX"),
            ("SCALAR", "JSONLD-EX"),
        ]
        comparisons = {}
        for cond_a, cond_b in pairs:
            overall = mcnemars_test(em_vectors[cond_a], em_vectors[cond_b])

            # Per-difficulty
            by_diff = {}
            for diff in ["easy", "medium", "hard"]:
                indices = [
                    i for i, pq in enumerate(answers_only_pq)
                    if pq["difficulty"] == diff
                ]
                if len(indices) < 5:
                    by_diff[diff] = {"n": len(indices), "skipped": True}
                    continue
                d_a = [em_vectors[cond_a][i] for i in indices]
                d_b = [em_vectors[cond_b][i] for i in indices]
                by_diff[diff] = mcnemars_test(d_a, d_b)
                by_diff[diff]["n"] = len(indices)

            comparisons[f"{cond_a}_vs_{cond_b}"] = {
                "overall": overall,
                "by_difficulty": by_diff,
            }

            sig = "***" if overall["significant"] else "n.s."
            print(f"    {cond_a} vs {cond_b}: p={overall['p_value']:.4f} {sig}  "
                  f"n01={overall['n_01']} n10={overall['n_10']}")

        # --- Assemble result ---
        # Include existing condition metrics for easy comparison
        pr_results = {
            "conditions": {},
            "comparisons": comparisons,
            "difficulty_distribution": full_run.get("difficulty_distribution", {}),
        }

        # Copy existing condition metrics from full run
        for cond in ["PLAIN", "SCALAR", "JSONLD-EX"]:
            pr_results["conditions"][cond] = full_run["conditions"][cond]
        pr_results["conditions"]["ANSWERS-ONLY"] = ao_metrics

        # Save per-question answers for ANSWERS-ONLY
        pr_results["per_question_answers_only"] = [
            {
                "question_id": pq["question_id"],
                "answer": pq["answer"],
                "exact_match": pq["exact_match"],
                "difficulty": pq["difficulty"],
            }
            for pq in answers_only_pq
        ]

        # Save checkpoint
        with open(str(ckpt_path), "w") as f:
            json.dump(pr_results, f, indent=2)
        print(f"\n  Saved checkpoint: {ckpt_path.name}")

        all_results[pr_tag] = pr_results

    # ── Final Summary ──
    total_time = time.time() - t_start

    print("\n" + "=" * 70)
    print("SUMMARY — EN3.2-H3 ANSWERS-ONLY Ablation")
    print("=" * 70)

    for pr_tag, pr_data in all_results.items():
        print(f"\n  {pr_tag}:")
        conds = pr_data["conditions"]
        print(f"    {'Condition':<14s} {'EM':>8s} {'EM CI':>22s} {'F1':>8s}   "
              f"{'Easy EM':>8s} {'Med EM':>8s} {'Hard EM':>8s}")
        print(f"    {'-'*14} {'-'*8} {'-'*22} {'-'*8}   {'-'*8} {'-'*8} {'-'*8}")
        for cond in ALL_CONDITIONS:
            m = conds.get(cond)
            if m is None:
                continue
            em_ci = f"[{m['exact_match_ci'][0]:.4f}, {m['exact_match_ci'][1]:.4f}]"
            easy_em = m["by_difficulty"].get("easy", {}).get("exact_match")
            med_em = m["by_difficulty"].get("medium", {}).get("exact_match")
            hard_em = m["by_difficulty"].get("hard", {}).get("exact_match")
            print(
                f"    {cond:<14s} {m['exact_match']:>8.4f} {em_ci:>22s} "
                f"{m['token_f1']:>8.4f}   "
                f"{easy_em if easy_em is not None else 'N/A':>8} "
                f"{med_em if med_em is not None else 'N/A':>8} "
                f"{hard_em if hard_em is not None else 'N/A':>8}"
            )

        # Key ablation comparisons
        print(f"\n    Key ablation comparisons:")
        for pair_key in ["PLAIN_vs_ANSWERS-ONLY", "ANSWERS-ONLY_vs_JSONLD-EX"]:
            comp = pr_data["comparisons"].get(pair_key)
            if comp:
                ov = comp["overall"]
                sig = "***" if ov["significant"] else "n.s."
                print(f"      {pair_key}: p={ov['p_value']:.4f} {sig}  "
                      f"n01={ov['n_01']} n10={ov['n_10']}")

    print(f"\n  Total wall time: {total_time:.1f}s")

    # ── Save final results ──
    output_path = RESULTS_DIR / f"en3_2_h3_{run_tag}_results.json"
    total_api_calls = n_questions * len(poison_rates)
    experiment_result = ExperimentResult(
        experiment_id="EN3.2-H3-ablation",
        parameters={
            "global_seed": GLOBAL_SEED,
            "n_questions": n_questions,
            "poison_rates": poison_rates,
            "model": OPENAI_MODEL,
            "temperature": 0.0,
            "n_bootstrap": N_BOOTSTRAP,
            "conditions": ALL_CONDITIONS,
            "new_condition": "ANSWERS-ONLY",
            "run_tag": run_tag,
        },
        metrics={
            "total_wall_time_seconds": round(total_time, 4),
            "total_api_calls": total_api_calls,
            "results_by_poison_rate": all_results,
        },
        raw_data={
            "n_questions_per_poison_rate": n_questions,
        },
        environment=env,
        notes=(
            "EN3.2-H3 Ablation: ANSWERS-ONLY condition to disentangle "
            "extracted-answer confound from SL metadata effect. "
            "ANSWERS-ONLY includes passages + extracted answers but NO "
            "SL (b,d,u) triples, NO conflict/agreement, NO fused assessment. "
            "Compared against PLAIN, SCALAR, and JSONLD-EX from the full run. "
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
