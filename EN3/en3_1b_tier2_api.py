"""EN3.1b Tier 2 — API-based LLM Answer Generation.

Loads Tier 1 checkpoints (embeddings, retrieval, QA extraction) from the
v1b experiment and evaluates answer quality using an API-based LLM.
Supports both Anthropic (Claude) and OpenAI APIs.

This produces a second set of Tier 2 results to complement the local
Qwen2.5-7B results, enabling cross-model validation.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex

    # Groq (Llama 3.3 70B, FREE):
    set GROQ_API_KEY=gsk_...
    python experiments/EN3/en3_1b_tier2_api.py --provider groq

    # Anthropic (Claude Sonnet):
    set ANTHROPIC_API_KEY=sk-ant-...
    python experiments/EN3/en3_1b_tier2_api.py --provider anthropic

    # OpenAI (GPT-4o-mini):
    set OPENAI_API_KEY=sk-...
    python experiments/EN3/en3_1b_tier2_api.py --provider openai
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
from typing import Any, Dict, List, Optional

import numpy as np

# ── Project paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_ROOT))

RESULTS_DIR = SCRIPT_DIR / "results"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"

from infra.config import set_global_seed
from infra.env_log import log_environment
from infra.results import ExperimentResult
from infra.stats import bootstrap_ci

from jsonld_ex.confidence_algebra import (
    Opinion, cumulative_fuse, pairwise_conflict,
)

# ═══════════════════════════════════════════════════════════════════
# Configuration (must match en3_1b_rag_pipeline.py)
# ═══════════════════════════════════════════════════════════════════

GLOBAL_SEED = 42
N_QUESTIONS = 500
TOP_K = 10
EVIDENCE_WEIGHT = 10
PRIOR_WEIGHT = 2
POISON_RATES = [0.05, 0.10, 0.20, 0.30]
SCALAR_THRESHOLDS = [0.3, 0.4, 0.5, 0.6]
N_BOOTSTRAP = 1000
ANSWER_SIMILARITY_THRESHOLD = 0.6

# API models
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
OPENAI_MODEL = "gpt-4o-mini"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Rate limiting
API_DELAY_SECONDS = 0.1  # delay between API calls to avoid rate limits
GROQ_DELAY_SECONDS = 1.2  # Groq free tier: ~30 RPM -> 1 req per 2s with margin
MAX_RETRIES = 5

# ═══════════════════════════════════════════════════════════════════
# SQuAD evaluation helpers
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
# Answer grouping + filtering methods (duplicated from v1b for
# self-contained execution)
# ═══════════════════════════════════════════════════════════════════

from difflib import SequenceMatcher

def answers_match(a: str, b: str, threshold: float = ANSWER_SIMILARITY_THRESHOLD) -> bool:
    na, nb = normalize_answer(a), normalize_answer(b)
    if na == nb:
        return True
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def group_passages_by_answer(passages, extractions):
    augmented = []
    for p in passages:
        pid = p["passage_id"]
        ext = extractions.get(pid, {"answer": "", "qa_score": 0.0})
        augmented.append({**p, "extracted_answer": ext["answer"], "qa_score": ext["qa_score"]})

    groups = []
    for ap in augmented:
        placed = False
        for group in groups:
            if answers_match(ap["extracted_answer"], group[0]["extracted_answer"]):
                group.append(ap)
                placed = True
                break
        if not placed:
            groups.append([ap])
    return groups


def method_a_no_filter(retrieval, **kw):
    return {"method": "A_no_filter", "kept_passages": retrieval["retrieved"], "abstained": False}


def method_b_scalar_threshold(retrieval, threshold, **kw):
    kept = [r for r in retrieval["retrieved"] if r["score"] >= threshold]
    return {"method": f"B_scalar_{threshold:.2f}", "kept_passages": kept, "abstained": len(kept) == 0}


def method_c_majority_vote(retrieval, extractions, **kw):
    passages = retrieval["retrieved"]
    if not passages:
        return {"method": "C_majority_vote", "kept_passages": [], "abstained": True}
    groups = group_passages_by_answer(passages, extractions)
    groups.sort(key=len, reverse=True)
    majority_group = groups[0]
    kept_pids = {p["passage_id"] for p in majority_group}
    return {"method": "C_majority_vote", "kept_passages": [p for p in passages if p["passage_id"] in kept_pids], "abstained": False}


def method_e_sl_answer_fusion(retrieval, extractions, **kw):
    passages = retrieval["retrieved"]
    if not passages:
        return {"method": "E_sl_fusion", "kept_passages": [], "abstained": True}

    groups = group_passages_by_answer(passages, extractions)
    if len(groups) == 1:
        return {"method": "E_sl_fusion", "kept_passages": passages, "abstained": False}

    group_opinions = []
    group_data = []
    for group in groups:
        ops = []
        for p in group:
            cos_sim = max(0.0, min(1.0, p["score"]))
            qa_conf = max(0.0, min(1.0, p["qa_score"]))
            combined = (cos_sim * qa_conf) ** 0.5
            ops.append(Opinion.from_evidence(combined * EVIDENCE_WEIGHT, (1.0 - combined) * EVIDENCE_WEIGHT, prior_weight=PRIOR_WEIGHT))
        fused = cumulative_fuse(*ops) if len(ops) > 1 else ops[0]
        group_opinions.append(fused)
        group_data.append({"group": group, "fused_proj": fused.projected_probability(), "fused_uncertainty": fused.uncertainty})

    max_conflict = 0.0
    for i in range(len(group_opinions)):
        for j in range(i + 1, len(group_opinions)):
            max_conflict = max(max_conflict, pairwise_conflict(group_opinions[i], group_opinions[j]))

    avg_u = np.mean([gd["fused_uncertainty"] for gd in group_data])
    if max_conflict <= avg_u:
        return {"method": "E_sl_fusion", "kept_passages": passages, "abstained": False}

    best_idx = max(range(len(group_data)), key=lambda k: group_data[k]["fused_proj"])
    best_group = group_data[best_idx]["group"]
    kept_pids = {p["passage_id"] for p in best_group}
    return {"method": "E_sl_fusion", "kept_passages": [p for p in passages if p["passage_id"] in kept_pids], "abstained": False}


def method_f_sl_remove_outlier(retrieval, extractions, **kw):
    """Method F: Remove only the conflicting minority group, keep everything else."""
    passages = retrieval["retrieved"]
    if not passages:
        return {"method": "F_sl_remove_outlier", "kept_passages": [], "abstained": True}

    groups = group_passages_by_answer(passages, extractions)
    if len(groups) == 1:
        return {"method": "F_sl_remove_outlier", "kept_passages": passages, "abstained": False}

    group_opinions = []
    group_data = []
    for group in groups:
        ops = []
        for p in group:
            cos_sim = max(0.0, min(1.0, p["score"]))
            qa_conf = max(0.0, min(1.0, p["qa_score"]))
            combined = (cos_sim * qa_conf) ** 0.5
            ops.append(Opinion.from_evidence(combined * EVIDENCE_WEIGHT, (1.0 - combined) * EVIDENCE_WEIGHT, prior_weight=PRIOR_WEIGHT))
        fused = cumulative_fuse(*ops) if len(ops) > 1 else ops[0]
        group_opinions.append(fused)
        group_data.append({"group": group, "fused_proj": fused.projected_probability(), "fused_uncertainty": fused.uncertainty})

    max_conflict = 0.0
    for i in range(len(group_opinions)):
        for j in range(i + 1, len(group_opinions)):
            max_conflict = max(max_conflict, pairwise_conflict(group_opinions[i], group_opinions[j]))

    avg_u = np.mean([gd["fused_uncertainty"] for gd in group_data])
    if max_conflict <= avg_u:
        return {"method": "F_sl_remove_outlier", "kept_passages": passages, "abstained": False}

    # Remove only the lowest-evidence group
    worst_idx = min(range(len(group_data)), key=lambda k: group_data[k]["fused_proj"])
    outlier_pids = {p["passage_id"] for p in group_data[worst_idx]["group"]}
    kept = [p for p in passages if p["passage_id"] not in outlier_pids]
    return {"method": "F_sl_remove_outlier", "kept_passages": kept, "abstained": len(kept) == 0}


# ═══════════════════════════════════════════════════════════════════
# API LLM generation
# ═══════════════════════════════════════════════════════════════════

def build_prompt(question: str, passages: List[str]) -> str:
    context = "\n\n".join(f"[Passage {i+1}] {p}" for i, p in enumerate(passages))
    return (
        f"Answer the following question based ONLY on the provided passages. "
        f"Give a short, direct answer (a few words or a short phrase). "
        f"If the passages do not contain enough information, say 'unanswerable'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def generate_answer_anthropic(client, question: str, passages: List[str]) -> str:
    prompt = build_prompt(question, passages)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=64,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip().split("\n")[0].strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"      API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"      API failed after {MAX_RETRIES} attempts: {e}")
                return ""


def generate_answer_openai(client, question: str, passages: List[str]) -> str:
    prompt = build_prompt(question, passages)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=64,
                temperature=0.0,
                seed=GLOBAL_SEED,
                messages=[
                    {"role": "system", "content": "You are a question-answering assistant. Give short, direct answers."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip().split("\n")[0].strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"      API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"      API failed after {MAX_RETRIES} attempts: {e}")
                return ""


def generate_answer_groq(client, question: str, passages: List[str]) -> str:
    """Generate answer via Groq API (OpenAI-compatible interface)."""
    prompt = build_prompt(question, passages)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=64,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "You are a question-answering assistant. Give short, direct answers."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip().split("\n")[0].strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                # Longer backoff for Groq rate limits
                wait = (2 ** attempt) * 2
                print(f"      Groq error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"      Groq failed after {MAX_RETRIES} attempts: {e}")
                return ""


# ═══════════════════════════════════════════════════════════════════
# Data loading (from v1b checkpoints ONLY — no independent corpus rebuild)
# ═══════════════════════════════════════════════════════════════════

def load_v1b_checkpoints(pr_tag: str):
    """Load all v1b checkpoints for a given poison rate.

    Returns (questions, pid_to_text, retrieval_results, extractions).
    Raises FileNotFoundError if any checkpoint is missing.
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
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EN3.1b Tier 2 — API LLM Evaluation")
    parser.add_argument("--provider", choices=["groq", "anthropic", "openai"], default="groq")
    parser.add_argument("--poison-rates", type=float, nargs="+", default=POISON_RATES)
    parser.add_argument("--limited", action="store_true",
                        help="Run limited experiment (100 questions, 2 poison rates) "
                             "for free-tier API validation")
    args = parser.parse_args()

    if args.limited:
        args.poison_rates = [0.10, 0.30]  # two representative rates
        args.n_questions = 100
    else:
        args.n_questions = N_QUESTIONS

    set_global_seed(GLOBAL_SEED)
    env = log_environment()

    provider = args.provider
    if provider == "groq":
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("ERROR: Set GROQ_API_KEY environment variable")
            print("  Get a free key at: https://console.groq.com/keys")
            sys.exit(1)
        client = Groq(api_key=api_key)
        model_name = GROQ_MODEL
        generate_fn = lambda q, p: generate_answer_groq(client, q, p)
        delay = GROQ_DELAY_SECONDS
    elif provider == "anthropic":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: Set ANTHROPIC_API_KEY environment variable")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
        model_name = ANTHROPIC_MODEL
        generate_fn = lambda q, p: generate_answer_anthropic(client, q, p)
        delay = API_DELAY_SECONDS
    else:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: Set OPENAI_API_KEY environment variable")
            sys.exit(1)
        client = openai.OpenAI(api_key=api_key)
        model_name = OPENAI_MODEL
        generate_fn = lambda q, p: generate_answer_openai(client, q, p)
        delay = API_DELAY_SECONDS

    print("=" * 70)
    print(f"EN3.1b Tier 2 — API LLM Evaluation ({provider})")
    print("=" * 70)
    print(f"  Provider:    {provider}")
    print(f"  Model:       {model_name}")
    print(f"  Poison rates: {args.poison_rates}")
    print(f"  Seed:        {GLOBAL_SEED}")
    print()

    t_start = time.time()
    methods_to_eval = ["A_no_filter", "B_scalar_0.50", "C_majority_vote", "E_sl_fusion", "F_sl_remove_outlier"]
    all_tier2_results = {}

    for poison_rate in args.poison_rates:
        pr_tag = f"pr{int(poison_rate * 100):02d}"

        # Check for existing checkpoint
        limited_tag = "_limited" if args.limited else ""
        ckpt_path = CHECKPOINT_DIR / f"v1b_tier2_api_{provider}{limited_tag}_{pr_tag}.json"
        if ckpt_path.exists():
            print(f"  Loading Tier 2 API checkpoint: {ckpt_path.name}")
            with open(str(ckpt_path), "r") as f:
                all_tier2_results[pr_tag] = json.load(f)
            continue

        print("=" * 70)
        print(f"  POISON RATE: {poison_rate:.0%}")
        print("=" * 70)

        # Load ALL data from v1b checkpoints (no independent corpus rebuild)
        print(f"  Loading v1b checkpoints for {pr_tag}...")
        try:
            questions, pid_to_text, retrieval_results, extractions = load_v1b_checkpoints(pr_tag)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)

        qid_to_question = {q["id"]: q for q in questions}

        # Subset for limited mode
        if hasattr(args, 'n_questions') and args.n_questions < len(retrieval_results):
            retrieval_results = retrieval_results[:args.n_questions]
            subset_qids = {rr["question_id"] for rr in retrieval_results}
            questions = [q for q in questions if q["id"] in subset_qids]
            extractions = {k: v for k, v in extractions.items() if k in subset_qids}
            qid_to_question = {q["id"]: q for q in questions}

        print(f"  Loaded {len(retrieval_results)} retrieval results, "
              f"{len(pid_to_text)} corpus passages, "
              f"{sum(len(v) for v in extractions.values())} QA extractions")

        results = {}
        for method_name in methods_to_eval:
            print(f"\n  Generating answers for {method_name}...")
            em_scores, f1_scores = [], []
            abstained_count = 0

            for i, rr in enumerate(retrieval_results):
                qid = rr["question_id"]
                q = qid_to_question[qid]
                ex = extractions.get(qid, {})

                if method_name == "A_no_filter":
                    filtered = method_a_no_filter(rr)
                elif method_name.startswith("B_scalar_"):
                    thr = float(method_name.split("_")[-1])
                    filtered = method_b_scalar_threshold(rr, threshold=thr)
                elif method_name == "C_majority_vote":
                    filtered = method_c_majority_vote(rr, extractions=ex)
                elif method_name == "E_sl_fusion":
                    filtered = method_e_sl_answer_fusion(rr, extractions=ex)
                elif method_name == "F_sl_remove_outlier":
                    filtered = method_f_sl_remove_outlier(rr, extractions=ex)
                else:
                    raise ValueError(f"Unknown method: {method_name}")

                if filtered["abstained"]:
                    abstained_count += 1
                    em_scores.append(0.0)
                    f1_scores.append(0.0)
                    continue

                passage_texts = [pid_to_text[p["passage_id"]] for p in filtered["kept_passages"]]
                answer = generate_fn(q["question"], passage_texts)

                em = metric_over_answers(answer, q["answers"], exact_match_score)
                f1 = metric_over_answers(answer, q["answers"], token_f1_score)
                em_scores.append(em)
                f1_scores.append(f1)

                if (i + 1) % 50 == 0:
                    print(f"    [{i+1}/{len(retrieval_results)}] "
                          f"EM={np.mean(em_scores):.3f} F1={np.mean(f1_scores):.3f}")

                time.sleep(delay)

            em_lo, em_mean, em_hi = bootstrap_ci(em_scores, N_BOOTSTRAP, seed=GLOBAL_SEED)
            f1_lo, f1_mean, f1_hi = bootstrap_ci(f1_scores, N_BOOTSTRAP, seed=GLOBAL_SEED)

            results[method_name] = {
                "exact_match": round(em_mean, 4),
                "exact_match_ci": [round(em_lo, 4), round(em_hi, 4)],
                "token_f1": round(f1_mean, 4),
                "token_f1_ci": [round(f1_lo, 4), round(f1_hi, 4)],
                "abstention_rate": round(abstained_count / len(retrieval_results), 4),
                "n_questions": len(retrieval_results),
            }
            print(f"    {method_name}: EM={em_mean:.4f} [{em_lo:.4f}, {em_hi:.4f}] "
                  f"F1={f1_mean:.4f} [{f1_lo:.4f}, {f1_hi:.4f}]")

        # Save per-poison-rate checkpoint
        with open(str(ckpt_path), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved checkpoint: {ckpt_path.name}")

        all_tier2_results[pr_tag] = results

    # ── Summary ──
    total_time = time.time() - t_start

    print("\n" + "=" * 70)
    print(f"SUMMARY — Tier 2 API ({provider}: {model_name})")
    print("=" * 70)

    for pr_tag, tier2 in all_tier2_results.items():
        print(f"\n  {pr_tag}:")
        print(f"    {'Method':<25s} {'EM':>8s} {'EM CI':>20s} {'F1':>8s} {'F1 CI':>20s}")
        print(f"    {'-'*25} {'-'*8} {'-'*20} {'-'*8} {'-'*20}")
        for method, m in tier2.items():
            em_ci = f"[{m['exact_match_ci'][0]:.4f}, {m['exact_match_ci'][1]:.4f}]"
            f1_ci = f"[{m['token_f1_ci'][0]:.4f}, {m['token_f1_ci'][1]:.4f}]"
            print(f"    {method:<25s} {m['exact_match']:>8.4f} {em_ci:>20s} "
                  f"{m['token_f1']:>8.4f} {f1_ci:>20s}")

    print(f"\n  Total wall time: {total_time:.1f}s")

    # ── Save final results ──
    limited_suffix = "_limited" if args.limited else ""
    output_path = RESULTS_DIR / f"en3_1b_tier2_{provider}{limited_suffix}_results.json"
    experiment_result = ExperimentResult(
        experiment_id=f"EN3.1b-Tier2-{provider}",
        parameters={
            "global_seed": GLOBAL_SEED,
            "n_questions": N_QUESTIONS,
            "top_k": TOP_K,
            "poison_rates": args.poison_rates,
            "provider": provider,
            "model": model_name,
            "temperature": 0.0,
            "n_bootstrap": N_BOOTSTRAP,
            "methods_evaluated": methods_to_eval,
            "limited_mode": args.limited,
            "n_questions_per_poison_rate": args.n_questions,
        },
        metrics={
            "total_wall_time_seconds": round(total_time, 4),
            "tier2_full": all_tier2_results,
        },
        raw_data={
            "n_questions_per_method_per_poison_rate": N_QUESTIONS,
            "total_api_calls": len(methods_to_eval) * N_QUESTIONS * len(args.poison_rates),
        },
        environment=env,
        notes=(
            f"EN3.1b Tier 2: API-based LLM answer generation using {provider} "
            f"({model_name}). Evaluates {len(methods_to_eval)} filtering methods "
            f"across {len(args.poison_rates)} poison rates. "
            f"Complements local Qwen2.5-7B results for cross-model validation."
        ),
    )
    experiment_result.save_json(str(output_path))
    print(f"\n  Results saved to: {output_path}")

    ts = experiment_result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    archive_path = RESULTS_DIR / f"en3_1b_tier2_{provider}{limited_suffix}_results_{ts}.json"
    experiment_result.save_json(str(archive_path))
    print(f"  Archived to:      {archive_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
