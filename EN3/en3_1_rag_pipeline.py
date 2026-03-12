"""EN3.1 — RAG Pipeline with Confidence-Aware Retrieval.

Hypothesis
----------
Filtering RAG context by SL confidence and detecting conflict between
retrieved passages reduces the inclusion rate of poisoned passages and
improves downstream answer quality compared to scalar thresholding.

Design
------
Tier 1 (filtering evaluation, no LLM):
    - Build retrieval corpus from SQuAD 1.1 dev passages
    - Inject poisoned passages (answer-entity replacement)
    - Retrieve top-k passages per question
    - Compare three filtering methods on poison-inclusion / gold-retention
Tier 2 (end-to-end, LLM):
    - Feed filtered passages to a local LLM (Qwen2.5-7B-Instruct, 4-bit)
    - Measure Exact Match and token-F1 against gold answers

Ablation: poison rate in {0.05, 0.10, 0.20, 0.30}

Metrics
-------
Tier 1: poison_inclusion_rate, gold_retention_rate, abstention_rate
Tier 2: exact_match, token_f1

References
----------
Josang (2016) Subjective Logic, Springer.
Rajpurkar et al. (2016) SQuAD: 100,000+ Questions for Machine
    Comprehension of Text, EMNLP.

Run
---
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN3/en3_1_rag_pipeline.py            # full run
    python experiments/EN3/en3_1_rag_pipeline.py --tier1     # filtering only
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Project paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_ROOT))

RESULTS_DIR = SCRIPT_DIR / "results"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

from infra.config import set_global_seed, get_global_seed
from infra.env_log import log_environment
from infra.results import ExperimentResult
from infra.stats import bootstrap_ci

# ── jsonld-ex imports ──
from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    conflict_metric,
    pairwise_conflict,
)

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

GLOBAL_SEED = 42
N_QUESTIONS = 500          # subset of SQuAD dev for manageable runtime
TOP_K = 10                 # passages to retrieve per question
EVIDENCE_WEIGHT = 10       # W for Opinion.from_evidence (controls uncertainty)
PRIOR_WEIGHT = 2           # Dirichlet prior weight
POISON_RATES = [0.05, 0.10, 0.20, 0.30]  # fraction of corpus with poisoned passages
SCALAR_THRESHOLDS = [0.3, 0.4, 0.5, 0.6]  # for Method B sweep
SL_CONFLICT_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]  # for Method C sweep
N_BOOTSTRAP = 1000         # bootstrap resamples for CIs
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LLM_MAX_NEW_TOKENS = 64

# ═══════════════════════════════════════════════════════════════════
# Evaluation helpers (standard QA metrics from SQuAD)
# ═══════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens) if prediction_tokens else 0.0
    recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def metric_over_answers(prediction: str, gold_answers: List[str], metric_fn) -> float:
    """Take max metric over all gold answers (standard SQuAD evaluation)."""
    return max(metric_fn(prediction, ans) for ans in gold_answers) if gold_answers else 0.0


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Data Preparation
# ═══════════════════════════════════════════════════════════════════

def _stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def load_and_prepare_squad(
    n_questions: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict[str, Any]]]:
    """Load SQuAD 1.1 dev set and extract questions + passages.

    Returns:
        passages: List of {"id": str, "text": str, "source_question_id": str,
                           "is_gold_for": set, "is_poison": bool}
        questions: List of {"id": str, "question": str, "answers": List[str],
                            "gold_passage_id": str}
    """
    from datasets import load_dataset

    print("  Loading SQuAD 1.1 dev set...")
    ds = load_dataset("rajpurkar/squad", split="validation")

    # Deduplicate passages (SQuAD has many questions per passage)
    passage_map: Dict[str, Dict] = {}  # hash -> passage dict
    all_questions: List[Dict[str, Any]] = []

    for i, example in enumerate(ds):
        ctx = example["context"]
        pid = f"p_{_stable_hash(ctx)}"
        if pid not in passage_map:
            passage_map[pid] = {
                "id": pid,
                "text": ctx,
                "is_gold_for": set(),
                "is_poison": False,
                "poison_of": None,
            }

        answers = example["answers"]["text"]
        qid = f"q_{i}"
        passage_map[pid]["is_gold_for"].add(qid)
        all_questions.append({
            "id": qid,
            "question": example["question"],
            "answers": list(set(answers)),  # deduplicate
            "gold_passage_id": pid,
        })

    # Subsample questions deterministically
    rng = np.random.RandomState(seed)
    if n_questions < len(all_questions):
        indices = rng.choice(len(all_questions), size=n_questions, replace=False)
        indices.sort()
        questions = [all_questions[i] for i in indices]
    else:
        questions = all_questions

    # Collect only passages that are gold for at least one selected question
    selected_qids = {q["id"] for q in questions}
    relevant_pids = set()
    for q in questions:
        relevant_pids.add(q["gold_passage_id"])

    # Also include some "distractor" passages for realistic retrieval
    all_pids = list(passage_map.keys())
    rng.shuffle(all_pids)
    # Add extra passages (up to 2x the number of relevant ones) for realism
    n_distractors = min(len(relevant_pids) * 2, len(all_pids) - len(relevant_pids))
    for pid in all_pids:
        if pid not in relevant_pids:
            relevant_pids.add(pid)
            n_distractors -= 1
            if n_distractors <= 0:
                break

    passages = []
    for pid in relevant_pids:
        p = dict(passage_map[pid])
        p["is_gold_for"] = list(p["is_gold_for"] & selected_qids)
        passages.append(p)

    print(f"  Selected {len(questions)} questions, {len(passages)} base passages")
    return passages, questions


def create_poisoned_passages(
    passages: List[Dict],
    questions: List[Dict],
    poison_rate: float,
    seed: int,
) -> List[Dict]:
    """Create poisoned passages by answer-entity substitution.

    For a fraction of questions (controlled by poison_rate), we take the gold
    passage and replace the answer text with a plausible wrong answer drawn
    from other questions. This creates passages that are topically similar
    but contain wrong information.

    Returns the full corpus (original + poisoned passages).
    """
    rng = np.random.RandomState(seed)

    # Build answer pool for replacement
    all_answers = []
    for q in questions:
        all_answers.extend(q["answers"])
    all_answers = list(set(all_answers))

    # Select which questions get poisoned
    n_poison = max(1, int(len(questions) * poison_rate))
    poison_indices = rng.choice(len(questions), size=n_poison, replace=False)

    pid_to_passage = {p["id"]: p for p in passages}
    new_passages = []

    for idx in poison_indices:
        q = questions[idx]
        gold_pid = q["gold_passage_id"]
        if gold_pid not in pid_to_passage:
            continue

        gold_passage = pid_to_passage[gold_pid]
        gold_text = gold_passage["text"]
        answer = q["answers"][0]  # use first answer for replacement

        # Find a plausible replacement: different answer of similar length
        candidates = [
            a for a in all_answers
            if a.lower() != answer.lower()
            and 0.3 < len(a) / max(len(answer), 1) < 3.0
        ]
        if not candidates:
            candidates = [a for a in all_answers if a.lower() != answer.lower()]
        if not candidates:
            continue

        replacement = rng.choice(candidates)
        poisoned_text = gold_text.replace(answer, replacement, 1)

        # Only add if we actually changed something
        if poisoned_text != gold_text:
            poison_pid = f"poison_{gold_pid}_{_stable_hash(poisoned_text)}"
            new_passages.append({
                "id": poison_pid,
                "text": poisoned_text,
                "is_gold_for": [],
                "is_poison": True,
                "poison_of": gold_pid,
                "original_answer": answer,
                "replacement_answer": replacement,
                "target_question_id": q["id"],
            })

    corpus = list(passages) + new_passages
    n_actual_poison = len(new_passages)
    print(f"  Poison rate={poison_rate:.0%}: added {n_actual_poison} poisoned passages "
          f"(corpus size: {len(corpus)})")
    return corpus


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Embedding and Indexing
# ═══════════════════════════════════════════════════════════════════

def embed_and_index(
    corpus: List[Dict],
    model_name: str,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[Any, np.ndarray]:
    """Embed passages and build a FAISS index.

    Returns (faiss_index, embeddings_matrix).
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    # Check for checkpoint
    if checkpoint_path and checkpoint_path.exists():
        print(f"  Loading embeddings from checkpoint: {checkpoint_path.name}")
        data = np.load(str(checkpoint_path))
        embeddings = data["embeddings"]
    else:
        print(f"  Embedding {len(corpus)} passages with {model_name}...")
        model = SentenceTransformer(model_name)
        texts = [p["text"] for p in corpus]
        embeddings = model.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,  # for cosine similarity via dot product
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        if checkpoint_path:
            np.savez_compressed(str(checkpoint_path), embeddings=embeddings)
            print(f"  Saved embeddings checkpoint: {checkpoint_path.name}")

        # Free model memory
        del model
        gc.collect()

    # Build FAISS index (inner product = cosine similarity for normalized vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, embeddings


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Retrieval
# ═══════════════════════════════════════════════════════════════════

def retrieve_passages(
    questions: List[Dict],
    corpus: List[Dict],
    index: Any,
    model_name: str,
    top_k: int,
    checkpoint_path: Optional[Path] = None,
) -> List[Dict]:
    """Retrieve top-k passages per question.

    Returns list of {"question_id", "retrieved": [{"passage_idx", "score", ...}]}
    """
    from sentence_transformers import SentenceTransformer

    if checkpoint_path and checkpoint_path.exists():
        print(f"  Loading retrieval results from checkpoint: {checkpoint_path.name}")
        with open(str(checkpoint_path), "r") as f:
            return json.load(f)

    print(f"  Encoding {len(questions)} queries...")
    model = SentenceTransformer(model_name)
    query_texts = [q["question"] for q in questions]
    query_embeddings = model.encode(
        query_texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    query_embeddings = np.array(query_embeddings, dtype=np.float32)

    del model
    gc.collect()

    print(f"  Retrieving top-{top_k} passages per question...")
    scores, indices = index.search(query_embeddings, top_k)

    retrieval_results = []
    for i, q in enumerate(questions):
        retrieved = []
        for rank in range(top_k):
            cidx = int(indices[i, rank])
            if cidx < 0:
                continue  # FAISS returns -1 if fewer than top_k results
            p = corpus[cidx]
            retrieved.append({
                "passage_idx": cidx,
                "passage_id": p["id"],
                "score": float(scores[i, rank]),  # cosine similarity
                "is_poison": p.get("is_poison", False),
                "is_gold": q["id"] in p.get("is_gold_for", []),
            })
        retrieval_results.append({
            "question_id": q["id"],
            "retrieved": retrieved,
        })

    if checkpoint_path:
        with open(str(checkpoint_path), "w") as f:
            json.dump(retrieval_results, f)
        print(f"  Saved retrieval checkpoint: {checkpoint_path.name}")

    return retrieval_results


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Filtering Methods
# ═══════════════════════════════════════════════════════════════════

def cosine_to_opinion(
    cosine_sim: float,
    evidence_weight: float = EVIDENCE_WEIGHT,
    prior_weight: float = PRIOR_WEIGHT,
) -> Opinion:
    """Convert cosine similarity to an SL opinion with meaningful uncertainty.

    Rather than using Opinion.from_confidence (which yields dogmatic opinions
    with near-zero uncertainty), we treat cosine similarity as evidence
    strength and construct an opinion that reflects our actual evidential
    basis. This preserves the epistemic information that SL needs for
    conflict detection and principled fusion.

    Parameters
    ----------
    cosine_sim : float
        Cosine similarity in [0, 1] (after normalization).
    evidence_weight : float
        Total evidence weight W.  Higher values yield lower uncertainty.
    prior_weight : float
        Dirichlet prior weight (default 2).
    """
    # Clamp to [0, 1] (sentence-transformers normalized embeddings are >= 0)
    sim = max(0.0, min(1.0, cosine_sim))
    pos_evidence = sim * evidence_weight
    neg_evidence = (1.0 - sim) * evidence_weight
    return Opinion.from_evidence(pos_evidence, neg_evidence, prior_weight=prior_weight)


def method_a_no_filter(retrieval: Dict) -> Dict:
    """Baseline A: use all retrieved passages, no filtering."""
    return {
        "method": "A_no_filter",
        "kept_indices": [r["passage_idx"] for r in retrieval["retrieved"]],
        "kept_passages": retrieval["retrieved"],
        "abstained": False,
    }


def method_b_scalar_threshold(retrieval: Dict, threshold: float) -> Dict:
    """Baseline B: drop passages below scalar cosine similarity threshold."""
    kept = [r for r in retrieval["retrieved"] if r["score"] >= threshold]
    return {
        "method": f"B_scalar_{threshold:.2f}",
        "kept_indices": [r["passage_idx"] for r in kept],
        "kept_passages": kept,
        "abstained": len(kept) == 0,
    }


def method_c_sl_fusion(
    retrieval: Dict,
    conflict_threshold: float,
    evidence_weight: float = EVIDENCE_WEIGHT,
    prior_weight: float = PRIOR_WEIGHT,
) -> Dict:
    """Method C: SL confidence fusion + conflict detection.

    1. Convert each passage's cosine similarity to an SL opinion.
    2. Compute pairwise conflict between passages.
    3. Fuse passage opinions via cumulative fusion.
    4. If the fused opinion's conflict exceeds the threshold, flag conflict.
    5. When conflict is detected, keep only passages whose individual
       projected probability exceeds the fused projected probability
       (i.e., keep the "consensus" passages, discard outliers).
    6. Abstain if fewer than 1 passage remains after filtering.
    """
    passages = retrieval["retrieved"]
    if not passages:
        return {
            "method": f"C_sl_{conflict_threshold:.2f}",
            "kept_indices": [],
            "kept_passages": [],
            "abstained": True,
            "conflict_detected": False,
            "fused_opinion": None,
        }

    # Step 1: Convert to SL opinions
    opinions = []
    for p in passages:
        op = cosine_to_opinion(p["score"], evidence_weight, prior_weight)
        opinions.append(op)

    # Step 2: Pairwise conflict
    max_conflict = 0.0
    if len(opinions) >= 2:
        for i in range(len(opinions)):
            for j in range(i + 1, len(opinions)):
                c = pairwise_conflict(opinions[i], opinions[j])
                max_conflict = max(max_conflict, c)

    # Step 3: Fuse all passage opinions
    if len(opinions) == 1:
        fused = opinions[0]
    else:
        fused = cumulative_fuse(*opinions)

    fused_proj = fused.projected_probability()

    # Step 4: Check conflict
    conflict_detected = max_conflict > conflict_threshold

    if not conflict_detected:
        # No significant conflict — keep all passages (like Method A)
        return {
            "method": f"C_sl_{conflict_threshold:.2f}",
            "kept_indices": [r["passage_idx"] for r in passages],
            "kept_passages": passages,
            "abstained": False,
            "conflict_detected": False,
            "fused_opinion": {
                "b": fused.belief, "d": fused.disbelief,
                "u": fused.uncertainty, "a": fused.base_rate,
                "P": fused_proj,
            },
            "max_pairwise_conflict": max_conflict,
        }

    # Step 5: Conflict detected — filter outliers
    # Keep passages whose opinion's projected probability is above median
    proj_probs = [op.projected_probability() for op in opinions]
    median_proj = float(np.median(proj_probs))

    kept_passages = []
    kept_indices = []
    for p, op in zip(passages, opinions):
        if op.projected_probability() >= median_proj:
            kept_passages.append(p)
            kept_indices.append(p["passage_idx"])

    return {
        "method": f"C_sl_{conflict_threshold:.2f}",
        "kept_indices": kept_indices,
        "kept_passages": kept_passages,
        "abstained": len(kept_passages) == 0,
        "conflict_detected": True,
        "fused_opinion": {
            "b": fused.belief, "d": fused.disbelief,
            "u": fused.uncertainty, "a": fused.base_rate,
            "P": fused_proj,
        },
        "max_pairwise_conflict": max_conflict,
    }


# ═══════════════════════════════════════════════════════════════════
# Tier 1 Evaluation: Filtering Quality (no LLM)
# ═══════════════════════════════════════════════════════════════════

def evaluate_filtering(
    retrieval_results: List[Dict],
    questions: List[Dict],
    corpus: List[Dict],
    poison_rate: float,
) -> Dict[str, Any]:
    """Evaluate all filtering methods on poison-inclusion and gold-retention.

    Returns a dict of method_name -> {metrics}.
    """
    method_configs = []

    # Method A: no filter
    method_configs.append(("A_no_filter", lambda r: method_a_no_filter(r)))

    # Method B: scalar threshold sweep
    for thr in SCALAR_THRESHOLDS:
        method_configs.append(
            (f"B_scalar_{thr:.2f}", lambda r, t=thr: method_b_scalar_threshold(r, t))
        )

    # Method C: SL fusion + conflict detection sweep
    for thr in SL_CONFLICT_THRESHOLDS:
        method_configs.append(
            (f"C_sl_{thr:.2f}", lambda r, t=thr: method_c_sl_fusion(r, t))
        )

    qid_to_question = {q["id"]: q for q in questions}
    results = {}

    for method_name, method_fn in method_configs:
        poison_included = []    # per-question: was any poison passage kept?
        gold_retained = []      # per-question: was the gold passage kept?
        abstained = []          # per-question: did the method abstain?
        n_kept_list = []        # per-question: how many passages kept?

        for rr in retrieval_results:
            qid = rr["question_id"]
            q = qid_to_question[qid]
            gold_pid = q["gold_passage_id"]

            filtered = method_fn(rr)
            kept_pids = set()
            has_poison = False
            has_gold = False

            for p in filtered["kept_passages"]:
                kept_pids.add(p["passage_id"])
                if p["is_poison"]:
                    has_poison = True
                if p["is_gold"]:
                    has_gold = True

            poison_included.append(float(has_poison))
            gold_retained.append(float(has_gold))
            abstained.append(float(filtered["abstained"]))
            n_kept_list.append(len(filtered["kept_passages"]))

        # Compute metrics with bootstrap CIs
        pi_lo, pi_mean, pi_hi = bootstrap_ci(poison_included, N_BOOTSTRAP, seed=GLOBAL_SEED)
        gr_lo, gr_mean, gr_hi = bootstrap_ci(gold_retained, N_BOOTSTRAP, seed=GLOBAL_SEED)
        ab_lo, ab_mean, ab_hi = bootstrap_ci(abstained, N_BOOTSTRAP, seed=GLOBAL_SEED)

        results[method_name] = {
            "poison_inclusion_rate": round(pi_mean, 4),
            "poison_inclusion_ci": [round(pi_lo, 4), round(pi_hi, 4)],
            "gold_retention_rate": round(gr_mean, 4),
            "gold_retention_ci": [round(gr_lo, 4), round(gr_hi, 4)],
            "abstention_rate": round(ab_mean, 4),
            "abstention_ci": [round(ab_lo, 4), round(ab_hi, 4)],
            "mean_passages_kept": round(float(np.mean(n_kept_list)), 2),
            "n_questions": len(retrieval_results),
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# Phase 5: LLM Generation (Tier 2)
# ═══════════════════════════════════════════════════════════════════

def load_llm(model_name: str):
    """Load a quantized LLM for answer generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"  Loading LLM: {model_name} (4-bit quantized)...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  LLM loaded. Device map: {model.hf_device_map}")
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    question: str,
    passages: List[str],
    max_new_tokens: int = LLM_MAX_NEW_TOKENS,
) -> str:
    """Generate an answer given a question and context passages."""
    import torch

    if not passages:
        return ""

    # Build prompt
    context = "\n\n".join(f"[Passage {i+1}] {p}" for i, p in enumerate(passages))
    prompt = (
        f"Answer the following question based ONLY on the provided passages. "
        f"Give a short, direct answer. If the passages do not contain enough "
        f"information, say 'unanswerable'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # greedy for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    # Take first line / sentence as the answer
    answer = answer.split("\n")[0].strip()
    return answer


def evaluate_tier2(
    retrieval_results: List[Dict],
    questions: List[Dict],
    corpus: List[Dict],
    model,
    tokenizer,
    methods_to_eval: List[str],
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Tier 2: generate answers with LLM and evaluate EM/F1.

    Only evaluates the specified methods (to avoid redundant LLM calls).
    """
    # Load checkpoint if available
    if checkpoint_path and checkpoint_path.exists():
        print(f"  Loading Tier 2 checkpoint: {checkpoint_path.name}")
        with open(str(checkpoint_path), "r") as f:
            return json.load(f)

    qid_to_question = {q["id"]: q for q in questions}
    pid_to_text = {p["id"]: p["text"] for p in corpus}

    results = {}

    for method_name in methods_to_eval:
        print(f"  Generating answers for {method_name}...")
        em_scores = []
        f1_scores = []
        abstained_count = 0

        for i, rr in enumerate(retrieval_results):
            qid = rr["question_id"]
            q = qid_to_question[qid]

            # Apply filtering method
            if method_name == "A_no_filter":
                filtered = method_a_no_filter(rr)
            elif method_name.startswith("B_scalar_"):
                thr = float(method_name.split("_")[-1])
                filtered = method_b_scalar_threshold(rr, thr)
            elif method_name.startswith("C_sl_"):
                thr = float(method_name.split("_")[-1])
                filtered = method_c_sl_fusion(rr, thr)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            if filtered["abstained"]:
                abstained_count += 1
                em_scores.append(0.0)
                f1_scores.append(0.0)
                continue

            # Get passage texts
            passage_texts = [pid_to_text[p["passage_id"]] for p in filtered["kept_passages"]]

            # Generate answer
            answer = generate_answer(model, tokenizer, q["question"], passage_texts)

            # Evaluate
            em = metric_over_answers(answer, q["answers"], exact_match_score)
            f1 = metric_over_answers(answer, q["answers"], token_f1_score)
            em_scores.append(em)
            f1_scores.append(f1)

            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(retrieval_results)}] "
                      f"EM={np.mean(em_scores):.3f} F1={np.mean(f1_scores):.3f}")

        # Bootstrap CIs
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

    # Save checkpoint
    if checkpoint_path:
        with open(str(checkpoint_path), "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved Tier 2 checkpoint: {checkpoint_path.name}")

    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EN3.1 — RAG Pipeline")
    parser.add_argument("--tier1", action="store_true",
                        help="Run only Tier 1 (filtering evaluation, no LLM)")
    parser.add_argument("--poison-rates", type=float, nargs="+",
                        default=POISON_RATES,
                        help="Poison rates to sweep")
    parser.add_argument("--n-questions", type=int, default=N_QUESTIONS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    set_global_seed(GLOBAL_SEED)
    env = log_environment()

    run_tier2 = not args.tier1

    print("=" * 70)
    print("EN3.1 — RAG Pipeline with Confidence-Aware Retrieval")
    print("=" * 70)
    print(f"  Mode:          {'Tier 1 + Tier 2' if run_tier2 else 'Tier 1 only'}")
    print(f"  Questions:     {args.n_questions}")
    print(f"  Top-K:         {args.top_k}")
    print(f"  Poison rates:  {args.poison_rates}")
    print(f"  Seed:          {GLOBAL_SEED}")
    print()

    t_start = time.time()

    # ── Phase 1: Load data ──
    print("Phase 1: Data Preparation")
    print("-" * 40)
    base_passages, questions = load_and_prepare_squad(args.n_questions, GLOBAL_SEED)
    print()

    # ── Run for each poison rate ──
    all_tier1_results = {}
    all_tier2_results = {}

    for poison_rate in args.poison_rates:
        pr_tag = f"pr{int(poison_rate * 100):02d}"
        print("=" * 70)
        print(f"  POISON RATE: {poison_rate:.0%}")
        print("=" * 70)

        # ── Phase 1b: Create poisoned corpus ──
        corpus = create_poisoned_passages(
            base_passages, questions, poison_rate,
            seed=GLOBAL_SEED + int(poison_rate * 1000),
        )

        # ── Phase 2: Embed and index ──
        print(f"\nPhase 2: Embedding and Indexing ({pr_tag})")
        print("-" * 40)
        emb_ckpt = CHECKPOINT_DIR / f"embeddings_{pr_tag}.npz"
        index, embeddings = embed_and_index(corpus, EMBEDDING_MODEL, emb_ckpt)
        print()

        # ── Phase 3: Retrieve ──
        print(f"Phase 3: Retrieval ({pr_tag})")
        print("-" * 40)
        retr_ckpt = CHECKPOINT_DIR / f"retrieval_{pr_tag}.json"
        retrieval_results = retrieve_passages(
            questions, corpus, index, EMBEDDING_MODEL, args.top_k, retr_ckpt
        )

        # Quick retrieval stats
        n_with_gold = sum(
            1 for rr in retrieval_results
            if any(p["is_gold"] for p in rr["retrieved"])
        )
        n_with_poison = sum(
            1 for rr in retrieval_results
            if any(p["is_poison"] for p in rr["retrieved"])
        )
        print(f"  Retrieval stats: {n_with_gold}/{len(retrieval_results)} have gold in top-{args.top_k}, "
              f"{n_with_poison}/{len(retrieval_results)} have poison in top-{args.top_k}")
        print()

        # ── Phase 4 + Tier 1 eval ──
        print(f"Phase 4: Tier 1 Filtering Evaluation ({pr_tag})")
        print("-" * 40)
        tier1 = evaluate_filtering(retrieval_results, questions, corpus, poison_rate)
        all_tier1_results[pr_tag] = tier1

        # Print Tier 1 results
        print(f"\n  {'Method':<25s} {'PoisonIncl':>12s} {'GoldRetain':>12s} {'Abstain':>10s} {'Kept':>6s}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*6}")
        for method, m in tier1.items():
            print(f"  {method:<25s} {m['poison_inclusion_rate']:>12.4f} "
                  f"{m['gold_retention_rate']:>12.4f} {m['abstention_rate']:>10.4f} "
                  f"{m['mean_passages_kept']:>6.1f}")
        print()

        # Free memory
        del index, embeddings
        gc.collect()

        # ── Tier 2: LLM Generation ──
        if run_tier2:
            print(f"Phase 5: Tier 2 LLM Evaluation ({pr_tag})")
            print("-" * 40)

            # Select representative methods for Tier 2 (avoid running all combos)
            # Best scalar threshold, best SL threshold, and no-filter baseline
            best_scalar = min(
                [k for k in tier1 if k.startswith("B_scalar_")],
                key=lambda k: tier1[k]["poison_inclusion_rate"],
            )
            best_sl = min(
                [k for k in tier1 if k.startswith("C_sl_")],
                key=lambda k: tier1[k]["poison_inclusion_rate"],
            )
            tier2_methods = ["A_no_filter", best_scalar, best_sl]

            # Load LLM (only once across poison rates — reload if first time)
            if poison_rate == args.poison_rates[0]:
                model, tokenizer = load_llm(LLM_MODEL)

            tier2_ckpt = CHECKPOINT_DIR / f"tier2_{pr_tag}.json"
            tier2 = evaluate_tier2(
                retrieval_results, questions, corpus,
                model, tokenizer, tier2_methods, tier2_ckpt,
            )
            all_tier2_results[pr_tag] = tier2
            print()

    # ── Summary ──
    total_time = time.time() - t_start

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Tier 1 summary: best method per poison rate
    print("\nTier 1 — Filtering Quality (lower poison inclusion = better)")
    print(f"  {'Poison Rate':<15s} {'Best Scalar':<25s} {'Best SL':<25s} {'SL Advantage':<15s}")
    print(f"  {'-'*15} {'-'*25} {'-'*25} {'-'*15}")

    tier1_summary = {}
    for pr_tag in all_tier1_results:
        tier1 = all_tier1_results[pr_tag]
        scalar_methods = {k: v for k, v in tier1.items() if k.startswith("B_scalar_")}
        sl_methods = {k: v for k, v in tier1.items() if k.startswith("C_sl_")}

        # Best = lowest poison inclusion that retains gold >= 0.5
        def score(m):
            return m["poison_inclusion_rate"] if m["gold_retention_rate"] >= 0.5 else 999

        best_scalar_name = min(scalar_methods, key=lambda k: score(scalar_methods[k]))
        best_sl_name = min(sl_methods, key=lambda k: score(sl_methods[k]))
        bs = scalar_methods[best_scalar_name]
        bsl = sl_methods[best_sl_name]

        advantage = bs["poison_inclusion_rate"] - bsl["poison_inclusion_rate"]
        tier1_summary[pr_tag] = {
            "best_scalar": best_scalar_name,
            "best_scalar_poison_incl": bs["poison_inclusion_rate"],
            "best_sl": best_sl_name,
            "best_sl_poison_incl": bsl["poison_inclusion_rate"],
            "sl_advantage": round(advantage, 4),
        }

        print(f"  {pr_tag:<15s} "
              f"{best_scalar_name} ({bs['poison_inclusion_rate']:.4f})  "
              f"{best_sl_name} ({bsl['poison_inclusion_rate']:.4f})  "
              f"{'+' if advantage > 0 else ''}{advantage:.4f}")

    if all_tier2_results:
        print("\nTier 2 — End-to-End Answer Quality")
        for pr_tag, tier2 in all_tier2_results.items():
            print(f"\n  {pr_tag}:")
            print(f"    {'Method':<25s} {'EM':>8s} {'F1':>8s}")
            print(f"    {'-'*25} {'-'*8} {'-'*8}")
            for method, m in tier2.items():
                print(f"    {method:<25s} {m['exact_match']:>8.4f} {m['token_f1']:>8.4f}")

    print(f"\n  Total wall time: {total_time:.1f}s")

    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "en3_1_results.json"

    experiment_result = ExperimentResult(
        experiment_id="EN3.1",
        parameters={
            "global_seed": GLOBAL_SEED,
            "n_questions": args.n_questions,
            "top_k": args.top_k,
            "evidence_weight": EVIDENCE_WEIGHT,
            "prior_weight": PRIOR_WEIGHT,
            "poison_rates": args.poison_rates,
            "scalar_thresholds": SCALAR_THRESHOLDS,
            "sl_conflict_thresholds": SL_CONFLICT_THRESHOLDS,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL if run_tier2 else "N/A (Tier 1 only)",
            "n_bootstrap": N_BOOTSTRAP,
            "tier2_run": run_tier2,
        },
        metrics={
            "total_wall_time_seconds": round(total_time, 4),
            "tier1_summary": tier1_summary,
            "tier1_full": all_tier1_results,
            "tier2_full": all_tier2_results if run_tier2 else None,
        },
        raw_data={
            "n_base_passages": len(base_passages),
            "n_questions_selected": len(questions),
        },
        environment=env,
        notes=(
            f"EN3.1: RAG pipeline with confidence-aware retrieval. "
            f"{len(args.poison_rates)} poison rates x "
            f"{len(SCALAR_THRESHOLDS)} scalar thresholds x "
            f"{len(SL_CONFLICT_THRESHOLDS)} SL conflict thresholds. "
            f"{'Tier 1 + Tier 2' if run_tier2 else 'Tier 1 only'}. "
            f"Dataset: SQuAD 1.1 dev ({args.n_questions} questions). "
            f"Embedding: {EMBEDDING_MODEL}."
        ),
    )
    experiment_result.save_json(str(output_path))
    print(f"\n  Results saved to: {output_path}")

    # Timestamped archive
    ts = experiment_result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    archive_path = RESULTS_DIR / f"en3_1_results_{ts}.json"
    experiment_result.save_json(str(archive_path))
    print(f"  Archived to:      {archive_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
