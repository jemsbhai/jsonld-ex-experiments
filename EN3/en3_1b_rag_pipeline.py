"""EN3.1b — Improved RAG Pipeline with Answer-Agreement-Based SL Filtering.

Motivation
----------
EN3.1 (v1) showed that SL conflict detection on cosine-similarity-derived
opinions does NOT outperform scalar thresholding. The root cause: poisoned
passages have nearly identical cosine scores to gold passages (same topic,
swapped answer entity), so relevance-based opinions cannot distinguish them.
SL conflict detection was picking up trivial score spread, not factual
disagreement.

This improved version tests the RIGHT hypothesis: SL's value in RAG is
detecting **answer-level conflict between passages**, not filtering by
relevance score.

Design
------
For each question, we:
  1. Retrieve top-k passages (same as v1)
  2. Extract a candidate answer from each passage using a lightweight
     extractive QA model (distilbert-base-cased-distilled-squad)
  3. Group passages by answer agreement (fuzzy matching)
  4. Apply filtering methods that reason about answer agreement

Methods
-------
  A  — No filter (baseline, all top-k passages)
  B  — Scalar cosine threshold (sweep)
  C  — Majority-vote answer filter (keep passages agreeing with plurality
       answer; non-SL baseline)
  D  — Score-weighted answer vote (weight each passage's answer vote by
       cosine similarity; non-SL baseline)
  E  — SL answer-agreement fusion (the core contribution):
       - For each answer group, construct an SL opinion from evidence
         (group size, extraction confidence, cosine similarity)
       - Use cumulative_fuse within groups, pairwise_conflict between groups
       - If conflict detected: keep only the highest-evidence group
       - SL advantage: formal uncertainty tracking + conflict detection

Ablation: poison rate in {0.05, 0.10, 0.20, 0.30}

Metrics
-------
  poison_inclusion_rate, gold_retention_rate, abstention_rate

References
----------
Josang (2016) Subjective Logic, Springer.
Rajpurkar et al. (2016) SQuAD, EMNLP.
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
from difflib import SequenceMatcher
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
N_QUESTIONS = 500
TOP_K = 10
EVIDENCE_WEIGHT = 10
PRIOR_WEIGHT = 2
POISON_RATES = [0.05, 0.10, 0.20, 0.30]
SCALAR_THRESHOLDS = [0.3, 0.4, 0.5, 0.6]
N_BOOTSTRAP = 1000
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QA_MODEL = "distilbert/distilbert-base-cased-distilled-squad"
ANSWER_SIMILARITY_THRESHOLD = 0.6   # fuzzy match threshold for answer grouping
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LLM_MAX_NEW_TOKENS = 64

# ═══════════════════════════════════════════════════════════════════
# QA evaluation helpers (from SQuAD)
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


def answers_match(a: str, b: str, threshold: float = ANSWER_SIMILARITY_THRESHOLD) -> bool:
    """Check if two answers are similar enough to be considered the same.

    Uses both exact normalized match and fuzzy SequenceMatcher ratio.
    """
    na, nb = normalize_answer(a), normalize_answer(b)
    if na == nb:
        return True
    if not na or not nb:
        return False
    # Substring containment (handles "Denver" vs "Denver, Colorado")
    if na in nb or nb in na:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


# ═══════════════════════════════════════════════════════════════════
# Data loading (reuses v1 checkpoints where possible)
# ═══════════════════════════════════════════════════════════════════

def _stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def load_and_prepare_squad(n_questions: int, seed: int):
    """Load SQuAD 1.1 dev set. Identical to v1."""
    from datasets import load_dataset

    print("  Loading SQuAD 1.1 dev set...")
    ds = load_dataset("rajpurkar/squad", split="validation")

    passage_map: Dict[str, Dict] = {}
    all_questions: List[Dict[str, Any]] = []

    for i, example in enumerate(ds):
        ctx = example["context"]
        pid = f"p_{_stable_hash(ctx)}"
        if pid not in passage_map:
            passage_map[pid] = {
                "id": pid, "text": ctx,
                "is_gold_for": set(), "is_poison": False, "poison_of": None,
            }
        answers = example["answers"]["text"]
        qid = f"q_{i}"
        passage_map[pid]["is_gold_for"].add(qid)
        all_questions.append({
            "id": qid, "question": example["question"],
            "answers": sorted(set(answers)), "gold_passage_id": pid,
        })

    rng = np.random.RandomState(seed)
    if n_questions < len(all_questions):
        indices = rng.choice(len(all_questions), size=n_questions, replace=False)
        indices.sort()
        questions = [all_questions[i] for i in indices]
    else:
        questions = all_questions

    selected_qids = {q["id"] for q in questions}
    relevant_pids = {q["gold_passage_id"] for q in questions}

    all_pids = list(passage_map.keys())
    rng.shuffle(all_pids)
    n_distractors = min(len(relevant_pids) * 2, len(all_pids) - len(relevant_pids))
    for pid in all_pids:
        if pid not in relevant_pids:
            relevant_pids.add(pid)
            n_distractors -= 1
            if n_distractors <= 0:
                break

    passages = []
    for pid in sorted(relevant_pids):
        p = dict(passage_map[pid])
        p["is_gold_for"] = list(p["is_gold_for"] & selected_qids)
        passages.append(p)

    print(f"  Selected {len(questions)} questions, {len(passages)} base passages")
    return passages, questions


def create_poisoned_passages(passages, questions, poison_rate, seed):
    """Identical to v1."""
    rng = np.random.RandomState(seed)
    all_answers = sorted({a for q in questions for a in q["answers"]})

    n_poison = max(1, int(len(questions) * poison_rate))
    poison_indices = rng.choice(len(questions), size=n_poison, replace=False)
    pid_to_passage = {p["id"]: p for p in passages}
    new_passages = []

    for idx in poison_indices:
        q = questions[idx]
        gold_pid = q["gold_passage_id"]
        if gold_pid not in pid_to_passage:
            continue
        gold_text = pid_to_passage[gold_pid]["text"]
        answer = q["answers"][0]

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
        if poisoned_text != gold_text:
            poison_pid = f"poison_{gold_pid}_{_stable_hash(poisoned_text)}"
            new_passages.append({
                "id": poison_pid, "text": poisoned_text,
                "is_gold_for": [], "is_poison": True,
                "poison_of": gold_pid,
                "original_answer": answer,
                "replacement_answer": replacement,
                "target_question_id": q["id"],
            })

    corpus = list(passages) + new_passages
    print(f"  Poison rate={poison_rate:.0%}: added {len(new_passages)} poisoned "
          f"passages (corpus size: {len(corpus)})")
    return corpus


def embed_and_index(corpus, model_name, checkpoint_path=None):
    """Identical to v1."""
    import faiss
    from sentence_transformers import SentenceTransformer

    if checkpoint_path and checkpoint_path.exists():
        print(f"  Loading embeddings from checkpoint: {checkpoint_path.name}")
        embeddings = np.load(str(checkpoint_path))["embeddings"]
    else:
        print(f"  Embedding {len(corpus)} passages with {model_name}...")
        model = SentenceTransformer(model_name)
        texts = [p["text"] for p in corpus]
        embeddings = model.encode(
            texts, batch_size=128, show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        if checkpoint_path:
            np.savez_compressed(str(checkpoint_path), embeddings=embeddings)
            print(f"  Saved embeddings checkpoint: {checkpoint_path.name}")
        del model; gc.collect()

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, embeddings


def retrieve_passages(questions, corpus, index, model_name, top_k, checkpoint_path=None):
    """Identical to v1."""
    from sentence_transformers import SentenceTransformer

    if checkpoint_path and checkpoint_path.exists():
        print(f"  Loading retrieval from checkpoint: {checkpoint_path.name}")
        with open(str(checkpoint_path), "r") as f:
            return json.load(f)

    print(f"  Encoding {len(questions)} queries...")
    model = SentenceTransformer(model_name)
    query_embeddings = model.encode(
        [q["question"] for q in questions],
        batch_size=128, show_progress_bar=True, normalize_embeddings=True,
    )
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    del model; gc.collect()

    print(f"  Retrieving top-{top_k}...")
    scores, indices = index.search(query_embeddings, top_k)

    retrieval_results = []
    for i, q in enumerate(questions):
        retrieved = []
        for rank in range(top_k):
            cidx = int(indices[i, rank])
            if cidx < 0:
                continue
            p = corpus[cidx]
            retrieved.append({
                "passage_idx": cidx, "passage_id": p["id"],
                "score": float(scores[i, rank]),
                "is_poison": p.get("is_poison", False),
                "is_gold": q["id"] in p.get("is_gold_for", []),
            })
        retrieval_results.append({"question_id": q["id"], "retrieved": retrieved})

    if checkpoint_path:
        with open(str(checkpoint_path), "w") as f:
            json.dump(retrieval_results, f)
        print(f"  Saved retrieval checkpoint: {checkpoint_path.name}")
    return retrieval_results


# ═══════════════════════════════════════════════════════════════════
# Phase: Answer Extraction
# ═══════════════════════════════════════════════════════════════════

def extract_answers_for_retrieval(
    questions: List[Dict],
    retrieval_results: List[Dict],
    corpus: List[Dict],
    qa_model_name: str,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Dict]]:
    """Extract candidate answers from each retrieved passage.

    Returns:
        {question_id: {passage_id: {"answer": str, "qa_score": float}}}
    """
    if checkpoint_path and checkpoint_path.exists():
        print(f"  Loading QA extraction from checkpoint: {checkpoint_path.name}")
        with open(str(checkpoint_path), "r") as f:
            return json.load(f)

    from transformers import pipeline
    import torch

    print(f"  Loading QA model: {qa_model_name}...")
    qa_pipe = pipeline(
        "question-answering",
        model=qa_model_name,
        device=0 if torch.cuda.is_available() else -1,
    )

    pid_to_text = {p["id"]: p["text"] for p in corpus}
    qid_to_question = {q["id"]: q["question"] for q in questions}

    extractions: Dict[str, Dict[str, Dict]] = {}
    total = sum(len(rr["retrieved"]) for rr in retrieval_results)
    done = 0

    for rr in retrieval_results:
        qid = rr["question_id"]
        question_text = qid_to_question[qid]
        extractions[qid] = {}

        for p in rr["retrieved"]:
            pid = p["passage_id"]
            passage_text = pid_to_text[pid]

            try:
                result = qa_pipe(
                    question=question_text,
                    context=passage_text,
                    max_answer_len=50,
                )
                extractions[qid][pid] = {
                    "answer": result["answer"],
                    "qa_score": float(result["score"]),
                }
            except Exception:
                extractions[qid][pid] = {
                    "answer": "",
                    "qa_score": 0.0,
                }

            done += 1
            if done % 500 == 0:
                print(f"    Extracted {done}/{total} answers...")

    print(f"  Extracted {done} answers total.")

    if checkpoint_path:
        with open(str(checkpoint_path), "w") as f:
            json.dump(extractions, f)
        print(f"  Saved QA extraction checkpoint: {checkpoint_path.name}")

    del qa_pipe; gc.collect()
    return extractions


# ═══════════════════════════════════════════════════════════════════
# Answer Grouping
# ═══════════════════════════════════════════════════════════════════

def group_passages_by_answer(
    passages: List[Dict],
    extractions: Dict[str, Dict],
) -> List[List[Dict]]:
    """Group passages by answer agreement using fuzzy matching.

    Each passage dict is augmented with 'extracted_answer' and 'qa_score'.
    Returns list of groups, where each group is a list of augmented passages.
    """
    augmented = []
    for p in passages:
        pid = p["passage_id"]
        ext = extractions.get(pid, {"answer": "", "qa_score": 0.0})
        augmented.append({
            **p,
            "extracted_answer": ext["answer"],
            "qa_score": ext["qa_score"],
        })

    # Greedy grouping: assign each passage to the first matching group
    groups: List[List[Dict]] = []
    for ap in augmented:
        placed = False
        for group in groups:
            # Compare against the first passage in the group (representative)
            if answers_match(ap["extracted_answer"], group[0]["extracted_answer"]):
                group.append(ap)
                placed = True
                break
        if not placed:
            groups.append([ap])

    return groups


# ═══════════════════════════════════════════════════════════════════
# Filtering Methods
# ═══════════════════════════════════════════════════════════════════

def method_a_no_filter(retrieval: Dict, **kwargs) -> Dict:
    """Baseline A: no filtering."""
    return {
        "method": "A_no_filter",
        "kept_passages": retrieval["retrieved"],
        "abstained": False,
    }


def method_b_scalar_threshold(retrieval: Dict, threshold: float, **kwargs) -> Dict:
    """Baseline B: scalar cosine similarity threshold."""
    kept = [r for r in retrieval["retrieved"] if r["score"] >= threshold]
    return {
        "method": f"B_scalar_{threshold:.2f}",
        "kept_passages": kept,
        "abstained": len(kept) == 0,
    }


def method_c_majority_vote(
    retrieval: Dict,
    extractions: Dict[str, Dict],
    **kwargs,
) -> Dict:
    """Method C: majority-vote answer filtering (non-SL baseline).

    Extract answers, find the plurality answer, keep only passages
    that agree with it.
    """
    passages = retrieval["retrieved"]
    if not passages:
        return {"method": "C_majority_vote", "kept_passages": [], "abstained": True}

    groups = group_passages_by_answer(passages, extractions)

    # Plurality = largest group
    groups.sort(key=len, reverse=True)
    majority_group = groups[0]

    kept_pids = {p["passage_id"] for p in majority_group}
    kept = [p for p in passages if p["passage_id"] in kept_pids]

    return {
        "method": "C_majority_vote",
        "kept_passages": kept,
        "abstained": len(kept) == 0,
        "n_answer_groups": len(groups),
        "majority_size": len(majority_group),
    }


def method_d_score_weighted_vote(
    retrieval: Dict,
    extractions: Dict[str, Dict],
    **kwargs,
) -> Dict:
    """Method D: score-weighted answer voting (non-SL baseline).

    Like majority vote but weights each passage's vote by cosine similarity.
    """
    passages = retrieval["retrieved"]
    if not passages:
        return {"method": "D_weighted_vote", "kept_passages": [], "abstained": True}

    groups = group_passages_by_answer(passages, extractions)

    # Weighted score per group
    group_scores = []
    for group in groups:
        total_score = sum(p["score"] for p in group)
        group_scores.append((total_score, group))

    group_scores.sort(key=lambda x: x[0], reverse=True)
    best_group = group_scores[0][1]

    kept_pids = {p["passage_id"] for p in best_group}
    kept = [p for p in passages if p["passage_id"] in kept_pids]

    return {
        "method": "D_weighted_vote",
        "kept_passages": kept,
        "abstained": len(kept) == 0,
        "n_answer_groups": len(groups),
        "best_group_size": len(best_group),
    }


def method_e_sl_answer_fusion(
    retrieval: Dict,
    extractions: Dict[str, Dict],
    **kwargs,
) -> Dict:
    """Method E: SL answer-agreement fusion (core contribution).

    For each answer group:
      1. Construct an SL opinion from evidence within the group:
         - Each passage contributes evidence proportional to both its
           cosine similarity (retrieval relevance) and QA extraction
           confidence (answer extraction quality).
      2. Fuse passage opinions within each group via cumulative_fuse.

    Then between groups:
      3. Compute pairwise_conflict between the fused group opinions.
      4. If conflict is significant AND there's a clear winner:
         keep only the highest-evidence group.
      5. If no significant conflict: keep all passages.

    SL advantage over Methods C/D:
      - Tracks UNCERTAINTY: a group of 2 high-confidence passages beats
        a group of 3 low-confidence passages (Methods C/D can't see this)
      - Detects genuine conflict: pairwise_conflict between group opinions
        captures when groups actively disagree vs merely differ
      - Principled fusion: cumulative_fuse accumulates evidence correctly
        per Josang's algebra
    """
    passages = retrieval["retrieved"]
    if not passages:
        return {"method": "E_sl_fusion", "kept_passages": [], "abstained": True,
                "conflict_detected": False}

    groups = group_passages_by_answer(passages, extractions)

    if len(groups) == 1:
        # All passages agree — no conflict, keep all
        return {
            "method": "E_sl_fusion",
            "kept_passages": passages,
            "abstained": False,
            "conflict_detected": False,
            "n_answer_groups": 1,
        }

    # Build SL opinion per group
    group_opinions = []
    group_data = []
    for group in groups:
        opinions_in_group = []
        for p in group:
            # Evidence from two signals: cosine sim and QA extraction confidence
            cos_sim = max(0.0, min(1.0, p["score"]))
            qa_conf = max(0.0, min(1.0, p["qa_score"]))
            # Combined evidence weight: geometric mean of both signals
            combined = (cos_sim * qa_conf) ** 0.5
            pos_ev = combined * EVIDENCE_WEIGHT
            neg_ev = (1.0 - combined) * EVIDENCE_WEIGHT
            op = Opinion.from_evidence(pos_ev, neg_ev, prior_weight=PRIOR_WEIGHT)
            opinions_in_group.append(op)

        if len(opinions_in_group) == 1:
            fused = opinions_in_group[0]
        else:
            fused = cumulative_fuse(*opinions_in_group)

        group_opinions.append(fused)
        group_data.append({
            "group": group,
            "fused_opinion": fused,
            "n_passages": len(group),
            "fused_proj": fused.projected_probability(),
            "fused_uncertainty": fused.uncertainty,
        })

    # Compute max pairwise conflict between groups
    max_conflict = 0.0
    for i in range(len(group_opinions)):
        for j in range(i + 1, len(group_opinions)):
            c = pairwise_conflict(group_opinions[i], group_opinions[j])
            max_conflict = max(max_conflict, c)

    # Decision: is there meaningful conflict?
    # Use a dynamic threshold based on group uncertainty:
    # conflict is meaningful if it exceeds the average uncertainty
    avg_uncertainty = np.mean([gd["fused_uncertainty"] for gd in group_data])
    conflict_detected = max_conflict > avg_uncertainty and len(groups) >= 2

    if not conflict_detected:
        return {
            "method": "E_sl_fusion",
            "kept_passages": passages,
            "abstained": False,
            "conflict_detected": False,
            "n_answer_groups": len(groups),
            "max_pairwise_conflict": max_conflict,
            "avg_uncertainty": avg_uncertainty,
        }

    # Conflict detected — keep the group with highest projected probability
    # (highest evidence)
    best_idx = max(range(len(group_data)),
                   key=lambda k: group_data[k]["fused_proj"])
    best_group = group_data[best_idx]["group"]
    kept_pids = {p["passage_id"] for p in best_group}
    kept = [p for p in passages if p["passage_id"] in kept_pids]

    return {
        "method": "E_sl_fusion",
        "kept_passages": kept,
        "abstained": len(kept) == 0,
        "conflict_detected": True,
        "n_answer_groups": len(groups),
        "best_group_size": len(best_group),
        "max_pairwise_conflict": max_conflict,
        "avg_uncertainty": avg_uncertainty,
    }


def method_e2_sl_abstention(
    retrieval: Dict,
    extractions: Dict[str, Dict],
    **kwargs,
) -> Dict:
    """Method E2: SL fusion with abstention on high-uncertainty conflict.

    Like E, but when conflict is detected AND the best group has high
    uncertainty (low evidence), abstain rather than guess.
    """
    passages = retrieval["retrieved"]
    if not passages:
        return {"method": "E2_sl_abstain", "kept_passages": [], "abstained": True,
                "conflict_detected": False}

    groups = group_passages_by_answer(passages, extractions)

    if len(groups) == 1:
        return {
            "method": "E2_sl_abstain",
            "kept_passages": passages,
            "abstained": False,
            "conflict_detected": False,
            "n_answer_groups": 1,
        }

    # Build SL opinions per group (same as E)
    group_opinions = []
    group_data = []
    for group in groups:
        opinions_in_group = []
        for p in group:
            cos_sim = max(0.0, min(1.0, p["score"]))
            qa_conf = max(0.0, min(1.0, p["qa_score"]))
            combined = (cos_sim * qa_conf) ** 0.5
            pos_ev = combined * EVIDENCE_WEIGHT
            neg_ev = (1.0 - combined) * EVIDENCE_WEIGHT
            op = Opinion.from_evidence(pos_ev, neg_ev, prior_weight=PRIOR_WEIGHT)
            opinions_in_group.append(op)

        fused = cumulative_fuse(*opinions_in_group) if len(opinions_in_group) > 1 else opinions_in_group[0]
        group_opinions.append(fused)
        group_data.append({
            "group": group,
            "fused_opinion": fused,
            "n_passages": len(group),
            "fused_proj": fused.projected_probability(),
            "fused_uncertainty": fused.uncertainty,
        })

    max_conflict = 0.0
    for i in range(len(group_opinions)):
        for j in range(i + 1, len(group_opinions)):
            c = pairwise_conflict(group_opinions[i], group_opinions[j])
            max_conflict = max(max_conflict, c)

    avg_uncertainty = np.mean([gd["fused_uncertainty"] for gd in group_data])
    conflict_detected = max_conflict > avg_uncertainty and len(groups) >= 2

    if not conflict_detected:
        return {
            "method": "E2_sl_abstain",
            "kept_passages": passages,
            "abstained": False,
            "conflict_detected": False,
            "n_answer_groups": len(groups),
        }

    # Conflict detected — check if best group has enough evidence
    best_idx = max(range(len(group_data)),
                   key=lambda k: group_data[k]["fused_proj"])
    best_gd = group_data[best_idx]

    # Abstain if best group uncertainty > 0.3 (not enough evidence to decide)
    if best_gd["fused_uncertainty"] > 0.3:
        return {
            "method": "E2_sl_abstain",
            "kept_passages": [],
            "abstained": True,
            "conflict_detected": True,
            "abstention_reason": "high_uncertainty_conflict",
            "best_group_uncertainty": best_gd["fused_uncertainty"],
        }

    best_group = best_gd["group"]
    kept_pids = {p["passage_id"] for p in best_group}
    kept = [p for p in passages if p["passage_id"] in kept_pids]

    return {
        "method": "E2_sl_abstain",
        "kept_passages": kept,
        "abstained": len(kept) == 0,
        "conflict_detected": True,
        "n_answer_groups": len(groups),
        "best_group_size": len(best_group),
    }


def method_f_sl_remove_outlier(
    retrieval: Dict,
    extractions: Dict[str, Dict],
    **kwargs,
) -> Dict:
    """Method F: SL conflict-guided outlier removal (context-preserving).

    Unlike Method E (which keeps only the best answer group, discarding
    ~87% of passages), Method F REMOVES only the conflicting minority
    group and keeps everything else.  This preserves most context while
    surgically excising poisoned passages.

    Algorithm:
      1. Group passages by extracted answer (same as E).
      2. If only one group: no conflict, keep all (same as no-filter).
      3. If multiple groups: build SL opinions per group, detect conflict.
      4. If conflict detected: identify the LOWEST-evidence group(s)
         and remove them.  Keep all passages from the remaining groups.
      5. If no significant conflict: keep all passages.

    SL advantage:  Uses cumulative_fuse evidence strength (not just
    group size) to decide which group is the outlier.  A single
    high-confidence poison passage in a small group gets correctly
    identified as an outlier vs a large low-confidence group.
    """
    passages = retrieval["retrieved"]
    if not passages:
        return {"method": "F_sl_remove_outlier", "kept_passages": [],
                "abstained": True, "conflict_detected": False}

    groups = group_passages_by_answer(passages, extractions)

    if len(groups) == 1:
        # All passages agree — keep all, identical to no-filter
        return {
            "method": "F_sl_remove_outlier",
            "kept_passages": passages,
            "abstained": False,
            "conflict_detected": False,
            "n_answer_groups": 1,
        }

    # Build SL opinion per group
    group_opinions = []
    group_data = []
    for group in groups:
        opinions_in_group = []
        for p in group:
            cos_sim = max(0.0, min(1.0, p["score"]))
            qa_conf = max(0.0, min(1.0, p["qa_score"]))
            combined = (cos_sim * qa_conf) ** 0.5
            pos_ev = combined * EVIDENCE_WEIGHT
            neg_ev = (1.0 - combined) * EVIDENCE_WEIGHT
            op = Opinion.from_evidence(pos_ev, neg_ev, prior_weight=PRIOR_WEIGHT)
            opinions_in_group.append(op)

        fused = cumulative_fuse(*opinions_in_group) if len(opinions_in_group) > 1 else opinions_in_group[0]
        group_opinions.append(fused)
        group_data.append({
            "group": group,
            "fused_opinion": fused,
            "n_passages": len(group),
            "fused_proj": fused.projected_probability(),
            "fused_uncertainty": fused.uncertainty,
        })

    # Pairwise conflict
    max_conflict = 0.0
    for i in range(len(group_opinions)):
        for j in range(i + 1, len(group_opinions)):
            c = pairwise_conflict(group_opinions[i], group_opinions[j])
            max_conflict = max(max_conflict, c)

    avg_u = np.mean([gd["fused_uncertainty"] for gd in group_data])
    conflict_detected = max_conflict > avg_u and len(groups) >= 2

    if not conflict_detected:
        # No significant conflict — keep all passages
        return {
            "method": "F_sl_remove_outlier",
            "kept_passages": passages,
            "abstained": False,
            "conflict_detected": False,
            "n_answer_groups": len(groups),
            "max_pairwise_conflict": max_conflict,
        }

    # Conflict detected — remove the LOWEST-evidence group
    worst_idx = min(range(len(group_data)),
                    key=lambda k: group_data[k]["fused_proj"])
    outlier_group = group_data[worst_idx]["group"]
    outlier_pids = {p["passage_id"] for p in outlier_group}

    # Keep everything EXCEPT the outlier group
    kept = [p for p in passages if p["passage_id"] not in outlier_pids]

    return {
        "method": "F_sl_remove_outlier",
        "kept_passages": kept,
        "abstained": len(kept) == 0,
        "conflict_detected": True,
        "n_answer_groups": len(groups),
        "outlier_group_size": len(outlier_group),
        "kept_count": len(kept),
        "max_pairwise_conflict": max_conflict,
    }


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_filtering(
    retrieval_results: List[Dict],
    questions: List[Dict],
    corpus: List[Dict],
    all_extractions: Dict[str, Dict[str, Dict]],
) -> Dict[str, Any]:
    """Evaluate all filtering methods."""
    method_configs = [
        ("A_no_filter", lambda r, ex: method_a_no_filter(r)),
    ]

    for thr in SCALAR_THRESHOLDS:
        method_configs.append(
            (f"B_scalar_{thr:.2f}", lambda r, ex, t=thr: method_b_scalar_threshold(r, threshold=t))
        )

    method_configs.extend([
        ("C_majority_vote", lambda r, ex: method_c_majority_vote(r, extractions=ex)),
        ("D_weighted_vote", lambda r, ex: method_d_score_weighted_vote(r, extractions=ex)),
        ("E_sl_fusion", lambda r, ex: method_e_sl_answer_fusion(r, extractions=ex)),
        ("E2_sl_abstain", lambda r, ex: method_e2_sl_abstention(r, extractions=ex)),
        ("F_sl_remove_outlier", lambda r, ex: method_f_sl_remove_outlier(r, extractions=ex)),
    ])

    qid_to_question = {q["id"]: q for q in questions}
    results = {}

    for method_name, method_fn in method_configs:
        poison_included = []
        gold_retained = []
        abstained = []
        n_kept_list = []

        for rr in retrieval_results:
            qid = rr["question_id"]
            q = qid_to_question[qid]
            ex = all_extractions.get(qid, {})

            filtered = method_fn(rr, ex)
            has_poison = False
            has_gold = False

            for p in filtered["kept_passages"]:
                if p["is_poison"]:
                    has_poison = True
                if p["is_gold"]:
                    has_gold = True

            poison_included.append(float(has_poison))
            gold_retained.append(float(has_gold))
            abstained.append(float(filtered["abstained"]))
            n_kept_list.append(len(filtered["kept_passages"]))

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
# Tier 2: LLM Answer Generation
# ═══════════════════════════════════════════════════════════════════

def load_llm(model_name: str):
    """Load a 4-bit quantized LLM for answer generation."""
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
    model, tokenizer, question: str, passages: List[str],
    max_new_tokens: int = LLM_MAX_NEW_TOKENS,
) -> str:
    """Generate an answer given a question and context passages."""
    import torch

    if not passages:
        return ""

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
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer.split("\n")[0].strip()


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


def evaluate_tier2(
    retrieval_results: List[Dict],
    questions: List[Dict],
    corpus: List[Dict],
    all_extractions: Dict[str, Dict[str, Dict]],
    model, tokenizer,
    methods_to_eval: List[str],
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Tier 2: generate answers with LLM and evaluate EM/F1."""
    if checkpoint_path and checkpoint_path.exists():
        print(f"  Loading Tier 2 checkpoint: {checkpoint_path.name}")
        with open(str(checkpoint_path), "r") as f:
            return json.load(f)

    qid_to_question = {q["id"]: q for q in questions}
    pid_to_text = {p["id"]: p["text"] for p in corpus}
    results = {}

    for method_name in methods_to_eval:
        print(f"  Generating answers for {method_name}...")
        em_scores, f1_scores = [], []
        abstained_count = 0

        for i, rr in enumerate(retrieval_results):
            qid = rr["question_id"]
            q = qid_to_question[qid]
            ex = all_extractions.get(qid, {})

            # Apply filtering
            if method_name == "A_no_filter":
                filtered = method_a_no_filter(rr)
            elif method_name.startswith("B_scalar_"):
                thr = float(method_name.split("_")[-1])
                filtered = method_b_scalar_threshold(rr, threshold=thr)
            elif method_name == "C_majority_vote":
                filtered = method_c_majority_vote(rr, extractions=ex)
            elif method_name == "D_weighted_vote":
                filtered = method_d_score_weighted_vote(rr, extractions=ex)
            elif method_name == "E_sl_fusion":
                filtered = method_e_sl_answer_fusion(rr, extractions=ex)
            elif method_name == "E2_sl_abstain":
                filtered = method_e2_sl_abstention(rr, extractions=ex)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            if filtered["abstained"]:
                abstained_count += 1
                em_scores.append(0.0)
                f1_scores.append(0.0)
                continue

            passage_texts = [pid_to_text[p["passage_id"]] for p in filtered["kept_passages"]]
            answer = generate_answer(model, tokenizer, q["question"], passage_texts)

            em = metric_over_answers(answer, q["answers"], exact_match_score)
            f1 = metric_over_answers(answer, q["answers"], token_f1_score)
            em_scores.append(em)
            f1_scores.append(f1)

            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(retrieval_results)}] "
                      f"EM={np.mean(em_scores):.3f} F1={np.mean(f1_scores):.3f}")

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

    if checkpoint_path:
        with open(str(checkpoint_path), "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved Tier 2 checkpoint: {checkpoint_path.name}")

    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EN3.1b — Improved RAG Pipeline")
    parser.add_argument("--tier1", action="store_true",
                        help="Run only Tier 1 (filtering evaluation, no LLM)")
    parser.add_argument("--poison-rates", type=float, nargs="+", default=POISON_RATES)
    parser.add_argument("--n-questions", type=int, default=N_QUESTIONS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    set_global_seed(GLOBAL_SEED)
    env = log_environment()
    run_tier2 = not args.tier1

    print("=" * 70)
    print("EN3.1b — Improved RAG Pipeline (Answer-Agreement SL Filtering)")
    print("=" * 70)
    print(f"  Mode:          {'Tier 1 + Tier 2' if run_tier2 else 'Tier 1 only'}")
    print(f"  Questions:     {args.n_questions}")
    print(f"  Top-K:         {args.top_k}")
    print(f"  Poison rates:  {args.poison_rates}")
    print(f"  QA model:      {QA_MODEL}")
    if run_tier2:
        print(f"  LLM model:     {LLM_MODEL}")
    print(f"  Seed:          {GLOBAL_SEED}")
    print()

    t_start = time.time()

    # ── Phase 1: Load data ──
    print("Phase 1: Data Preparation")
    print("-" * 40)
    base_passages, questions = load_and_prepare_squad(args.n_questions, GLOBAL_SEED)
    print()

    all_tier1_results = {}
    all_tier2_results = {}
    llm_model = None
    llm_tokenizer = None

    for poison_rate in args.poison_rates:
        pr_tag = f"pr{int(poison_rate * 100):02d}"
        print("=" * 70)
        print(f"  POISON RATE: {poison_rate:.0%}")
        print("=" * 70)

        # ── Create poisoned corpus ──
        corpus = create_poisoned_passages(
            base_passages, questions, poison_rate,
            seed=GLOBAL_SEED + int(poison_rate * 1000),
        )

        # ── Save corpus text checkpoint (for API script) ──
        corpus_ckpt = CHECKPOINT_DIR / f"v1b_corpus_texts_{pr_tag}.json"
        if not corpus_ckpt.exists():
            pid_to_text = {p["id"]: p["text"] for p in corpus}
            with open(str(corpus_ckpt), "w", encoding="utf-8") as f:
                json.dump(pid_to_text, f)
            print(f"  Saved corpus text checkpoint: {corpus_ckpt.name}")

        # ── Also save questions checkpoint ──
        questions_ckpt = CHECKPOINT_DIR / f"v1b_questions_{pr_tag}.json"
        if not questions_ckpt.exists():
            with open(str(questions_ckpt), "w", encoding="utf-8") as f:
                json.dump(questions, f)
            print(f"  Saved questions checkpoint: {questions_ckpt.name}")

        # ── Embed and index ──
        print(f"\nPhase 2: Embedding and Indexing ({pr_tag})")
        print("-" * 40)
        emb_ckpt = CHECKPOINT_DIR / f"v1b_embeddings_{pr_tag}.npz"
        index, embeddings = embed_and_index(corpus, EMBEDDING_MODEL, emb_ckpt)
        print()

        # ── Retrieve (reuse v1 checkpoints) ──
        print(f"Phase 3: Retrieval ({pr_tag})")
        print("-" * 40)
        retr_ckpt = CHECKPOINT_DIR / f"v1b_retrieval_{pr_tag}.json"
        retrieval_results = retrieve_passages(
            questions, corpus, index, EMBEDDING_MODEL, args.top_k, retr_ckpt
        )

        n_with_gold = sum(1 for rr in retrieval_results if any(p["is_gold"] for p in rr["retrieved"]))
        n_with_poison = sum(1 for rr in retrieval_results if any(p["is_poison"] for p in rr["retrieved"]))
        print(f"  Retrieval: {n_with_gold}/{len(retrieval_results)} have gold, "
              f"{n_with_poison}/{len(retrieval_results)} have poison in top-{args.top_k}")
        print()

        # ── Answer extraction (new in v1b) ──
        print(f"Phase 3b: Answer Extraction ({pr_tag})")
        print("-" * 40)
        qa_ckpt = CHECKPOINT_DIR / f"v1b_qa_extraction_{pr_tag}.json"
        extractions = extract_answers_for_retrieval(
            questions, retrieval_results, corpus, QA_MODEL, qa_ckpt,
        )
        print()

        # ── Evaluate filtering ──
        print(f"Phase 4: Filtering Evaluation ({pr_tag})")
        print("-" * 40)
        tier1 = evaluate_filtering(retrieval_results, questions, corpus, extractions)
        all_tier1_results[pr_tag] = tier1

        print(f"\n  {'Method':<25s} {'PoisonIncl':>12s} {'GoldRetain':>12s} {'Abstain':>10s} {'Kept':>6s}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*6}")
        for method, m in tier1.items():
            print(f"  {method:<25s} {m['poison_inclusion_rate']:>12.4f} "
                  f"{m['gold_retention_rate']:>12.4f} {m['abstention_rate']:>10.4f} "
                  f"{m['mean_passages_kept']:>6.1f}")
        print()

        del index, embeddings
        gc.collect()

        # ── Tier 2: LLM Answer Generation ──
        if run_tier2:
            print(f"Phase 5: Tier 2 LLM Evaluation ({pr_tag})")
            print("-" * 40)

            # Load LLM once
            if llm_model is None:
                llm_model, llm_tokenizer = load_llm(LLM_MODEL)

            # Evaluate representative methods: no-filter, best scalar, majority, SL
            tier2_methods = ["A_no_filter", "B_scalar_0.50", "C_majority_vote", "E_sl_fusion"]
            tier2_ckpt = CHECKPOINT_DIR / f"v1b_tier2_{pr_tag}.json"
            tier2 = evaluate_tier2(
                retrieval_results, questions, corpus, extractions,
                llm_model, llm_tokenizer, tier2_methods, tier2_ckpt,
            )
            all_tier2_results[pr_tag] = tier2
            print()

    # ── Summary ──
    total_time = time.time() - t_start

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  {'PR':<8s} {'A_noF':>8s} {'BestB':>8s} {'C_maj':>8s} {'D_wvt':>8s} "
          f"{'E_sl':>8s} {'E2_ab':>8s}  (poison inclusion rate)")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    tier1_summary = {}
    for pr_tag in all_tier1_results:
        tier1 = all_tier1_results[pr_tag]
        a_pi = tier1["A_no_filter"]["poison_inclusion_rate"]

        # Best scalar
        best_b_name = min(
            [k for k in tier1 if k.startswith("B_scalar_")],
            key=lambda k: tier1[k]["poison_inclusion_rate"]
            if tier1[k]["gold_retention_rate"] >= 0.5 else 999,
        )
        best_b_pi = tier1[best_b_name]["poison_inclusion_rate"]

        c_pi = tier1["C_majority_vote"]["poison_inclusion_rate"]
        d_pi = tier1["D_weighted_vote"]["poison_inclusion_rate"]
        e_pi = tier1["E_sl_fusion"]["poison_inclusion_rate"]
        e2_pi = tier1["E2_sl_abstain"]["poison_inclusion_rate"]

        tier1_summary[pr_tag] = {
            "A_no_filter": a_pi,
            "best_scalar": best_b_name,
            "best_scalar_pi": best_b_pi,
            "C_majority_vote": c_pi,
            "D_weighted_vote": d_pi,
            "E_sl_fusion": e_pi,
            "E2_sl_abstain": e2_pi,
            "sl_vs_majority": round(c_pi - e_pi, 4),
            "sl_vs_weighted": round(d_pi - e_pi, 4),
            "sl_vs_best_scalar": round(best_b_pi - e_pi, 4),
        }

        print(f"  {pr_tag:<8s} {a_pi:>8.4f} {best_b_pi:>8.4f} {c_pi:>8.4f} "
              f"{d_pi:>8.4f} {e_pi:>8.4f} {e2_pi:>8.4f}")

    # Gold retention comparison
    print(f"\n  {'PR':<8s} {'A_noF':>8s} {'BestB':>8s} {'C_maj':>8s} {'D_wvt':>8s} "
          f"{'E_sl':>8s} {'E2_ab':>8s}  (gold retention rate)")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for pr_tag in all_tier1_results:
        tier1 = all_tier1_results[pr_tag]
        best_b_name = tier1_summary[pr_tag]["best_scalar"]
        print(f"  {pr_tag:<8s} "
              f"{tier1['A_no_filter']['gold_retention_rate']:>8.4f} "
              f"{tier1[best_b_name]['gold_retention_rate']:>8.4f} "
              f"{tier1['C_majority_vote']['gold_retention_rate']:>8.4f} "
              f"{tier1['D_weighted_vote']['gold_retention_rate']:>8.4f} "
              f"{tier1['E_sl_fusion']['gold_retention_rate']:>8.4f} "
              f"{tier1['E2_sl_abstain']['gold_retention_rate']:>8.4f}")

    if all_tier2_results:
        print("\nTier 2 — End-to-End Answer Quality")
        for pr_tag, tier2 in all_tier2_results.items():
            print(f"\n  {pr_tag}:")
            print(f"    {'Method':<25s} {'EM':>8s} {'F1':>8s} {'Abstain':>10s}")
            print(f"    {'-'*25} {'-'*8} {'-'*8} {'-'*10}")
            for method, m in tier2.items():
                print(f"    {method:<25s} {m['exact_match']:>8.4f} {m['token_f1']:>8.4f} "
                      f"{m['abstention_rate']:>10.4f}")

    print(f"\n  Total wall time: {total_time:.1f}s")

    # ── Save results ──
    output_path = RESULTS_DIR / "en3_1b_results.json"
    experiment_result = ExperimentResult(
        experiment_id="EN3.1b",
        parameters={
            "global_seed": GLOBAL_SEED,
            "n_questions": args.n_questions,
            "top_k": args.top_k,
            "evidence_weight": EVIDENCE_WEIGHT,
            "prior_weight": PRIOR_WEIGHT,
            "poison_rates": args.poison_rates,
            "scalar_thresholds": SCALAR_THRESHOLDS,
            "embedding_model": EMBEDDING_MODEL,
            "qa_model": QA_MODEL,
            "llm_model": LLM_MODEL if run_tier2 else "N/A (Tier 1 only)",
            "answer_similarity_threshold": ANSWER_SIMILARITY_THRESHOLD,
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
            "v1_diagnosis": (
                "EN3.1 v1 showed SL conflict on cosine-similarity opinions does not "
                "outperform scalar thresholding because poisoned passages have nearly "
                "identical cosine scores to gold (same topic, swapped answer). "
                "v1b redesigns to test answer-level conflict detection."
            ),
        },
        environment=env,
        notes=(
            f"EN3.1b: Improved RAG pipeline with answer-agreement-based SL filtering. "
            f"v1 showed relevance-based conflict detection fails; v1b uses extractive "
            f"QA to detect factual conflict between passages. "
            f"6 methods: no-filter, scalar threshold, majority vote, weighted vote, "
            f"SL fusion, SL fusion+abstention. "
            f"{'Tier 1 + Tier 2' if run_tier2 else 'Tier 1 only'}. "
            f"Dataset: SQuAD 1.1 dev ({args.n_questions} questions). "
            f"QA model: {QA_MODEL}."
            f"{' LLM: ' + LLM_MODEL + '.' if run_tier2 else ''}"
        ),
    )
    experiment_result.save_json(str(output_path))
    print(f"\n  Results saved to: {output_path}")

    ts = experiment_result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    archive_path = RESULTS_DIR / f"en3_1b_results_{ts}.json"
    experiment_result.save_json(str(archive_path))
    print(f"  Archived to:      {archive_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
