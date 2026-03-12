"""EN3.2-H1c v2 Ablation Core — Per-Difficulty, Model Subset, Param Sweep, Precision-at-Coverage.

Pure computation module (no I/O, no API calls).

Ablation analyses for the per-passage multi-model fusion experiment:
  1. Per-difficulty breakdown (easy/medium/hard)
  2. Model subset ablation (which models matter?)
  3. Parameter sweep (evidence_weight × prior_weight for SL strategies)
  4. Precision-at-coverage curves (abstention via fused confidence)
  5. McNemar contingency tables (pairwise strategy comparison)
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure sibling modules in EN3/ are importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from scipy.stats import binomtest  # type: ignore

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    pairwise_conflict,
    trust_discount,
)

from en3_2_h1c_v2_core import (
    evaluate_all_strategies,
    fuse_passage_sl_fusion,
    fuse_passage_sl_trust_discount,
    rank_passages_by_confidence_x_cosine,
    select_answer_from_ranking,
    _score_to_opinion,
    _fuse_opinions,
    _group_within_passage,
    _WEAKEST_MODEL,
)

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

# Difficulty thresholds (0-indexed ranks) — matches H3's classifier
_EASY_MAX_RANK = 2        # gold in top-3 (index 0, 1, 2)
_MEDIUM_MAX_RANK = 6      # gold in ranks 4-7 (index 3..6)
_HARD_POISON_THRESHOLD = 2  # >= 2 poison passages → hard

# Named model subsets for ablation
MODEL_SUBSETS: Dict[str, List[str]] = {
    "all_4": ["distilbert", "roberta", "electra", "bert_tiny"],
    "drop_bert_tiny": ["distilbert", "roberta", "electra"],
    "drop_distilbert": ["roberta", "electra", "bert_tiny"],
    "drop_roberta": ["distilbert", "electra", "bert_tiny"],
    "drop_electra": ["distilbert", "roberta", "bert_tiny"],
    "top_2": ["roberta", "electra"],
    "diverse_pair": ["roberta", "distilbert"],
}

# SL strategy names (the ones that depend on evidence_weight / prior_weight)
_SL_STRATEGY_NAMES = [
    "sl_fusion",
    "sl_trust_discount",
    "sl_3strong",
    "sl_conflict_weighted",
]


# ═══════════════════════════════════════════════════════════════════
# 1. Difficulty classification
# ═══════════════════════════════════════════════════════════════════

def classify_question_difficulty_h1c(
    gold_passage_id: str,
    retrieved: List[Dict[str, Any]],
    poison_pids: Set[str],
) -> str:
    """Classify a question as easy / medium / hard.

    Identical logic to H3's ``classify_question_difficulty`` but accepts
    the H1c retrieval data structure (list of dicts with ``passage_id``
    and ``score`` keys).

    Criteria:
      - easy:   gold in top-3 (rank <= 2) AND 0 poison in retrieved
      - medium: (gold top-3 + 1 poison) OR (gold rank 3-6 + <=1 poison)
      - hard:   gold rank >= 7 OR gold missing OR >= 2 poison passages
    """
    gold_rank: Optional[int] = None
    for idx, r in enumerate(retrieved):
        if r["passage_id"] == gold_passage_id:
            gold_rank = idx
            break

    n_poison = sum(1 for r in retrieved if r["passage_id"] in poison_pids)

    # Hard conditions (checked first — override everything)
    if n_poison >= _HARD_POISON_THRESHOLD:
        return "hard"
    if gold_rank is None:
        return "hard"
    if gold_rank > _MEDIUM_MAX_RANK:
        return "hard"

    # Easy: gold in top-3, no poison
    if gold_rank <= _EASY_MAX_RANK and n_poison == 0:
        return "easy"

    # Everything else is medium
    return "medium"


# ═══════════════════════════════════════════════════════════════════
# 2. Model subset evaluation
# ═══════════════════════════════════════════════════════════════════

def _filter_extractions_to_subset(
    question_passages: List[Dict[str, Any]],
    models: List[str],
) -> List[Dict[str, Any]]:
    """Return a copy of question_passages with extractions filtered
    to include only the specified models."""
    filtered = []
    model_set = set(models)
    for passage in question_passages:
        new_passage = {
            "cosine": passage["cosine"],
            "passage_id": passage["passage_id"],
            "extractions": {
                k: v for k, v in passage["extractions"].items()
                if k in model_set
            },
        }
        filtered.append(new_passage)
    return filtered


def evaluate_model_subset(
    question_passages: List[Dict[str, Any]],
    models: List[str],
    model_trust: Optional[Dict[str, float]] = None,
    evidence_weight: float = 10.0,
    prior_weight: float = 2.0,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all strategies on a question using only the specified model subset.

    Filters extractions to include only models in ``models``, then
    delegates to ``evaluate_all_strategies``.

    Returns: {strategy_name: {answer, confidence}}
    """
    filtered = _filter_extractions_to_subset(question_passages, models)
    return evaluate_all_strategies(
        filtered,
        model_trust=model_trust,
        evidence_weight=evidence_weight,
        prior_weight=prior_weight,
    )


# ═══════════════════════════════════════════════════════════════════
# 3. Parameter sweep (SL strategies only)
# ═══════════════════════════════════════════════════════════════════

def evaluate_param_combo(
    question_passages: List[Dict[str, Any]],
    model_trust: Optional[Dict[str, float]] = None,
    evidence_weight: float = 10.0,
    prior_weight: float = 2.0,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate only SL strategies with a specific (ew, pw) combo.

    Scalar strategies are parameter-independent and excluded to avoid
    redundant computation during sweeps.

    Returns: {sl_strategy_name: {answer, confidence}}
    """
    all_results = evaluate_all_strategies(
        question_passages,
        model_trust=model_trust,
        evidence_weight=evidence_weight,
        prior_weight=prior_weight,
    )
    return {
        name: all_results[name]
        for name in _SL_STRATEGY_NAMES
        if name in all_results
    }


# ═══════════════════════════════════════════════════════════════════
# 4. Precision-at-coverage
# ═══════════════════════════════════════════════════════════════════

def compute_precision_at_coverage(
    per_question: List[Dict[str, Any]],
    n_levels: int = 20,
) -> List[Dict[str, float]]:
    """Compute precision-at-coverage curve for a strategy.

    Args:
        per_question: List of dicts with ``confidence`` (float) and
            ``correct`` (bool) keys.
        n_levels: Number of evenly-spaced coverage levels between
            1/N and 1.0.

    Returns:
        List of {coverage, precision, n_answered} dicts sorted by
        ascending coverage.
    """
    if not per_question:
        return []

    n = len(per_question)
    sorted_q = sorted(per_question, key=lambda q: -q["confidence"])

    curve: List[Dict[str, float]] = []

    if n <= n_levels:
        ks = list(range(1, n + 1))
    else:
        step = n / n_levels
        ks = sorted(set(
            [max(1, round(step * i)) for i in range(1, n_levels + 1)]
            + [n]
        ))

    for k in ks:
        subset = sorted_q[:k]
        n_correct = sum(1 for q in subset if q["correct"])
        precision = n_correct / k
        coverage = k / n
        curve.append({
            "coverage": round(coverage, 6),
            "precision": round(precision, 6),
            "n_answered": k,
        })

    return curve


# ═══════════════════════════════════════════════════════════════════
# 5. Per-difficulty breakdown
# ═══════════════════════════════════════════════════════════════════

def compute_per_difficulty_breakdown(
    per_question: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Aggregate per-question results by difficulty level.

    Args:
        per_question: List of dicts with ``difficulty`` (str) and
            ``strategies`` (dict mapping strategy_name -> bool correct).

    Returns:
        {difficulty: {strategy: {accuracy, n, n_correct}}}
        All three difficulty levels are always present (with n=0 if empty).
    """
    all_strategies: set = set()
    for q in per_question:
        all_strategies.update(q["strategies"].keys())

    by_diff: Dict[str, List[Dict[str, Any]]] = {
        "easy": [], "medium": [], "hard": [],
    }
    for q in per_question:
        diff = q["difficulty"]
        if diff in by_diff:
            by_diff[diff].append(q)

    breakdown: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for diff in ["easy", "medium", "hard"]:
        breakdown[diff] = {}
        questions = by_diff[diff]
        for strategy in sorted(all_strategies):
            correct_list = [
                q["strategies"].get(strategy, False)
                for q in questions
            ]
            n = len(correct_list)
            n_correct = sum(correct_list)
            accuracy = n_correct / n if n > 0 else 0.0
            breakdown[diff][strategy] = {
                "accuracy": accuracy,
                "n": n,
                "n_correct": n_correct,
            }

    return breakdown


# ═══════════════════════════════════════════════════════════════════
# 6. McNemar contingency table
# ═══════════════════════════════════════════════════════════════════

def compute_mcnemar_contingency(
    per_question: List[Dict[str, Any]],
    strategy_a: str,
    strategy_b: str,
) -> Dict[str, Any]:
    """Compute McNemar's contingency table and exact test between two strategies.

    Args:
        per_question: List of dicts, each with keys ``strategy_a`` and
            ``strategy_b`` mapping to bool (correct/incorrect).

    Returns:
        Dict with keys: both_correct, a_only, b_only, both_wrong, p_value.
    """
    both_correct = 0
    a_only = 0
    b_only = 0
    both_wrong = 0

    for q in per_question:
        a_correct = bool(q[strategy_a])
        b_correct = bool(q[strategy_b])
        if a_correct and b_correct:
            both_correct += 1
        elif a_correct and not b_correct:
            a_only += 1
        elif not a_correct and b_correct:
            b_only += 1
        else:
            both_wrong += 1

    n_discordant = a_only + b_only
    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(a_only, n_discordant, 0.5).pvalue

    return {
        "both_correct": both_correct,
        "a_only": a_only,
        "b_only": b_only,
        "both_wrong": both_wrong,
        "n_discordant": n_discordant,
        "p_value": p_value,
    }
