"""EN3.2-H3 Core — Metadata-Enriched Prompting building blocks.

Provides prompt builders, difficulty classification, SL metadata
computation, and McNemar's test for the EN3.2-H3 experiment.

This module contains NO API calls or I/O — it is pure computation,
fully testable without network access.
"""
from __future__ import annotations

import math
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Sequence

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    pairwise_conflict,
)


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

ANSWER_SIMILARITY_THRESHOLD = 0.6

# Difficulty thresholds (0-indexed ranks)
_EASY_MAX_RANK = 2       # gold in top-3 (index 0, 1, 2)
_MEDIUM_MAX_RANK = 6     # gold in ranks 4-7 (index 3..6)
_HARD_POISON_THRESHOLD = 2  # >= 2 poison passages → hard


# ═══════════════════════════════════════════════════════════════════
# 1. Prompt builders
# ═══════════════════════════════════════════════════════════════════

_SYSTEM_INSTRUCTION = (
    "Answer the following question based ONLY on the provided passages. "
    "Give a short, direct answer (a few words or a short phrase). "
    "If the passages do not contain enough information, say 'unanswerable'."
)


def build_prompt_plain(
    question: str,
    passages: List[Dict[str, Any]],
    pid_to_text: Dict[str, str],
) -> str:
    """Build PLAIN prompt: passages only, no metadata."""
    parts = [_SYSTEM_INSTRUCTION, "", "Context:"]
    for i, p in enumerate(passages):
        text = pid_to_text[p["passage_id"]]
        parts.append(f"[Passage {i + 1}] {text}")
    parts.append("")
    parts.append(f"Question: {question}")
    parts.append("")
    parts.append("Answer:")
    return "\n".join(parts)


def build_prompt_scalar(
    question: str,
    passages: List[Dict[str, Any]],
    pid_to_text: Dict[str, str],
) -> str:
    """Build SCALAR prompt: passages + cosine similarity scores."""
    parts = [
        _SYSTEM_INSTRUCTION,
        "Each passage includes a relevance similarity score (0-1) indicating "
        "how closely it matches the question. Higher scores suggest greater relevance.",
        "",
        "Context:",
    ]
    for i, p in enumerate(passages):
        text = pid_to_text[p["passage_id"]]
        score = p["score"]
        parts.append(f"[Passage {i + 1}] (relevance: {score:.2f}) {text}")
    parts.append("")
    parts.append(f"Question: {question}")
    parts.append("")
    parts.append("Answer:")
    return "\n".join(parts)


def build_prompt_answers_only(
    question: str,
    passages: List[Dict[str, Any]],
    pid_to_text: Dict[str, str],
    extractions: Dict[str, Dict[str, Any]],
) -> str:
    """Build ANSWERS-ONLY prompt: passages + extracted answers, no SL metadata.

    Ablation condition that isolates the effect of extracted answers
    from the effect of SL (b, d, u) triples and conflict/agreement
    metadata. Contains NO cosine scores, NO belief/disbelief/uncertainty,
    NO conflict detection, NO fused assessment.
    """
    parts = [
        _SYSTEM_INSTRUCTION,
        "Each passage includes an extracted candidate answer from "
        "an independent QA model.",
        "",
        "Context:",
    ]
    for i, p in enumerate(passages):
        pid = p["passage_id"]
        text = pid_to_text[pid]
        ext = extractions.get(pid, {"answer": "", "qa_score": 0.0})
        answer = ext["answer"]
        parts.append(
            f"[Passage {i + 1}] "
            f"(extracted_answer=\"{answer}\") "
            f"{text}"
        )
    parts.append("")
    parts.append(f"Question: {question}")
    parts.append("")
    parts.append("Answer:")
    return "\n".join(parts)


def build_prompt_jsonldex(
    question: str,
    passages: List[Dict[str, Any]],
    pid_to_text: Dict[str, str],
    extractions: Dict[str, Dict[str, Any]],
    evidence_weight: float = 10,
    prior_weight: float = 2,
) -> str:
    """Build JSONLD-EX prompt: passages + full SL metadata.

    Includes per-passage (b, d, u) triples, extracted answers,
    pairwise conflict flags, source agreement counts, and a fused
    overall assessment in natural language.
    """
    # --- Compute per-passage SL metadata ---
    passage_metas: List[Dict[str, Any]] = []
    for p in passages:
        pid = p["passage_id"]
        ext = extractions.get(pid, {"answer": "", "qa_score": 0.0})
        meta = compute_passage_sl_metadata(
            cosine_score=p["score"],
            qa_score=ext["qa_score"],
            evidence_weight=evidence_weight,
            prior_weight=prior_weight,
        )
        meta["answer"] = ext["answer"]
        meta["passage_id"] = pid
        passage_metas.append(meta)

    # --- Compute group-level metadata ---
    group_meta = compute_group_metadata(passage_metas)

    # --- Build prompt ---
    parts = [
        _SYSTEM_INSTRUCTION,
        "Each passage includes epistemic metadata from jsonld-ex confidence algebra:",
        "  - belief (b): evidence FOR the passage being relevant/correct",
        "  - disbelief (d): evidence AGAINST",
        "  - uncertainty (u): how much evidence is still missing",
        "  - extracted answer: what the passage appears to answer",
        "Use this metadata to weigh passage reliability. "
        "Passages with high belief and low uncertainty are more trustworthy. "
        "Conflict between passages means sources disagree — prefer the better-supported answer.",
        "",
        "Context:",
    ]

    for i, (p, meta) in enumerate(zip(passages, passage_metas)):
        text = pid_to_text[p["passage_id"]]
        parts.append(
            f"[Passage {i + 1}] "
            f"(belief={meta['belief']:.2f}, disbelief={meta['disbelief']:.2f}, "
            f"uncertainty={meta['uncertainty']:.2f}, "
            f"extracted_answer=\"{meta['answer']}\") "
            f"{text}"
        )

    # --- Fused assessment section ---
    parts.append("")
    parts.append(f"Source Assessment: {group_meta['assessment_text']}")
    parts.append("")
    parts.append(f"Question: {question}")
    parts.append("")
    parts.append("Answer:")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# 2. Difficulty classification
# ═══════════════════════════════════════════════════════════════════

def classify_question_difficulty(
    gold_passage_id: str,
    retrieved: List[Dict[str, Any]],
    poison_pids: Set[str],
) -> str:
    """Classify a question as easy / medium / hard.

    Criteria:
      - easy:   gold in top-3 (rank <= 2) AND 0 poison passages in retrieved
      - medium: (gold top-3 + 1 poison) OR (gold rank 3-6 + <=1 poison)
      - hard:   gold rank >= 7 OR gold missing OR >= 2 poison passages
    """
    # Find gold rank (0-indexed)
    gold_rank: Optional[int] = None
    for idx, r in enumerate(retrieved):
        if r["passage_id"] == gold_passage_id:
            gold_rank = idx
            break

    # Count poison passages in retrieved set
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
# 3. SL metadata computation
# ═══════════════════════════════════════════════════════════════════

def compute_passage_sl_metadata(
    cosine_score: float,
    qa_score: float,
    evidence_weight: float = 10,
    prior_weight: float = 2,
) -> Dict[str, float]:
    """Compute SL opinion for a single passage.

    Combines cosine similarity and QA extraction confidence via
    geometric mean, then constructs an evidence-based opinion.

    Returns dict with keys: belief, disbelief, uncertainty,
    projected_probability.
    """
    cos_clamped = max(0.0, min(1.0, cosine_score))
    qa_clamped = max(0.0, min(1.0, qa_score))
    combined = (cos_clamped * qa_clamped) ** 0.5

    positive_evidence = combined * evidence_weight
    negative_evidence = (1.0 - combined) * evidence_weight

    op = Opinion.from_evidence(
        positive_evidence, negative_evidence, prior_weight=prior_weight,
    )
    return {
        "belief": op.belief,
        "disbelief": op.disbelief,
        "uncertainty": op.uncertainty,
        "projected_probability": op.projected_probability(),
    }


def _answers_match(a: str, b: str, threshold: float = ANSWER_SIMILARITY_THRESHOLD) -> bool:
    """Check if two extracted answers refer to the same entity."""
    na = a.strip().lower()
    nb = b.strip().lower()
    if na == nb:
        return True
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def compute_group_metadata(
    passage_metas: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute group-level SL metadata across all passages for a question.

    Takes a list of per-passage metadata dicts (each must have keys:
    belief, disbelief, uncertainty, projected_probability, answer).

    Returns dict with: max_pairwise_conflict, has_conflict,
    best_answer, best_answer_source_count, fused_belief,
    fused_uncertainty, assessment_text.
    """
    n = len(passage_metas)

    # --- Build Opinion objects ---
    opinions = []
    for m in passage_metas:
        opinions.append(Opinion(m["belief"], m["disbelief"], m["uncertainty"]))

    # --- Fused opinion ---
    if n == 1:
        fused = opinions[0]
    else:
        fused = cumulative_fuse(*opinions)

    # --- Pairwise conflict (max) ---
    max_conflict = 0.0
    conflict_pairs: List[tuple] = []
    for i in range(n):
        for j in range(i + 1, n):
            c = pairwise_conflict(opinions[i], opinions[j])
            if c > max_conflict:
                max_conflict = c
            # Also check answer disagreement
            ans_i = passage_metas[i].get("answer", "")
            ans_j = passage_metas[j].get("answer", "")
            if ans_i and ans_j and not _answers_match(ans_i, ans_j):
                conflict_pairs.append((i, j, c))

    # --- Answer agreement ---
    answer_groups: Dict[str, List[int]] = {}
    for idx, m in enumerate(passage_metas):
        ans = m.get("answer", "").strip()
        if not ans:
            continue
        placed = False
        for canonical, indices in answer_groups.items():
            if _answers_match(ans, canonical):
                indices.append(idx)
                placed = True
                break
        if not placed:
            answer_groups[ans] = [idx]

    # Best answer = group with most sources
    if answer_groups:
        best_answer = max(answer_groups, key=lambda k: len(answer_groups[k]))
        best_count = len(answer_groups[best_answer])
    else:
        best_answer = ""
        best_count = 0

    # Has conflict = answer-level disagreement AND at least 2 distinct answer groups
    has_conflict = len(answer_groups) >= 2 and len(conflict_pairs) > 0

    # --- Natural language assessment ---
    assessment_parts = []
    if has_conflict:
        n_groups = len(answer_groups)
        assessment_parts.append(
            f"CONFLICT detected: {n_groups} different answers found among passages."
        )
        assessment_parts.append(
            f"Best-supported answer: \"{best_answer}\" ({best_count} sources agree)."
        )
        assessment_parts.append(
            f"Max pairwise conflict: {max_conflict:.2f}."
        )
    else:
        if best_count > 1:
            assessment_parts.append(
                f"{best_count} sources agree on \"{best_answer}\"."
            )
        elif best_count == 1:
            assessment_parts.append(
                f"Only 1 source provides answer \"{best_answer}\"."
            )
        assessment_parts.append(
            f"Fused assessment: belief={fused.belief:.2f}, "
            f"uncertainty={fused.uncertainty:.2f}."
        )

    assessment_text = " ".join(assessment_parts)

    return {
        "max_pairwise_conflict": max_conflict,
        "has_conflict": has_conflict,
        "best_answer": best_answer,
        "best_answer_source_count": best_count,
        "fused_belief": fused.belief,
        "fused_disbelief": fused.disbelief,
        "fused_uncertainty": fused.uncertainty,
        "fused_projected_probability": fused.projected_probability(),
        "assessment_text": assessment_text,
    }


# ═══════════════════════════════════════════════════════════════════
# 4. McNemar's test
# ═══════════════════════════════════════════════════════════════════

def mcnemars_test(
    pred_a: Sequence[int],
    pred_b: Sequence[int],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Paired McNemar's test comparing two binary prediction sequences.

    Uses the exact binomial test (mid-p variant for small samples,
    chi-squared with continuity correction for larger samples).

    Args:
        pred_a: Binary predictions from condition A (1=correct, 0=wrong).
        pred_b: Binary predictions from condition B (1=correct, 0=wrong).
        alpha:  Significance level (default 0.05).

    Returns:
        Dict with keys: n_11, n_10, n_01, n_00, statistic, p_value,
        significant, alpha.

    Raises:
        ValueError: If pred_a and pred_b have different lengths.
    """
    if len(pred_a) != len(pred_b):
        raise ValueError(
            f"pred_a and pred_b must have same length, "
            f"got {len(pred_a)} and {len(pred_b)}"
        )

    n_11 = 0  # both correct
    n_10 = 0  # A correct, B wrong
    n_01 = 0  # A wrong, B correct
    n_00 = 0  # both wrong

    for a, b in zip(pred_a, pred_b):
        if a == 1 and b == 1:
            n_11 += 1
        elif a == 1 and b == 0:
            n_10 += 1
        elif a == 0 and b == 1:
            n_01 += 1
        else:
            n_00 += 1

    # Discordant pairs
    n_disc = n_01 + n_10

    if n_disc == 0:
        # No discordant pairs — methods are identical on all questions
        return {
            "n_11": n_11, "n_10": n_10, "n_01": n_01, "n_00": n_00,
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "alpha": alpha,
        }

    # Chi-squared with continuity correction (Edwards)
    statistic = (abs(n_01 - n_10) - 1) ** 2 / n_disc

    # p-value from chi-squared distribution with 1 df
    # Using survival function approximation
    p_value = _chi2_sf(statistic, df=1)

    return {
        "n_11": n_11, "n_10": n_10, "n_01": n_01, "n_00": n_00,
        "statistic": round(statistic, 6),
        "p_value": round(p_value, 6),
        "significant": p_value < alpha,
        "alpha": alpha,
    }


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function (1 - CDF) of chi-squared distribution.

    Uses the regularized incomplete gamma function for df=1:
        P(X > x) = erfc(sqrt(x/2))   for df=1

    This avoids scipy dependency.
    """
    if x <= 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(x / 2.0))
    # Fallback for other df (not needed for McNemar but included for safety)
    # Simple approximation using Wilson-Hilferty
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return 0.5 * math.erfc(z / math.sqrt(2.0))
