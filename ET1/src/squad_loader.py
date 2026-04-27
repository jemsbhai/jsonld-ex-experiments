"""SQuAD 2.0 dataset loader with SL opinion derivation for ET1 experiments.

Loads SQuAD 2.0 and derives Subjective Logic opinions from:
  - Answerability (answerable vs unanswerable)
  - Annotator agreement (pairwise token-level F1 across answer spans)

Answerable + high agreement → high belief, low uncertainty
Answerable + low agreement  → moderate belief, moderate uncertainty
Unanswerable                → low belief, high uncertainty
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Optional

from src.fact import Fact, Provenance, SLOpinion, Source


# ---------------------------------------------------------------------------
# Token-level F1 (standard SQuAD metric)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenization."""
    return text.lower().split()


def _token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between two answer strings."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_answer_agreement(answers: list[str]) -> float:
    """Compute mean pairwise token-level F1 across annotator answers.

    Args:
        answers: List of answer strings from different annotators.

    Returns:
        Mean pairwise F1 in [0, 1]. Returns 0.0 for empty list,
        1.0 for a single annotator (no disagreement possible).
    """
    if len(answers) == 0:
        return 0.0
    if len(answers) == 1:
        return 1.0

    total_f1 = 0.0
    n_pairs = 0
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            total_f1 += _token_f1(answers[i], answers[j])
            n_pairs += 1

    return total_f1 / n_pairs if n_pairs > 0 else 0.0


# ---------------------------------------------------------------------------
# SL Opinion derivation
# ---------------------------------------------------------------------------

def derive_squad_opinion(is_answerable: bool, agreement: float) -> SLOpinion:
    """Derive an SL opinion from answerability and annotator agreement.

    Args:
        is_answerable: Whether the question has an answer in the context.
        agreement: Mean pairwise F1 across annotator answers (0-1).

    Returns:
        A valid SLOpinion.
    """
    if not is_answerable:
        # Unanswerable: high uncertainty, low belief
        b = 0.05
        d = 0.15
        u = 0.80
        return SLOpinion(belief=b, disbelief=d, uncertainty=u)

    # Answerable: belief scales with agreement
    # agreement=1.0 → b=0.95, u=0.03
    # agreement=0.0 → b=0.30, u=0.50
    b = 0.30 + 0.65 * agreement
    # Uncertainty decreases as agreement increases
    u = 0.50 - 0.47 * agreement
    # Clamp
    b = min(max(b, 0.0), 0.99)
    u = min(max(u, 0.01), 1.0)
    d = 1.0 - b - u
    d = max(d, 0.0)

    # Normalize via the same pattern as knowledge_base._make_opinion
    b = round(b, 4)
    d = round(d, 4)
    if b + d > 1.0:
        d = round(1.0 - b, 4)
    u = 1.0 - b - d
    if u < 0.0:
        u = 0.0
        d = 1.0 - b

    return SLOpinion(belief=b, disbelief=d, uncertainty=u)


def derive_squad_tier(is_answerable: bool, agreement: float) -> str:
    """Map SQuAD answerability and agreement to a confidence tier."""
    if not is_answerable:
        return "T4_speculative"
    if agreement >= 0.8:
        return "T1_established"
    if agreement >= 0.5:
        return "T2_probable"
    if agreement >= 0.3:
        return "T3_uncertain"
    return "T4_speculative"


# ---------------------------------------------------------------------------
# Conversion to Fact objects
# ---------------------------------------------------------------------------

def squad_rows_to_facts(
    rows: list[dict],
    max_facts: Optional[int] = None,
    seed: int = 42,
) -> list[Fact]:
    """Convert SQuAD 2.0 rows to Fact objects.

    Each row is one question. Answerable questions have answer spans;
    unanswerable questions have empty answer lists.

    Args:
        rows: Raw SQuAD 2.0 dataset rows.
        max_facts: If set, subsample to this many facts.
        seed: Random seed for deterministic subsampling.

    Returns:
        List of Fact objects with dataset="squad".
    """
    rng = random.Random(seed)

    facts: list[Fact] = []
    for row in rows:
        qid = row["id"]
        title = row.get("title", "Unknown")
        context = row["context"]
        question = row["question"]
        answer_texts = row["answers"]["text"]

        is_answerable = len(answer_texts) > 0
        agreement = compute_answer_agreement(answer_texts)

        opinion = derive_squad_opinion(is_answerable, agreement)
        tier = derive_squad_tier(is_answerable, agreement)

        if is_answerable:
            # Use the first (or most common) answer
            answer = answer_texts[0]
        else:
            answer = (
                "This question cannot be answered based on the provided context."
            )

        # Provenance from the Wikipedia article title
        source = Source(
            id=f"urn:squad:{qid}",
            name=title,
            reliability=0.85 if is_answerable else 0.0,
        )
        provenance = Provenance(
            sources=[source],
            method="extractive_qa_annotation",
        )

        if not question.endswith("?"):
            question = question + "?"

        fact = Fact(
            id=f"SQUAD-{qid}",
            dataset="squad",
            question=question,
            answer=answer,
            entity_name=title,
            entity_type="Answer",
            relation="extractive_qa",
            opinion=opinion,
            provenance=provenance,
            tier=tier,
            context=context,
        )
        facts.append(fact)

    # Subsample if requested
    if max_facts is not None and len(facts) > max_facts:
        rng.shuffle(facts)
        facts = facts[:max_facts]

    # Stable sort for determinism
    facts.sort(key=lambda f: f.id)

    return facts


# ---------------------------------------------------------------------------
# HuggingFace download
# ---------------------------------------------------------------------------

def load_squad_from_hf(
    split: str = "train",
    cache_dir: Optional[str] = None,
) -> list[dict]:
    """Download SQuAD 2.0 from HuggingFace and return raw rows.

    Requires: pip install datasets

    Args:
        split: "train" or "validation".
        cache_dir: Where to cache the download.

    Returns:
        List of dicts with SQuAD fields.
    """
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split=split, cache_dir=cache_dir)
    return [dict(row) for row in ds]
