"""FEVER dataset loader with SL opinion derivation for ET1 experiments.

Loads the FEVER (Fact Extraction and VERification) dataset and derives
Subjective Logic opinions from verification labels and evidence metadata.

Label mapping:
  SUPPORTS       → high belief, low disbelief, low uncertainty
  REFUTES        → low belief, high disbelief, low uncertainty
  NOT ENOUGH INFO → low belief, low disbelief, high uncertainty

Evidence count modulates confidence within each label category.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Optional

from src.fact import Fact, Provenance, SLOpinion, Source


# ---------------------------------------------------------------------------
# SL Opinion derivation from FEVER labels
# ---------------------------------------------------------------------------

def derive_fever_opinion(label: str, evidence_count: int) -> SLOpinion:
    """Derive an SL opinion from a FEVER label and evidence count.

    The label determines the base opinion profile; evidence count modulates
    the strength within that profile (more evidence → higher confidence).

    Args:
        label: One of "SUPPORTS", "REFUTES", "NOT ENOUGH INFO".
        evidence_count: Number of distinct evidence sentences for this claim.

    Returns:
        A valid SLOpinion (b + d + u = 1.0).
    """
    # Evidence bonus: each extra evidence sentence adds a small increment,
    # capped to prevent exceeding bounds.
    ev_bonus = min(evidence_count - 1, 5) * 0.02 if evidence_count > 0 else 0.0

    if label == "SUPPORTS":
        b = min(0.85 + ev_bonus, 0.99)
        d = 0.02
        u = 1.0 - b - d
        return SLOpinion(
            belief=round(b, 4),
            disbelief=round(d, 4),
            uncertainty=1.0 - round(b, 4) - round(d, 4),
        )

    elif label == "REFUTES":
        d = min(0.80 + ev_bonus, 0.99)
        b = 0.05
        u = 1.0 - b - d
        return SLOpinion(
            belief=round(b, 4),
            disbelief=round(d, 4),
            uncertainty=1.0 - round(b, 4) - round(d, 4),
        )

    elif label == "NOT ENOUGH INFO":
        b = 0.15
        d = 0.10
        u = 1.0 - b - d  # = 0.75
        return SLOpinion(belief=b, disbelief=d, uncertainty=u)

    else:
        raise ValueError(f"Unknown FEVER label: {label}")


def derive_fever_tier(label: str, evidence_count: int) -> str:
    """Map a FEVER label to a confidence tier for evaluation."""
    if label == "SUPPORTS":
        if evidence_count >= 2:
            return "T1_established"
        return "T2_probable"
    elif label == "REFUTES":
        return "T5_contested"
    elif label == "NOT ENOUGH INFO":
        if evidence_count == 0:
            return "T4_speculative"
        return "T3_uncertain"
    else:
        raise ValueError(f"Unknown FEVER label: {label}")


# ---------------------------------------------------------------------------
# Row grouping: FEVER has one row per evidence sentence, not per claim
# ---------------------------------------------------------------------------

def group_fever_rows(rows: list[dict]) -> dict[int, dict]:
    """Group FEVER rows by claim ID, aggregating evidence.

    FEVER stores one row per (claim, evidence_sentence) pair. A single claim
    may have multiple evidence sentences. We group by claim ID and count
    distinct evidence sources.

    Returns:
        Dict mapping claim_id → {
            "claim": str,
            "label": str,
            "evidence_count": int,
            "evidence_wiki_urls": list[str],
        }
    """
    grouped: dict[int, dict] = {}

    for row in rows:
        cid = row["id"]
        if cid not in grouped:
            grouped[cid] = {
                "claim": row["claim"],
                "label": row["label"],
                "evidence_count": 0,
                "evidence_wiki_urls": [],
            }

        # Count valid evidence (NEI has sentinel values like -1 or empty)
        wiki_url = row.get("evidence_wiki_url", "")
        if wiki_url and wiki_url.strip() and row.get("evidence_sentence_id", -1) >= 0:
            grouped[cid]["evidence_count"] += 1
            if wiki_url not in grouped[cid]["evidence_wiki_urls"]:
                grouped[cid]["evidence_wiki_urls"].append(wiki_url)

    return grouped


# ---------------------------------------------------------------------------
# Conversion to Fact objects
# ---------------------------------------------------------------------------

def _claim_to_question(claim: str) -> str:
    """Convert a FEVER claim to a verification question."""
    return f'Is the following claim true? "{claim}"?'


def _claim_to_answer(claim: str, label: str) -> str:
    """Generate a natural-language answer based on the FEVER label."""
    if label == "SUPPORTS":
        return f"The claim is supported. {claim}"
    elif label == "REFUTES":
        return f"The claim is refuted. The evidence contradicts: {claim}"
    elif label == "NOT ENOUGH INFO":
        return (
            f"There is not enough information to verify this claim. "
            f"The claim \"{claim}\" cannot be confirmed or denied "
            f"based on available evidence."
        )
    else:
        raise ValueError(f"Unknown label: {label}")


def _build_provenance(label: str, wiki_urls: list[str]) -> Provenance:
    """Build provenance from FEVER evidence metadata."""
    if not wiki_urls or label == "NOT ENOUGH INFO":
        # NEI claims have no evidence; use a placeholder source
        sources = [
            Source(
                id="urn:fever:no-evidence",
                name="No sufficient evidence",
                reliability=0.0,
            )
        ]
        method = "insufficient_evidence"
    else:
        sources = [
            Source(
                id=f"urn:wikipedia:{url}",
                name=url.replace("_", " "),
                reliability=0.85,
            )
            for url in wiki_urls
        ]
        method = "evidence_based_verification"

    return Provenance(sources=sources, method=method)


def fever_rows_to_facts(
    rows: list[dict],
    max_facts: Optional[int] = None,
    seed: int = 42,
) -> list[Fact]:
    """Convert raw FEVER rows to a list of Fact objects.

    Groups rows by claim ID, derives SL opinions, and optionally subsamples.

    Args:
        rows: Raw FEVER dataset rows (one per evidence sentence).
        max_facts: If set, subsample to this many facts.
        seed: Random seed for deterministic subsampling.

    Returns:
        List of Fact objects with dataset="fever".
    """
    rng = random.Random(seed)
    grouped = group_fever_rows(rows)

    # Convert each grouped claim to a Fact
    facts: list[Fact] = []
    for cid, info in grouped.items():
        label = info["label"]
        ev_count = info["evidence_count"]
        claim = info["claim"]
        wiki_urls = info["evidence_wiki_urls"]

        opinion = derive_fever_opinion(label, ev_count)
        tier = derive_fever_tier(label, ev_count)
        provenance = _build_provenance(label, wiki_urls)
        question = _claim_to_question(claim)
        answer = _claim_to_answer(claim, label)

        fact = Fact(
            id=f"FEVER-{cid}",
            dataset="fever",
            question=question,
            answer=answer,
            entity_name=claim[:80],  # Truncated claim as entity name
            entity_type="ClaimReview",
            relation="verification",
            opinion=opinion,
            provenance=provenance,
            tier=tier,
        )
        facts.append(fact)

    # Subsample if requested
    if max_facts is not None and len(facts) > max_facts:
        rng.shuffle(facts)
        facts = facts[:max_facts]

    # Stable sort by ID for determinism
    facts.sort(key=lambda f: f.id)

    return facts


# ---------------------------------------------------------------------------
# HuggingFace download (for real data, not used in unit tests)
# ---------------------------------------------------------------------------

def load_fever_from_hf(
    split: str = "train",
    cache_dir: Optional[str] = None,
) -> list[dict]:
    """Download FEVER from HuggingFace and return raw rows.

    Requires: pip install datasets

    Args:
        split: "train", "labelled_dev", or "paper_dev".
        cache_dir: Where to cache the download.

    Returns:
        List of dicts with FEVER fields.
    """
    from datasets import load_dataset

    ds = load_dataset("fever/fever", "v1.0", split=split, cache_dir=cache_dir)
    return [dict(row) for row in ds]
