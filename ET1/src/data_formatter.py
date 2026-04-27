"""7-condition data formatter for ET1 experiments.

Takes Fact objects and renders them into instruction-tuning (prompt, response)
pairs for each experimental condition. The prompt is IDENTICAL across all
conditions; only the response format differs.

Conditions:
  C1 — Plain text (primary control)
  C2 — Plain JSON (structural control)
  C3 — JSON-LD with @context/@type/@id (semantic control)
  C4 — jsonld-ex full: SL opinions, provenance, temporal (treatment)
  C5 — Verbose text, token-matched to C4 (length control)
  C6 — jsonld-ex with randomized SL values (signal control)
  C7 — JSON with scalar confidence only (representation control)
"""

from __future__ import annotations

import json
import random
from typing import Optional

from src.fact import Fact


_VALID_CONDITIONS = {"C1", "C2", "C3", "C4", "C5", "C6", "C7"}


# ---------------------------------------------------------------------------
# Prompt generation (identical across conditions)
# ---------------------------------------------------------------------------

def _build_prompt(fact: Fact) -> str:
    """Build the instruction prompt for a fact.

    The prompt is the SAME for all 7 conditions — only the response differs.
    For SQuAD facts, includes the context paragraph.
    """
    if fact.context:
        return (
            f"Based on the following context, answer the question.\n\n"
            f"Context: {fact.context}\n\n"
            f"Question: {fact.question}"
        )
    return fact.question


# ---------------------------------------------------------------------------
# Response formatters (one per condition)
# ---------------------------------------------------------------------------

def _format_c1(fact: Fact) -> str:
    """C1: Plain text response."""
    return f"{fact.entity_name}: {fact.relation.replace('_', ' ')} is {fact.answer}."


def _format_c2(fact: Fact) -> str:
    """C2: Flat key-value JSON."""
    obj = {
        "entity": fact.entity_name,
        "relation": fact.relation,
        "value": fact.answer,
    }
    return json.dumps(obj, indent=2)


def _format_c3(fact: Fact) -> str:
    """C3: JSON-LD with @context, @type, @id — no SL annotations."""
    obj = {
        "@context": "https://schema.org/",
        "@type": fact.entity_type,
        "@id": f"urn:{fact.dataset}:{fact.id}",
        "name": fact.entity_name,
        fact.relation: fact.answer,
    }
    return json.dumps(obj, indent=2)


def _format_c4(fact: Fact) -> str:
    """C4: Full jsonld-ex with SL opinions, provenance, temporal."""
    answer_obj = {
        fact.relation: fact.answer,
        "@opinion": fact.opinion.to_dict(),
        "@source": [s.id for s in fact.provenance.sources],
        "@method": fact.provenance.method,
    }
    if fact.valid_from:
        answer_obj["@valid_from"] = fact.valid_from
    if fact.valid_until:
        answer_obj["@valid_until"] = fact.valid_until

    obj = {
        "@context": ["https://schema.org/", "https://jsonld-ex.org/context/v1"],
        "@type": fact.entity_type,
        "@id": f"urn:{fact.dataset}:{fact.id}",
        "name": fact.entity_name,
        **answer_obj,
    }
    return json.dumps(obj, indent=2)


def _format_c5(fact: Fact, c4_response: str) -> str:
    """C5: Verbose text padded to approximately match C4 token count."""
    c4_word_count = len(c4_response.split())

    # Base answer
    base = f"{fact.entity_name}: {fact.relation.replace('_', ' ')} is {fact.answer}."

    # Add metadata prose to pad toward C4 length
    source_names = [s.name for s in fact.provenance.sources]
    source_str = ", ".join(source_names) if source_names else "available records"

    padding_parts = [
        f"This information was verified through {fact.provenance.method.replace('_', ' ')}.",
        f"The sources consulted include {source_str}.",
    ]

    if fact.valid_from:
        padding_parts.append(
            f"This information has been valid since {fact.valid_from}."
        )
    if fact.valid_until:
        padding_parts.append(
            f"The validity extends until {fact.valid_until}."
        )

    # Add generic filler if still too short
    fillers = [
        "Multiple independent references confirm this information.",
        "The verification process involved cross-checking across sources.",
        "This finding is consistent with other available records in the domain.",
        "Further corroboration may be available through additional research.",
        "The reliability of the primary sources has been assessed and deemed adequate.",
    ]

    result = base + " " + " ".join(padding_parts)
    filler_idx = 0
    while len(result.split()) < c4_word_count * 0.7 and filler_idx < len(fillers):
        result += " " + fillers[filler_idx]
        filler_idx += 1

    return result


def _format_c6(fact: Fact, rng: random.Random) -> str:
    """C6: jsonld-ex structure with RANDOMIZED SL values.

    Same JSON structure as C4, but opinion values are uniformly random
    (subject to b+d+u=1). This controls for annotation structure vs signal.
    """
    # Generate random SL opinion: sample from Dirichlet(1,1,1) via gamma
    # Simpler: uniform on simplex
    r1 = rng.random()
    r2 = rng.random()
    vals = sorted([0.0, r1, r2, 1.0])
    rand_b = round(vals[1] - vals[0], 4)
    rand_d = round(vals[2] - vals[1], 4)
    rand_u = 1.0 - rand_b - rand_d  # exact complement

    rand_opinion = {
        "belief": rand_b,
        "disbelief": rand_d,
        "uncertainty": rand_u,
        "base_rate": 0.5,
    }

    answer_obj = {
        fact.relation: fact.answer,
        "@opinion": rand_opinion,
        "@source": [s.id for s in fact.provenance.sources],
        "@method": fact.provenance.method,
    }
    if fact.valid_from:
        answer_obj["@valid_from"] = fact.valid_from
    if fact.valid_until:
        answer_obj["@valid_until"] = fact.valid_until

    obj = {
        "@context": ["https://schema.org/", "https://jsonld-ex.org/context/v1"],
        "@type": fact.entity_type,
        "@id": f"urn:{fact.dataset}:{fact.id}",
        "name": fact.entity_name,
        **answer_obj,
    }
    return json.dumps(obj, indent=2)


def _format_c7(fact: Fact) -> str:
    """C7: JSON with scalar confidence (projected probability), no SL tuple."""
    obj = {
        "entity": fact.entity_name,
        "relation": fact.relation,
        "value": fact.answer,
        "confidence": round(fact.opinion.projected_probability, 4),
    }
    return json.dumps(obj, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_fact(
    fact: Fact,
    condition: str,
    seed: int = 42,
) -> dict[str, str]:
    """Format a single Fact into an instruction-tuning (prompt, response) pair.

    Args:
        fact: The Fact to format.
        condition: One of C1..C7.
        seed: Random seed (used only by C6 for randomized opinions).

    Returns:
        Dict with "prompt" and "response" keys.
    """
    if condition not in _VALID_CONDITIONS:
        raise ValueError(
            f"Invalid condition '{condition}'. "
            f"Must be one of {sorted(_VALID_CONDITIONS)}"
        )

    prompt = _build_prompt(fact)
    rng = random.Random(seed)

    if condition == "C1":
        response = _format_c1(fact)
    elif condition == "C2":
        response = _format_c2(fact)
    elif condition == "C3":
        response = _format_c3(fact)
    elif condition == "C4":
        response = _format_c4(fact)
    elif condition == "C5":
        c4_response = _format_c4(fact)
        response = _format_c5(fact, c4_response)
    elif condition == "C6":
        response = _format_c6(fact, rng)
    elif condition == "C7":
        response = _format_c7(fact)
    else:
        raise ValueError(f"Unhandled condition: {condition}")

    return {"prompt": prompt, "response": response}


def format_facts(
    facts: list[Fact],
    condition: str,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Format a batch of Facts into instruction-tuning pairs.

    Each fact gets a unique seed derived from the base seed and its index,
    ensuring C6 randomization is different per fact but deterministic.

    Args:
        facts: List of Facts to format.
        condition: One of C1..C7.
        seed: Base random seed.

    Returns:
        List of dicts with "prompt" and "response" keys.
    """
    results = []
    for i, fact in enumerate(facts):
        fact_seed = seed + i  # Deterministic per-fact seed
        results.append(format_fact(fact, condition=condition, seed=fact_seed))
    return results
