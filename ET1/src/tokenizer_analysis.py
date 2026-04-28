"""Tokenizer analysis for ET1 experiments.

Quantifies the token-level cost of each experimental condition (C1-C7)
and analyzes how the tokenizer fragments JSON-LD syntax tokens.

This is a critical confound analysis: if the tokenizer fragments JSON
syntax badly, structured conditions (C2-C7) pay a token tax that plain
text (C1) does not, which must be documented and controlled for.

Usage:
    python -m src.tokenizer_analysis [--model MODEL_NAME] [--n-facts N]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import Optional

from transformers import AutoTokenizer

from src.data_formatter import format_fact
from src.fact import Fact


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

_CONDITIONS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def compute_token_stats(
    facts: list[Fact],
    model_name: str,
    seed: int = 42,
) -> dict[str, dict]:
    """Compute token count statistics for each condition.

    For each condition, formats every fact and tokenizes the response.
    Reports mean, std, min, max, and total token counts.

    Args:
        facts: List of Facts to analyze.
        model_name: HuggingFace model ID (for tokenizer).
        seed: Random seed for C6 randomization.

    Returns:
        Dict mapping condition ID to stats dict.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    stats = {}
    for condition in _CONDITIONS:
        token_counts = []
        for i, fact in enumerate(facts):
            formatted = format_fact(fact, condition=condition, seed=seed + i)
            response = formatted["response"]
            tokens = tokenizer.encode(response, add_special_tokens=False)
            token_counts.append(len(tokens))

        mean = statistics.mean(token_counts)
        std = statistics.stdev(token_counts) if len(token_counts) > 1 else 0.0

        stats[condition] = {
            "mean_tokens": round(mean, 2),
            "std_tokens": round(std, 2),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "total_tokens": sum(token_counts),
            "n_facts": len(token_counts),
        }

    return stats


def analyze_token_fragmentation(
    model_name: str,
) -> dict[str, dict]:
    """Analyze how the tokenizer splits key JSON-LD syntax tokens.

    Tests a curated list of strings that appear frequently in C3/C4/C6
    responses. A well-trained tokenizer should keep common JSON patterns
    as single or few tokens; heavy fragmentation indicates a tax on
    structured conditions.

    Args:
        model_name: HuggingFace model ID (for tokenizer).

    Returns:
        Dict mapping test string to {n_tokens, tokens, token_ids}.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Strings to test: JSON syntax, JSON-LD keywords, SL opinion fields
    test_strings = [
        # JSON syntax
        '{"',
        '"}',
        '": ',
        '": "',
        '"entity"',
        '"value"',
        # JSON-LD keywords
        '"@context"',
        '"@type"',
        '"@id"',
        # jsonld-ex specific
        '"@opinion"',
        '"@source"',
        '"@method"',
        '"@valid_from"',
        # SL opinion fields
        '"belief"',
        '"disbelief"',
        '"uncertainty"',
        '"base_rate"',
        # Full small structures
        '"belief": 0.9',
        '"confidence": 0.85',
        '"@context": "https://schema.org/"',
        '"@context": ["https://schema.org/", "https://jsonld-ex.org/context/v1"]',
        # Comparison: plain text equivalents
        "belief",
        "disbelief",
        "uncertainty",
        "confidence",
        "headquartered in",
        "cross reference verification",
    ]

    results = {}
    for text in test_strings:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_strings = [tokenizer.decode([tid]) for tid in token_ids]

        results[text] = {
            "n_tokens": len(token_ids),
            "tokens": token_strings,
            "token_ids": token_ids,
        }

    return results


def compute_efficiency_ratios(
    token_stats: dict[str, dict],
    baseline: str = "C1",
) -> dict[str, float]:
    """Compute token efficiency ratios relative to a baseline condition.

    A ratio > 1 means the condition uses more tokens than baseline.

    Args:
        token_stats: Output of compute_token_stats().
        baseline: Condition to use as denominator.

    Returns:
        Dict mapping condition to ratio (baseline condition = 1.0).
    """
    baseline_mean = token_stats[baseline]["mean_tokens"]
    return {
        cond: round(stats["mean_tokens"] / baseline_mean, 3)
        for cond, stats in token_stats.items()
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_token_report(
    token_stats: dict[str, dict],
    fragmentation: dict[str, dict],
    ratios: dict[str, float],
) -> None:
    """Print a human-readable tokenizer analysis report.

    Args:
        token_stats: Per-condition token statistics.
        fragmentation: Per-string fragmentation analysis.
        ratios: Token efficiency ratios.
    """
    print("\n" + "=" * 70)
    print("TOKENIZER ANALYSIS REPORT — ET1")
    print("=" * 70)

    # --- Section 1: Token counts per condition ---
    print("\n--- Token Counts by Condition ---\n")
    print(f"  {'Cond':<6} {'Mean':>8} {'Std':>8} {'Min':>6} {'Max':>6} "
          f"{'Ratio':>7}")
    print("  " + "-" * 50)
    for cond in _CONDITIONS:
        s = token_stats.get(cond, {})
        r = ratios.get(cond, 0)
        print(f"  {cond:<6} {s.get('mean_tokens', 0):>8.1f} "
              f"{s.get('std_tokens', 0):>8.1f} "
              f"{s.get('min_tokens', 0):>6d} "
              f"{s.get('max_tokens', 0):>6d} "
              f"{r:>7.2f}x")

    # --- Section 2: Fragmentation ---
    print("\n--- JSON-LD Token Fragmentation ---\n")
    print(f"  {'String':<55} {'Tokens':>6}  Breakdown")
    print("  " + "-" * 80)

    for text, info in fragmentation.items():
        display = text if len(text) <= 52 else text[:49] + "..."
        tokens_str = " | ".join(info["tokens"])
        print(f"  {display:<55} {info['n_tokens']:>6}  [{tokens_str}]")

    # --- Section 3: Key findings ---
    print("\n--- Key Findings ---\n")

    c1_mean = token_stats.get("C1", {}).get("mean_tokens", 1)
    c4_mean = token_stats.get("C4", {}).get("mean_tokens", 1)
    c5_mean = token_stats.get("C5", {}).get("mean_tokens", 1)
    c7_mean = token_stats.get("C7", {}).get("mean_tokens", 1)

    print(f"  C4 (jsonld-ex) uses {ratios.get('C4', 0):.1f}x the tokens of "
          f"C1 (plain text)")
    print(f"  C5 (verbose text) uses {ratios.get('C5', 0):.1f}x the tokens "
          f"of C1 — designed to match C4")
    print(f"  C5/C4 ratio: {c5_mean / c4_mean:.2f} "
          f"({'well matched' if 0.7 < c5_mean / c4_mean < 1.3 else 'MISMATCH — investigate'})")

    # Count heavily fragmented JSON-LD terms (>3 tokens for a short string)
    heavy = [t for t, info in fragmentation.items()
             if info["n_tokens"] > 3 and len(t) < 20]
    if heavy:
        print(f"\n  WARNING: {len(heavy)} short strings fragment into >3 tokens:")
        for t in heavy:
            print(f"    '{t}' → {fragmentation[t]['n_tokens']} tokens")
    else:
        print("\n  No severe fragmentation detected for short JSON-LD tokens.")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Run tokenizer analysis from command line."""
    parser = argparse.ArgumentParser(
        description="ET1 Tokenizer Analysis: quantify token cost per condition"
    )
    parser.add_argument(
        "--model", type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--n-facts", type=int, default=50,
        help="Number of Meridian facts to generate for analysis",
    )
    args = parser.parse_args()

    # Generate sample facts
    print(f"Generating {args.n_facts} sample facts...")
    from src.knowledge_base import generate_knowledge_base
    from src.fact import Fact as FactCls

    kb_config = {
        "seed": 42,
        "world_name": "Meridian",
        "total_facts": args.n_facts,
        "confidence_tiers": {
            "T1_established": {"belief_range": [0.90, 0.99], "fraction": 0.30},
            "T2_probable": {"belief_range": [0.70, 0.89], "fraction": 0.25},
            "T3_uncertain": {"belief_range": [0.40, 0.69], "fraction": 0.20},
            "T4_speculative": {"belief_range": [0.15, 0.39], "fraction": 0.15},
            "T5_contested": {
                "belief_range": [0.30, 0.60],
                "disbelief_range": [0.20, 0.50],
                "fraction": 0.10,
            },
        },
    }
    kb = generate_knowledge_base(kb_config)
    facts = [FactCls.from_meridian(f) for f in kb["facts"]]

    print(f"Analyzing tokenizer: {args.model}")
    print(f"Facts: {len(facts)}")

    # Run analyses
    token_stats = compute_token_stats(facts, model_name=args.model)
    fragmentation = analyze_token_fragmentation(model_name=args.model)
    ratios = compute_efficiency_ratios(token_stats, baseline="C1")

    # Report
    print_token_report(token_stats, fragmentation, ratios)


if __name__ == "__main__":
    main()
