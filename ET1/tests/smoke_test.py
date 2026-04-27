"""Smoke test: exercises the full data pipeline end-to-end.

Generates Meridian KB, converts to Facts, formats across all 7 conditions,
and saves sample outputs for inspection. Optionally loads SQuAD 2.0 if
the `datasets` library is available.

Usage:
    cd experiments/ET1
    python -m tests.smoke_test
"""

import json
import sys
import os
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_base import generate_knowledge_base, split_knowledge_base, save_knowledge_base
from src.fact import Fact
from src.data_formatter import format_fact

# Use the small config for speed
SMALL_CONFIG = {
    "seed": 42,
    "world_name": "Meridian",
    "total_facts": 100,
    "splits": {
        "train": 50,
        "val": 15,
        "test_id": 20,
        "test_ood": 15,
    },
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

ALL_CONDS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

DIVIDER = "=" * 70


def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def main():
    output_dir = Path("data/smoke_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate Meridian KB
    # ------------------------------------------------------------------
    section("1. Generating Meridian Knowledge Base (100 facts)")

    kb = generate_knowledge_base(SMALL_CONFIG)
    print(f"   Generated {kb['metadata']['total_facts']} facts")
    print(f"   Entity types: {kb['metadata']['entity_types']}")
    print(f"   Main facts: {kb['metadata']['main_fact_count']}")
    print(f"   OOD facts:  {kb['metadata']['ood_fact_count']}")

    # Save KB
    kb_path = output_dir / "meridian_kb.json"
    save_knowledge_base(kb, kb_path)
    print(f"   Saved to {kb_path}")

    # ------------------------------------------------------------------
    # 2. Split KB
    # ------------------------------------------------------------------
    section("2. Splitting Knowledge Base")

    splits = split_knowledge_base(kb, SMALL_CONFIG)
    for name, facts in splits.items():
        tiers = {}
        for f in facts:
            tiers[f["tier"]] = tiers.get(f["tier"], 0) + 1
        print(f"   {name:10s}: {len(facts):3d} facts | tiers: {dict(sorted(tiers.items()))}")

    # ------------------------------------------------------------------
    # 3. Convert to Fact objects
    # ------------------------------------------------------------------
    section("3. Converting to Fact objects")

    train_facts = [Fact.from_meridian(raw) for raw in splits["train"]]
    print(f"   Converted {len(train_facts)} training facts")

    # Pick representative examples: one high-confidence, one uncertain, one contested
    examples = []
    for tier_prefix in ["T1_", "T4_", "T5_"]:
        for f in train_facts:
            if f.tier and f.tier.startswith(tier_prefix):
                examples.append(f)
                break

    if len(examples) < 3:
        examples = train_facts[:3]

    print(f"   Selected {len(examples)} examples for formatting demo")
    for ex in examples:
        print(f"     - [{ex.tier}] {ex.entity_name}: {ex.relation} = {ex.answer}")
        print(f"       Opinion: b={ex.opinion.belief:.3f} d={ex.opinion.disbelief:.3f} "
              f"u={ex.opinion.uncertainty:.3f}")

    # ------------------------------------------------------------------
    # 4. Format across all 7 conditions
    # ------------------------------------------------------------------
    section("4. Formatting examples across all 7 conditions")

    all_formatted = {}
    for ex in examples:
        fact_formatted = {}
        for cond in ALL_CONDS:
            result = format_fact(ex, condition=cond, seed=42)
            fact_formatted[cond] = result
        all_formatted[ex.id] = fact_formatted

    # Print one example in all conditions
    demo_id = examples[0].id
    demo = all_formatted[demo_id]
    print(f"\n   Example fact: {examples[0].entity_name} [{examples[0].tier}]")
    print(f"   Question: {examples[0].question}")
    print()

    for cond in ALL_CONDS:
        cond_names = {
            "C1": "Plain Text",
            "C2": "Plain JSON",
            "C3": "JSON-LD",
            "C4": "jsonld-ex Full (TREATMENT)",
            "C5": "Verbose Text (length control)",
            "C6": "jsonld-ex Randomized (signal control)",
            "C7": "Scalar Confidence",
        }
        print(f"   --- {cond}: {cond_names[cond]} ---")
        print(f"   Prompt: {demo[cond]['prompt'][:80]}...")
        resp = demo[cond]["response"]
        # Truncate long responses for display
        if len(resp) > 300:
            resp = resp[:300] + "..."
        for line in resp.split("\n"):
            print(f"   | {line}")
        print()

    # ------------------------------------------------------------------
    # 5. Save all formatted outputs
    # ------------------------------------------------------------------
    section("5. Saving formatted outputs")

    for cond in ALL_CONDS:
        cond_dir = output_dir / cond
        cond_dir.mkdir(exist_ok=True)

        cond_data = []
        for f in train_facts:
            result = format_fact(f, condition=cond, seed=42)
            cond_data.append({
                "fact_id": f.id,
                "tier": f.tier,
                "prompt": result["prompt"],
                "response": result["response"],
            })

        out_path = cond_dir / "train.json"
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(cond_data, fp, indent=2, ensure_ascii=False)
        print(f"   {cond}: {len(cond_data)} examples -> {out_path}")

    # ------------------------------------------------------------------
    # 6. Token count comparison
    # ------------------------------------------------------------------
    section("6. Token count comparison across conditions")

    print(f"   {'Condition':<8} {'Mean tokens':>12} {'Min':>6} {'Max':>6}")
    print(f"   {'-'*8:<8} {'-'*12:>12} {'-'*6:>6} {'-'*6:>6}")

    for cond in ALL_CONDS:
        lengths = []
        for f in train_facts:
            result = format_fact(f, condition=cond, seed=42)
            lengths.append(len(result["response"].split()))

        mean_len = sum(lengths) / len(lengths)
        print(f"   {cond:<8} {mean_len:>12.1f} {min(lengths):>6} {max(lengths):>6}")

    # ------------------------------------------------------------------
    # 7. SQuAD 2.0 (optional)
    # ------------------------------------------------------------------
    section("7. SQuAD 2.0 integration (optional)")

    try:
        from datasets import load_dataset
        print("   `datasets` library found. Loading SQuAD 2.0 validation split...")

        from src.squad_loader import squad_rows_to_facts

        ds = load_dataset("rajpurkar/squad_v2", split="validation")
        rows = [dict(row) for row in ds]
        print(f"   Loaded {len(rows)} SQuAD validation rows")

        squad_facts = squad_rows_to_facts(rows, max_facts=20, seed=42)
        print(f"   Converted {len(squad_facts)} SQuAD facts")

        # Show one SQuAD example in C4
        if squad_facts:
            sf = squad_facts[0]
            print(f"\n   SQuAD example: {sf.question}")
            print(f"   Answer: {sf.answer}")
            print(f"   Tier: {sf.tier}")
            print(f"   Opinion: b={sf.opinion.belief:.3f} d={sf.opinion.disbelief:.3f} "
                  f"u={sf.opinion.uncertainty:.3f}")

            c4 = format_fact(sf, condition="C4", seed=42)
            print(f"\n   C4 response:")
            for line in c4["response"].split("\n"):
                print(f"   | {line}")

    except ImportError:
        print("   `datasets` library not installed. Skipping SQuAD.")
        print("   Install with: pip install datasets")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("SMOKE TEST COMPLETE")
    print(f"   Meridian KB: {kb['metadata']['total_facts']} facts generated and split")
    print(f"   Formatted:   {len(train_facts)} facts x {len(ALL_CONDS)} conditions = "
          f"{len(train_facts) * len(ALL_CONDS)} training examples")
    print(f"   Output dir:  {output_dir.resolve()}")
    print(f"\n   Inspect the output files in data/smoke_test/C1..C7/train.json")
    print()


if __name__ == "__main__":
    main()
