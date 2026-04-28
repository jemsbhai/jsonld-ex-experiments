"""Quick diagnostic: inspect raw model outputs from pilot results."""

import json
import sys
from pathlib import Path

results_dir = Path("results")

for condition in ["C1", "C4"]:
    result_file = results_dir / f"smollm2-135m__{condition}__seed42.json"
    if not result_file.exists():
        print(f"MISSING: {result_file}")
        continue

    r = json.load(open(result_file))
    print(f"\n{'='*70}")
    print(f"CONDITION: {condition}")
    print(f"{'='*70}")

    agg = r["test_id"]["aggregate"]
    print(f"Accuracy: {agg['accuracy']:.4f}")
    print(f"N with confidence: {agg['n_with_confidence']}")
    print(f"N parseable: {agg['n_parseable']}")
    print(f"N abstained: {agg['n_abstained']}")

    print(f"\n--- First 8 examples (test_id) ---\n")
    for p in r["test_id"]["per_fact"][:8]:
        print(f"  Ground truth: {p['ground_truth']}")
        raw = p['raw_response'][:250].replace('\n', ' \\n ')
        print(f"  Raw response: {raw}")
        print(f"  Parsed answer: {p['parsed_answer']}")
        print(f"  Confidence: {p['stated_confidence']}")
        print(f"  Correct: {p['is_correct']}")
        print(f"  ---")
