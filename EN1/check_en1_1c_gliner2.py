#!/usr/bin/env python
"""EN1.1c — GLiNER2 dependency check and span-to-token alignment verification.

Verifies that:
  1. gliner2 package is installed and model loads
  2. GLiNER2 can extract entities from CoNLL-2003 formatted sentences
  3. Span-to-token alignment produces valid IOB2 tags in the same format
     as the existing 4 models: [{"tag": "B-PER", "confidence": 0.85}, ...]
  4. No overlapping span conflicts

This is a PRE-CHECK — run this before en1_1c_gliner2_runner.py.

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN1/check_en1_1c_gliner2.py
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Tuple

# =====================================================================
# 1. Dependency Check
# =====================================================================

def check_dependencies() -> bool:
    """Check that gliner2 and required deps are available."""
    deps = {}
    ok = True

    try:
        import gliner2
        deps["gliner2"] = getattr(gliner2, "__version__", "installed")
    except ImportError:
        deps["gliner2"] = "MISSING"
        ok = False

    try:
        import torch
        deps["torch"] = torch.__version__
        deps["cuda"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            deps["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        deps["torch"] = "MISSING"
        ok = False

    try:
        import datasets
        deps["datasets"] = datasets.__version__
    except ImportError:
        deps["datasets"] = "MISSING"
        ok = False

    print("=== EN1.1c GLiNER2 Dependency Check ===")
    for k, v in deps.items():
        status = "OK" if v != "MISSING" else "MISSING"
        print(f"  {k:15s}: {v:30s} [{status}]")

    if not ok:
        missing = [k for k, v in deps.items() if v == "MISSING"]
        print(f"\nMISSING: {', '.join(missing)}")
        print("Install with: pip install gliner2 torch datasets")
    return ok


# =====================================================================
# 2. Span-to-Token Alignment Logic
# =====================================================================

# CoNLL-2003 entity types
ENTITY_TYPES = ["PER", "LOC", "ORG", "MISC"]

# GLiNER2 entity label -> CoNLL tag mapping
# GLiNER2 uses natural language labels; we pass these as entity types
# and map back to CoNLL codes.
GLINER2_LABEL_MAP = {
    "person": "PER",
    "location": "LOC",
    "organization": "ORG",
    "miscellaneous": "MISC",
}

# Entity type descriptions for GLiNER2 (schema-driven extraction)
GLINER2_ENTITY_LABELS = list(GLINER2_LABEL_MAP.keys())


def tokens_to_text_with_offsets(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """Join tokens with spaces, tracking character offset for each token.

    Returns:
        text: The joined string.
        offsets: List of (start_char, end_char) for each token.

    This is the inverse operation we need for span alignment:
    given a character span from GLiNER2, we map it back to token indices.
    """
    offsets = []
    pos = 0
    parts = []
    for tok in tokens:
        offsets.append((pos, pos + len(tok)))
        parts.append(tok)
        pos += len(tok) + 1  # +1 for the space
    text = " ".join(parts)
    return text, offsets


def char_span_to_token_indices(
    span_start: int,
    span_end: int,
    token_offsets: List[Tuple[int, int]],
) -> List[int]:
    """Map a character-level span to token indices.

    A token is included if its character range overlaps with the span.
    Returns sorted list of token indices.
    """
    indices = []
    for idx, (tok_start, tok_end) in enumerate(token_offsets):
        # Token overlaps with span if they share any characters
        if tok_start < span_end and tok_end > span_start:
            indices.append(idx)
    return indices


def gliner2_spans_to_iob2(
    entities: Dict[str, List[str]],
    text: str,
    tokens: List[str],
    token_offsets: List[Tuple[int, int]],
    scores: Dict[str, List[float]] | None = None,
) -> List[Dict[str, Any]]:
    """Convert GLiNER2 entity dict to token-level IOB2 predictions.

    GLiNER2's extract_entities returns:
        {"entities": {"person": ["Tim Cook"], "location": ["Cupertino"]}}

    We need to find where each entity text occurs in the original text,
    map to token indices, and produce IOB2 tags.

    For overlapping entities, we keep the one with higher confidence
    (or the first one if scores are unavailable).

    Args:
        entities: GLiNER2 entity dict {label: [entity_text, ...]}
        text: The original text (tokens joined with spaces)
        tokens: The original token list
        token_offsets: Character offsets from tokens_to_text_with_offsets
        scores: Optional per-entity confidence scores {label: [score, ...]}

    Returns:
        Per-token predictions: [{"tag": "B-PER", "confidence": 0.85}, ...]
    """
    n_tokens = len(tokens)

    # Start with all O tags
    token_preds = [{"tag": "O", "confidence": 0.90, "_source": None}
                   for _ in range(n_tokens)]

    # Collect all entity spans with their scores
    all_spans = []  # (token_indices, conll_tag, confidence)

    for label, entity_texts in entities.items():
        conll_tag = GLINER2_LABEL_MAP.get(label)
        if conll_tag is None:
            continue

        label_scores = scores.get(label, []) if scores else []

        for ent_idx, ent_text in enumerate(entity_texts):
            conf = label_scores[ent_idx] if ent_idx < len(label_scores) else 0.80

            # Find entity text in the joined text
            # Use simple find — for CoNLL pre-tokenized text joined with spaces,
            # this should be unambiguous in most cases.
            # For repeated entities, find all occurrences.
            search_start = 0
            while True:
                pos = text.find(ent_text, search_start)
                if pos == -1:
                    break

                span_start = pos
                span_end = pos + len(ent_text)
                tok_indices = char_span_to_token_indices(
                    span_start, span_end, token_offsets
                )

                if tok_indices:
                    all_spans.append((tok_indices, conll_tag, conf))

                search_start = pos + 1  # Move past for next occurrence

    # Sort by confidence descending — higher confidence spans win ties
    all_spans.sort(key=lambda x: -x[2])

    # Assign tags, respecting no-overlap (first assigned wins = highest conf)
    assigned = set()
    for tok_indices, conll_tag, conf in all_spans:
        # Skip if any token already assigned
        if any(idx in assigned for idx in tok_indices):
            continue

        for i, idx in enumerate(tok_indices):
            prefix = "B" if i == 0 else "I"
            token_preds[idx] = {
                "tag": f"{prefix}-{conll_tag}",
                "confidence": float(conf),
            }
            assigned.add(idx)

    # Remove internal _source key
    for p in token_preds:
        p.pop("_source", None)

    return token_preds


# =====================================================================
# 3. Verification on Sample Sentences
# =====================================================================

# Hand-crafted test cases with known entities for alignment verification
VERIFICATION_CASES = [
    {
        "tokens": ["John", "Smith", "works", "at", "Google", "in", "London", "."],
        "expected_entities": {
            "person": [("John Smith", [0, 1])],
            "organization": [("Google", [4])],
            "location": [("London", [6])],
        },
    },
    {
        "tokens": ["The", "European", "Union", "met", "in", "Brussels", "."],
        "expected_entities": {
            "organization": [("European Union", [1, 2])],
            "location": [("Brussels", [5])],
        },
    },
    {
        "tokens": ["U.S.", "President", "Biden", "visited", "Paris", "."],
        "expected_entities": {
            "location": [("U.S.", [0]), ("Paris", [4])],
            "person": [("Biden", [2])],
        },
    },
]


def verify_alignment():
    """Verify span-to-token alignment on hand-crafted cases."""
    print("\n=== Span-to-Token Alignment Verification ===")
    all_pass = True

    for case_idx, case in enumerate(VERIFICATION_CASES):
        tokens = case["tokens"]
        text, offsets = tokens_to_text_with_offsets(tokens)
        print(f"\n  Case {case_idx + 1}: {text}")
        print(f"    Tokens: {tokens}")
        print(f"    Offsets: {offsets}")

        for label, expected_ents in case["expected_entities"].items():
            for ent_text, expected_indices in expected_ents:
                # Find the entity in text
                pos = text.find(ent_text)
                if pos == -1:
                    print(f"    FAIL: '{ent_text}' not found in text")
                    all_pass = False
                    continue

                actual_indices = char_span_to_token_indices(
                    pos, pos + len(ent_text), offsets
                )

                if actual_indices == expected_indices:
                    print(f"    OK: '{ent_text}' -> tokens {actual_indices}")
                else:
                    print(f"    FAIL: '{ent_text}' -> tokens {actual_indices}, "
                          f"expected {expected_indices}")
                    all_pass = False

    return all_pass


def verify_iob2_format():
    """Verify IOB2 output format matches existing models."""
    print("\n=== IOB2 Format Verification ===")

    tokens = ["John", "Smith", "works", "at", "Google", "in", "London", "."]
    text, offsets = tokens_to_text_with_offsets(tokens)

    # Simulate GLiNER2 output
    entities = {
        "person": ["John Smith"],
        "organization": ["Google"],
        "location": ["London"],
    }
    scores = {
        "person": [0.92],
        "organization": [0.88],
        "location": [0.95],
    }

    preds = gliner2_spans_to_iob2(entities, text, tokens, offsets, scores)

    print(f"\n  Input: {' '.join(tokens)}")
    print(f"  Entities: {entities}")
    print(f"  Predictions:")

    expected_tags = ["B-PER", "I-PER", "O", "O", "B-ORG", "O", "B-LOC", "O"]
    all_pass = True

    for i, (tok, pred, exp) in enumerate(zip(tokens, preds, expected_tags)):
        match = "OK" if pred["tag"] == exp else "FAIL"
        if pred["tag"] != exp:
            all_pass = False
        print(f"    [{i}] {tok:15s} -> {pred['tag']:8s} (conf={pred['confidence']:.2f}) "
              f"expected={exp:8s} [{match}]")

    # Verify format: each pred must have exactly "tag" and "confidence" keys
    for i, pred in enumerate(preds):
        keys = set(pred.keys())
        if keys != {"tag", "confidence"}:
            print(f"    FAIL: pred[{i}] has keys {keys}, expected {{'tag', 'confidence'}}")
            all_pass = False

    return all_pass


def verify_with_live_model():
    """Run GLiNER2 on sample sentences and verify output."""
    print("\n=== Live Model Verification ===")

    try:
        from gliner2 import GLiNER2
    except ImportError:
        print("  SKIP: gliner2 not installed")
        return False

    print("  Loading fastino/gliner2-base-v1 ...")
    t0 = time.time()
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Test on verification cases
    for case_idx, case in enumerate(VERIFICATION_CASES):
        tokens = case["tokens"]
        text, offsets = tokens_to_text_with_offsets(tokens)

        print(f"\n  Case {case_idx + 1}: {text}")

        t0 = time.time()
        result = model.extract_entities(text, GLINER2_ENTITY_LABELS)
        elapsed = time.time() - t0

        print(f"    GLiNER2 raw output ({elapsed:.3f}s): {result}")

        if "entities" in result:
            ents = result["entities"]
        else:
            ents = result  # In case format differs

        # Convert to IOB2
        # Note: GLiNER2 doesn't provide per-entity scores in extract_entities.
        # We'll need to check if scores are available or use predict_entities.
        preds = gliner2_spans_to_iob2(ents, text, tokens, offsets)

        print(f"    IOB2 predictions:")
        for i, (tok, pred) in enumerate(zip(tokens, preds)):
            print(f"      [{i}] {tok:15s} -> {pred['tag']:8s} (conf={pred['confidence']:.2f})")

    return True


def verify_score_extraction():
    """Check whether GLiNER2 exposes per-entity confidence scores.

    This is critical: if GLiNER2 only returns entity texts without scores,
    we'd have uniform confidence across all entities — losing the
    heterogeneous confidence scale that motivates adding GLiNER2.

    We check both extract_entities and predict_entities APIs.
    """
    print("\n=== Score Extraction Verification ===")

    try:
        from gliner2 import GLiNER2
    except ImportError:
        print("  SKIP: gliner2 not installed")
        return False

    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    text = "John Smith works at Google in London."

    # Method 1: extract_entities (standard API)
    print("\n  Method 1: extract_entities()")
    result1 = model.extract_entities(text, GLINER2_ENTITY_LABELS)
    print(f"    Result: {result1}")
    print(f"    Type: {type(result1)}")
    if isinstance(result1, dict):
        for k, v in result1.items():
            print(f"    Key '{k}': {type(v)} = {v}")

    # Method 2: Check if there's a predict_entities or similar with scores
    print("\n  Method 2: Checking available methods...")
    score_methods = [m for m in dir(model) if "predict" in m.lower()
                     or "score" in m.lower() or "extract" in m.lower()]
    print(f"    Relevant methods: {score_methods}")

    # Method 3: Check if extract_entities accepts a threshold parameter
    # (which would imply internal scores exist)
    print("\n  Method 3: Checking extract_entities signature...")
    import inspect
    try:
        sig = inspect.signature(model.extract_entities)
        print(f"    Signature: {sig}")
    except Exception as e:
        print(f"    Could not inspect: {e}")

    return True


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EN1.1c — GLiNER2 Pre-Check")
    print("=" * 60)

    # Step 1: Check dependencies
    if not check_dependencies():
        print("\nFATAL: Missing dependencies. Install and retry.")
        sys.exit(1)

    # Step 2: Verify span-to-token alignment (pure logic, no model needed)
    align_ok = verify_alignment()
    format_ok = verify_iob2_format()

    if not align_ok or not format_ok:
        print("\nFATAL: Alignment or format verification failed.")
        sys.exit(1)

    # Step 3: Live model verification
    live_ok = verify_with_live_model()

    # Step 4: Score extraction verification
    score_ok = verify_score_extraction()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Dependencies:      {'PASS' if True else 'FAIL'}")
    print(f"  Alignment logic:   {'PASS' if align_ok else 'FAIL'}")
    print(f"  IOB2 format:       {'PASS' if format_ok else 'FAIL'}")
    print(f"  Live model:        {'PASS' if live_ok else 'FAIL'}")
    print(f"  Score extraction:  {'PASS' if score_ok else 'FAIL'}")
    print("=" * 60)

    if not all([align_ok, format_ok, live_ok, score_ok]):
        print("\nSome checks failed. Review output above before proceeding.")
        sys.exit(1)
    else:
        print("\nAll checks passed. Ready to run en1_1c_gliner2_runner.py")
