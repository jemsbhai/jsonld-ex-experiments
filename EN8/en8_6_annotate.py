"""EN8.6 Phase 2 -- LLM-Assisted Conflict Annotation via Gemini Flash 2.5.

Protocol:
    1. Load cached Phase 2 conflicts from DBpedia x Wikidata
    2. Send each conflict to Gemini Flash 2.5 for classification:
       - conflict_type: granularity | synonym | temporal | factual | format
       - correct_source: dbpedia | wikidata | both_valid | unclear
       - reasoning: brief explanation
    3. Cache all annotations to avoid re-querying
    4. Output annotated CSV for human verification of a random sample

Usage:
    $env:GEMINI_API_KEY = "your-key"
    python EN8/en8_6_annotate.py
    python EN8/en8_6_annotate.py --verify-sample 50
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests
import numpy as np

_EN8_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _EN8_DIR.parent
for _p in [str(_EN8_DIR), str(_EXPERIMENTS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE_DIR = _EN8_DIR / "data" / "phase2_cache"
RESULTS_DIR = _EN8_DIR / "results"
ANNOTATION_CACHE = CACHE_DIR / "conflict_annotations.json"

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# ═══════════════════════════════════════════════════════════════════
# GEMINI API
# ═══════════════════════════════════════════════════════════════════


def call_gemini(
    prompt: str,
    api_key: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> Optional[str]:
    """Call Gemini API and return text response."""
    url = GEMINI_URL.format(model=model)
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 8192,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url, headers=headers, params=params,
                json=body, timeout=30,
            )
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    API error after {max_retries} attempts: {e}")
                return None
    return None


# ═══════════════════════════════════════════════════════════════════
# CONFLICT ANNOTATION
# ═══════════════════════════════════════════════════════════════════

CLASSIFICATION_PROMPT = """You are annotating conflicts between two knowledge graphs (DBpedia and Wikidata) for a research paper.

Entity: {entity_name}
Property: {prop_name}
DBpedia value: {dbpedia_value}
Wikidata value: {wikidata_value}

Classify this conflict into EXACTLY ONE of these types:
- granularity: Both values are correct but at different levels of specificity (e.g., "Switzerland" vs "Zurich" — country vs city)
- synonym: Both values refer to the same concept using different words (e.g., "aerospace" vs "space industry")
- temporal: Both values were correct at different points in time (e.g., different employers at different career stages)
- factual: The values genuinely disagree and at most one can be correct (e.g., "Russia" vs "Saudi Arabia")
- format: Same information in different formats (e.g., "1974-02-28" vs "1974-01-01T00:00:00Z")

Then determine which source is correct:
- dbpedia: DBpedia has the more accurate/current value
- wikidata: Wikidata has the more accurate/current value
- both_valid: Both values are valid (typical for granularity, synonym, temporal)
- unclear: Cannot determine without additional research

Respond ONLY with a JSON object, no other text:
{{"conflict_type": "...", "correct_source": "...", "reasoning": "brief explanation"}}"""


def annotate_conflict(
    entity_name: str,
    prop_name: str,
    dbpedia_value: str,
    wikidata_value: str,
    api_key: str,
) -> dict[str, str]:
    """Classify a single conflict using Gemini."""
    prompt = CLASSIFICATION_PROMPT.format(
        entity_name=entity_name,
        prop_name=prop_name,
        dbpedia_value=dbpedia_value,
        wikidata_value=wikidata_value,
    )

    response = call_gemini(prompt, api_key, temperature=0.0)
    if response is None:
        return {
            "conflict_type": "unknown",
            "correct_source": "unclear",
            "reasoning": "API call failed",
        }

    # Detect truncated response (no closing brace)
    if response and not response.rstrip().endswith(("}", "```")):
        return {
            "conflict_type": "unknown",
            "correct_source": "unclear",
            "reasoning": f"Truncated response ({len(response)} chars)",
        }

    # Parse JSON from response
    try:
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()
        result = json.loads(text)
        # Validate fields
        valid_types = {"granularity", "synonym", "temporal", "factual", "format"}
        valid_sources = {"dbpedia", "wikidata", "both_valid", "unclear"}
        if result.get("conflict_type") not in valid_types:
            result["conflict_type"] = "unknown"
        if result.get("correct_source") not in valid_sources:
            result["correct_source"] = "unclear"
        return result
    except (json.JSONDecodeError, KeyError, TypeError):
        return {
            "conflict_type": "unknown",
            "correct_source": "unclear",
            "reasoning": f"Parse error: {response[:100]}",
        }


def load_conflicts_from_phase2() -> list[dict[str, Any]]:
    """Load all conflicts from the Phase 2 results."""
    results_file = RESULTS_DIR / "en8_6_phase2.json"
    if not results_file.exists():
        print("ERROR: Phase 2 results not found. Run en8_6_real_world.py first.")
        sys.exit(1)

    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)

    conflicts: list[dict[str, Any]] = []
    for domain in ["scientist", "company"]:
        d = data.get(domain, {})
        for c in d.get("sample_conflicts", []):
            c["domain"] = domain
            conflicts.append(c)

    # sample_conflicts only has 10 per domain — we need ALL conflicts.
    # Re-extract from cached SPARQL data.
    return conflicts


def load_all_conflicts() -> list[dict[str, Any]]:
    """Re-extract ALL conflicts from cached Phase 2 data."""
    all_conflicts: list[dict[str, Any]] = []

    for domain, entity_class, props_module in [
        ("scientist", "Scientist", "SCIENTIST_PROPS"),
        ("company", "Company", "COMPANY_PROPS"),
    ]:
        cache_dbp = CACHE_DIR / f"dbpedia_{entity_class.lower()}.json"
        cache_wd = CACHE_DIR / f"wikidata_{domain.lower()}.json"

        if not cache_dbp.exists() or not cache_wd.exists():
            print(f"  WARNING: Missing cache for {domain}, skipping")
            continue

        with open(cache_dbp, encoding="utf-8") as f:
            dbp_entities = json.load(f)
        with open(cache_wd, encoding="utf-8") as f:
            wd_data = json.load(f)

        # Import property definitions
        from en8_6_real_world import SCIENTIST_PROPS, COMPANY_PROPS, normalize_value, values_match
        props = SCIENTIST_PROPS if domain == "scientist" else COMPANY_PROPS

        for ent in dbp_entities:
            qid = ent.get("wikidata_qid", "")
            wd_props = wd_data.get(qid, {})
            dbp_props = ent.get("properties", {})
            name = ent.get("name", "")

            for prop_name, prop_def in props.items():
                dbp_val = dbp_props.get(prop_name)
                wd_val = wd_props.get(prop_name)
                prop_type = prop_def["type"]

                if dbp_val is None or wd_val is None:
                    continue
                if values_match(str(dbp_val), str(wd_val), prop_type):
                    continue

                all_conflicts.append({
                    "domain": domain,
                    "entity": name,
                    "entity_qid": qid,
                    "property": prop_name,
                    "prop_type": prop_type,
                    "dbpedia": str(dbp_val),
                    "wikidata": str(wd_val),
                })

    return all_conflicts


def annotate_all_conflicts(
    conflicts: list[dict[str, Any]],
    api_key: str,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Annotate all conflicts, using cache where available."""
    # Load existing annotations
    existing: dict[str, dict] = {}
    if ANNOTATION_CACHE.exists() and not force_refresh:
        with open(ANNOTATION_CACHE, encoding="utf-8") as f:
            cached = json.load(f)
        for item in cached:
            key = f"{item['entity_qid']}|{item['property']}"
            existing[key] = item
        print(f"  Loaded {len(existing)} cached annotations")

    annotated: list[dict[str, Any]] = []
    n_cached = 0
    n_new = 0
    n_failed = 0

    for i, conflict in enumerate(conflicts):
        key = f"{conflict['entity_qid']}|{conflict['property']}"

        if key in existing and not force_refresh:
            annotated.append(existing[key])
            n_cached += 1
            continue

        # Call Gemini
        result = annotate_conflict(
            entity_name=conflict["entity"],
            prop_name=conflict["property"],
            dbpedia_value=conflict["dbpedia"],
            wikidata_value=conflict["wikidata"],
            api_key=api_key,
        )

        entry = {**conflict, **result}
        annotated.append(entry)

        if result["conflict_type"] == "unknown":
            n_failed += 1
        else:
            n_new += 1

        # Progress
        if (i + 1) % 10 == 0 or i == len(conflicts) - 1:
            print(f"  [{i+1}/{len(conflicts)}] "
                  f"cached={n_cached} new={n_new} failed={n_failed}")

        # Rate limiting: ~15 requests per minute for free tier
        if n_new > 0 and n_new % 14 == 0:
            print("    Pausing 5s for rate limit...")
            time.sleep(5)
        else:
            time.sleep(0.5)

    # Save cache
    with open(ANNOTATION_CACHE, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(annotated)} annotations to cache")

    return annotated


# ═══════════════════════════════════════════════════════════════════
# OUTPUT AND ANALYSIS
# ═══════════════════════════════════════════════════════════════════


def export_verification_csv(
    annotated: list[dict[str, Any]],
    sample_size: int = 50,
    seed: int = 42,
) -> Path:
    """Export a random sample for human verification."""
    rng = np.random.RandomState(seed)
    n = min(sample_size, len(annotated))
    indices = rng.choice(len(annotated), size=n, replace=False)
    sample = [annotated[i] for i in sorted(indices)]

    csv_path = RESULTS_DIR / "en8_6_verification_sample.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "domain", "entity", "entity_qid", "property", "prop_type",
            "dbpedia", "wikidata",
            "conflict_type", "correct_source", "reasoning",
            "human_conflict_type", "human_correct_source", "human_notes",
        ])
        writer.writeheader()
        for item in sample:
            row = {k: item.get(k, "") for k in writer.fieldnames}
            # Leave human columns blank for manual fill
            row["human_conflict_type"] = ""
            row["human_correct_source"] = ""
            row["human_notes"] = ""
            writer.writerow(row)

    print(f"\n  Verification CSV: {csv_path}")
    print(f"  {n} conflicts sampled for human review")
    return csv_path


def print_annotation_summary(annotated: list[dict[str, Any]]) -> None:
    """Print summary statistics of the annotations."""
    print(f"\n{'='*60}")
    print("ANNOTATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total conflicts annotated: {len(annotated)}")

    # By type
    from collections import Counter
    type_counts = Counter(a.get("conflict_type", "unknown") for a in annotated)
    print(f"\n  Conflict Types:")
    for ct, count in type_counts.most_common():
        pct = count / len(annotated) * 100
        print(f"    {ct:15s}: {count:4d} ({pct:5.1f}%)")

    # By correct source
    source_counts = Counter(a.get("correct_source", "unclear") for a in annotated)
    print(f"\n  Correct Source:")
    for cs, count in source_counts.most_common():
        pct = count / len(annotated) * 100
        print(f"    {cs:15s}: {count:4d} ({pct:5.1f}%)")

    # By domain
    for domain in ["scientist", "company"]:
        domain_items = [a for a in annotated if a.get("domain") == domain]
        if not domain_items:
            continue
        print(f"\n  {domain.upper()} ({len(domain_items)} conflicts):")
        dt = Counter(a.get("conflict_type", "unknown") for a in domain_items)
        for ct, count in dt.most_common():
            print(f"    {ct:15s}: {count}")

    # Factual disagreements (the ones we'll evaluate merge accuracy on)
    factual = [a for a in annotated if a.get("conflict_type") == "factual"]
    print(f"\n  Factual disagreements: {len(factual)}")
    if factual:
        wd_correct = sum(1 for a in factual if a.get("correct_source") == "wikidata")
        dbp_correct = sum(1 for a in factual if a.get("correct_source") == "dbpedia")
        unclear = sum(1 for a in factual if a.get("correct_source") in ("unclear", "both_valid"))
        print(f"    Wikidata correct: {wd_correct}")
        print(f"    DBpedia correct:  {dbp_correct}")
        print(f"    Unclear:          {unclear}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EN8.6 LLM Conflict Annotation")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-annotation (ignore cache)")
    parser.add_argument("--verify-sample", type=int, default=0,
                        help="Export N-item verification CSV")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    # Load all conflicts
    print("Loading conflicts from Phase 2 cache...")
    conflicts = load_all_conflicts()
    print(f"  Found {len(conflicts)} conflicts")

    if len(conflicts) == 0:
        print("No conflicts to annotate. Run en8_6_real_world.py first.")
        sys.exit(1)

    # Annotate
    print(f"\nAnnotating with {GEMINI_MODEL}...")
    annotated = annotate_all_conflicts(
        conflicts, api_key, force_refresh=args.refresh
    )

    # Summary
    print_annotation_summary(annotated)

    # Verification CSV
    sample_size = args.verify_sample if args.verify_sample > 0 else 50
    export_verification_csv(annotated, sample_size=sample_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
