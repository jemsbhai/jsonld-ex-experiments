"""Meridian knowledge base generator for ET1 experiments.

Generates a fictional world ('Meridian') of entities, facts, and SL opinions
for controlled fine-tuning experiments. All entities are fictional to avoid
base-model contamination.
"""

import json
import math
import random
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Name generation (syllable-based, deterministic)
# ---------------------------------------------------------------------------

_ONSETS = [
    "b", "br", "c", "ch", "cr", "d", "dr", "f", "fl", "fr", "g", "gl",
    "gr", "h", "j", "k", "kl", "kr", "l", "m", "n", "p", "pl", "pr",
    "qu", "r", "s", "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st",
    "str", "sw", "t", "th", "tr", "v", "w", "z",
]
_NUCLEI = ["a", "e", "i", "o", "u", "ai", "ei", "ou", "ar", "or", "er", "an", "en", "on", "al"]
_CODAS = ["", "", "", "n", "r", "l", "s", "th", "x", "nd", "nt", "rn", "rd", "lt", "st"]


def _gen_syllable(rng: random.Random) -> str:
    return rng.choice(_ONSETS) + rng.choice(_NUCLEI) + rng.choice(_CODAS)


def _gen_name(rng: random.Random, syllables: int = 2) -> str:
    name = "".join(_gen_syllable(rng) for _ in range(syllables))
    return name.capitalize()


def _gen_org_name(rng: random.Random) -> str:
    suffixes = ["Industries", "Corp", "Labs", "Systems", "Group",
                "Holdings", "Technologies", "Dynamics", "Solutions", "Inc"]
    return f"{_gen_name(rng, rng.randint(2, 3))} {rng.choice(suffixes)}"


def _gen_person_name(rng: random.Random) -> str:
    return f"{_gen_name(rng, 2)} {_gen_name(rng, 2)}"


def _gen_location_name(rng: random.Random) -> str:
    suffixes = ["", "ville", "port", "burgh", "dale", "haven",
                "stead", "ford", "ridge", "vale"]
    base = _gen_name(rng, rng.randint(2, 3))
    suffix = rng.choice(suffixes)
    return base + suffix


def _gen_product_name(rng: random.Random) -> str:
    prefixes = ["Nova", "Flux", "Aero", "Vex", "Tera", "Cryo", "Pyro",
                "Hexa", "Quad", "Orbi"]
    suffixes = ["X", "Pro", "One", "Max", "Core", "Link", "Net", "Plus"]
    return f"{rng.choice(prefixes)}{_gen_name(rng, 1)}-{rng.choice(suffixes)}"


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "-").replace(".", "")


# ---------------------------------------------------------------------------
# Entity schema: types, relations, and value generators
# ---------------------------------------------------------------------------

# Train/val/test_id entity types
_MAIN_ENTITY_TYPES = ["Organization", "Person", "Location",
                      "ResearchFinding", "Product", "Event"]

# OOD-only entity types
_OOD_ENTITY_TYPES = ["SportsTeam", "GeologicalFeature", "ArtWork"]

_RELATIONS = {
    "Organization": [
        ("headquartered_in", "Location"),
        ("founded_by", "Person"),
        ("ceo", "Person"),
        ("industry", "field"),
        ("employee_count", "number"),
        ("founded_year", "year"),
    ],
    "Person": [
        ("works_at", "Organization"),
        ("born_in", "Location"),
        ("studied_at", "institution"),
        ("field_of_expertise", "field"),
        ("nationality", "demonym"),
    ],
    "Location": [
        ("located_in", "region"),
        ("population", "number"),
        ("climate", "climate"),
        ("elevation_m", "number"),
    ],
    "ResearchFinding": [
        ("discovered_by", "Person"),
        ("published_in", "journal"),
        ("year_published", "year"),
        ("field", "field"),
        ("key_result", "description"),
    ],
    "Product": [
        ("developed_by", "Organization"),
        ("release_year", "year"),
        ("category", "category"),
        ("price_usd", "number"),
    ],
    "Event": [
        ("occurred_in", "Location"),
        ("date", "date"),
        ("involved_org", "Organization"),
        ("outcome", "description"),
    ],
    # OOD types
    "SportsTeam": [
        ("based_in", "Location"),
        ("founded_year", "year"),
        ("sport", "sport"),
        ("league", "league"),
        ("coach", "Person"),
    ],
    "GeologicalFeature": [
        ("located_in", "region"),
        ("formation_period", "period"),
        ("elevation_m", "number"),
        ("feature_type", "geo_type"),
    ],
    "ArtWork": [
        ("created_by", "Person"),
        ("year_created", "year"),
        ("medium", "medium"),
        ("current_location", "Location"),
    ],
}

_FIELDS = ["quantum computing", "renewable energy", "marine biology",
           "materials science", "urban planning", "neuroscience",
           "agricultural tech", "cryptography", "climate modeling",
           "synthetic biology", "robotics", "linguistics"]

_INSTITUTIONS = ["Valorian Institute of Technology", "Cedarpoint University",
                 "Kasmir Academy of Sciences", "Thornfield College",
                 "Meridian National University", "Orindale Polytechnic"]

_JOURNALS = ["Meridian Science Review", "Valorian Journal of Physics",
             "Advances in Meridian Engineering", "Cedarpoint Letters",
             "Journal of Applied Meridian Studies"]

_CLIMATES = ["temperate oceanic", "continental", "subtropical humid",
             "arid steppe", "Mediterranean", "subarctic"]

_REGIONS = ["Valoria", "Kasmir", "Thornreach", "Orindale", "Cedarmark",
            "Westhollow", "Ashenmoor", "Brightvale"]

_DEMONYMS = ["Valorian", "Kasmirite", "Thornish", "Orindalean",
             "Cedarmarkian", "Westhollowan"]

_CATEGORIES = ["enterprise software", "consumer electronics", "biotech device",
               "industrial sensor", "communication platform", "analytics tool"]

_SPORTS = ["skyball", "waverun", "gridshot", "stonecast", "vaultrace"]

_LEAGUES = ["Meridian Premier League", "Valorian Championship",
            "Continental Skyball Association", "Inter-Regional Waverun Circuit"]

_GEO_TYPES = ["volcanic plateau", "submarine canyon", "crystalline ridge",
              "salt flat", "glacial valley", "basalt mesa"]

_PERIODS = ["Late Cretaceous", "Miocene", "Paleocene", "Pleistocene",
            "Ordovician", "Devonian"]

_MEDIUMS = ["oil on canvas", "marble sculpture", "bronze casting",
            "mixed media", "digital holograph", "woven textile"]

_METHODS = [
    "cross_reference_verification", "single_source_report",
    "automated_extraction", "expert_assessment", "sensor_measurement",
    "public_records_query", "survey_aggregation", "archival_research",
]


def _gen_value(rng: random.Random, value_type: str,
               entities: dict[str, list[dict]]) -> str:
    """Generate a plausible value string for a given value type."""
    if value_type == "Location":
        locs = entities.get("Location", [])
        if locs:
            return rng.choice(locs)["name"]
        return _gen_location_name(rng)
    elif value_type == "Person":
        persons = entities.get("Person", [])
        if persons:
            return rng.choice(persons)["name"]
        return _gen_person_name(rng)
    elif value_type == "Organization":
        orgs = entities.get("Organization", [])
        if orgs:
            return rng.choice(orgs)["name"]
        return _gen_org_name(rng)
    elif value_type == "field":
        return rng.choice(_FIELDS)
    elif value_type == "number":
        return str(rng.randint(50, 500000))
    elif value_type == "year":
        return str(rng.randint(1950, 2025))
    elif value_type == "institution":
        return rng.choice(_INSTITUTIONS)
    elif value_type == "journal":
        return rng.choice(_JOURNALS)
    elif value_type == "climate":
        return rng.choice(_CLIMATES)
    elif value_type == "region":
        return rng.choice(_REGIONS)
    elif value_type == "demonym":
        return rng.choice(_DEMONYMS)
    elif value_type == "category":
        return rng.choice(_CATEGORIES)
    elif value_type == "date":
        y = rng.randint(2015, 2025)
        m = rng.randint(1, 12)
        d = rng.randint(1, 28)
        return f"{y}-{m:02d}-{d:02d}"
    elif value_type == "description":
        descs = [
            f"demonstrated a {rng.choice(['significant', 'modest', 'preliminary'])} "
            f"{rng.choice(['correlation', 'improvement', 'reduction', 'increase'])} "
            f"in {rng.choice(_FIELDS)}",
            f"achieved {rng.choice(['partial', 'full', 'conditional'])} "
            f"{rng.choice(['agreement', 'resolution', 'integration', 'compliance'])}",
        ]
        return rng.choice(descs)
    elif value_type == "sport":
        return rng.choice(_SPORTS)
    elif value_type == "league":
        return rng.choice(_LEAGUES)
    elif value_type == "geo_type":
        return rng.choice(_GEO_TYPES)
    elif value_type == "period":
        return rng.choice(_PERIODS)
    elif value_type == "medium":
        return rng.choice(_MEDIUMS)
    else:
        return _gen_name(rng, 2)


# ---------------------------------------------------------------------------
# Question templates
# ---------------------------------------------------------------------------

_QUESTION_TEMPLATES = {
    "headquartered_in": [
        "Where is {entity} headquartered?",
        "In which city is {entity} located?",
    ],
    "founded_by": [
        "Who founded {entity}?",
    ],
    "ceo": [
        "Who is the CEO of {entity}?",
    ],
    "industry": [
        "What industry does {entity} operate in?",
    ],
    "employee_count": [
        "How many employees does {entity} have?",
    ],
    "founded_year": [
        "When was {entity} founded?",
    ],
    "works_at": [
        "Where does {entity} work?",
    ],
    "born_in": [
        "Where was {entity} born?",
    ],
    "studied_at": [
        "Where did {entity} study?",
    ],
    "field_of_expertise": [
        "What is {entity}'s field of expertise?",
    ],
    "nationality": [
        "What is {entity}'s nationality?",
    ],
    "located_in": [
        "Where is {entity} located?",
        "In which region is {entity}?",
    ],
    "population": [
        "What is the population of {entity}?",
    ],
    "climate": [
        "What is the climate of {entity}?",
    ],
    "elevation_m": [
        "What is the elevation of {entity}?",
    ],
    "discovered_by": [
        "Who discovered {entity}?",
    ],
    "published_in": [
        "Where was {entity} published?",
    ],
    "year_published": [
        "When was {entity} published?",
    ],
    "field": [
        "What field does {entity} relate to?",
    ],
    "key_result": [
        "What was the key finding of {entity}?",
    ],
    "developed_by": [
        "Who developed {entity}?",
    ],
    "release_year": [
        "When was {entity} released?",
    ],
    "category": [
        "What category is {entity}?",
    ],
    "price_usd": [
        "What is the price of {entity}?",
    ],
    "occurred_in": [
        "Where did {entity} take place?",
    ],
    "date": [
        "When did {entity} occur?",
    ],
    "involved_org": [
        "Which organization was involved in {entity}?",
    ],
    "outcome": [
        "What was the outcome of {entity}?",
    ],
    "based_in": [
        "Where is {entity} based?",
    ],
    "sport": [
        "What sport does {entity} play?",
    ],
    "league": [
        "What league does {entity} compete in?",
    ],
    "coach": [
        "Who coaches {entity}?",
    ],
    "formation_period": [
        "When was {entity} formed geologically?",
    ],
    "feature_type": [
        "What type of geological feature is {entity}?",
    ],
    "created_by": [
        "Who created {entity}?",
    ],
    "year_created": [
        "When was {entity} created?",
    ],
    "medium": [
        "What medium is {entity}?",
    ],
    "current_location": [
        "Where is {entity} currently housed?",
    ],
}


def _gen_questions(entity_name: str, relation: str,
                   rng: random.Random) -> list[str]:
    """Generate natural-language questions for a fact."""
    templates = _QUESTION_TEMPLATES.get(relation, [
        f"What is the {relation.replace('_', ' ')} of {{entity}}?",
    ])
    # Pick 1-2 questions
    n = min(len(templates), rng.randint(1, 2))
    selected = rng.sample(templates, n)
    return [t.format(entity=entity_name) for t in selected]


# ---------------------------------------------------------------------------
# SL Opinion generation
# ---------------------------------------------------------------------------

def _make_opinion(b: float, d: float, u: float,
                  base_rate: float = 0.5) -> dict[str, float]:
    """Build an SL opinion dict, guaranteeing b + d + u == 1.0 exactly.

    Strategy: round b and d to 4 decimal places, then compute u as the
    exact float complement.  This avoids the triple-independent-rounding
    bug where three separate round() calls can compound to >1e-4 error.
    """
    b = round(max(b, 0.0), 4)
    d = round(max(d, 0.0), 4)
    # If rounding pushed b+d above 1, clamp d
    if b + d > 1.0:
        d = round(1.0 - b, 4)
    # u absorbs all residual — NOT independently rounded
    u = 1.0 - b - d
    # Clamp u in case of float dust (e.g. -1e-16)
    if u < 0.0:
        u = 0.0
        d = 1.0 - b
    return {
        "belief": b,
        "disbelief": d,
        "uncertainty": u,
        "base_rate": round(base_rate, 4),
    }


def _gen_opinion_for_tier(tier: str, tier_cfg: dict,
                          rng: random.Random) -> dict[str, float]:
    """Generate a valid SL opinion consistent with a confidence tier."""
    if tier == "T5_contested":
        b_lo, b_hi = tier_cfg["belief_range"]
        d_lo, d_hi = tier_cfg["disbelief_range"]
        # Rejection sampling: b and d in range, b+d <= 1, u <= 0.30
        for _ in range(100):
            b = rng.uniform(b_lo, b_hi)
            d = rng.uniform(d_lo, d_hi)
            u = 1.0 - b - d
            if u >= 0.0 and u <= 0.30:
                return _make_opinion(b, d, u)
        # Fallback: construct valid values deterministically
        b = rng.uniform(0.38, 0.42)
        d = rng.uniform(0.35, 0.40)
        u = 1.0 - b - d
        return _make_opinion(b, d, max(u, 0.0))
    else:
        b_lo, b_hi = tier_cfg["belief_range"]
        b = rng.uniform(b_lo, b_hi)
        remainder = 1.0 - b
        # Higher tiers -> lower uncertainty; lower tiers -> higher uncertainty
        if tier == "T1_established":
            u_fraction = rng.uniform(0.3, 0.7)
        elif tier == "T2_probable":
            u_fraction = rng.uniform(0.3, 0.6)
        elif tier == "T3_uncertain":
            u_fraction = rng.uniform(0.4, 0.8)
        elif tier == "T4_speculative":
            u_fraction = rng.uniform(0.5, 0.9)
        else:
            u_fraction = 0.5
        u = remainder * u_fraction
        d = remainder - u
        return _make_opinion(b, d, u)


# ---------------------------------------------------------------------------
# Provenance generation
# ---------------------------------------------------------------------------

def _gen_provenance(tier: str, rng: random.Random) -> dict[str, Any]:
    """Generate provenance metadata consistent with the confidence tier."""
    if tier == "T1_established":
        n_sources = rng.randint(2, 4)
    elif tier == "T2_probable":
        n_sources = rng.randint(2, 3)
    elif tier in ("T3_uncertain", "T5_contested"):
        n_sources = rng.randint(1, 3)
    else:  # T4_speculative
        n_sources = 1

    sources = []
    for _ in range(n_sources):
        src_name = _gen_name(rng, 2)
        sources.append({
            "id": f"urn:meridian:source:{_slugify(src_name)}-{rng.randint(2020, 2025)}",
            "name": src_name,
            "reliability": round(rng.uniform(0.5, 0.99), 2),
        })

    return {
        "sources": sources,
        "method": rng.choice(_METHODS),
    }


# ---------------------------------------------------------------------------
# Entity creation
# ---------------------------------------------------------------------------

def _create_entities(rng: random.Random, entity_types: list[str],
                     count_per_type: int) -> dict[str, list[dict]]:
    """Create a pool of named entities by type."""
    entities: dict[str, list[dict]] = {}
    for etype in entity_types:
        ents = []
        used_names: set[str] = set()
        for _ in range(count_per_type):
            for _attempt in range(50):
                if etype == "Organization":
                    name = _gen_org_name(rng)
                elif etype == "Person":
                    name = _gen_person_name(rng)
                elif etype in ("Location", "GeologicalFeature"):
                    name = _gen_location_name(rng)
                elif etype == "ResearchFinding":
                    name = f"The {_gen_name(rng, 2)} Effect"
                elif etype == "Product":
                    name = _gen_product_name(rng)
                elif etype == "Event":
                    name = (
                        f"The {_gen_name(rng, 2)} "
                        f"{rng.choice(['Summit', 'Accord', 'Incident', 'Convention', 'Initiative'])}"
                    )
                elif etype == "SportsTeam":
                    name = (
                        f"{_gen_location_name(rng)} "
                        f"{rng.choice(['Strikers', 'Wolves', 'Titans', 'Falcons', 'Storms'])}"
                    )
                elif etype == "ArtWork":
                    name = (
                        f"{rng.choice(['The', 'A'])} {_gen_name(rng, 2)} "
                        f"{rng.choice(['Ascending', 'at Dusk', 'in Repose', 'Unbound', 'Convergence'])}"
                    )
                else:
                    name = _gen_name(rng, 3)
                if name not in used_names:
                    used_names.add(name)
                    break
            ents.append({
                "name": name,
                "type": etype,
                "id": f"urn:meridian:{_slugify(etype)}:{_slugify(name)}",
            })
        entities[etype] = ents
    return entities


# ---------------------------------------------------------------------------
# Fact generation from an entity pool
# ---------------------------------------------------------------------------

def _build_tier_sequence(n: int, tier_cfgs: dict,
                         rng: random.Random) -> list[str]:
    """Build a shuffled sequence of n tier labels respecting configured fractions.

    Guarantees at least 1 fact per tier when n >= len(tiers).
    """
    tier_names = list(tier_cfgs.keys())
    tier_sequence: list[str] = []

    if n >= len(tier_names):
        # Guarantee at least 1 per tier
        tier_sequence.extend(tier_names)
        remaining = n - len(tier_names)

        # Fill the rest proportionally
        for tier_name, tier_cfg in tier_cfgs.items():
            count = round(remaining * tier_cfg["fraction"])
            tier_sequence.extend([tier_name] * count)
    else:
        # Fewer facts than tiers: sample n distinct tiers
        tier_sequence = rng.sample(tier_names, n)

    # Pad or trim to exact count
    while len(tier_sequence) < n:
        tier_sequence.append(rng.choice(tier_names))
    tier_sequence = tier_sequence[:n]

    rng.shuffle(tier_sequence)
    return tier_sequence


def _generate_fact_pool(
    entities: dict[str, list[dict]],
    target_count: int,
    tier_cfgs: dict,
    rng: random.Random,
    all_entities_for_values: dict[str, list[dict]],
    fact_id_start: int,
) -> list[dict]:
    """Generate facts from an entity pool up to target_count.

    Generates facts by iterating over entities and their relations.
    If one pass isn't enough, cycles through entities again with
    randomly chosen relations.
    """
    tier_sequence = _build_tier_sequence(target_count, tier_cfgs, rng)
    facts: list[dict] = []
    fact_counter = fact_id_start
    tier_idx = 0

    # First pass: one fact per (entity, relation) pair
    for etype, ents in entities.items():
        relations = _RELATIONS.get(etype, [])
        for ent in ents:
            for relation_name, value_type in relations:
                if tier_idx >= target_count:
                    break
                fact = _make_fact(
                    ent, etype, relation_name, value_type,
                    tier_sequence[tier_idx], tier_cfgs,
                    rng, all_entities_for_values, fact_counter,
                )
                facts.append(fact)
                fact_counter += 1
                tier_idx += 1
            if tier_idx >= target_count:
                break
        if tier_idx >= target_count:
            break

    # Second pass: cycle if we haven't reached target
    all_etype_list = list(entities.keys())
    while len(facts) < target_count and tier_idx < len(tier_sequence):
        etype = rng.choice(all_etype_list)
        ent = rng.choice(entities[etype])
        relations = _RELATIONS.get(etype, [])
        if not relations:
            continue
        relation_name, value_type = rng.choice(relations)
        fact = _make_fact(
            ent, etype, relation_name, value_type,
            tier_sequence[tier_idx], tier_cfgs,
            rng, all_entities_for_values, fact_counter,
        )
        facts.append(fact)
        fact_counter += 1
        tier_idx += 1

    return facts[:target_count]


def _make_fact(
    ent: dict, etype: str, relation_name: str, value_type: str,
    tier: str, tier_cfgs: dict,
    rng: random.Random, all_entities: dict[str, list[dict]],
    fact_counter: int,
) -> dict:
    """Construct a single fact dict."""
    value = _gen_value(rng, value_type, all_entities)
    opinion = _gen_opinion_for_tier(tier, tier_cfgs[tier], rng)

    # Temporal metadata (~40% of facts)
    has_temporal = rng.random() < 0.40
    valid_from = None
    valid_until = None
    if has_temporal:
        y = rng.randint(2015, 2024)
        m = rng.randint(1, 12)
        d = rng.randint(1, 28)
        valid_from = f"{y}-{m:02d}-{d:02d}"
        if rng.random() < 0.5:
            y2 = rng.randint(y + 1, 2026)
            m2 = rng.randint(1, 12)
            d2 = rng.randint(1, 28)
            valid_until = f"{y2}-{m2:02d}-{d2:02d}"

    fact = {
        "id": f"F{fact_counter:05d}",
        "entity_id": ent["id"],
        "entity_name": ent["name"],
        "entity_type": etype,
        "relation": relation_name,
        "value": value,
        "tier": tier,
        "opinion": opinion,
        "provenance": _gen_provenance(tier, rng),
        "questions": _gen_questions(ent["name"], relation_name, rng),
    }
    if valid_from is not None:
        fact["valid_from"] = valid_from
    if valid_until is not None:
        fact["valid_until"] = valid_until

    return fact


# ---------------------------------------------------------------------------
# Core generation — main and OOD pools generated SEPARATELY
# ---------------------------------------------------------------------------

def generate_knowledge_base(config: dict) -> dict[str, Any]:
    """Generate the Meridian knowledge base.

    Main and OOD entity pools are generated independently with separate
    tier stratification.  This guarantees:
    - OOD facts actually exist (not starved by main facts hitting total first)
    - Each pool has its own tier distribution matching the config

    Args:
        config: The knowledge_base section of training_config.yaml.

    Returns:
        Dict with 'metadata' and 'facts' keys.
    """
    rng = random.Random(config["seed"])
    total_facts = config["total_facts"]
    tier_cfgs = config["confidence_tiers"]

    # Split target between main and OOD pools.
    # OOD pool size is derived from the splits config: test_ood gets ALL
    # OOD facts, so we generate exactly what it needs.  Main pool gets
    # everything else and must satisfy train + val + test_id.
    ood_fact_target = config["splits"]["test_ood"]
    main_fact_target = total_facts - ood_fact_target

    # Sanity check: main pool must be large enough for the other 3 splits
    main_needed = (
        config["splits"]["train"]
        + config["splits"]["val"]
        + config["splits"]["test_id"]
    )
    assert main_fact_target >= main_needed, (
        f"Main pool ({main_fact_target}) too small for "
        f"train+val+test_id ({main_needed}). "
        f"Increase total_facts or decrease test_ood."
    )

    # Entity counts: each entity yields ~4-5 facts (one per relation).
    # Over-provision entities so we don't run short.
    avg_relations_per_type = 4.5
    ents_per_main_type = max(
        3,
        math.ceil(main_fact_target / (len(_MAIN_ENTITY_TYPES) * avg_relations_per_type)),
    )
    ents_per_ood_type = max(
        2,
        math.ceil(ood_fact_target / (len(_OOD_ENTITY_TYPES) * avg_relations_per_type)),
    )

    # Create entity pools (deterministic order)
    main_entities = _create_entities(rng, _MAIN_ENTITY_TYPES, ents_per_main_type)
    ood_entities = _create_entities(rng, _OOD_ENTITY_TYPES, ents_per_ood_type)
    # Combined pool for value lookups (so OOD relations like "based_in"
    # can reference Location entities from the main pool)
    all_entities = {**main_entities, **ood_entities}

    # Generate fact pools SEPARATELY
    main_facts = _generate_fact_pool(
        main_entities, main_fact_target, tier_cfgs,
        rng, all_entities, fact_id_start=0,
    )
    ood_facts = _generate_fact_pool(
        ood_entities, ood_fact_target, tier_cfgs,
        rng, all_entities, fact_id_start=len(main_facts),
    )

    # Combine and re-id for a clean sequence
    all_facts = main_facts + ood_facts
    for i, fact in enumerate(all_facts):
        fact["id"] = f"F{i:05d}"

    assert len(all_facts) == total_facts, (
        f"Generated {len(all_facts)} facts, expected {total_facts}"
    )

    return {
        "metadata": {
            "world_name": config["world_name"],
            "seed": config["seed"],
            "total_facts": len(all_facts),
            "entity_types": list(all_entities.keys()),
            "main_fact_count": len(main_facts),
            "ood_fact_count": len(ood_facts),
        },
        "facts": all_facts,
    }


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_knowledge_base(kb: dict, config: dict) -> dict[str, list[dict]]:
    """Split KB into train/val/test_id/test_ood.

    Structural invariants enforced:
    1. test_ood contains ONLY OOD entity types -- never main types.
    2. train/val/test_id contain ONLY main entity types -- never OOD types.
    3. Every split has all 5 confidence tiers.
    4. Split sizes exactly match config.
    5. Splits are disjoint and cover all facts.
    """
    rng = random.Random(config["seed"])
    split_sizes = config["splits"]
    ood_type_set = set(_OOD_ENTITY_TYPES)

    # Hard partition: OOD facts vs main facts -- this boundary is never crossed
    ood_facts = [f for f in kb["facts"] if f["entity_type"] in ood_type_set]
    main_facts = [f for f in kb["facts"] if f["entity_type"] not in ood_type_set]

    rng.shuffle(ood_facts)
    rng.shuffle(main_facts)

    # --- test_ood: stratified sample from OOD pool only ---
    ood_target = split_sizes["test_ood"]
    test_ood = _stratified_sample(ood_facts, ood_target, rng)

    # --- Remaining main facts -> train, val, test_id ---
    test_id_target = split_sizes["test_id"]
    val_target = split_sizes["val"]
    train_target = split_sizes["train"]

    test_id = _stratified_sample(main_facts, test_id_target, rng)
    test_id_ids = {f["id"] for f in test_id}
    remaining = [f for f in main_facts if f["id"] not in test_id_ids]

    val = _stratified_sample(remaining, val_target, rng)
    val_ids = {f["id"] for f in val}
    remaining = [f for f in remaining if f["id"] not in val_ids]

    train = remaining[:train_target]

    # --- Ensure all tiers in each main-pool split via size-preserving swaps ---
    _ensure_all_tiers_by_swap(train, val, test_id, rng)

    return {
        "train": train,
        "val": val,
        "test_id": test_id,
        "test_ood": test_ood,
    }


def _stratified_sample(facts: list[dict], n: int,
                       rng: random.Random) -> list[dict]:
    """Sample n facts ensuring all 5 tiers are represented.

    Works on a COPY of the tier buckets to avoid mutating the input list.
    """
    all_tiers = ["T1_established", "T2_probable", "T3_uncertain",
                 "T4_speculative", "T5_contested"]
    by_tier: dict[str, list[dict]] = {t: [] for t in all_tiers}
    for f in facts:
        by_tier[f["tier"]].append(f)

    result: list[dict] = []

    # First: guarantee at least 1 per tier (if available)
    for tier in all_tiers:
        if by_tier[tier]:
            pick = by_tier[tier].pop(rng.randint(0, len(by_tier[tier]) - 1))
            result.append(pick)

    # Fill remainder randomly from what's left
    remaining_pool = []
    for tier_facts in by_tier.values():
        remaining_pool.extend(tier_facts)
    rng.shuffle(remaining_pool)

    needed = n - len(result)
    result.extend(remaining_pool[:max(0, needed)])

    return result[:n]


def _ensure_all_tiers_by_swap(train: list, val: list, test_id: list,
                               rng: random.Random) -> None:
    """Ensure all tiers present in train/val/test_id via SIZE-PRESERVING swaps.

    If a split is missing a tier, we find a donor split that has >=2 of
    that tier, then SWAP: move one fact of the needed tier from donor to
    target, and move one fact (any tier the target has in surplus) from
    target to donor.  This keeps both split sizes constant.

    NEVER touches test_ood -- the main/OOD boundary is inviolate.
    """
    all_tiers = {"T1_established", "T2_probable", "T3_uncertain",
                 "T4_speculative", "T5_contested"}
    splits = {"train": train, "val": val, "test_id": test_id}

    for split_name, split_facts in splits.items():
        present_tiers = {f["tier"] for f in split_facts}
        missing = all_tiers - present_tiers
        if not missing:
            continue

        for needed_tier in missing:
            # Find a donor split with >=2 of the needed tier
            for donor_name, donor_facts in splits.items():
                if donor_name == split_name:
                    continue
                donor_tier_facts = [
                    (i, f) for i, f in enumerate(donor_facts)
                    if f["tier"] == needed_tier
                ]
                if len(donor_tier_facts) < 2:
                    continue

                # Find a surplus tier in target to give back
                target_tier_counts: dict[str, int] = {}
                for f in split_facts:
                    target_tier_counts[f["tier"]] = (
                        target_tier_counts.get(f["tier"], 0) + 1
                    )
                surplus_tier = None
                for t, c in sorted(target_tier_counts.items(),
                                   key=lambda x: -x[1]):
                    if c >= 2:
                        surplus_tier = t
                        break

                if surplus_tier is None:
                    continue  # Target too small to swap safely

                # Execute the swap:
                # 1) Move needed-tier fact from donor -> target
                donor_idx, donor_fact = donor_tier_facts[0]
                donor_facts.pop(donor_idx)
                split_facts.append(donor_fact)

                # 2) Move surplus-tier fact from target -> donor
                for j, f in enumerate(split_facts):
                    if f["tier"] == surplus_tier and f["id"] != donor_fact["id"]:
                        moved_back = split_facts.pop(j)
                        donor_facts.append(moved_back)
                        break

                break  # Done with this needed_tier


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_knowledge_base(kb: dict, path: Path | str) -> None:
    """Save KB to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)


def load_knowledge_base(path: Path | str) -> dict:
    """Load KB from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
