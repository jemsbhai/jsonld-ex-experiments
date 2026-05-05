"""EN8.6 â€” Graph Merge and Diff Operations: Core Module.

Provides:
  - Synthetic KG generation with controlled ground truth
  - 4 calibration regimes (ideal, noisy, uncalibrated, adversarial)
  - 5 baselines (rdflib union, random, majority, most-recent, rdflib+conf)
  - Evaluation metrics (accuracy, ECE, diff P/R, audit completeness)
  - High-level experiment runner

All functions are deterministic given a seed.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np

from jsonld_ex.merge import merge_graphs, diff_graphs, MergeReport


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CONSTANTS â€” Value pools for synthetic KG generation
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

_FIRST_NAMES = [
    "Alice", "Bob", "Carlos", "Diana", "Elena", "Faisal", "Grace", "Hiro",
    "Ingrid", "Juan", "Kira", "Liam", "Mei", "Nadia", "Oscar", "Priya",
    "Quinn", "Raj", "Sara", "Tomasz", "Uma", "Viktor", "Wendy", "Xander",
    "Yuki", "Zara", "Amir", "Bianca", "Cheng", "Daria", "Erik", "Fatima",
    "Georg", "Hana", "Ivan", "Jia", "Karim", "Lucia", "Marco", "Nina",
    "Olga", "Pavel", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wei",
    "Ximena", "Yosef", "Amara", "Boris", "Cleo", "Dante", "Elise",
    "Felix", "Greta", "Hugo", "Isla", "Jasper", "Kaya", "Leo", "Mila",
    "Nico", "Opal", "Petra", "Remy", "Suki", "Theo", "Ursa", "Viggo",
    "Wren", "Xara", "Yael", "Zeke", "Anya", "Basil", "Cyra", "Dion",
    "Enya", "Flynn", "Gaia", "Heath", "Ivor", "Jade", "Knox", "Luna",
    "Mars", "Neve", "Odin", "Pike", "Rhea", "Seth", "Thea", "Vale",
    "Zion", "Abel", "Bree", "Cade", "Dove", "Ezra",
]

_LAST_NAMES = [
    "Smith", "Chen", "Kumar", "Garcia", "Mueller", "Tanaka", "Petrov",
    "Kim", "Ali", "Johansson", "Santos", "Williams", "Brown", "Jones",
    "Wilson", "Anderson", "Thomas", "Jackson", "White", "Harris",
    "Martin", "Lee", "Walker", "Hall", "Young", "King", "Wright",
    "Lopez", "Hill", "Scott", "Green", "Adams", "Baker", "Nelson",
    "Carter", "Mitchell", "Perez", "Roberts", "Turner", "Phillips",
    "Campbell", "Parker", "Evans", "Edwards", "Collins", "Stewart",
    "Sanchez", "Morris", "Rogers", "Reed", "Cook", "Morgan", "Bell",
    "Murphy", "Bailey", "Rivera", "Cooper", "Richardson", "Cox",
    "Howard", "Ward", "Torres", "Peterson", "Gray", "Ramirez",
    "James", "Watson", "Brooks", "Kelly", "Sanders", "Price",
    "Bennett", "Wood", "Barnes", "Ross", "Henderson", "Coleman",
    "Jenkins", "Perry", "Powell", "Long", "Patterson", "Hughes",
    "Flores", "Washington", "Butler", "Simmons", "Foster", "Gonzales",
    "Bryant", "Alexander", "Russell", "Griffin", "Diaz", "Hayes",
    "Myers", "Ford", "Hamilton", "Graham", "Sullivan", "Wallace",
]

_AFFILIATIONS = [
    "MIT", "Stanford", "Oxford", "Cambridge", "ETH Zurich", "Tsinghua",
    "Harvard", "Berkeley", "Princeton", "Caltech", "CMU", "Columbia",
    "Yale", "Imperial College", "Toronto", "Max Planck", "EPFL",
    "Georgia Tech", "Michigan", "Cornell", "Penn", "UCLA", "NYU",
    "UW Seattle", "UIUC", "USC", "Duke", "Northwestern", "Johns Hopkins",
    "Peking University", "NUS Singapore", "KAIST", "TU Munich",
    "University of Tokyo", "Melbourne", "Edinburgh", "KU Leuven",
    "Sorbonne", "Hebrew University", "IIT Bombay",
]

_FIELDS = [
    "Machine Learning", "NLP", "Computer Vision", "Robotics",
    "Reinforcement Learning", "Graph Neural Networks", "Bayesian Methods",
    "Optimization", "Information Theory", "Computational Biology",
    "Quantum Computing", "Distributed Systems", "Security",
    "Human-Computer Interaction", "Knowledge Graphs", "Causal Inference",
    "Generative Models", "Federated Learning", "Time Series",
    "Signal Processing", "Speech Recognition", "Recommender Systems",
    "Autonomous Driving", "Medical Imaging", "Formal Verification",
]

_COUNTRIES = [
    "USA", "UK", "China", "Germany", "Japan", "Canada", "France",
    "India", "South Korea", "Australia", "Switzerland", "Netherlands",
    "Sweden", "Israel", "Singapore", "Italy", "Spain", "Brazil",
    "Russia", "Taiwan",
]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DATA STRUCTURES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class CalibrationRegime(Enum):
    """Confidence calibration regimes for synthetic data."""
    IDEAL = "ideal"
    NOISY = "noisy"
    UNCALIBRATED = "uncalibrated"
    ADVERSARIAL = "adversarial"


@dataclass
class ExperimentConfig:
    """Configuration for a single EN8.6 experiment run."""
    n_entities: int
    n_sources: int
    corruption_rate: float
    calibration: CalibrationRegime
    overlap_rate: float
    seed: int
    majority_correct_fraction: float = 0.20


@dataclass
class AccuracyResult:
    """Result of merge accuracy evaluation."""
    accuracy: float
    n_correct: int
    n_conflicts: int  # n_evaluated - n_correct (or total evaluated)
    n_evaluated: int


@dataclass
class MergeResult:
    """Complete results from a single experiment configuration."""
    # H8.6a â€” overall conflict resolution accuracy
    jsonldex_highest_accuracy: float = 0.0
    jsonldex_weighted_vote_accuracy: float = 0.0
    rdflib_union_accuracy: float = 0.0
    random_accuracy: float = 0.0
    majority_vote_accuracy: float = 0.0
    most_recent_accuracy: float = 0.0
    b5_confidence_argmax_accuracy: float = 0.0

    # H8.6b â€” partitioned accuracy
    highest_majority_correct_acc: float = 0.0
    weighted_vote_majority_correct_acc: float = 0.0
    highest_standard_acc: float = 0.0
    weighted_vote_standard_acc: float = 0.0

    # H8.6d â€” ECE
    ece_noisy_or: float = 0.0
    ece_max: float = 0.0

    # H8.6e â€” diff
    diff_precision: float = 0.0
    diff_recall: float = 0.0

    # H8.6f â€” audit
    audit_completeness: float = 0.0

    # H8.6g â€” throughput
    throughput_p50_ms: float = 0.0
    throughput_p95_ms: float = 0.0
    throughput_p99_ms: float = 0.0

    # Metadata
    n_conflicts_total: int = 0
    n_majority_correct_conflicts: int = 0
    config: Optional[ExperimentConfig] = None


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1. DATA GENERATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


def generate_ground_truth_kg(
    n_entities: int,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Generate a ground-truth knowledge graph of researchers.

    Returns dict mapping entity URIs to property dicts.
    All names and ORCIDs are unique.
    """
    rng = np.random.RandomState(seed)
    gt: dict[str, dict[str, Any]] = {}

    # Generate unique names
    all_name_combos = []
    for fn in _FIRST_NAMES:
        for ln in _LAST_NAMES:
            all_name_combos.append(f"{fn} {ln}")
    rng.shuffle(all_name_combos)
    if n_entities > len(all_name_combos):
        # Extend with numbered names
        for i in range(len(all_name_combos), n_entities):
            all_name_combos.append(f"Researcher_{i:05d}")

    for i in range(n_entities):
        entity_id = f"urn:researcher:{i:05d}"
        name = all_name_combos[i]
        name_slug = name.lower().replace(" ", ".")

        gt[entity_id] = {
            "name": name,
            "affiliation": _AFFILIATIONS[rng.randint(len(_AFFILIATIONS))],
            "field": _FIELDS[rng.randint(len(_FIELDS))],
            "h_index": int(rng.randint(1, 101)),
            "email": f"{name_slug}@{_AFFILIATIONS[rng.randint(len(_AFFILIATIONS))].lower().replace(' ', '')}.edu",
            "publications_count": int(rng.randint(5, 501)),
            "country": _COUNTRIES[rng.randint(len(_COUNTRIES))],
            "active": bool(rng.random() > 0.15),
            "homepage": f"https://{name_slug}.github.io",
            "orcid": f"0000-{rng.randint(1000, 10000)}-{rng.randint(1000, 10000)}-{rng.randint(1000, 10000)}",
        }

    return gt


def generate_source_views(
    ground_truth: dict[str, dict[str, Any]],
    n_sources: int,
    overlap_rate: float,
    seed: int,
) -> list[dict[str, dict[str, Any]]]:
    """Generate overlapping source views of the ground truth.

    Each source contains a subset of entities. The overlap_rate controls
    the approximate Jaccard overlap between consecutive source pairs.

    All entities are guaranteed to appear in at least one source.
    """
    rng = np.random.RandomState(seed)
    entity_ids = sorted(ground_truth.keys())
    n = len(entity_ids)

    # Per-entity inclusion probability to achieve target Jaccard overlap
    # Jaccard(A,B) = p / (2 - p)  â†’  p = 2*J / (1 + J)
    p_include = 2.0 * overlap_rate / (1.0 + overlap_rate)
    p_include = max(0.3, min(0.95, p_include))

    sources: list[dict[str, dict[str, Any]]] = []
    for _ in range(n_sources):
        mask = rng.random(n) < p_include
        source = {}
        for j, eid in enumerate(entity_ids):
            if mask[j]:
                source[eid] = dict(ground_truth[eid])
        sources.append(source)

    # Ensure all entities appear in at least one source
    all_covered: set[str] = set()
    for s in sources:
        all_covered.update(s.keys())
    uncovered = set(entity_ids) - all_covered
    for eid in uncovered:
        idx = int(rng.randint(n_sources))
        sources[idx][eid] = dict(ground_truth[eid])

    return sources


# Properties that can be corrupted (identity props excluded)
_MUTABLE_PROPS = [
    "affiliation", "field", "h_index", "email",
    "publications_count", "country", "active", "homepage",
]


def generate_corrupted_source(
    source: dict[str, dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
    corruption_rate: float,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Corrupt a fraction of mutable properties in a source view.

    Identity properties (name, orcid) are NEVER corrupted.
    Corrupted values are plausible alternatives from the same domain.
    """
    rng = np.random.RandomState(seed)
    corrupted = {}

    for eid, entity in source.items():
        new_entity = dict(entity)
        for prop in _MUTABLE_PROPS:
            if prop not in new_entity:
                continue
            if rng.random() < corruption_rate:
                new_entity[prop] = _generate_alternative(
                    prop, entity[prop], rng
                )
        corrupted[eid] = new_entity

    return corrupted


def _generate_alternative(
    prop: str, current_value: Any, rng: np.random.RandomState,
) -> Any:
    """Generate a plausible but different value for a property."""
    if prop == "affiliation":
        alts = [a for a in _AFFILIATIONS if a != current_value]
        return alts[rng.randint(len(alts))]
    elif prop == "field":
        alts = [f for f in _FIELDS if f != current_value]
        return alts[rng.randint(len(alts))]
    elif prop == "h_index":
        # Different integer, same range
        new_val = int(rng.randint(1, 101))
        while new_val == current_value:
            new_val = int(rng.randint(1, 101))
        return new_val
    elif prop == "email":
        # Generate a different email
        fn = _FIRST_NAMES[rng.randint(len(_FIRST_NAMES))].lower()
        ln = _LAST_NAMES[rng.randint(len(_LAST_NAMES))].lower()
        dom = _AFFILIATIONS[rng.randint(len(_AFFILIATIONS))].lower().replace(" ", "")
        return f"{fn}.{ln}@{dom}.edu"
    elif prop == "publications_count":
        new_val = int(rng.randint(5, 501))
        while new_val == current_value:
            new_val = int(rng.randint(5, 501))
        return new_val
    elif prop == "country":
        alts = [c for c in _COUNTRIES if c != current_value]
        return alts[rng.randint(len(alts))]
    elif prop == "active":
        return not current_value
    elif prop == "homepage":
        fn = _FIRST_NAMES[rng.randint(len(_FIRST_NAMES))].lower()
        ln = _LAST_NAMES[rng.randint(len(_LAST_NAMES))].lower()
        return f"https://{fn}.{ln}.github.io"
    else:
        return f"alt_{current_value}"


# â”€â”€ Calibration parameters for Beta distributions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_CALIBRATION_PARAMS = {
    CalibrationRegime.IDEAL: {
        "correct": (8.0, 2.0),      # mean â‰ˆ 0.80
        "incorrect": (3.0, 5.0),     # mean â‰ˆ 0.375
    },
    CalibrationRegime.NOISY: {
        "correct": (5.0, 3.0),       # mean â‰ˆ 0.625
        "incorrect": (3.0, 4.0),     # mean â‰ˆ 0.43
    },
    CalibrationRegime.UNCALIBRATED: {
        "correct": (4.0, 4.0),       # mean â‰ˆ 0.50
        "incorrect": (4.0, 4.0),     # mean â‰ˆ 0.50
    },
    CalibrationRegime.ADVERSARIAL: {
        "correct": (3.0, 5.0),       # mean â‰ˆ 0.375
        "incorrect": (8.0, 2.0),     # mean â‰ˆ 0.80
    },
}


def assign_confidence(
    corrupted_source: dict[str, dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
    regime: CalibrationRegime,
    seed: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Assign confidence to each property based on calibration regime.

    Returns: entity_id -> {prop: {"value": val, "confidence": float}}
    """
    rng = np.random.RandomState(seed)
    params = _CALIBRATION_PARAMS[regime]
    annotated: dict[str, dict[str, dict[str, Any]]] = {}

    for eid, entity in corrupted_source.items():
        gt_entity = ground_truth[eid]
        props: dict[str, dict[str, Any]] = {}

        for prop in list(entity.keys()):
            val = entity[prop]
            is_correct = (val == gt_entity.get(prop))

            if prop in ("name", "orcid"):
                # Identity â€” no confidence annotation needed, just pass through
                props[prop] = {"value": val, "confidence": 1.0}
                continue

            if is_correct:
                a, b = params["correct"]
            else:
                a, b = params["incorrect"]

            conf = float(np.clip(rng.beta(a, b), 0.01, 0.99))
            props[prop] = {"value": val, "confidence": conf}

        annotated[eid] = props

    return annotated


def source_to_jsonld(
    annotated_source: dict[str, dict[str, dict[str, Any]]],
    source_name: str,
    extracted_at: str = "2025-06-01T00:00:00Z",
) -> dict[str, Any]:
    """Convert an annotated source to JSON-LD format.

    Each property becomes {"@value": val, "@confidence": conf, "@source": name}.
    """
    nodes = []
    for eid, props in annotated_source.items():
        node: dict[str, Any] = {
            "@id": eid,
            "@type": "Researcher",
        }
        for prop, meta in props.items():
            if prop in ("name", "orcid"):
                # Identity props: annotate but with @value wrapper
                node[prop] = {
                    "@value": meta["value"],
                    "@confidence": meta["confidence"],
                    "@source": source_name,
                    "@extractedAt": extracted_at,
                }
            else:
                node[prop] = {
                    "@value": meta["value"],
                    "@confidence": meta["confidence"],
                    "@source": source_name,
                    "@extractedAt": extracted_at,
                }
        nodes.append(node)

    return {
        "@context": {
            "@vocab": "http://example.org/researcher#",
            "confidence": "http://www.w3.org/ns/jsonld-ex/confidence",
        },
        "@graph": nodes,
    }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2. BASELINES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


def _extract_nodes_by_id(
    graphs: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Index all nodes across graphs by @id."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for g in graphs:
        nodes = g.get("@graph", [])
        if isinstance(nodes, dict):
            nodes = [nodes]
        for node in nodes:
            nid = node.get("@id")
            if nid is not None:
                buckets.setdefault(nid, []).append(node)
    return buckets


def _get_bare_value(val: Any) -> Any:
    """Extract bare value from possibly-annotated property."""
    if isinstance(val, dict) and "@value" in val:
        return val["@value"]
    return val


def _get_confidence(val: Any) -> float:
    """Extract @confidence from a property value, defaulting to 0.5."""
    if isinstance(val, dict) and "@confidence" in val:
        return float(val["@confidence"])
    return 0.5


def _get_extracted_at(val: Any) -> str:
    """Extract @extractedAt from a property value."""
    if isinstance(val, dict) and "@extractedAt" in val:
        return str(val["@extractedAt"])
    return ""


_SKIP_KEYS = frozenset({"@id", "@type", "@context"})


def baseline_rdflib_union(
    graphs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """B1: rdflib-style naive union â€” last source wins for conflicts."""
    buckets = _extract_nodes_by_id(graphs)
    merged_nodes = []

    for nid, nodes in buckets.items():
        merged: dict[str, Any] = {"@id": nid}
        # Collect types
        types: set[str] = set()
        for n in nodes:
            t = n.get("@type")
            if isinstance(t, str):
                types.add(t)
            elif isinstance(t, list):
                types.update(t)
        if len(types) == 1:
            merged["@type"] = types.pop()
        elif len(types) > 1:
            merged["@type"] = sorted(types)

        # Last-source-wins for each property
        for node in nodes:  # iterate in order; last overwrites
            for k, v in node.items():
                if k not in _SKIP_KEYS:
                    merged[k] = copy.deepcopy(v)
        merged_nodes.append(merged)

    return {"@graph": merged_nodes}


def baseline_random_choice(
    graphs: Sequence[dict[str, Any]],
    seed: int = 42,
) -> dict[str, Any]:
    """B2: For each conflict, pick uniformly at random."""
    rng = np.random.RandomState(seed)
    buckets = _extract_nodes_by_id(graphs)
    merged_nodes = []

    for nid, nodes in buckets.items():
        merged: dict[str, Any] = {"@id": nid}
        types: set[str] = set()
        for n in nodes:
            t = n.get("@type")
            if isinstance(t, str):
                types.add(t)
            elif isinstance(t, list):
                types.update(t)
        if len(types) == 1:
            merged["@type"] = types.pop()
        elif len(types) > 1:
            merged["@type"] = sorted(types)

        # Collect all values for each property
        all_props: dict[str, list[Any]] = {}
        for node in nodes:
            for k, v in node.items():
                if k not in _SKIP_KEYS:
                    all_props.setdefault(k, []).append(v)

        for k, vals in all_props.items():
            merged[k] = copy.deepcopy(vals[int(rng.randint(len(vals)))])
        merged_nodes.append(merged)

    return {"@graph": merged_nodes}


def baseline_majority_vote(
    graphs: Sequence[dict[str, Any]],
    seed: int = 42,
) -> dict[str, Any]:
    """B3: Pick the value asserted by the most sources. Ties broken randomly."""
    rng = np.random.RandomState(seed)
    buckets = _extract_nodes_by_id(graphs)
    merged_nodes = []

    for nid, nodes in buckets.items():
        merged: dict[str, Any] = {"@id": nid}
        types: set[str] = set()
        for n in nodes:
            t = n.get("@type")
            if isinstance(t, str):
                types.add(t)
            elif isinstance(t, list):
                types.update(t)
        if len(types) == 1:
            merged["@type"] = types.pop()
        elif len(types) > 1:
            merged["@type"] = sorted(types)

        all_props: dict[str, list[Any]] = {}
        for node in nodes:
            for k, v in node.items():
                if k not in _SKIP_KEYS:
                    all_props.setdefault(k, []).append(v)

        for k, vals in all_props.items():
            if len(vals) == 1:
                merged[k] = copy.deepcopy(vals[0])
            else:
                # Group by bare value
                groups: dict[Any, list[Any]] = {}
                for v in vals:
                    bv = _get_bare_value(v)
                    key = str(bv)  # hashable key
                    groups.setdefault(key, []).append(v)
                # Find max count
                max_count = max(len(g) for g in groups.values())
                winners = [g for g in groups.values() if len(g) == max_count]
                # Break ties randomly
                winner_group = winners[int(rng.randint(len(winners)))]
                merged[k] = copy.deepcopy(winner_group[0])
        merged_nodes.append(merged)

    return {"@graph": merged_nodes}


def baseline_most_recent(
    graphs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """B4: Pick the value with the most recent @extractedAt timestamp."""
    buckets = _extract_nodes_by_id(graphs)
    merged_nodes = []

    for nid, nodes in buckets.items():
        merged: dict[str, Any] = {"@id": nid}
        types: set[str] = set()
        for n in nodes:
            t = n.get("@type")
            if isinstance(t, str):
                types.add(t)
            elif isinstance(t, list):
                types.update(t)
        if len(types) == 1:
            merged["@type"] = types.pop()
        elif len(types) > 1:
            merged["@type"] = sorted(types)

        all_props: dict[str, list[Any]] = {}
        for node in nodes:
            for k, v in node.items():
                if k not in _SKIP_KEYS:
                    all_props.setdefault(k, []).append(v)

        for k, vals in all_props.items():
            if len(vals) == 1:
                merged[k] = copy.deepcopy(vals[0])
            else:
                # Pick by most recent @extractedAt
                best = vals[0]
                best_ts = _get_extracted_at(vals[0])
                for v in vals[1:]:
                    ts = _get_extracted_at(v)
                    if ts > best_ts:
                        best_ts = ts
                        best = v
                merged[k] = copy.deepcopy(best)
        merged_nodes.append(merged)

    return {"@graph": merged_nodes}


def baseline_rdflib_confidence_argmax(
    graphs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """B5: Custom rdflib merge picking highest @confidence.

    Same algorithm as jsonld-ex 'highest' strategy, implemented
    without using jsonld-ex â€” honest control for algorithmic parity.
    """
    buckets = _extract_nodes_by_id(graphs)
    merged_nodes = []

    for nid, nodes in buckets.items():
        merged: dict[str, Any] = {"@id": nid}
        types: set[str] = set()
        for n in nodes:
            t = n.get("@type")
            if isinstance(t, str):
                types.add(t)
            elif isinstance(t, list):
                types.update(t)
        if len(types) == 1:
            merged["@type"] = types.pop()
        elif len(types) > 1:
            merged["@type"] = sorted(types)

        all_props: dict[str, list[Any]] = {}
        for node in nodes:
            for k, v in node.items():
                if k not in _SKIP_KEYS:
                    all_props.setdefault(k, []).append(v)

        for k, vals in all_props.items():
            if len(vals) == 1:
                merged[k] = copy.deepcopy(vals[0])
            else:
                # Pick by highest @confidence
                best = vals[0]
                best_conf = _get_confidence(vals[0])
                for v in vals[1:]:
                    c = _get_confidence(v)
                    if c > best_conf:
                        best_conf = c
                        best = v
                merged[k] = copy.deepcopy(best)
        merged_nodes.append(merged)

    return {"@graph": merged_nodes}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 3. EVALUATION FUNCTIONS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


def evaluate_merge_accuracy(
    merged: dict[str, Any],
    ground_truth: dict[str, dict[str, Any]],
    properties_to_evaluate: set[tuple[str, str]] | None = None,
) -> AccuracyResult:
    """Compare merged JSON-LD graph against ground truth.

    Args:
        merged: Merged JSON-LD document with @graph array.
        ground_truth: Dict mapping entity_id -> {prop: value}.
        properties_to_evaluate: If provided, only evaluate these
            (entity_id, property_name) pairs. Otherwise evaluate all.

    Returns:
        AccuracyResult with accuracy, n_correct, n_conflicts, n_evaluated.
    """
    nodes = {n["@id"]: n for n in merged.get("@graph", [])}
    n_correct = 0
    n_evaluated = 0

    for eid, gt_entity in ground_truth.items():
        node = nodes.get(eid)
        if node is None:
            continue
        for prop, gt_value in gt_entity.items():
            if prop in ("name", "orcid"):
                continue  # identity props, not meaningful for conflict eval
            if properties_to_evaluate is not None:
                if (eid, prop) not in properties_to_evaluate:
                    continue

            merged_val = node.get(prop)
            if merged_val is None:
                continue

            bare = _get_bare_value(merged_val)
            n_evaluated += 1
            if bare == gt_value:
                n_correct += 1

    accuracy = n_correct / n_evaluated if n_evaluated > 0 else 1.0
    return AccuracyResult(
        accuracy=accuracy,
        n_correct=n_correct,
        n_conflicts=n_evaluated - n_correct,
        n_evaluated=n_evaluated,
    )


def compute_ece(
    predictions: Sequence[dict[str, Any]],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        predictions: List of {"confidence": float, "correct": bool}.
        n_bins: Number of equal-width bins.

    Returns:
        ECE value in [0, 1].
    """
    if len(predictions) == 0:
        return 0.0

    bins: list[list[dict]] = [[] for _ in range(n_bins)]
    for pred in predictions:
        conf = pred["confidence"]
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append(pred)

    ece = 0.0
    total = len(predictions)
    for bin_preds in bins:
        if len(bin_preds) == 0:
            continue
        avg_conf = sum(p["confidence"] for p in bin_preds) / len(bin_preds)
        avg_acc = sum(1 for p in bin_preds if p["correct"]) / len(bin_preds)
        ece += (len(bin_preds) / total) * abs(avg_acc - avg_conf)

    return ece


def evaluate_diff(
    a: dict[str, Any],
    b: dict[str, Any],
) -> dict[str, int]:
    """Evaluate diff_graphs by counting changes.

    Returns dict with n_added, n_removed, n_modified, n_unchanged.
    """
    result = diff_graphs(a, b)
    return {
        "n_added": len(result.get("added", [])),
        "n_removed": len(result.get("removed", [])),
        "n_modified": len(result.get("modified", [])),
        "n_unchanged": len(result.get("unchanged", [])),
    }


def evaluate_audit_trail(
    report: MergeReport,
    known_conflicts: set[tuple[str, str]],
) -> float:
    """Check that MergeReport records all known conflicts.

    Args:
        report: MergeReport from merge_graphs.
        known_conflicts: Set of (entity_id, property_name) pairs
            where conflicts are expected.

    Returns:
        Completeness ratio (0 to 1). 1.0 means all conflicts recorded.
    """
    if len(known_conflicts) == 0:
        return 1.0

    reported = set()
    for conflict in report.conflicts:
        reported.add((conflict.node_id, conflict.property_name))

    matched = known_conflicts & reported
    return len(matched) / len(known_conflicts)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 4. CONFLICT ANALYSIS HELPERS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


def _identify_conflicts(
    jsonld_graphs: list[dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Identify conflicted properties across multiple JSON-LD graphs.

    Returns:
        (all_conflicts, agreed_properties):
            all_conflicts: set of (entity_id, property) where at least
                two sources provide different bare values.
            agreed_properties: set of (entity_id, property) where all
                sources providing the property agree on the bare value.
    """
    buckets = _extract_nodes_by_id(jsonld_graphs)
    conflicts: set[tuple[str, str]] = set()
    agreed: set[tuple[str, str]] = set()

    for nid, nodes in buckets.items():
        if nid not in ground_truth:
            continue
        # Collect property values across sources
        prop_vals: dict[str, list[Any]] = {}
        for node in nodes:
            for k, v in node.items():
                if k not in _SKIP_KEYS and k not in ("name", "orcid"):
                    prop_vals.setdefault(k, []).append(v)

        for prop, vals in prop_vals.items():
            if len(vals) < 2:
                continue  # only one source â€” no conflict possible
            bare_vals = [_get_bare_value(v) for v in vals]
            if len(set(str(bv) for bv in bare_vals)) > 1:
                conflicts.add((nid, prop))
            else:
                agreed.add((nid, prop))

    return conflicts, agreed


def _force_majority_correct_conflicts(
    jsonld_graphs: list[dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
    conflicts: set[tuple[str, str]],
    fraction: float,
    seed: int,
) -> set[tuple[str, str]]:
    """Force a fraction of conflicts into the majority-correct pattern.

    Pattern: 2+ sources have correct value with moderate confidence
    (0.55-0.70), 1 source has wrong value with high confidence (0.85-0.95).

    Modifies jsonld_graphs IN PLACE. Returns set of (eid, prop) pairs
    that were forced into this pattern.
    """
    rng = np.random.RandomState(seed)

    # Find conflicts where 3+ sources participate
    buckets = _extract_nodes_by_id(jsonld_graphs)
    eligible = []
    for eid, prop in conflicts:
        if eid not in ground_truth:
            continue
        nodes_with_prop = []
        for gi, g in enumerate(jsonld_graphs):
            for node in g.get("@graph", []):
                if node.get("@id") == eid and prop in node:
                    nodes_with_prop.append((gi, node))
        if len(nodes_with_prop) >= 3:
            eligible.append((eid, prop, nodes_with_prop))

    n_to_force = max(1, int(len(eligible) * fraction))
    if n_to_force > len(eligible):
        n_to_force = len(eligible)

    indices = rng.choice(len(eligible), size=n_to_force, replace=False)
    majority_correct_set: set[tuple[str, str]] = set()

    for idx in indices:
        eid, prop, nodes_with_prop = eligible[idx]
        gt_value = ground_truth[eid].get(prop)
        if gt_value is None:
            continue

        # Pick one source to be the "wrong high-confidence" one
        wrong_idx = int(rng.randint(len(nodes_with_prop)))

        for i, (gi, node) in enumerate(nodes_with_prop):
            old_val = node[prop]
            if i == wrong_idx:
                # Wrong value, high confidence
                wrong_val = _generate_alternative(
                    prop, gt_value, rng
                )
                node[prop] = {
                    "@value": wrong_val,
                    "@confidence": float(rng.uniform(0.82, 0.90)),
                    "@source": old_val.get("@source", f"source_{gi}") if isinstance(old_val, dict) else f"source_{gi}",
                    "@extractedAt": old_val.get("@extractedAt", "2025-06-01T00:00:00Z") if isinstance(old_val, dict) else "2025-06-01T00:00:00Z",
                }
            else:
                # Correct value, moderate confidence
                node[prop] = {
                    "@value": gt_value,
                    "@confidence": float(rng.uniform(0.65, 0.80)),
                    "@source": old_val.get("@source", f"source_{gi}") if isinstance(old_val, dict) else f"source_{gi}",
                    "@extractedAt": old_val.get("@extractedAt", "2025-06-01T00:00:00Z") if isinstance(old_val, dict) else "2025-06-01T00:00:00Z",
                }

        majority_correct_set.add((eid, prop))

    return majority_correct_set


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 5. DIFF TEST HELPERS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


def _generate_diff_test_pairs(
    ground_truth: dict[str, dict[str, Any]],
    n_pairs: int,
    seed: int,
) -> list[tuple[dict, dict, dict[str, int]]]:
    """Generate graph pairs with known diffs for H8.6e testing.

    Returns list of (graph_a, graph_b, expected_counts) triples.
    expected_counts: {"n_added": int, "n_removed": int, "n_modified": int}
    """
    rng = np.random.RandomState(seed)
    eids = sorted(ground_truth.keys())
    pairs = []

    for i in range(n_pairs):
        # Select a subset of entities
        n_select = min(20, len(eids))
        selected = list(rng.choice(eids, size=n_select, replace=False))

        # Graph A: base graph
        nodes_a = []
        for eid in selected:
            node = {"@id": eid, "@type": "Researcher"}
            for prop in _MUTABLE_PROPS[:5]:  # use first 5 mutable props
                if prop in ground_truth[eid]:
                    node[prop] = {"@value": ground_truth[eid][prop]}
            nodes_a.append(node)
        graph_a = {"@graph": nodes_a}

        # Graph B: modified version
        nodes_b = copy.deepcopy(nodes_a)
        expected = {"n_added": 0, "n_removed": 0, "n_modified": 0}

        # Add a new node
        extra_eid = [e for e in eids if e not in selected]
        if extra_eid:
            new_eid = extra_eid[0]
            new_node = {"@id": new_eid, "@type": "Researcher"}
            new_node["field"] = {"@value": ground_truth[new_eid].get("field", "Unknown")}
            nodes_b.append(new_node)
            expected["n_added"] += 1  # whole node = ONE added entry

        # Remove a node
        if len(nodes_b) > 2:
            removed = nodes_b.pop(0)
            expected["n_removed"] += 1  # whole node = ONE removed entry

        # Modify some properties
        n_modify = min(3, len(nodes_b))
        for j in range(n_modify):
            node = nodes_b[j]
            for prop in list(node.keys()):
                if prop not in _SKIP_KEYS and isinstance(node[prop], dict):
                    old_val = node[prop].get("@value")
                    new_val = _generate_alternative(
                        prop, old_val, rng
                    ) if old_val is not None else "modified"
                    node[prop] = {"@value": new_val}
                    expected["n_modified"] += 1
                    break  # only modify one prop per node

        graph_b = {"@graph": nodes_b}
        pairs.append((graph_a, graph_b, expected))

    return pairs


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 6. HIGH-LEVEL RUNNER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


def run_single_config(config: ExperimentConfig) -> MergeResult:
    """Run a complete EN8.6 experiment for one configuration.

    Steps:
        1. Generate ground truth and source views
        2. Corrupt and assign confidence
        3. Convert to JSON-LD
        4. Force majority-correct conflicts
        5. Run jsonld-ex merge (highest + weighted_vote)
        6. Run all baselines
        7. Evaluate accuracy (overall + partitioned)
        8. Compute ECE, diff, audit trail, throughput
    """
    result = MergeResult(config=config)

    # â”€â”€ Step 1: Data generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    gt = generate_ground_truth_kg(config.n_entities, seed=config.seed)
    views = generate_source_views(
        gt, config.n_sources, config.overlap_rate, seed=config.seed
    )

    # â”€â”€ Step 2: Corrupt and assign confidence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    jsonld_graphs: list[dict[str, Any]] = []
    timestamps = [
        "2025-01-01T00:00:00Z",
        "2025-03-01T00:00:00Z",
        "2025-06-01T00:00:00Z",
        "2025-09-01T00:00:00Z",
        "2025-12-01T00:00:00Z",
    ]
    for i, view in enumerate(views):
        corrupted = generate_corrupted_source(
            view, gt,
            corruption_rate=config.corruption_rate,
            seed=config.seed + i + 1,
        )
        annotated = assign_confidence(
            corrupted, gt, config.calibration,
            seed=config.seed + i + 100,
        )
        doc = source_to_jsonld(
            annotated,
            source_name=f"source_{chr(65 + i)}",
            extracted_at=timestamps[i % len(timestamps)],
        )
        jsonld_graphs.append(doc)

    # â”€â”€ Step 3: Identify conflicts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    all_conflicts, agreed_props = _identify_conflicts(jsonld_graphs, gt)

    # â”€â”€ Step 4: Force majority-correct pattern â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    majority_correct_conflicts: set[tuple[str, str]] = set()
    if len(all_conflicts) > 0 and config.n_sources >= 3:
        majority_correct_conflicts = _force_majority_correct_conflicts(
            jsonld_graphs, gt, all_conflicts,
            fraction=config.majority_correct_fraction,
            seed=config.seed + 999,
        )
        # Re-identify conflicts after forcing
        all_conflicts, agreed_props = _identify_conflicts(jsonld_graphs, gt)

    standard_conflicts = all_conflicts - majority_correct_conflicts
    result.n_conflicts_total = len(all_conflicts)
    result.n_majority_correct_conflicts = len(majority_correct_conflicts)

    # â”€â”€ Step 5: Run jsonld-ex merge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    merged_highest, report_highest = merge_graphs(
        jsonld_graphs, conflict_strategy="highest"
    )
    merged_weighted, report_weighted = merge_graphs(
        jsonld_graphs, conflict_strategy="weighted_vote"
    )

    # â”€â”€ Step 6: Run baselines â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    b1_merged = baseline_rdflib_union(jsonld_graphs)
    b2_merged = baseline_random_choice(jsonld_graphs, seed=config.seed + 200)
    b3_merged = baseline_majority_vote(jsonld_graphs, seed=config.seed + 300)
    b4_merged = baseline_most_recent(jsonld_graphs)
    b5_merged = baseline_rdflib_confidence_argmax(jsonld_graphs)

    # â”€â”€ Step 7: Evaluate accuracy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if len(all_conflicts) > 0:
        # Overall accuracy on all conflicts
        r_highest = evaluate_merge_accuracy(merged_highest, gt, all_conflicts)
        r_weighted = evaluate_merge_accuracy(merged_weighted, gt, all_conflicts)
        r_b1 = evaluate_merge_accuracy(b1_merged, gt, all_conflicts)
        r_b2 = evaluate_merge_accuracy(b2_merged, gt, all_conflicts)
        r_b3 = evaluate_merge_accuracy(b3_merged, gt, all_conflicts)
        r_b4 = evaluate_merge_accuracy(b4_merged, gt, all_conflicts)
        r_b5 = evaluate_merge_accuracy(b5_merged, gt, all_conflicts)

        result.jsonldex_highest_accuracy = r_highest.accuracy
        result.jsonldex_weighted_vote_accuracy = r_weighted.accuracy
        result.rdflib_union_accuracy = r_b1.accuracy
        result.random_accuracy = r_b2.accuracy
        result.majority_vote_accuracy = r_b3.accuracy
        result.most_recent_accuracy = r_b4.accuracy
        result.b5_confidence_argmax_accuracy = r_b5.accuracy

        # Partitioned accuracy (H8.6b)
        if len(majority_correct_conflicts) > 0:
            r_h_mc = evaluate_merge_accuracy(
                merged_highest, gt, majority_correct_conflicts
            )
            r_w_mc = evaluate_merge_accuracy(
                merged_weighted, gt, majority_correct_conflicts
            )
            result.highest_majority_correct_acc = r_h_mc.accuracy
            result.weighted_vote_majority_correct_acc = r_w_mc.accuracy

        if len(standard_conflicts) > 0:
            r_h_std = evaluate_merge_accuracy(
                merged_highest, gt, standard_conflicts
            )
            r_w_std = evaluate_merge_accuracy(
                merged_weighted, gt, standard_conflicts
            )
            result.highest_standard_acc = r_h_std.accuracy
            result.weighted_vote_standard_acc = r_w_std.accuracy

    # â”€â”€ Step 8: ECE on agreed properties (H8.6d) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if len(agreed_props) > 0:
        # Merge with noisy_or combination
        merged_noisyor, _ = merge_graphs(
            jsonld_graphs,
            conflict_strategy="highest",
            confidence_combination="noisy_or",
        )
        # Merge with max combination
        merged_max, _ = merge_graphs(
            jsonld_graphs,
            conflict_strategy="highest",
            confidence_combination="max",
        )

        preds_noisyor = []
        preds_max = []
        nodes_noisyor = {n["@id"]: n for n in merged_noisyor.get("@graph", [])}
        nodes_max = {n["@id"]: n for n in merged_max.get("@graph", [])}

        for eid, prop in agreed_props:
            gt_val = gt.get(eid, {}).get(prop)
            if gt_val is None:
                continue

            # noisy-OR merged
            n_no = nodes_noisyor.get(eid)
            if n_no and prop in n_no:
                bv = _get_bare_value(n_no[prop])
                conf = _get_confidence(n_no[prop])
                preds_noisyor.append({
                    "confidence": conf,
                    "correct": bv == gt_val,
                })

            # max merged
            n_mx = nodes_max.get(eid)
            if n_mx and prop in n_mx:
                bv = _get_bare_value(n_mx[prop])
                conf = _get_confidence(n_mx[prop])
                preds_max.append({
                    "confidence": conf,
                    "correct": bv == gt_val,
                })

        if preds_noisyor:
            result.ece_noisy_or = compute_ece(preds_noisyor)
        if preds_max:
            result.ece_max = compute_ece(preds_max)

    # â”€â”€ Step 9: Diff testing (H8.6e) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    diff_pairs = _generate_diff_test_pairs(gt, n_pairs=10, seed=config.seed + 500)
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for graph_a, graph_b, expected in diff_pairs:
        actual = evaluate_diff(graph_a, graph_b)
        # We check structural correctness: counts should match expectations
        # TP = min(actual, expected), FP = max(0, actual - expected),
        # FN = max(0, expected - actual) for each category
        for category in ["n_added", "n_removed", "n_modified"]:
            exp = expected[category]
            act = actual[category]
            total_tp += min(act, exp)
            total_fp += max(0, act - exp)
            total_fn += max(0, exp - act)

    result.diff_precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 1.0
    )
    result.diff_recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 1.0
    )

    # â”€â”€ Step 10: Audit trail fidelity (H8.6f) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    result.audit_completeness = evaluate_audit_trail(
        report_highest, all_conflicts
    )

    # â”€â”€ Step 11: Throughput (H8.6g) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Warmup
    for _ in range(3):
        merge_graphs(jsonld_graphs, conflict_strategy="highest")

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        merge_graphs(jsonld_graphs, conflict_strategy="highest")
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    latencies.sort()
    n = len(latencies)
    result.throughput_p50_ms = latencies[n // 2]
    result.throughput_p95_ms = latencies[int(n * 0.95)]
    result.throughput_p99_ms = latencies[int(n * 0.99)]

    return result
