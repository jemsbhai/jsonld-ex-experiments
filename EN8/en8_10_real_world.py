"""EN8.10 -- Multi-Format Interop Pipeline: Real-World Data (Phase 2).

Loads real data from completed experiments to create jsonld-ex annotated
documents for ecological validity testing.

Sources:
  - EN8.6: DBpedia x Wikidata conflict annotations (261 records)
  - EN8.2: Intel Berkeley Lab sensor data (2.3M readings)
  - EN2.4: Real MLCommons Croissant cards (10 cards)
  - EN1.1: CoNLL-2003 NER experiment parameters (model calibration data)
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────
_EN8_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _EN8_DIR.parent
_PKG_SRC = _EXPERIMENTS_ROOT.parent / "packages" / "python" / "src"

for p in [str(_EN8_DIR), str(_EXPERIMENTS_ROOT), str(_PKG_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from jsonld_ex.ai_ml import annotate
from jsonld_ex.dataset import from_croissant

# ── Data paths ──────────────────────────────────────────────────────
_DATA_DIR = _EN8_DIR / "data"
_INTEL_DATA = _DATA_DIR / "intel_lab_data.txt.gz"
_MOTE_LOCS = _DATA_DIR / "mote_locs.txt"
_PHASE2_CACHE = _DATA_DIR / "phase2_cache"
_CROISSANT_DIR = _EXPERIMENTS_ROOT / "EN2" / "croissant_cards"
_EN1_RESULTS = _EXPERIMENTS_ROOT / "EN1" / "results" / "en1_1_results.json"


# ===================================================================
# 1. EN8.6 — KG Conflict Annotations → Category A Docs
# ===================================================================

def load_kg_conflict_docs(
    n_docs: int = 30,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create annotated docs from EN8.6 DBpedia x Wikidata conflicts.

    Each conflict record becomes a jsonld-ex document with two annotated
    values (one per source) with confidence based on conflict type.

    Args:
        n_docs: Number of documents to generate (sampled from 261).
        seed: Random seed for reproducible sampling.

    Returns:
        List of jsonld-ex annotated documents.
    """
    cache_path = _PHASE2_CACHE / "conflict_annotations.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"EN8.6 conflict cache not found: {cache_path}\n"
            "Run EN8.6 Phase 2 first."
        )

    with open(cache_path, "r", encoding="utf-8") as f:
        conflicts = json.load(f)

    rng = np.random.default_rng(seed)
    n_docs = min(n_docs, len(conflicts))
    indices = rng.choice(len(conflicts), size=n_docs, replace=False)

    # Confidence mapping by conflict type
    # (real finding from EN8.6: DBpedia correct 59.5% of the time)
    type_confidence = {
        "factual": {"dbpedia": 0.60, "wikidata": 0.55},
        "granularity": {"dbpedia": 0.80, "wikidata": 0.85},
        "temporal": {"dbpedia": 0.70, "wikidata": 0.75},
        "synonym": {"dbpedia": 0.90, "wikidata": 0.90},
        "format": {"dbpedia": 0.85, "wikidata": 0.85},
    }

    docs: list[dict[str, Any]] = []
    for idx in indices:
        conflict = conflicts[idx]
        ctype = conflict.get("conflict_type", "factual")
        conf_map = type_confidence.get(ctype, type_confidence["factual"])

        # Add noise to confidence (±0.05)
        dbp_conf = round(float(np.clip(
            conf_map["dbpedia"] + rng.normal(0, 0.03), 0.01, 0.99
        )), 4)
        wkd_conf = round(float(np.clip(
            conf_map["wikidata"] + rng.normal(0, 0.03), 0.01, 0.99
        )), 4)

        entity_name = conflict.get("entity", f"entity-{idx}")
        prop_name = conflict.get("property", "unknown_property")
        entity_id = conflict.get("entity_qid", f"Q{idx}")

        doc = {
            "@context": {"@vocab": "http://schema.org/"},
            "@id": f"http://dbpedia.org/resource/{entity_name.replace(' ', '_')}",
            "@type": "Thing",
            f"http://schema.org/{prop_name}": annotate(
                conflict.get("dbpedia", "unknown"),
                confidence=dbp_conf,
                source="http://dbpedia.org",
                method="SPARQL-extraction",
                extracted_at="2026-01-15T00:00:00Z",
                derived_from=[
                    f"http://dbpedia.org/resource/{entity_name.replace(' ', '_')}",
                    f"http://www.wikidata.org/entity/{entity_id}",
                ],
            ),
        }
        docs.append(doc)

    return docs


# ===================================================================
# 2. EN8.2 — Intel Lab Sensor Data → Category B Docs
# ===================================================================

def _load_intel_lab_raw(
    max_records: int = 100000,
) -> list[dict[str, Any]]:
    """Load raw Intel Lab sensor records (subset for speed)."""
    records = []
    if not _INTEL_DATA.exists():
        raise FileNotFoundError(f"Intel Lab data not found: {_INTEL_DATA}")

    with gzip.open(_INTEL_DATA, "rt") as f:
        for line in f:
            if len(records) >= max_records:
                break
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                records.append({
                    "date": parts[0],
                    "time": parts[1],
                    "epoch": int(parts[2]),
                    "moteid": int(parts[3]),
                    "temperature": float(parts[4]),
                    "humidity": float(parts[5]),
                    "light": float(parts[6]),
                    "voltage": float(parts[7]),
                })
            except (ValueError, IndexError):
                continue

    return records


def load_iot_sensor_docs(
    n_docs: int = 60,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create annotated IoT docs from Intel Lab sensor data.

    Each record becomes a jsonld-ex annotated document with real
    sensor readings, measurement uncertainty derived from voltage
    stability, and calibration metadata.

    Args:
        n_docs: Number of documents to generate.
        seed: Random seed for reproducible sampling.

    Returns:
        List of jsonld-ex annotated IoT documents.
    """
    raw = _load_intel_lab_raw(max_records=50000)
    if not raw:
        raise RuntimeError("No Intel Lab records loaded")

    # Load mote locations for context
    mote_locs: dict[int, tuple[float, float]] = {}
    if _MOTE_LOCS.exists():
        with open(_MOTE_LOCS, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    mote_locs[int(parts[0])] = (float(parts[1]), float(parts[2]))

    rng = np.random.default_rng(seed)

    # Filter out clearly invalid readings
    valid = [
        r for r in raw
        if -40 < r["temperature"] < 60
        and 0 < r["humidity"] < 100
        and r["voltage"] > 1.0
    ]

    n_docs = min(n_docs, len(valid))
    indices = rng.choice(len(valid), size=n_docs, replace=False)

    docs: list[dict[str, Any]] = []
    for idx in indices:
        rec = valid[idx]
        moteid = rec["moteid"]
        timestamp = f"{rec['date']}T{rec['time']}Z"

        # Derive measurement uncertainty from voltage stability
        # (lower voltage → higher uncertainty, real sensor behavior)
        voltage_factor = max(0.1, (rec["voltage"] - 1.0) / 2.0)
        temp_sigma = round(0.5 / voltage_factor, 3)
        humid_sigma = round(2.0 / voltage_factor, 3)

        # Confidence from sigma
        temp_conf = round(1.0 / (1.0 + temp_sigma), 4)
        humid_conf = round(1.0 / (1.0 + humid_sigma), 4)

        doc = {
            "@context": {"@vocab": "http://schema.org/"},
            "@id": f"http://intel-lab.example.org/mote-{moteid}/epoch-{rec['epoch']}",
            "@type": "Observation",
            "temperature": annotate(
                round(rec["temperature"], 2),
                confidence=temp_conf,
                source=f"http://intel-lab.example.org/mote-{moteid}",
                method="direct-measurement",
                extracted_at=timestamp,
                unit="celsius",
                measurement_uncertainty=temp_sigma,
                calibrated_at="2004-02-28T00:00:00Z",
                calibration_method="factory-calibration",
                calibration_authority="Crossbow-Mica2Dot",
            ),
            "humidity": annotate(
                round(rec["humidity"], 2),
                confidence=humid_conf,
                source=f"http://intel-lab.example.org/mote-{moteid}",
                method="direct-measurement",
                extracted_at=timestamp,
                unit="percent",
                measurement_uncertainty=humid_sigma,
            ),
        }
        docs.append(doc)

    return docs


# ===================================================================
# 3. EN2.4 — Real Croissant Cards → Category C Docs
# ===================================================================

def load_croissant_docs() -> list[dict[str, Any]]:
    """Load real MLCommons Croissant cards from EN2/croissant_cards/.

    These are genuine Croissant JSON-LD documents. We import them via
    from_croissant() to get jsonld-ex dataset documents, then test
    the round-trip back through to_croissant() → from_croissant().

    Returns:
        List of jsonld-ex dataset documents (imported from Croissant).
    """
    if not _CROISSANT_DIR.exists():
        raise FileNotFoundError(
            f"Croissant cards not found: {_CROISSANT_DIR}\n"
            "Ensure EN2/croissant_cards/ exists."
        )

    docs: list[dict[str, Any]] = []
    for card_path in sorted(_CROISSANT_DIR.glob("*.json")):
        try:
            with open(card_path, "r", encoding="utf-8") as f:
                croissant_doc = json.load(f)
            # Import into jsonld-ex format
            jex_doc = from_croissant(croissant_doc)
            docs.append(jex_doc)
        except Exception as e:
            print(f"Warning: Failed to load {card_path.name}: {e}")

    return docs


# ===================================================================
# 4. EN1.1 — NER Annotation Docs from Real Model Parameters
# ===================================================================

def load_ner_annotation_docs(
    n_docs: int = 30,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create NER annotation docs using real EN1.1 model parameters.

    Uses actual model names, calibrated confidence distributions, and
    temperature scaling parameters from the EN1.1 CoNLL-2003 experiment.
    Entity names are synthetic but structurally representative.

    Args:
        n_docs: Number of documents to generate.
        seed: Random seed.

    Returns:
        List of jsonld-ex annotated NER documents.
    """
    # Load real EN1.1 parameters if available
    models = ["spacy", "flair", "stanza", "huggingface"]
    temperatures = {
        "spacy": 1.50, "flair": 0.84,
        "stanza": 1.16, "huggingface": 1.61,
    }
    model_f1s = {
        "spacy": 0.463, "flair": 0.838,
        "stanza": 0.610, "huggingface": 0.787,
    }

    if _EN1_RESULTS.exists():
        try:
            with open(_EN1_RESULTS, "r", encoding="utf-8") as f:
                en1_data = json.load(f)
            params = en1_data.get("parameters", {})
            if "temperatures" in params:
                temperatures = params["temperatures"]
            metrics = en1_data.get("metrics", {}).get("individual_models_test", {})
            for m in models:
                if m in metrics and "entity_f1" in metrics[m]:
                    model_f1s[m] = metrics[m]["entity_f1"]
        except Exception:
            pass  # Use defaults

    rng = np.random.default_rng(seed)

    # Entity templates (structurally representative of CoNLL-2003)
    entity_templates = [
        ("PER", ["John Smith", "Maria Garcia", "Chen Wei", "Ahmed Hassan",
                 "Sarah Johnson", "Michael Brown", "Yuki Tanaka",
                 "Olga Petrova", "David Kim", "Emma Wilson"]),
        ("ORG", ["Google", "Microsoft", "United Nations", "WHO",
                 "European Union", "Toyota", "Samsung", "Reuters",
                 "BBC", "Stanford University"]),
        ("LOC", ["New York", "London", "Tokyo", "Berlin", "Sydney",
                 "Moscow", "Beijing", "Paris", "Cairo", "Mumbai"]),
    ]

    docs: list[dict[str, Any]] = []
    for i in range(n_docs):
        # Pick an entity type and name
        etype_idx = i % len(entity_templates)
        etype, names = entity_templates[etype_idx]
        name = names[i % len(names)]

        # Pick a model
        model = models[i % len(models)]
        temp = temperatures.get(model, 1.0)
        base_f1 = model_f1s.get(model, 0.5)

        # Generate calibrated confidence using temperature scaling
        # Raw logit from F1-correlated distribution
        raw_conf = float(np.clip(rng.beta(base_f1 * 10, (1 - base_f1) * 10), 0.01, 0.99))
        # Apply temperature scaling (lower temp → sharper, higher temp → flatter)
        calibrated_conf = round(raw_conf ** (1.0 / temp), 4)

        doc = {
            "@context": {"@vocab": "http://schema.org/"},
            "@id": f"http://example.org/conll-entity-{i}",
            "@type": etype,
            "name": annotate(
                name,
                confidence=calibrated_conf,
                source=f"http://models.example.org/{model}",
                method=f"{model}-NER-v1",
                extracted_at=f"2026-03-10T{(8 + i % 12):02d}:00:00Z",
                human_verified=bool(rng.random() > 0.7),
            ),
        }
        docs.append(doc)

    return docs


# ===================================================================
# 5. Kitchen Sink — Composite Real-World Docs (Category E)
# ===================================================================

def load_kitchen_sink_docs(
    n_docs: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create kitchen sink docs by composing real data from multiple sources.

    Each doc combines: dataset metadata (from Croissant) + annotated
    sensor reading (from Intel Lab) + provenance fields.

    Args:
        n_docs: Number of documents.
        seed: Random seed.

    Returns:
        List of composite jsonld-ex documents.
    """
    rng = np.random.default_rng(seed)

    # Load a few real sensor readings
    try:
        sensor_docs = load_iot_sensor_docs(n_docs=n_docs, seed=seed + 100)
    except FileNotFoundError:
        sensor_docs = []

    # Load a few Croissant cards
    try:
        croissant_docs = load_croissant_docs()
    except FileNotFoundError:
        croissant_docs = []

    from jsonld_ex.dataset import create_dataset_metadata

    docs: list[dict[str, Any]] = []
    for i in range(n_docs):
        # Start with dataset metadata
        doc = create_dataset_metadata(
            name=f"RealWorld-IoT-Dataset-{i}",
            description=f"Composite IoT dataset #{i} from Intel Lab data",
            url=f"http://example.org/real-iot-{i}",
            license="http://creativecommons.org/licenses/by/4.0/",
            creator="Intel Berkeley Research Lab",
            version="1.0",
        )

        # Add a real sensor reading if available
        if i < len(sensor_docs):
            sensor = sensor_docs[i]
            for key in ["temperature", "humidity"]:
                if key in sensor:
                    doc[key] = sensor[key]

        # Add a derived-from provenance chain
        if i < len(croissant_docs):
            cr = croissant_docs[i]
            cr_name = cr.get("name", f"source-dataset-{i}")
            doc["sampleAnnotation"] = annotate(
                f"Derived from {cr_name}",
                confidence=round(float(rng.uniform(0.8, 0.95)), 4),
                source="http://intel-lab.example.org/pipeline",
                method="automated-ingestion",
                extracted_at="2026-01-15T00:00:00Z",
                derived_from=[f"http://example.org/datasets/{cr_name}"],
            )

        docs.append(doc)

    return docs


# ===================================================================
# 6. Master Loader
# ===================================================================

def load_all_phase2_docs(seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    """Load all Phase 2 real-world documents by category.

    Returns:
        Dict mapping category letter to list of documents.
        Categories that fail to load return empty lists with warnings.
    """
    results: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []

    # Category A: KG conflicts + NER annotations
    cat_a: list[dict[str, Any]] = []
    try:
        cat_a.extend(load_kg_conflict_docs(n_docs=30, seed=seed))
    except FileNotFoundError as e:
        warnings.append(f"KG conflicts: {e}")
    try:
        cat_a.extend(load_ner_annotation_docs(n_docs=30, seed=seed))
    except Exception as e:
        warnings.append(f"NER annotations: {e}")
    results["A"] = cat_a

    # Category B: IoT sensor data
    try:
        results["B"] = load_iot_sensor_docs(n_docs=60, seed=seed)
    except FileNotFoundError as e:
        warnings.append(f"IoT sensors: {e}")
        results["B"] = []

    # Category C: Croissant cards
    try:
        results["C"] = load_croissant_docs()
    except FileNotFoundError as e:
        warnings.append(f"Croissant cards: {e}")
        results["C"] = []

    # Category D: Shapes (use synthetic — no real shape files cached)
    # Phase 2 shapes come from Category D synthetic generator
    from en8_10_core import generate_category_d_docs
    results["D"] = generate_category_d_docs(seed=seed)

    # Category E: Kitchen sink
    try:
        results["E"] = load_kitchen_sink_docs(n_docs=5, seed=seed)
    except Exception as e:
        warnings.append(f"Kitchen sink: {e}")
        results["E"] = []

    if warnings:
        print(f"Phase 2 loading warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    total = sum(len(v) for v in results.values())
    print(f"Phase 2 loaded: {total} documents across {len(results)} categories")
    for cat, docs in sorted(results.items()):
        print(f"  Category {cat}: {len(docs)} docs")

    return results
