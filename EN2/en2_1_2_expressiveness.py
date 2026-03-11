#!/usr/bin/env python
"""EN2.1 + EN2.2 — Format Expressiveness Comparison.

NeurIPS 2026 D&B, Suite EN2 (Format Expressiveness), Experiments 1 & 2.

EN2.1 — Verbosity Comparison:
    Expresses 10 ML-relevant scenarios in 6 formats and measures byte sizes.
    Formats: jsonld-ex, PROV-O (via W3C ontology), SHACL, Croissant,
    Plain JSON, Raw JSON-LD 1.1.

EN2.2 — Feature Coverage Matrix:
    Assesses 35 ML-relevant features across 8 format ecosystems.
    Support levels: "native" / "workaround" / "not_possible".

Methodology:
    - All representations convey the SAME information per scenario.
    - Byte measurement: len(json.dumps(doc, indent=2, sort_keys=True).encode("utf-8"))
      with consistent indent=2 and sorted keys for fair comparison.
    - Alternative representations constructed manually to be as compact as
      reasonably possible — we do NOT strawman the alternatives.
    - Features assessed conservatively: "workaround" means the format can
      express it with extra effort; "native" means it's a first-class feature.

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN2/en2_1_2_expressiveness.py

Output:
    experiments/EN2/results/en2_1_2_results.json           (latest)
    experiments/EN2/results/en2_1_2_results_YYYYMMDD_HHMMSS.json (archived)

References:
    Croissant: Akhtar et al. (2024). Croissant: A Metadata Format for ML-
    Ready Datasets. NeurIPS 2024 D&B.
    PROV-O: Lebo et al. (2013). W3C PROV-O Ontology. W3C Recommendation.
    SHACL: Knublauch & Kontokostas (2017). W3C SHACL. W3C Recommendation.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Path setup ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "packages" / "python" / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

# ── Imports: jsonld-ex ─────────────────────────────────────────────
from jsonld_ex.ai_ml import annotate
from jsonld_ex.owl_interop import (
    to_prov_o,
    shape_to_shacl,
    to_rdf_star_ntriples,
    to_ssn,
    compare_with_prov_o,
    compare_with_shacl,
)
from jsonld_ex.temporal import add_temporal
from jsonld_ex.confidence_algebra import Opinion

# ── Imports: experiment infrastructure ─────────────────────────────
from experiments.infra.results import ExperimentResult
from experiments.infra.env_log import log_environment

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# =====================================================================
# EN2.1 — Scenario Definitions
# =====================================================================

def measure_bytes(obj: Any) -> int:
    """Consistent byte measurement: indent=2, sorted keys, UTF-8."""
    return len(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def build_scenarios() -> List[Dict[str, Any]]:
    """Build 10 ML-relevant scenarios in all 6 formats.

    Each scenario returns a dict with:
        id, name, description,
        jsonld_ex: the jsonld-ex representation,
        prov_o: equivalent PROV-O representation,
        shacl: equivalent SHACL (where applicable, else None),
        croissant: equivalent Croissant (where applicable, else None),
        plain_json: plain JSON (no semantics),
        jsonld_11: raw JSON-LD 1.1 (standard, no extensions).
    """
    scenarios = []

    # ── S1: Dataset Card with Splits ──
    s1_jsonld_ex = {
        "@context": "https://json-ld.org/ns/jsonld-ex/v1",
        "@type": "Dataset",
        "name": annotate("CoNLL-2003", confidence=0.99, source="original-paper",
                         extracted_at="2026-01-15T10:00:00Z"),
        "description": "Named Entity Recognition benchmark dataset",
        "license": "custom-research",
        "distribution": [
            {"@type": "FileObject", "name": "train.txt", "contentSize": "3.3MB",
             "encodingFormat": "text/plain"},
            {"@type": "FileObject", "name": "test.txt", "contentSize": "0.7MB",
             "encodingFormat": "text/plain"},
        ],
        "recordSet": {
            "@type": "RecordSet",
            "field": [
                {"@type": "Field", "name": "token", "dataType": "sc:Text"},
                {"@type": "Field", "name": "ner_tag", "dataType": "sc:Text",
                 "@confidence": 0.95, "@source": "annotator-panel"},
            ],
        },
    }

    s1_prov_o = {
        "@context": {"prov": "http://www.w3.org/ns/prov#",
                      "schema": "http://schema.org/"},
        "@graph": [
            {"@id": "dataset:conll2003", "@type": "schema:Dataset",
             "schema:name": "CoNLL-2003",
             "schema:description": "Named Entity Recognition benchmark dataset",
             "schema:license": "custom-research"},
            {"@id": "dataset:conll2003/train", "@type": "schema:DataDownload",
             "schema:name": "train.txt", "schema:contentSize": "3.3MB",
             "schema:encodingFormat": "text/plain"},
            {"@id": "dataset:conll2003/test", "@type": "schema:DataDownload",
             "schema:name": "test.txt", "schema:contentSize": "0.7MB",
             "schema:encodingFormat": "text/plain"},
            {"@id": "_:name-assertion", "@type": "prov:Entity",
             "prov:value": "CoNLL-2003",
             "prov:wasGeneratedBy": {"@type": "prov:Activity",
                                      "prov:wasAssociatedWith": "original-paper",
                                      "prov:endedAtTime": "2026-01-15T10:00:00Z"},
             "schema:additionalProperty": {"schema:value": 0.99,
                                            "schema:name": "confidence"}},
            {"@id": "_:ner-field", "@type": "schema:PropertyValue",
             "schema:name": "ner_tag", "schema:additionalProperty": [
                 {"schema:name": "confidence", "schema:value": 0.95},
                 {"schema:name": "source", "schema:value": "annotator-panel"},
             ]},
        ],
    }

    s1_croissant = {
        "@context": {"@vocab": "http://mlcommons.org/croissant/",
                      "sc": "http://schema.org/"},
        "@type": "sc:Dataset",
        "sc:name": "CoNLL-2003",
        "sc:description": "Named Entity Recognition benchmark dataset",
        "sc:license": "custom-research",
        "distribution": [
            {"@type": "cr:FileObject", "sc:name": "train.txt",
             "sc:contentSize": "3.3MB", "sc:encodingFormat": "text/plain"},
            {"@type": "cr:FileObject", "sc:name": "test.txt",
             "sc:contentSize": "0.7MB", "sc:encodingFormat": "text/plain"},
        ],
        "recordSet": {
            "@type": "cr:RecordSet",
            "field": [
                {"@type": "cr:Field", "sc:name": "token",
                 "dataType": "sc:Text"},
                {"@type": "cr:Field", "sc:name": "ner_tag",
                 "dataType": "sc:Text"},
                # NOTE: Croissant has NO native way to express per-field
                # confidence or source attribution
            ],
        },
    }

    s1_plain_json = {
        "name": "CoNLL-2003",
        "name_metadata": {"confidence": 0.99, "source": "original-paper",
                          "extracted_at": "2026-01-15T10:00:00Z"},
        "description": "Named Entity Recognition benchmark dataset",
        "license": "custom-research",
        "files": [
            {"name": "train.txt", "size": "3.3MB", "format": "text/plain"},
            {"name": "test.txt", "size": "0.7MB", "format": "text/plain"},
        ],
        "fields": [
            {"name": "token", "type": "text"},
            {"name": "ner_tag", "type": "text", "confidence": 0.95,
             "source": "annotator-panel"},
        ],
    }

    s1_jsonld_11 = {
        "@context": {"schema": "http://schema.org/"},
        "@type": "schema:Dataset",
        "schema:name": "CoNLL-2003",
        "schema:description": "Named Entity Recognition benchmark dataset",
        "schema:license": "custom-research",
        "schema:distribution": [
            {"@type": "schema:DataDownload", "schema:name": "train.txt",
             "schema:contentSize": "3.3MB",
             "schema:encodingFormat": "text/plain"},
            {"@type": "schema:DataDownload", "schema:name": "test.txt",
             "schema:contentSize": "0.7MB",
             "schema:encodingFormat": "text/plain"},
        ],
        # JSON-LD 1.1 has no standard way to express per-field confidence
    }

    scenarios.append({
        "id": "S1", "name": "Dataset card with splits",
        "description": "ML dataset with file distributions, record schema, and per-field confidence",
        "jsonld_ex": s1_jsonld_ex, "prov_o": s1_prov_o, "shacl": None,
        "croissant": s1_croissant, "plain_json": s1_plain_json,
        "jsonld_11": s1_jsonld_11,
    })

    # ── S2: NER Annotations with Per-Token Confidence ──
    s2_jsonld_ex = {
        "@context": "https://json-ld.org/ns/jsonld-ex/v1",
        "tokens": [
            annotate("Barack", confidence=0.97, source="spacy-trf", method="NER"),
            annotate("Obama", confidence=0.99, source="spacy-trf", method="NER"),
            annotate("visited", confidence=0.85, source="spacy-trf", method="NER"),
            annotate("Paris", confidence=0.92, source="spacy-trf", method="NER"),
        ],
        "entities": [
            {"text": "Barack Obama", "label": annotate("PER", confidence=0.98,
             source="spacy-trf"), "span": [0, 1]},
            {"text": "Paris", "label": annotate("LOC", confidence=0.92,
             source="spacy-trf"), "span": [3, 3]},
        ],
    }

    s2_prov_o = {
        "@context": {"prov": "http://www.w3.org/ns/prov#",
                      "nif": "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#"},
        "@graph": [
            {"@id": "_:tok0", "@type": "nif:Word", "nif:anchorOf": "Barack",
             "prov:wasGeneratedBy": {"@type": "prov:Activity",
                                      "prov:wasAssociatedWith": "spacy-trf",
                                      "prov:qualifiedAssociation": {"prov:hadRole": "NER"}},
             "prov:value": "Barack",
             "nif:confidence": 0.97},
            {"@id": "_:tok1", "@type": "nif:Word", "nif:anchorOf": "Obama",
             "nif:confidence": 0.99,
             "prov:wasGeneratedBy": {"prov:wasAssociatedWith": "spacy-trf"}},
            {"@id": "_:tok2", "@type": "nif:Word", "nif:anchorOf": "visited",
             "nif:confidence": 0.85,
             "prov:wasGeneratedBy": {"prov:wasAssociatedWith": "spacy-trf"}},
            {"@id": "_:tok3", "@type": "nif:Word", "nif:anchorOf": "Paris",
             "nif:confidence": 0.92,
             "prov:wasGeneratedBy": {"prov:wasAssociatedWith": "spacy-trf"}},
            {"@id": "_:ent0", "@type": "nif:EntityOccurrence",
             "nif:anchorOf": "Barack Obama", "nif:entity": "PER",
             "nif:confidence": 0.98, "nif:beginIndex": 0, "nif:endIndex": 1,
             "prov:wasGeneratedBy": {"prov:wasAssociatedWith": "spacy-trf"}},
            {"@id": "_:ent1", "@type": "nif:EntityOccurrence",
             "nif:anchorOf": "Paris", "nif:entity": "LOC",
             "nif:confidence": 0.92, "nif:beginIndex": 3, "nif:endIndex": 3,
             "prov:wasGeneratedBy": {"prov:wasAssociatedWith": "spacy-trf"}},
        ],
    }

    s2_plain_json = {
        "tokens": [
            {"text": "Barack", "confidence": 0.97, "source": "spacy-trf", "method": "NER"},
            {"text": "Obama", "confidence": 0.99, "source": "spacy-trf", "method": "NER"},
            {"text": "visited", "confidence": 0.85, "source": "spacy-trf", "method": "NER"},
            {"text": "Paris", "confidence": 0.92, "source": "spacy-trf", "method": "NER"},
        ],
        "entities": [
            {"text": "Barack Obama", "label": "PER", "label_confidence": 0.98,
             "source": "spacy-trf", "span": [0, 1]},
            {"text": "Paris", "label": "LOC", "label_confidence": 0.92,
             "source": "spacy-trf", "span": [3, 3]},
        ],
    }

    s2_jsonld_11 = {
        "@context": {"nif": "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#"},
        "@graph": [
            {"@type": "nif:Word", "nif:anchorOf": "Barack"},
            {"@type": "nif:Word", "nif:anchorOf": "Obama"},
            {"@type": "nif:Word", "nif:anchorOf": "visited"},
            {"@type": "nif:Word", "nif:anchorOf": "Paris"},
            {"@type": "nif:EntityOccurrence", "nif:anchorOf": "Barack Obama",
             "nif:entity": "PER", "nif:beginIndex": 0, "nif:endIndex": 1},
            {"@type": "nif:EntityOccurrence", "nif:anchorOf": "Paris",
             "nif:entity": "LOC", "nif:beginIndex": 3, "nif:endIndex": 3},
            # JSON-LD 1.1 has no standard confidence or source annotation
        ],
    }

    scenarios.append({
        "id": "S2", "name": "NER annotations with per-token confidence",
        "description": "NLP token annotations with confidence scores and model attribution",
        "jsonld_ex": s2_jsonld_ex, "prov_o": s2_prov_o, "shacl": None,
        "croissant": None, "plain_json": s2_plain_json, "jsonld_11": s2_jsonld_11,
    })

    # ── S3: Sensor Reading with Uncertainty and Calibration ──
    s3_jsonld_ex = {
        "@context": "https://json-ld.org/ns/jsonld-ex/v1",
        "@type": "Observation",
        "observedProperty": "temperature",
        "result": annotate(23.4, confidence=0.95, source="sensor-DHT22",
                           measurement_uncertainty=0.5, unit="degC",
                           calibrated_at="2026-01-01T00:00:00Z",
                           calibration_method="NIST-traceable",
                           extracted_at="2026-03-10T14:30:00Z"),
        "phenomenonTime": "2026-03-10T14:30:00Z",
    }

    s3_prov_o_ssn = {
        "@context": {"sosa": "http://www.w3.org/ns/sosa/",
                      "ssn-system": "http://www.w3.org/ns/ssn/systems/",
                      "qudt": "http://qudt.org/schema/qudt/",
                      "prov": "http://www.w3.org/ns/prov#"},
        "@type": "sosa:Observation",
        "sosa:observedProperty": "temperature",
        "sosa:hasSimpleResult": 23.4,
        "sosa:resultTime": "2026-03-10T14:30:00Z",
        "sosa:madeBySensor": {
            "@id": "sensor:DHT22",
            "@type": "sosa:Sensor",
            "ssn-system:hasSystemCapability": {
                "@type": "ssn-system:SystemCapability",
                "ssn-system:hasSystemProperty": {
                    "@type": "ssn-system:Accuracy",
                    "schema:value": 0.5,
                    "qudt:unit": "qudt:DEG_C",
                },
            },
        },
        "prov:wasGeneratedBy": {
            "@type": "prov:Activity",
            "prov:wasAssociatedWith": "sensor:DHT22",
            "prov:endedAtTime": "2026-03-10T14:30:00Z",
        },
        # Calibration requires additional triples
        "ssn-system:hasDeployment": {
            "ssn-system:deployedOnPlatform": "lab-bench-1",
            "prov:qualifiedAssociation": {
                "prov:hadRole": "calibration",
                "prov:agent": "NIST-traceable",
                "prov:atTime": "2026-01-01T00:00:00Z",
            },
        },
    }

    s3_plain_json = {
        "type": "observation",
        "property": "temperature",
        "value": 23.4,
        "unit": "degC",
        "uncertainty": 0.5,
        "confidence": 0.95,
        "sensor": "DHT22",
        "timestamp": "2026-03-10T14:30:00Z",
        "calibration": {
            "date": "2026-01-01T00:00:00Z",
            "method": "NIST-traceable",
        },
    }

    s3_jsonld_11 = {
        "@context": {"sosa": "http://www.w3.org/ns/sosa/"},
        "@type": "sosa:Observation",
        "sosa:observedProperty": "temperature",
        "sosa:hasSimpleResult": 23.4,
        "sosa:resultTime": "2026-03-10T14:30:00Z",
        # No native uncertainty, calibration, or confidence fields
    }

    scenarios.append({
        "id": "S3", "name": "Sensor reading with uncertainty and calibration",
        "description": "IoT sensor observation with measurement uncertainty, calibration provenance",
        "jsonld_ex": s3_jsonld_ex, "prov_o": s3_prov_o_ssn, "shacl": None,
        "croissant": None, "plain_json": s3_plain_json, "jsonld_11": s3_jsonld_11,
    })

    # ── S4: Knowledge Graph Triple with Provenance Chain ──
    s4_jsonld_ex = {
        "@context": "https://json-ld.org/ns/jsonld-ex/v1",
        "@type": "Person",
        "name": "Marie Curie",
        "birthPlace": annotate("Warsaw", confidence=0.99,
                                source="wikidata:Q36",
                                extracted_at="2026-02-01T12:00:00Z",
                                derived_from=["wikidata:Q7186",
                                               "dbpedia:Marie_Curie"],
                                human_verified=True),
    }

    s4_prov_o = {
        "@context": {"prov": "http://www.w3.org/ns/prov#",
                      "schema": "http://schema.org/"},
        "@graph": [
            {"@id": "person:marie-curie", "@type": "schema:Person",
             "schema:name": "Marie Curie",
             "schema:birthPlace": {"@id": "_:birthplace-assertion"}},
            {"@id": "_:birthplace-assertion", "@type": "prov:Entity",
             "prov:value": "Warsaw",
             "prov:wasGeneratedBy": {
                 "@type": "prov:Activity",
                 "prov:wasAssociatedWith": "wikidata:Q36",
                 "prov:endedAtTime": "2026-02-01T12:00:00Z",
             },
             "prov:wasDerivedFrom": [
                 {"@id": "wikidata:Q7186"},
                 {"@id": "dbpedia:Marie_Curie"},
             ],
             "prov:wasAttributedTo": {"prov:hadRole": "human-verifier"},
             "schema:additionalProperty": {
                 "schema:name": "confidence", "schema:value": 0.99,
             }},
        ],
    }

    s4_plain_json = {
        "type": "Person",
        "name": "Marie Curie",
        "birthPlace": "Warsaw",
        "birthPlace_metadata": {
            "confidence": 0.99, "source": "wikidata:Q36",
            "extracted_at": "2026-02-01T12:00:00Z",
            "derived_from": ["wikidata:Q7186", "dbpedia:Marie_Curie"],
            "human_verified": True,
        },
    }

    s4_jsonld_11 = {
        "@context": {"schema": "http://schema.org/"},
        "@type": "schema:Person",
        "schema:name": "Marie Curie",
        "schema:birthPlace": "Warsaw",
        # No native provenance, confidence, or derivation chain
    }

    scenarios.append({
        "id": "S4", "name": "KG triple with provenance chain",
        "description": "Knowledge graph assertion with multi-source derivation and human verification",
        "jsonld_ex": s4_jsonld_ex, "prov_o": s4_prov_o, "shacl": None,
        "croissant": None, "plain_json": s4_plain_json, "jsonld_11": s4_jsonld_11,
    })

    # ── S5: Model Prediction with Confidence + Source + Timestamp ──
    s5_jsonld_ex = {
        "@context": "https://json-ld.org/ns/jsonld-ex/v1",
        "@type": "Prediction",
        "input": "chest_xray_0042.dcm",
        "label": annotate("pneumonia", confidence=0.87,
                           source="resnet50-chexpert-v2",
                           method="classification",
                           extracted_at="2026-03-10T09:15:00Z"),
        "secondary_label": annotate("normal", confidence=0.11,
                                     source="resnet50-chexpert-v2"),
    }

    s5_prov_o = {
        "@context": {"prov": "http://www.w3.org/ns/prov#"},
        "@graph": [
            {"@id": "_:prediction", "@type": "prov:Entity",
             "prov:wasGeneratedBy": {
                 "@type": "prov:Activity",
                 "prov:used": "chest_xray_0042.dcm",
                 "prov:wasAssociatedWith": {
                     "@type": "prov:SoftwareAgent",
                     "@id": "resnet50-chexpert-v2",
                 },
                 "prov:endedAtTime": "2026-03-10T09:15:00Z",
                 "prov:qualifiedAssociation": {
                     "prov:hadRole": "classification",
                 },
             }},
            {"@id": "_:label-primary", "@type": "prov:Entity",
             "prov:value": "pneumonia",
             "prov:wasGeneratedBy": {"@id": "_:prediction"},
             "schema:additionalProperty": {
                 "schema:name": "confidence", "schema:value": 0.87}},
            {"@id": "_:label-secondary", "@type": "prov:Entity",
             "prov:value": "normal",
             "prov:wasGeneratedBy": {"@id": "_:prediction"},
             "schema:additionalProperty": {
                 "schema:name": "confidence", "schema:value": 0.11}},
        ],
    }

    s5_plain_json = {
        "input": "chest_xray_0042.dcm",
        "predictions": [
            {"label": "pneumonia", "confidence": 0.87,
             "model": "resnet50-chexpert-v2", "method": "classification",
             "timestamp": "2026-03-10T09:15:00Z"},
            {"label": "normal", "confidence": 0.11,
             "model": "resnet50-chexpert-v2"},
        ],
    }

    s5_jsonld_11 = {
        "@context": {"schema": "http://schema.org/"},
        "@type": "schema:Action",
        "schema:object": "chest_xray_0042.dcm",
        "schema:result": "pneumonia",
        # No confidence, no multi-label, no model attribution natively
    }

    scenarios.append({
        "id": "S5", "name": "Model prediction with confidence",
        "description": "Classification output with confidence scores and model provenance",
        "jsonld_ex": s5_jsonld_ex, "prov_o": s5_prov_o, "shacl": None,
        "croissant": None, "plain_json": s5_plain_json, "jsonld_11": s5_jsonld_11,
    })

    # ── S6: Multi-Language Content with Translation Provenance ──
    s6_jsonld_ex = {
        "@context": "https://json-ld.org/ns/jsonld-ex/v1",
        "title": {"@value": "Neural Networks", "@language": "en"},
        "title_ja": annotate("ニューラルネットワーク",
                              translated_from="en",
                              translation_model="gpt-4-turbo",
                              confidence=0.94,
                              source="openai-api",
                              extracted_at="2026-03-01T08:00:00Z"),
    }

    s6_prov_o = {
        "@context": {"prov": "http://www.w3.org/ns/prov#"},
        "@graph": [
            {"@id": "_:title-en", "@type": "prov:Entity",
             "prov:value": {"@value": "Neural Networks", "@language": "en"}},
            {"@id": "_:title-ja", "@type": "prov:Entity",
             "prov:value": {"@value": "ニューラルネットワーク", "@language": "ja"},
             "prov:wasDerivedFrom": {"@id": "_:title-en"},
             "prov:wasGeneratedBy": {
                 "@type": "prov:Activity",
                 "prov:wasAssociatedWith": {
                     "@type": "prov:SoftwareAgent",
                     "@id": "gpt-4-turbo",
                     "prov:actedOnBehalfOf": "openai-api",
                 },
                 "prov:endedAtTime": "2026-03-01T08:00:00Z",
                 "prov:qualifiedAssociation": {
                     "prov:hadRole": "translation",
                 },
             },
             "schema:additionalProperty": {
                 "schema:name": "confidence", "schema:value": 0.94}},
        ],
    }

    s6_plain_json = {
        "title": {"text": "Neural Networks", "language": "en"},
        "title_ja": {
            "text": "ニューラルネットワーク", "language": "ja",
            "translated_from": "en", "translation_model": "gpt-4-turbo",
            "confidence": 0.94, "source": "openai-api",
            "timestamp": "2026-03-01T08:00:00Z",
        },
    }

    s6_jsonld_11 = {
        "@context": {},
        "title": {"@value": "Neural Networks", "@language": "en"},
        "title_ja": {"@value": "ニューラルネットワーク", "@language": "ja"},
        # No translation provenance natively
    }

    scenarios.append({
        "id": "S6", "name": "Multi-language with translation provenance",
        "description": "Multilingual content with source language, translation model, and confidence",
        "jsonld_ex": s6_jsonld_ex, "prov_o": s6_prov_o, "shacl": None,
        "croissant": None, "plain_json": s6_plain_json, "jsonld_11": s6_jsonld_11,
    })

    # ── S7: Temporal Validity Window ──
    s7_jsonld_ex = {
        "@context": "https://json-ld.org/ns/jsonld-ex/v1",
        "@type": "Regulation",
        "name": "GDPR Article 6(1)(a) - Consent",
        "status": annotate("active", confidence=1.0,
                            source="eur-lex.europa.eu"),
        "@validFrom": "2018-05-25T00:00:00Z",
        "@validUntil": "2099-12-31T23:59:59Z",
    }

    s7_prov_o = {
        "@context": {"prov": "http://www.w3.org/ns/prov#",
                      "schema": "http://schema.org/"},
        "@graph": [
            {"@id": "_:regulation", "@type": "prov:Entity",
             "schema:name": "GDPR Article 6(1)(a) - Consent",
             "prov:generatedAtTime": "2018-05-25T00:00:00Z",
             "prov:invalidatedAtTime": "2099-12-31T23:59:59Z",
             "schema:additionalProperty": [
                 {"schema:name": "status", "schema:value": "active"},
                 {"schema:name": "confidence", "schema:value": 1.0},
                 {"schema:name": "source", "schema:value": "eur-lex.europa.eu"},
             ]},
        ],
    }

    s7_plain_json = {
        "type": "regulation",
        "name": "GDPR Article 6(1)(a) - Consent",
        "status": "active",
        "valid_from": "2018-05-25T00:00:00Z",
        "valid_until": "2099-12-31T23:59:59Z",
    }

    s7_jsonld_11 = {
        "@context": {"schema": "http://schema.org/"},
        "@type": "schema:Legislation",
        "schema:name": "GDPR Article 6(1)(a) - Consent",
        "schema:datePublished": "2018-05-25",
        # No @validFrom/@validUntil, no confidence natively
    }

    scenarios.append({
        "id": "S7", "name": "Temporal validity window",
        "description": "Time-bounded assertion with validity period and status confidence",
        "jsonld_ex": s7_jsonld_ex, "prov_o": s7_prov_o, "shacl": None,
        "croissant": None, "plain_json": s7_plain_json, "jsonld_11": s7_jsonld_11,
    })

    # ── S8: Validation Shape for a Person Entity ──
    s8_shape = {
        "@type": "Person",
        "name": {"@required": True, "@type": "string", "@minLength": 1},
        "email": {"@required": True, "@type": "string", "@pattern": "^.+@.+$"},
        "age": {"@type": "integer", "@min": 0, "@max": 150},
    }

    s8_jsonld_ex = {
        "@context": "https://json-ld.org/ns/jsonld-ex/v1",
        "@shape": s8_shape,
    }

    s8_shacl = {
        "@context": {"sh": "http://www.w3.org/ns/shacl#",
                      "xsd": "http://www.w3.org/2001/XMLSchema#",
                      "schema": "http://schema.org/"},
        "@type": "sh:NodeShape",
        "sh:targetClass": "schema:Person",
        "sh:property": [
            {"sh:path": "schema:name", "sh:minCount": 1,
             "sh:datatype": "xsd:string", "sh:minLength": 1},
            {"sh:path": "schema:email", "sh:minCount": 1,
             "sh:datatype": "xsd:string",
             "sh:pattern": "^.+@.+$"},
            {"sh:path": "schema:age", "sh:datatype": "xsd:integer",
             "sh:minInclusive": 0, "sh:maxInclusive": 150},
        ],
    }

    s8_plain_json = {
        "type": "validation_schema",
        "target": "Person",
        "properties": {
            "name": {"type": "string", "required": True, "minLength": 1},
            "email": {"type": "string", "required": True,
                       "pattern": "^.+@.+$"},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
        },
    }

    # JSON Schema equivalent (closest standard)
    s8_jsonld_11 = {
        "@context": {"schema": "http://schema.org/"},
        "@type": "schema:Person",
        # No validation constraints in JSON-LD 1.1 natively — use SHACL
    }

    scenarios.append({
        "id": "S8", "name": "Validation shape for Person entity",
        "description": "Data validation constraints with type checking, required fields, ranges",
        "jsonld_ex": s8_jsonld_ex, "prov_o": None, "shacl": s8_shacl,
        "croissant": None, "plain_json": s8_plain_json, "jsonld_11": s8_jsonld_11,
    })

    # ── S9: Multi-Source Fused Assertion ──
    s9_jsonld_ex = {
        "@context": "https://json-ld.org/ns/jsonld-ex/v1",
        "@type": "Claim",
        "subject": "global_mean_temperature_2025",
        "value": annotate(15.02, confidence=0.96, source="fusion:3-sources",
                           method="cumulative_fuse",
                           derived_from=["NASA-GISS", "HadCRUT5", "NOAA-NCEI"]),
        "@opinion": {
            "belief": 0.91, "disbelief": 0.02, "uncertainty": 0.07,
            "baseRate": 0.5,
        },
    }

    s9_prov_o = {
        "@context": {"prov": "http://www.w3.org/ns/prov#"},
        "@graph": [
            {"@id": "_:fused-claim", "@type": "prov:Entity",
             "prov:value": 15.02,
             "prov:wasGeneratedBy": {
                 "@type": "prov:Activity",
                 "prov:qualifiedAssociation": {
                     "prov:hadRole": "cumulative_fuse",
                 },
                 "prov:used": [
                     {"@id": "_:src-nasa"},
                     {"@id": "_:src-hadcrut"},
                     {"@id": "_:src-noaa"},
                 ],
             },
             "schema:additionalProperty": [
                 {"schema:name": "confidence", "schema:value": 0.96},
             ]},
            {"@id": "_:src-nasa", "@type": "prov:Entity",
             "prov:wasAttributedTo": "NASA-GISS"},
            {"@id": "_:src-hadcrut", "@type": "prov:Entity",
             "prov:wasAttributedTo": "HadCRUT5"},
            {"@id": "_:src-noaa", "@type": "prov:Entity",
             "prov:wasAttributedTo": "NOAA-NCEI"},
            # PROV-O cannot express the (b,d,u,a) opinion natively
        ],
    }

    s9_plain_json = {
        "subject": "global_mean_temperature_2025",
        "value": 15.02,
        "confidence": 0.96,
        "fusion_method": "cumulative_fuse",
        "sources": ["NASA-GISS", "HadCRUT5", "NOAA-NCEI"],
        "opinion": {
            "belief": 0.91, "disbelief": 0.02, "uncertainty": 0.07,
            "base_rate": 0.5,
        },
    }

    s9_jsonld_11 = {
        "@context": {"schema": "http://schema.org/"},
        "@type": "schema:Claim",
        "schema:about": "global_mean_temperature_2025",
        "schema:value": 15.02,
        # No fusion, no opinion, no multi-source derivation
    }

    scenarios.append({
        "id": "S9", "name": "Multi-source fused assertion",
        "description": "Fused claim from 3 sources with SL opinion and fusion method attribution",
        "jsonld_ex": s9_jsonld_ex, "prov_o": s9_prov_o, "shacl": None,
        "croissant": None, "plain_json": s9_plain_json, "jsonld_11": s9_jsonld_11,
    })

    # ── S10: Invalidated/Retracted Claim ──
    s10_jsonld_ex = {
        "@context": "https://json-ld.org/ns/jsonld-ex/v1",
        "@type": "Claim",
        "subject": "drug_efficacy_trial_2024",
        "value": annotate("effective", confidence=0.45,
                           source="retracted-journal:10.1234/fake",
                           invalidated_at="2026-02-15T00:00:00Z",
                           invalidation_reason="data-fabrication"),
    }

    s10_prov_o = {
        "@context": {"prov": "http://www.w3.org/ns/prov#"},
        "@graph": [
            {"@id": "_:claim", "@type": "prov:Entity",
             "prov:value": "effective",
             "prov:wasAttributedTo": "retracted-journal:10.1234/fake",
             "prov:wasInvalidatedBy": {
                 "@type": "prov:Activity",
                 "prov:atTime": "2026-02-15T00:00:00Z",
                 "prov:qualifiedAssociation": {
                     "prov:hadRole": "retraction",
                 },
             },
             "schema:additionalProperty": [
                 {"schema:name": "confidence", "schema:value": 0.45},
                 {"schema:name": "invalidation_reason",
                  "schema:value": "data-fabrication"},
             ]},
        ],
    }

    s10_plain_json = {
        "subject": "drug_efficacy_trial_2024",
        "value": "effective",
        "confidence": 0.45,
        "source": "retracted-journal:10.1234/fake",
        "invalidated_at": "2026-02-15T00:00:00Z",
        "invalidation_reason": "data-fabrication",
    }

    s10_jsonld_11 = {
        "@context": {"schema": "http://schema.org/"},
        "@type": "schema:Claim",
        "schema:about": "drug_efficacy_trial_2024",
        "schema:text": "effective",
        # No invalidation, no confidence natively
    }

    scenarios.append({
        "id": "S10", "name": "Invalidated/retracted claim",
        "description": "Retracted assertion with invalidation timestamp and reason",
        "jsonld_ex": s10_jsonld_ex, "prov_o": s10_prov_o, "shacl": None,
        "croissant": None, "plain_json": s10_plain_json, "jsonld_11": s10_jsonld_11,
    })

    return scenarios


# =====================================================================
# EN2.1 — Byte Measurement
# =====================================================================

def measure_scenario_bytes(scenarios: List[Dict]) -> List[Dict[str, Any]]:
    """Measure byte sizes for all scenarios across all formats."""
    results = []
    formats = ["jsonld_ex", "prov_o", "shacl", "croissant", "plain_json", "jsonld_11"]

    for sc in scenarios:
        row = {
            "id": sc["id"],
            "name": sc["name"],
            "description": sc["description"],
        }
        for fmt in formats:
            doc = sc.get(fmt)
            if doc is not None:
                row[f"{fmt}_bytes"] = measure_bytes(doc)
            else:
                row[f"{fmt}_bytes"] = None

        # Compute ratios vs jsonld-ex
        ex_bytes = row["jsonld_ex_bytes"]
        for fmt in formats:
            if fmt == "jsonld_ex":
                continue
            alt_bytes = row.get(f"{fmt}_bytes")
            if alt_bytes is not None and ex_bytes > 0:
                row[f"{fmt}_ratio"] = round(alt_bytes / ex_bytes, 3)
            else:
                row[f"{fmt}_ratio"] = None

        results.append(row)

    return results


# =====================================================================
# EN2.2 — Feature Coverage Matrix
# =====================================================================

# Support levels: "native", "workaround", "not_possible"
# "native" = first-class feature with dedicated syntax
# "workaround" = expressible but requires ad-hoc extensions or verbose patterns
# "not_possible" = fundamentally cannot express this concept

FEATURES = [
    # Core annotation
    ("Scalar confidence per assertion", "confidence"),
    ("SL opinion (b,d,u,a) per assertion", "sl_opinion"),
    ("Source attribution per assertion", "source"),
    ("Extraction timestamp per assertion", "extraction_time"),
    ("Extraction method per assertion", "extraction_method"),
    ("Human verification flag", "human_verified"),
    # Provenance
    ("Derivation chain (derived_from)", "derivation_chain"),
    ("Delegation chain (delegated_by)", "delegation_chain"),
    ("Invalidation with reason", "invalidation"),
    # Temporal
    ("Validity window (@validFrom/@validUntil)", "validity_window"),
    ("Point-in-time query", "temporal_query"),
    ("Temporal diff between snapshots", "temporal_diff"),
    # Uncertainty algebra
    ("Cumulative fusion of opinions", "cumulative_fusion"),
    ("Averaging fusion of opinions", "averaging_fusion"),
    ("Trust discount through provenance", "trust_discount"),
    ("Deduction (uncertain conditionals)", "deduction"),
    ("Conflict detection between sources", "conflict_detection"),
    ("Byzantine-resistant fusion", "byzantine_fusion"),
    ("Temporal decay of opinions", "temporal_decay"),
    # Validation
    ("Schema validation (type, range, pattern)", "schema_validation"),
    ("Required field constraints", "required_fields"),
    ("Cross-field constraints", "cross_field"),
    # Data description
    ("Dataset metadata (name, license, etc.)", "dataset_metadata"),
    ("File distribution descriptions", "file_distributions"),
    ("Record set / field definitions", "record_set"),
    # Interop
    ("RDF/Linked Data compatibility", "rdf_compat"),
    ("PROV-O export/import", "prov_o_interop"),
    ("SHACL export/import", "shacl_interop"),
    ("OWL class restriction export", "owl_interop"),
    ("SSN/SOSA sensor interop", "ssn_sosa"),
    ("Croissant import/export", "croissant_interop"),
    # Advanced
    ("Vector embeddings (@vector)", "vector_embeddings"),
    ("Measurement uncertainty (IoT)", "measurement_uncertainty"),
    ("Calibration metadata", "calibration"),
    ("Translation provenance", "translation_provenance"),
    ("Multi-source graph merge", "graph_merge"),
]


def build_coverage_matrix() -> Dict[str, Any]:
    """Build the EN2.2 feature coverage matrix.

    Returns dict mapping format_name -> {feature_key: support_level}.

    Assessment methodology: each feature is evaluated based on the
    format's specification, not on what could theoretically be forced
    via arbitrary key-value pairs. "workaround" means there is a
    documented or conventional pattern; "not_possible" means the
    format's data model fundamentally cannot represent the concept.
    """
    # fmt: off
    matrix = {
        "jsonld_ex": {
            "confidence": "native", "sl_opinion": "native",
            "source": "native", "extraction_time": "native",
            "extraction_method": "native", "human_verified": "native",
            "derivation_chain": "native", "delegation_chain": "native",
            "invalidation": "native",
            "validity_window": "native", "temporal_query": "native",
            "temporal_diff": "native",
            "cumulative_fusion": "native", "averaging_fusion": "native",
            "trust_discount": "native", "deduction": "native",
            "conflict_detection": "native", "byzantine_fusion": "native",
            "temporal_decay": "native",
            "schema_validation": "native", "required_fields": "native",
            "cross_field": "native",
            "dataset_metadata": "native", "file_distributions": "native",
            "record_set": "native",
            "rdf_compat": "native", "prov_o_interop": "native",
            "shacl_interop": "native", "owl_interop": "native",
            "ssn_sosa": "native", "croissant_interop": "native",
            "vector_embeddings": "native", "measurement_uncertainty": "native",
            "calibration": "native", "translation_provenance": "native",
            "graph_merge": "native",
        },
        "croissant": {
            "confidence": "not_possible", "sl_opinion": "not_possible",
            "source": "workaround", "extraction_time": "not_possible",
            "extraction_method": "not_possible", "human_verified": "not_possible",
            "derivation_chain": "not_possible", "delegation_chain": "not_possible",
            "invalidation": "not_possible",
            "validity_window": "not_possible", "temporal_query": "not_possible",
            "temporal_diff": "not_possible",
            "cumulative_fusion": "not_possible", "averaging_fusion": "not_possible",
            "trust_discount": "not_possible", "deduction": "not_possible",
            "conflict_detection": "not_possible", "byzantine_fusion": "not_possible",
            "temporal_decay": "not_possible",
            "schema_validation": "workaround", "required_fields": "workaround",
            "cross_field": "not_possible",
            "dataset_metadata": "native", "file_distributions": "native",
            "record_set": "native",
            "rdf_compat": "native", "prov_o_interop": "not_possible",
            "shacl_interop": "not_possible", "owl_interop": "not_possible",
            "ssn_sosa": "not_possible", "croissant_interop": "native",
            "vector_embeddings": "not_possible", "measurement_uncertainty": "not_possible",
            "calibration": "not_possible", "translation_provenance": "not_possible",
            "graph_merge": "not_possible",
        },
        "prov_o": {
            "confidence": "workaround", "sl_opinion": "not_possible",
            "source": "native", "extraction_time": "native",
            "extraction_method": "native", "human_verified": "workaround",
            "derivation_chain": "native", "delegation_chain": "native",
            "invalidation": "native",
            "validity_window": "native", "temporal_query": "workaround",
            "temporal_diff": "not_possible",
            "cumulative_fusion": "not_possible", "averaging_fusion": "not_possible",
            "trust_discount": "not_possible", "deduction": "not_possible",
            "conflict_detection": "not_possible", "byzantine_fusion": "not_possible",
            "temporal_decay": "not_possible",
            "schema_validation": "not_possible", "required_fields": "not_possible",
            "cross_field": "not_possible",
            "dataset_metadata": "workaround", "file_distributions": "workaround",
            "record_set": "not_possible",
            "rdf_compat": "native", "prov_o_interop": "native",
            "shacl_interop": "not_possible", "owl_interop": "not_possible",
            "ssn_sosa": "workaround", "croissant_interop": "not_possible",
            "vector_embeddings": "not_possible", "measurement_uncertainty": "workaround",
            "calibration": "workaround", "translation_provenance": "workaround",
            "graph_merge": "workaround",
        },
        "shacl": {
            "confidence": "not_possible", "sl_opinion": "not_possible",
            "source": "not_possible", "extraction_time": "not_possible",
            "extraction_method": "not_possible", "human_verified": "not_possible",
            "derivation_chain": "not_possible", "delegation_chain": "not_possible",
            "invalidation": "not_possible",
            "validity_window": "not_possible", "temporal_query": "not_possible",
            "temporal_diff": "not_possible",
            "cumulative_fusion": "not_possible", "averaging_fusion": "not_possible",
            "trust_discount": "not_possible", "deduction": "not_possible",
            "conflict_detection": "not_possible", "byzantine_fusion": "not_possible",
            "temporal_decay": "not_possible",
            "schema_validation": "native", "required_fields": "native",
            "cross_field": "native",
            "dataset_metadata": "not_possible", "file_distributions": "not_possible",
            "record_set": "not_possible",
            "rdf_compat": "native", "prov_o_interop": "not_possible",
            "shacl_interop": "native", "owl_interop": "workaround",
            "ssn_sosa": "not_possible", "croissant_interop": "not_possible",
            "vector_embeddings": "not_possible", "measurement_uncertainty": "not_possible",
            "calibration": "not_possible", "translation_provenance": "not_possible",
            "graph_merge": "not_possible",
        },
        "jsonld_11": {
            "confidence": "not_possible", "sl_opinion": "not_possible",
            "source": "not_possible", "extraction_time": "not_possible",
            "extraction_method": "not_possible", "human_verified": "not_possible",
            "derivation_chain": "not_possible", "delegation_chain": "not_possible",
            "invalidation": "not_possible",
            "validity_window": "not_possible", "temporal_query": "not_possible",
            "temporal_diff": "not_possible",
            "cumulative_fusion": "not_possible", "averaging_fusion": "not_possible",
            "trust_discount": "not_possible", "deduction": "not_possible",
            "conflict_detection": "not_possible", "byzantine_fusion": "not_possible",
            "temporal_decay": "not_possible",
            "schema_validation": "not_possible", "required_fields": "not_possible",
            "cross_field": "not_possible",
            "dataset_metadata": "workaround", "file_distributions": "workaround",
            "record_set": "workaround",
            "rdf_compat": "native", "prov_o_interop": "workaround",
            "shacl_interop": "workaround", "owl_interop": "workaround",
            "ssn_sosa": "workaround", "croissant_interop": "workaround",
            "vector_embeddings": "not_possible", "measurement_uncertainty": "not_possible",
            "calibration": "not_possible", "translation_provenance": "not_possible",
            "graph_merge": "not_possible",
        },
        "plain_json": {
            "confidence": "workaround", "sl_opinion": "workaround",
            "source": "workaround", "extraction_time": "workaround",
            "extraction_method": "workaround", "human_verified": "workaround",
            "derivation_chain": "workaround", "delegation_chain": "workaround",
            "invalidation": "workaround",
            "validity_window": "workaround", "temporal_query": "not_possible",
            "temporal_diff": "not_possible",
            "cumulative_fusion": "not_possible", "averaging_fusion": "not_possible",
            "trust_discount": "not_possible", "deduction": "not_possible",
            "conflict_detection": "not_possible", "byzantine_fusion": "not_possible",
            "temporal_decay": "not_possible",
            "schema_validation": "not_possible", "required_fields": "not_possible",
            "cross_field": "not_possible",
            "dataset_metadata": "workaround", "file_distributions": "workaround",
            "record_set": "workaround",
            "rdf_compat": "not_possible", "prov_o_interop": "not_possible",
            "shacl_interop": "not_possible", "owl_interop": "not_possible",
            "ssn_sosa": "not_possible", "croissant_interop": "not_possible",
            "vector_embeddings": "workaround", "measurement_uncertainty": "workaround",
            "calibration": "workaround", "translation_provenance": "workaround",
            "graph_merge": "not_possible",
        },
        "hf_datasets": {
            "confidence": "not_possible", "sl_opinion": "not_possible",
            "source": "workaround", "extraction_time": "not_possible",
            "extraction_method": "not_possible", "human_verified": "not_possible",
            "derivation_chain": "not_possible", "delegation_chain": "not_possible",
            "invalidation": "not_possible",
            "validity_window": "not_possible", "temporal_query": "not_possible",
            "temporal_diff": "not_possible",
            "cumulative_fusion": "not_possible", "averaging_fusion": "not_possible",
            "trust_discount": "not_possible", "deduction": "not_possible",
            "conflict_detection": "not_possible", "byzantine_fusion": "not_possible",
            "temporal_decay": "not_possible",
            "schema_validation": "workaround", "required_fields": "workaround",
            "cross_field": "not_possible",
            "dataset_metadata": "native", "file_distributions": "native",
            "record_set": "native",
            "rdf_compat": "not_possible", "prov_o_interop": "not_possible",
            "shacl_interop": "not_possible", "owl_interop": "not_possible",
            "ssn_sosa": "not_possible", "croissant_interop": "workaround",
            "vector_embeddings": "workaround", "measurement_uncertainty": "not_possible",
            "calibration": "not_possible", "translation_provenance": "not_possible",
            "graph_merge": "not_possible",
        },
        "mlflow": {
            "confidence": "workaround", "sl_opinion": "not_possible",
            "source": "native", "extraction_time": "native",
            "extraction_method": "native", "human_verified": "not_possible",
            "derivation_chain": "workaround", "delegation_chain": "not_possible",
            "invalidation": "not_possible",
            "validity_window": "not_possible", "temporal_query": "not_possible",
            "temporal_diff": "not_possible",
            "cumulative_fusion": "not_possible", "averaging_fusion": "not_possible",
            "trust_discount": "not_possible", "deduction": "not_possible",
            "conflict_detection": "not_possible", "byzantine_fusion": "not_possible",
            "temporal_decay": "not_possible",
            "schema_validation": "workaround", "required_fields": "not_possible",
            "cross_field": "not_possible",
            "dataset_metadata": "native", "file_distributions": "workaround",
            "record_set": "workaround",
            "rdf_compat": "not_possible", "prov_o_interop": "not_possible",
            "shacl_interop": "not_possible", "owl_interop": "not_possible",
            "ssn_sosa": "not_possible", "croissant_interop": "not_possible",
            "vector_embeddings": "not_possible", "measurement_uncertainty": "not_possible",
            "calibration": "not_possible", "translation_provenance": "not_possible",
            "graph_merge": "not_possible",
        },
    }
    # fmt: on

    return matrix


def compute_coverage_summary(matrix: Dict) -> Dict[str, Dict[str, int]]:
    """Compute native/workaround/not_possible counts per format."""
    summary = {}
    for fmt, features in matrix.items():
        counts = {"native": 0, "workaround": 0, "not_possible": 0}
        for feat_key, level in features.items():
            counts[level] += 1
        counts["total"] = len(features)
        counts["native_pct"] = round(counts["native"] / counts["total"] * 100, 1)
        counts["expressible_pct"] = round(
            (counts["native"] + counts["workaround"]) / counts["total"] * 100, 1
        )
        summary[fmt] = counts
    return summary


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    """Run EN2.1 + EN2.2 — Format Expressiveness Comparison."""

    print("=" * 70)
    print("EN2.1 + EN2.2 — Format Expressiveness Comparison")
    print("NeurIPS 2026 D&B, Suite EN2")
    print("=" * 70)

    t_start = time.time()
    env = log_environment()

    # ── EN2.1: Verbosity Comparison ──
    print("\n--- EN2.1: Verbosity Comparison (10 scenarios x 6 formats) ---")
    scenarios = build_scenarios()
    byte_results = measure_scenario_bytes(scenarios)

    # Compute aggregates
    formats_for_ratio = ["prov_o", "shacl", "croissant", "plain_json", "jsonld_11"]
    ratio_sums = {f: [] for f in formats_for_ratio}

    for row in byte_results:
        ex_b = row["jsonld_ex_bytes"]
        print(f"\n  {row['id']}: {row['name']}")
        print(f"    jsonld-ex:   {ex_b:>6} bytes")
        for fmt in formats_for_ratio:
            alt_b = row.get(f"{fmt}_bytes")
            ratio = row.get(f"{fmt}_ratio")
            if alt_b is not None:
                print(f"    {fmt:12s}: {alt_b:>6} bytes (ratio: {ratio:.3f}x)")
                ratio_sums[fmt].append(ratio)
            else:
                print(f"    {fmt:12s}:    N/A")

    print("\n  === Aggregate Ratios (alt_bytes / jsonld_ex_bytes) ===")
    print("  > 1.0 = alternative is larger; < 1.0 = alternative is smaller")
    agg_ratios = {}
    for fmt in formats_for_ratio:
        vals = ratio_sums[fmt]
        if vals:
            import numpy as np
            arr = np.array(vals)
            mean_r = float(np.mean(arr))
            median_r = float(np.median(arr))
            agg_ratios[fmt] = {
                "mean": mean_r, "median": median_r,
                "min": float(np.min(arr)), "max": float(np.max(arr)),
                "n_scenarios": len(vals),
            }
            print(f"    {fmt:12s}: mean={mean_r:.3f}x  median={median_r:.3f}x "
                  f"(n={len(vals)})")

    # ── EN2.2: Feature Coverage Matrix ──
    print("\n--- EN2.2: Feature Coverage Matrix (35 features x 8 formats) ---")
    matrix = build_coverage_matrix()
    coverage_summary = compute_coverage_summary(matrix)

    print(f"\n  {'Format':15s} {'Native':>7} {'Workaround':>11} {'Not Possible':>13} "
          f"{'Native%':>8} {'Expressible%':>13}")
    print("  " + "-" * 70)
    for fmt, counts in coverage_summary.items():
        print(f"  {fmt:15s} {counts['native']:>7} {counts['workaround']:>11} "
              f"{counts['not_possible']:>13} {counts['native_pct']:>7.1f}% "
              f"{counts['expressible_pct']:>12.1f}%")

    # ── Timing ──
    total_time = time.time() - t_start
    print(f"\n  Total wall time: {total_time:.1f}s")

    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "en2_1_2_results.json"

    # Build feature list for results
    feature_list = [{"name": name, "key": key} for name, key in FEATURES]

    experiment_result = ExperimentResult(
        experiment_id="EN2.1+EN2.2",
        parameters={
            "n_scenarios": len(scenarios),
            "n_features": len(FEATURES),
            "formats_compared_verbosity": ["jsonld_ex", "prov_o", "shacl",
                                            "croissant", "plain_json", "jsonld_11"],
            "formats_compared_coverage": list(matrix.keys()),
            "byte_measurement": "json.dumps(indent=2, sort_keys=True).encode('utf-8')",
        },
        metrics={
            "en2_1_aggregate_ratios": agg_ratios,
            "en2_2_coverage_summary": coverage_summary,
            "en2_2_n_features": len(FEATURES),
            "total_wall_time_seconds": round(total_time, 4),
        },
        raw_data={
            "en2_1_scenario_bytes": byte_results,
            "en2_2_feature_matrix": matrix,
            "en2_2_feature_list": feature_list,
        },
        environment=env,
        notes=(
            "EN2.1: Verbosity comparison of 10 ML scenarios across 6 formats. "
            "EN2.2: Feature coverage matrix of 35 ML-relevant features across "
            "8 format ecosystems. All representations constructed to be fair "
            "to alternatives — no strawmanning."
        ),
    )
    experiment_result.save_json(str(output_path))
    print(f"\n  Results saved to: {output_path}")

    # ── Timestamped archive ──
    ts = experiment_result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    archive_path = RESULTS_DIR / f"en2_1_2_results_{ts}.json"
    experiment_result.save_json(str(archive_path))
    print(f"  Archived to:      {archive_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
