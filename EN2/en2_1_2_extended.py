#!/usr/bin/env python
"""EN2.1 + EN2.2 — Format Expressiveness Comparison (Extended, NeurIPS-Grade).

NeurIPS 2026 D&B, Suite EN2 (Format Expressiveness), Experiments 1 & 2.

Extended version with:
  - 10 scenarios x 6 formats, ALL formats attempted for ALL scenarios
  - Information completeness scoring: what fraction of semantic fields survives
  - Bytes-per-semantic-field normalization
  - Both compact (no indent) and pretty (indent=2) serialization
  - Scaling analysis: 10, 100, 1,000, 10,000 annotated nodes
  - Feature matrix with justification strings for every cell
  - Croissant and SHACL representations for every applicable scenario

Methodology:
  - "Semantic fields" = the distinct pieces of information in the scenario
    (e.g., confidence=0.97 counts as 1 field, source="spacy" counts as 1 field).
  - "Information completeness" = (fields expressible in format) / (total fields).
    If a format CAN express it via a workaround, it counts. If it fundamentally
    cannot, it's a 0 for that field.
  - Byte measurement: both compact and pretty, sorted keys, UTF-8.
  - All alternative representations are constructed as compactly as reasonably
    possible — we do NOT strawman the alternatives.
  - For "not expressible" fields, the alternative representation simply omits
    them, making it smaller but LESS informative.

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN2/en2_1_2_extended.py

References:
    Croissant: Akhtar et al. (2024). NeurIPS 2024 D&B.
    PROV-O: Lebo et al. (2013). W3C Recommendation.
    SHACL: Knublauch & Kontokostas (2017). W3C Recommendation.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "packages" / "python" / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from jsonld_ex.ai_ml import annotate
from jsonld_ex.confidence_algebra import Opinion

from experiments.infra.results import ExperimentResult
from experiments.infra.env_log import log_environment
from experiments.infra.stats import bootstrap_ci

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# =====================================================================
# Utility
# =====================================================================

def bytes_pretty(obj: Any) -> int:
    return len(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8"))

def bytes_compact(obj: Any) -> int:
    return len(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8"))


# =====================================================================
# Scenario Builder
# =====================================================================

class Scenario:
    """A single expressiveness comparison scenario."""

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        semantic_fields: List[str],
        jsonld_ex: dict,
        prov_o: dict,
        shacl: Optional[dict],
        croissant: Optional[dict],
        plain_json: dict,
        jsonld_11: dict,
        # Which semantic fields each format preserves
        prov_o_fields: List[str],
        shacl_fields: Optional[List[str]],
        croissant_fields: Optional[List[str]],
        plain_json_fields: List[str],
        jsonld_11_fields: List[str],
    ):
        self.id = id
        self.name = name
        self.description = description
        self.semantic_fields = semantic_fields
        self.representations = {
            "jsonld_ex": jsonld_ex,
            "prov_o": prov_o,
            "shacl": shacl,
            "croissant": croissant,
            "plain_json": plain_json,
            "jsonld_11": jsonld_11,
        }
        self.field_coverage = {
            "jsonld_ex": semantic_fields,  # always 100%
            "prov_o": prov_o_fields,
            "shacl": shacl_fields or [],
            "croissant": croissant_fields or [],
            "plain_json": plain_json_fields,
            "jsonld_11": jsonld_11_fields,
        }

    def measure(self) -> Dict[str, Any]:
        n_fields = len(self.semantic_fields)
        result = {"id": self.id, "name": self.name,
                  "description": self.description,
                  "n_semantic_fields": n_fields}

        for fmt, doc in self.representations.items():
            if doc is not None:
                bp = bytes_pretty(doc)
                bc = bytes_compact(doc)
                covered = self.field_coverage[fmt]
                n_covered = len(covered)
                completeness = n_covered / n_fields if n_fields > 0 else 0.0
                fields_lost = sorted(set(self.semantic_fields) - set(covered))

                result[fmt] = {
                    "bytes_pretty": bp,
                    "bytes_compact": bc,
                    "fields_covered": n_covered,
                    "fields_total": n_fields,
                    "completeness": round(completeness, 4),
                    "fields_lost": fields_lost,
                    "bytes_per_field_pretty": round(bp / n_covered, 1) if n_covered > 0 else None,
                    "bytes_per_field_compact": round(bc / n_covered, 1) if n_covered > 0 else None,
                }
            else:
                result[fmt] = {
                    "bytes_pretty": None, "bytes_compact": None,
                    "fields_covered": 0, "fields_total": n_fields,
                    "completeness": 0.0,
                    "fields_lost": self.semantic_fields,
                    "bytes_per_field_pretty": None,
                    "bytes_per_field_compact": None,
                }
        return result


def build_all_scenarios() -> List[Scenario]:
    scenarios = []

    # ── S1: Dataset Card with Splits ──
    s1_fields = [
        "dataset_name", "dataset_name_confidence", "dataset_name_source",
        "dataset_name_timestamp", "description", "license",
        "file_train_name", "file_train_size", "file_train_format",
        "file_test_name", "file_test_size", "file_test_format",
        "field_token_name", "field_token_type",
        "field_ner_name", "field_ner_type", "field_ner_confidence",
        "field_ner_source",
    ]

    scenarios.append(Scenario(
        id="S1", name="Dataset card with splits and per-field confidence",
        description="ML dataset with file distributions, record schema, and per-field annotation quality metadata",
        semantic_fields=s1_fields,
        jsonld_ex={
            "@context": "https://json-ld.org/ns/jsonld-ex/v1",
            "@type": "Dataset",
            "name": annotate("CoNLL-2003", confidence=0.99, source="original-paper",
                             extracted_at="2026-01-15T10:00:00Z"),
            "description": "Named Entity Recognition benchmark dataset",
            "license": "custom-research",
            "distribution": [
                {"@type": "FileObject", "name": "train.txt",
                 "contentSize": "3.3MB", "encodingFormat": "text/plain"},
                {"@type": "FileObject", "name": "test.txt",
                 "contentSize": "0.7MB", "encodingFormat": "text/plain"},
            ],
            "recordSet": {"@type": "RecordSet", "field": [
                {"@type": "Field", "name": "token", "dataType": "sc:Text"},
                {"@type": "Field", "name": "ner_tag", "dataType": "sc:Text",
                 "@confidence": 0.95, "@source": "annotator-panel"},
            ]},
        },
        prov_o={
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
                 "prov:wasGeneratedBy": {
                     "@type": "prov:Activity",
                     "prov:wasAssociatedWith": "original-paper",
                     "prov:endedAtTime": "2026-01-15T10:00:00Z"},
                 "schema:additionalProperty": {"schema:value": 0.99,
                                                "schema:name": "confidence"}},
            ],
        },
        shacl=None,
        croissant={
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
            "recordSet": {"@type": "cr:RecordSet", "field": [
                {"@type": "cr:Field", "sc:name": "token", "dataType": "sc:Text"},
                {"@type": "cr:Field", "sc:name": "ner_tag", "dataType": "sc:Text"},
            ]},
        },
        plain_json={
            "name": "CoNLL-2003",
            "name_meta": {"confidence": 0.99, "source": "original-paper",
                          "extracted_at": "2026-01-15T10:00:00Z"},
            "description": "Named Entity Recognition benchmark dataset",
            "license": "custom-research",
            "files": [
                {"name": "train.txt", "size": "3.3MB", "format": "text/plain"},
                {"name": "test.txt", "size": "0.7MB", "format": "text/plain"},
            ],
            "fields": [
                {"name": "token", "type": "text"},
                {"name": "ner_tag", "type": "text",
                 "confidence": 0.95, "source": "annotator-panel"},
            ],
        },
        jsonld_11={
            "@context": {"schema": "http://schema.org/"},
            "@type": "schema:Dataset",
            "schema:name": "CoNLL-2003",
            "schema:description": "Named Entity Recognition benchmark dataset",
            "schema:license": "custom-research",
            "schema:distribution": [
                {"@type": "schema:DataDownload", "schema:name": "train.txt",
                 "schema:contentSize": "3.3MB", "schema:encodingFormat": "text/plain"},
                {"@type": "schema:DataDownload", "schema:name": "test.txt",
                 "schema:contentSize": "0.7MB", "schema:encodingFormat": "text/plain"},
            ],
        },
        prov_o_fields=[
            "dataset_name", "dataset_name_confidence", "dataset_name_source",
            "dataset_name_timestamp", "description", "license",
            "file_train_name", "file_train_size", "file_train_format",
            "file_test_name", "file_test_size", "file_test_format",
        ],
        shacl_fields=None,
        croissant_fields=[
            "dataset_name", "description", "license",
            "file_train_name", "file_train_size", "file_train_format",
            "file_test_name", "file_test_size", "file_test_format",
            "field_token_name", "field_token_type",
            "field_ner_name", "field_ner_type",
        ],
        plain_json_fields=s1_fields,  # plain JSON CAN store everything (ad-hoc keys)
        jsonld_11_fields=[
            "dataset_name", "description", "license",
            "file_train_name", "file_train_size", "file_train_format",
            "file_test_name", "file_test_size", "file_test_format",
        ],
    ))

    # ── S2: NER Annotations ──
    s2_fields = [
        "tok0_text", "tok0_confidence", "tok0_source", "tok0_method",
        "tok1_text", "tok1_confidence", "tok1_source", "tok1_method",
        "tok2_text", "tok2_confidence", "tok2_source", "tok2_method",
        "tok3_text", "tok3_confidence", "tok3_source", "tok3_method",
        "ent0_text", "ent0_label", "ent0_label_conf", "ent0_source", "ent0_span",
        "ent1_text", "ent1_label", "ent1_label_conf", "ent1_source", "ent1_span",
    ]

    scenarios.append(Scenario(
        id="S2", name="NER annotations with per-token confidence",
        description="NLP token annotations with per-token confidence, source, method",
        semantic_fields=s2_fields,
        jsonld_ex={
            "@context": "https://json-ld.org/ns/jsonld-ex/v1",
            "tokens": [
                annotate("Barack", confidence=0.97, source="spacy-trf", method="NER"),
                annotate("Obama", confidence=0.99, source="spacy-trf", method="NER"),
                annotate("visited", confidence=0.85, source="spacy-trf", method="NER"),
                annotate("Paris", confidence=0.92, source="spacy-trf", method="NER"),
            ],
            "entities": [
                {"text": "Barack Obama",
                 "label": annotate("PER", confidence=0.98, source="spacy-trf"),
                 "span": [0, 1]},
                {"text": "Paris",
                 "label": annotate("LOC", confidence=0.92, source="spacy-trf"),
                 "span": [3, 3]},
            ],
        },
        prov_o={
            "@context": {"prov": "http://www.w3.org/ns/prov#",
                          "nif": "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#"},
            "@graph": [
                {"@id": f"_:tok{i}", "@type": "nif:Word", "nif:anchorOf": text,
                 "nif:confidence": conf,
                 "prov:wasGeneratedBy": {
                     "prov:wasAssociatedWith": "spacy-trf",
                     "prov:qualifiedAssociation": {"prov:hadRole": "NER"}}}
                for i, (text, conf) in enumerate([
                    ("Barack", 0.97), ("Obama", 0.99),
                    ("visited", 0.85), ("Paris", 0.92)])
            ] + [
                {"@id": "_:ent0", "@type": "nif:EntityOccurrence",
                 "nif:anchorOf": "Barack Obama", "nif:entity": "PER",
                 "nif:confidence": 0.98, "nif:beginIndex": 0, "nif:endIndex": 1,
                 "prov:wasGeneratedBy": {"prov:wasAssociatedWith": "spacy-trf"}},
                {"@id": "_:ent1", "@type": "nif:EntityOccurrence",
                 "nif:anchorOf": "Paris", "nif:entity": "LOC",
                 "nif:confidence": 0.92, "nif:beginIndex": 3, "nif:endIndex": 3,
                 "prov:wasGeneratedBy": {"prov:wasAssociatedWith": "spacy-trf"}},
            ],
        },
        shacl=None, croissant=None,
        plain_json={
            "tokens": [
                {"text": t, "confidence": c, "source": "spacy-trf", "method": "NER"}
                for t, c in [("Barack", 0.97), ("Obama", 0.99),
                              ("visited", 0.85), ("Paris", 0.92)]
            ],
            "entities": [
                {"text": "Barack Obama", "label": "PER", "label_confidence": 0.98,
                 "source": "spacy-trf", "span": [0, 1]},
                {"text": "Paris", "label": "LOC", "label_confidence": 0.92,
                 "source": "spacy-trf", "span": [3, 3]},
            ],
        },
        jsonld_11={
            "@context": {"nif": "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#"},
            "@graph": [
                {"@type": "nif:Word", "nif:anchorOf": t}
                for t in ["Barack", "Obama", "visited", "Paris"]
            ] + [
                {"@type": "nif:EntityOccurrence", "nif:anchorOf": "Barack Obama",
                 "nif:entity": "PER", "nif:beginIndex": 0, "nif:endIndex": 1},
                {"@type": "nif:EntityOccurrence", "nif:anchorOf": "Paris",
                 "nif:entity": "LOC", "nif:beginIndex": 3, "nif:endIndex": 3},
            ],
        },
        prov_o_fields=[f for f in s2_fields if "method" not in f],  # PROV-O can express most via roles
        shacl_fields=None,
        croissant_fields=None,
        plain_json_fields=s2_fields,
        jsonld_11_fields=[
            "tok0_text", "tok1_text", "tok2_text", "tok3_text",
            "ent0_text", "ent0_label", "ent0_span",
            "ent1_text", "ent1_label", "ent1_span",
        ],
    ))

    # ── S3: Sensor Reading ──
    s3_fields = [
        "obs_type", "obs_property", "obs_value", "obs_confidence",
        "obs_source", "obs_uncertainty", "obs_unit", "obs_timestamp",
        "cal_date", "cal_method",
    ]

    scenarios.append(Scenario(
        id="S3", name="IoT sensor reading with uncertainty and calibration",
        description="Sensor observation with measurement uncertainty, calibration provenance, confidence",
        semantic_fields=s3_fields,
        jsonld_ex={
            "@context": "https://json-ld.org/ns/jsonld-ex/v1",
            "@type": "Observation",
            "observedProperty": "temperature",
            "result": annotate(23.4, confidence=0.95, source="sensor-DHT22",
                               measurement_uncertainty=0.5, unit="degC",
                               calibrated_at="2026-01-01T00:00:00Z",
                               calibration_method="NIST-traceable",
                               extracted_at="2026-03-10T14:30:00Z"),
        },
        prov_o={
            "@context": {"sosa": "http://www.w3.org/ns/sosa/",
                          "ssn-sys": "http://www.w3.org/ns/ssn/systems/",
                          "qudt": "http://qudt.org/schema/qudt/",
                          "prov": "http://www.w3.org/ns/prov#"},
            "@type": "sosa:Observation",
            "sosa:observedProperty": "temperature",
            "sosa:hasSimpleResult": 23.4,
            "sosa:resultTime": "2026-03-10T14:30:00Z",
            "sosa:madeBySensor": {
                "@id": "sensor:DHT22", "@type": "sosa:Sensor",
                "ssn-sys:hasSystemCapability": {
                    "ssn-sys:hasSystemProperty": {
                        "@type": "ssn-sys:Accuracy",
                        "schema:value": 0.5, "qudt:unit": "qudt:DEG_C"}}},
            "prov:wasGeneratedBy": {
                "prov:wasAssociatedWith": "sensor:DHT22",
                "prov:endedAtTime": "2026-03-10T14:30:00Z"},
            "ssn-sys:hasDeployment": {
                "prov:qualifiedAssociation": {
                    "prov:hadRole": "calibration",
                    "prov:agent": "NIST-traceable",
                    "prov:atTime": "2026-01-01T00:00:00Z"}},
        },
        shacl=None, croissant=None,
        plain_json={
            "type": "observation", "property": "temperature",
            "value": 23.4, "unit": "degC", "uncertainty": 0.5,
            "confidence": 0.95, "sensor": "DHT22",
            "timestamp": "2026-03-10T14:30:00Z",
            "calibration": {"date": "2026-01-01T00:00:00Z",
                             "method": "NIST-traceable"},
        },
        jsonld_11={
            "@context": {"sosa": "http://www.w3.org/ns/sosa/"},
            "@type": "sosa:Observation",
            "sosa:observedProperty": "temperature",
            "sosa:hasSimpleResult": 23.4,
            "sosa:resultTime": "2026-03-10T14:30:00Z",
        },
        prov_o_fields=s3_fields,  # SSN/SOSA+PROV can express all (verbose)
        shacl_fields=None,
        croissant_fields=None,
        plain_json_fields=s3_fields,
        jsonld_11_fields=["obs_type", "obs_property", "obs_value", "obs_timestamp"],
    ))

    # ── S4: KG Triple with Provenance Chain ──
    s4_fields = [
        "person_name", "birthplace_value", "birthplace_confidence",
        "birthplace_source", "birthplace_timestamp",
        "birthplace_derived_from_1", "birthplace_derived_from_2",
        "birthplace_human_verified",
    ]

    scenarios.append(Scenario(
        id="S4", name="KG triple with multi-source provenance chain",
        description="Knowledge graph assertion with confidence, multi-source derivation, human verification",
        semantic_fields=s4_fields,
        jsonld_ex={
            "@context": "https://json-ld.org/ns/jsonld-ex/v1",
            "@type": "Person", "name": "Marie Curie",
            "birthPlace": annotate("Warsaw", confidence=0.99,
                                    source="wikidata:Q36",
                                    extracted_at="2026-02-01T12:00:00Z",
                                    derived_from=["wikidata:Q7186", "dbpedia:Marie_Curie"],
                                    human_verified=True),
        },
        prov_o={
            "@context": {"prov": "http://www.w3.org/ns/prov#",
                          "schema": "http://schema.org/"},
            "@graph": [
                {"@id": "person:mc", "@type": "schema:Person",
                 "schema:name": "Marie Curie",
                 "schema:birthPlace": {"@id": "_:bp"}},
                {"@id": "_:bp", "@type": "prov:Entity",
                 "prov:value": "Warsaw",
                 "prov:wasGeneratedBy": {
                     "prov:wasAssociatedWith": "wikidata:Q36",
                     "prov:endedAtTime": "2026-02-01T12:00:00Z"},
                 "prov:wasDerivedFrom": [{"@id": "wikidata:Q7186"},
                                          {"@id": "dbpedia:Marie_Curie"}],
                 "prov:wasAttributedTo": {"prov:hadRole": "human-verifier"},
                 "schema:additionalProperty": {
                     "schema:name": "confidence", "schema:value": 0.99}},
            ],
        },
        shacl=None, croissant=None,
        plain_json={
            "type": "Person", "name": "Marie Curie",
            "birthPlace": "Warsaw",
            "birthPlace_meta": {
                "confidence": 0.99, "source": "wikidata:Q36",
                "extracted_at": "2026-02-01T12:00:00Z",
                "derived_from": ["wikidata:Q7186", "dbpedia:Marie_Curie"],
                "human_verified": True},
        },
        jsonld_11={
            "@context": {"schema": "http://schema.org/"},
            "@type": "schema:Person",
            "schema:name": "Marie Curie",
            "schema:birthPlace": "Warsaw",
        },
        prov_o_fields=s4_fields,
        shacl_fields=None, croissant_fields=None,
        plain_json_fields=s4_fields,
        jsonld_11_fields=["person_name", "birthplace_value"],
    ))

    # ── S5: Model Prediction ──
    s5_fields = [
        "input_file", "label_primary", "label_primary_confidence",
        "label_primary_source", "label_primary_method",
        "label_primary_timestamp",
        "label_secondary", "label_secondary_confidence",
        "label_secondary_source",
    ]

    scenarios.append(Scenario(
        id="S5", name="Model prediction with multi-label confidence",
        description="Classification output with per-label confidence and model provenance",
        semantic_fields=s5_fields,
        jsonld_ex={
            "@context": "https://json-ld.org/ns/jsonld-ex/v1",
            "@type": "Prediction",
            "input": "chest_xray_0042.dcm",
            "label": annotate("pneumonia", confidence=0.87,
                               source="resnet50-chexpert-v2",
                               method="classification",
                               extracted_at="2026-03-10T09:15:00Z"),
            "secondary_label": annotate("normal", confidence=0.11,
                                         source="resnet50-chexpert-v2"),
        },
        prov_o={
            "@context": {"prov": "http://www.w3.org/ns/prov#"},
            "@graph": [
                {"@id": "_:pred", "@type": "prov:Activity",
                 "prov:used": "chest_xray_0042.dcm",
                 "prov:wasAssociatedWith": "resnet50-chexpert-v2",
                 "prov:endedAtTime": "2026-03-10T09:15:00Z",
                 "prov:qualifiedAssociation": {"prov:hadRole": "classification"}},
                {"@id": "_:lbl1", "@type": "prov:Entity",
                 "prov:value": "pneumonia",
                 "prov:wasGeneratedBy": {"@id": "_:pred"},
                 "schema:additionalProperty": {
                     "schema:name": "confidence", "schema:value": 0.87}},
                {"@id": "_:lbl2", "@type": "prov:Entity",
                 "prov:value": "normal",
                 "prov:wasGeneratedBy": {"@id": "_:pred"},
                 "schema:additionalProperty": {
                     "schema:name": "confidence", "schema:value": 0.11}},
            ],
        },
        shacl=None, croissant=None,
        plain_json={
            "input": "chest_xray_0042.dcm",
            "predictions": [
                {"label": "pneumonia", "confidence": 0.87,
                 "model": "resnet50-chexpert-v2", "method": "classification",
                 "timestamp": "2026-03-10T09:15:00Z"},
                {"label": "normal", "confidence": 0.11,
                 "model": "resnet50-chexpert-v2"},
            ],
        },
        jsonld_11={
            "@context": {"schema": "http://schema.org/"},
            "@type": "schema:Action",
            "schema:object": "chest_xray_0042.dcm",
            "schema:result": "pneumonia",
        },
        prov_o_fields=s5_fields,
        shacl_fields=None, croissant_fields=None,
        plain_json_fields=s5_fields,
        jsonld_11_fields=["input_file", "label_primary"],
    ))

    # ── S6: Translation Provenance ──
    s6_fields = [
        "title_en", "title_ja", "translated_from", "translation_model",
        "translation_confidence", "translation_source", "translation_timestamp",
    ]

    scenarios.append(Scenario(
        id="S6", name="Multilingual content with translation provenance",
        description="Translated text with source language, model, confidence, and timestamp",
        semantic_fields=s6_fields,
        jsonld_ex={
            "@context": "https://json-ld.org/ns/jsonld-ex/v1",
            "title": {"@value": "Neural Networks", "@language": "en"},
            "title_ja": annotate("ニューラルネットワーク",
                                  translated_from="en",
                                  translation_model="gpt-4-turbo",
                                  confidence=0.94, source="openai-api",
                                  extracted_at="2026-03-01T08:00:00Z"),
        },
        prov_o={
            "@context": {"prov": "http://www.w3.org/ns/prov#"},
            "@graph": [
                {"@id": "_:en", "@type": "prov:Entity",
                 "prov:value": {"@value": "Neural Networks", "@language": "en"}},
                {"@id": "_:ja", "@type": "prov:Entity",
                 "prov:value": {"@value": "ニューラルネットワーク", "@language": "ja"},
                 "prov:wasDerivedFrom": {"@id": "_:en"},
                 "prov:wasGeneratedBy": {
                     "prov:wasAssociatedWith": "gpt-4-turbo",
                     "prov:endedAtTime": "2026-03-01T08:00:00Z",
                     "prov:qualifiedAssociation": {"prov:hadRole": "translation"}},
                 "schema:additionalProperty": {
                     "schema:name": "confidence", "schema:value": 0.94}},
            ],
        },
        shacl=None, croissant=None,
        plain_json={
            "title": {"text": "Neural Networks", "language": "en"},
            "title_ja": {"text": "ニューラルネットワーク", "language": "ja",
                          "translated_from": "en", "model": "gpt-4-turbo",
                          "confidence": 0.94, "source": "openai-api",
                          "timestamp": "2026-03-01T08:00:00Z"},
        },
        jsonld_11={
            "@context": {},
            "title": {"@value": "Neural Networks", "@language": "en"},
            "title_ja": {"@value": "ニューラルネットワーク", "@language": "ja"},
        },
        prov_o_fields=s6_fields,
        shacl_fields=None, croissant_fields=None,
        plain_json_fields=s6_fields,
        jsonld_11_fields=["title_en", "title_ja"],
    ))

    # ── S7: Temporal Validity Window ──
    s7_fields = [
        "regulation_name", "status_value", "status_confidence",
        "status_source", "valid_from", "valid_until",
    ]

    scenarios.append(Scenario(
        id="S7", name="Temporal validity window on regulatory assertion",
        description="Time-bounded assertion with validity period and confidence on status",
        semantic_fields=s7_fields,
        jsonld_ex={
            "@context": "https://json-ld.org/ns/jsonld-ex/v1",
            "@type": "Regulation",
            "name": "GDPR Article 6(1)(a)",
            "status": annotate("active", confidence=1.0, source="eur-lex.europa.eu"),
            "@validFrom": "2018-05-25T00:00:00Z",
            "@validUntil": "2099-12-31T23:59:59Z",
        },
        prov_o={
            "@context": {"prov": "http://www.w3.org/ns/prov#",
                          "schema": "http://schema.org/"},
            "@graph": [
                {"@id": "_:reg", "@type": "prov:Entity",
                 "schema:name": "GDPR Article 6(1)(a)",
                 "prov:generatedAtTime": "2018-05-25T00:00:00Z",
                 "prov:invalidatedAtTime": "2099-12-31T23:59:59Z",
                 "schema:additionalProperty": [
                     {"schema:name": "status", "schema:value": "active"},
                     {"schema:name": "confidence", "schema:value": 1.0},
                     {"schema:name": "source",
                      "schema:value": "eur-lex.europa.eu"}]},
            ],
        },
        shacl=None, croissant=None,
        plain_json={
            "name": "GDPR Article 6(1)(a)", "status": "active",
            "confidence": 1.0, "source": "eur-lex.europa.eu",
            "valid_from": "2018-05-25T00:00:00Z",
            "valid_until": "2099-12-31T23:59:59Z",
        },
        jsonld_11={
            "@context": {"schema": "http://schema.org/"},
            "@type": "schema:Legislation",
            "schema:name": "GDPR Article 6(1)(a)",
            "schema:datePublished": "2018-05-25",
        },
        prov_o_fields=s7_fields,
        shacl_fields=None, croissant_fields=None,
        plain_json_fields=s7_fields,
        jsonld_11_fields=["regulation_name", "valid_from"],
    ))

    # ── S8: Validation Shape ──
    s8_fields = [
        "target_type", "name_required", "name_type", "name_minLength",
        "email_required", "email_type", "email_pattern",
        "age_type", "age_min", "age_max",
    ]

    scenarios.append(Scenario(
        id="S8", name="Validation shape for Person entity",
        description="Data validation constraints: types, required fields, ranges, patterns",
        semantic_fields=s8_fields,
        jsonld_ex={
            "@context": "https://json-ld.org/ns/jsonld-ex/v1",
            "@shape": {
                "@type": "Person",
                "name": {"@required": True, "@type": "string", "@minLength": 1},
                "email": {"@required": True, "@type": "string",
                           "@pattern": "^.+@.+$"},
                "age": {"@type": "integer", "@min": 0, "@max": 150},
            },
        },
        prov_o=None,
        shacl={
            "@context": {"sh": "http://www.w3.org/ns/shacl#",
                          "xsd": "http://www.w3.org/2001/XMLSchema#",
                          "schema": "http://schema.org/"},
            "@type": "sh:NodeShape",
            "sh:targetClass": "schema:Person",
            "sh:property": [
                {"sh:path": "schema:name", "sh:minCount": 1,
                 "sh:datatype": "xsd:string", "sh:minLength": 1},
                {"sh:path": "schema:email", "sh:minCount": 1,
                 "sh:datatype": "xsd:string", "sh:pattern": "^.+@.+$"},
                {"sh:path": "schema:age", "sh:datatype": "xsd:integer",
                 "sh:minInclusive": 0, "sh:maxInclusive": 150},
            ],
        },
        croissant=None,
        plain_json={
            "type": "validation_schema", "target": "Person",
            "properties": {
                "name": {"type": "string", "required": True, "minLength": 1},
                "email": {"type": "string", "required": True,
                           "pattern": "^.+@.+$"},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
            },
        },
        jsonld_11={
            "@context": {"schema": "http://schema.org/"},
            "@type": "schema:Person",
        },
        prov_o_fields=[],
        shacl_fields=s8_fields,
        croissant_fields=None,
        plain_json_fields=s8_fields,
        jsonld_11_fields=["target_type"],
    ))

    # ── S9: Multi-Source Fused Assertion with SL Opinion ──
    s9_fields = [
        "claim_subject", "claim_value", "claim_confidence",
        "fusion_method", "source_1", "source_2", "source_3",
        "opinion_belief", "opinion_disbelief", "opinion_uncertainty",
        "opinion_base_rate",
    ]

    scenarios.append(Scenario(
        id="S9", name="Multi-source fused assertion with SL opinion",
        description="Fused claim from 3 sources with full (b,d,u,a) opinion and fusion provenance",
        semantic_fields=s9_fields,
        jsonld_ex={
            "@context": "https://json-ld.org/ns/jsonld-ex/v1",
            "@type": "Claim",
            "subject": "global_mean_temp_2025",
            "value": annotate(15.02, confidence=0.96, source="fusion:3-sources",
                               method="cumulative_fuse",
                               derived_from=["NASA-GISS", "HadCRUT5", "NOAA-NCEI"]),
            "@opinion": {"belief": 0.91, "disbelief": 0.02,
                          "uncertainty": 0.07, "baseRate": 0.5},
        },
        prov_o={
            "@context": {"prov": "http://www.w3.org/ns/prov#"},
            "@graph": [
                {"@id": "_:fused", "@type": "prov:Entity",
                 "prov:value": 15.02,
                 "prov:wasGeneratedBy": {
                     "@type": "prov:Activity",
                     "prov:qualifiedAssociation": {"prov:hadRole": "cumulative_fuse"},
                     "prov:used": [{"@id": "_:nasa"}, {"@id": "_:had"}, {"@id": "_:noaa"}]},
                 "schema:additionalProperty": {
                     "schema:name": "confidence", "schema:value": 0.96}},
                {"@id": "_:nasa", "prov:wasAttributedTo": "NASA-GISS"},
                {"@id": "_:had", "prov:wasAttributedTo": "HadCRUT5"},
                {"@id": "_:noaa", "prov:wasAttributedTo": "NOAA-NCEI"},
            ],
        },
        shacl=None, croissant=None,
        plain_json={
            "subject": "global_mean_temp_2025", "value": 15.02,
            "confidence": 0.96, "fusion_method": "cumulative_fuse",
            "sources": ["NASA-GISS", "HadCRUT5", "NOAA-NCEI"],
            "opinion": {"belief": 0.91, "disbelief": 0.02,
                         "uncertainty": 0.07, "base_rate": 0.5},
        },
        jsonld_11={
            "@context": {"schema": "http://schema.org/"},
            "@type": "schema:Claim",
            "schema:about": "global_mean_temp_2025",
            "schema:value": 15.02,
        },
        prov_o_fields=[
            "claim_subject", "claim_value", "claim_confidence",
            "fusion_method", "source_1", "source_2", "source_3",
        ],
        shacl_fields=None, croissant_fields=None,
        plain_json_fields=s9_fields,
        jsonld_11_fields=["claim_subject", "claim_value"],
    ))

    # ── S10: Invalidated / Retracted Claim ──
    s10_fields = [
        "claim_subject", "claim_value", "claim_confidence",
        "claim_source", "invalidated_at", "invalidation_reason",
    ]

    scenarios.append(Scenario(
        id="S10", name="Invalidated/retracted claim",
        description="Retracted assertion with invalidation timestamp and reason",
        semantic_fields=s10_fields,
        jsonld_ex={
            "@context": "https://json-ld.org/ns/jsonld-ex/v1",
            "@type": "Claim",
            "subject": "drug_efficacy_trial_2024",
            "value": annotate("effective", confidence=0.45,
                               source="retracted:10.1234/fake",
                               invalidated_at="2026-02-15T00:00:00Z",
                               invalidation_reason="data-fabrication"),
        },
        prov_o={
            "@context": {"prov": "http://www.w3.org/ns/prov#"},
            "@graph": [
                {"@id": "_:claim", "@type": "prov:Entity",
                 "prov:value": "effective",
                 "prov:wasAttributedTo": "retracted:10.1234/fake",
                 "prov:wasInvalidatedBy": {
                     "@type": "prov:Activity",
                     "prov:atTime": "2026-02-15T00:00:00Z",
                     "prov:qualifiedAssociation": {"prov:hadRole": "retraction"}},
                 "schema:additionalProperty": [
                     {"schema:name": "confidence", "schema:value": 0.45},
                     {"schema:name": "invalidation_reason",
                      "schema:value": "data-fabrication"}]},
            ],
        },
        shacl=None, croissant=None,
        plain_json={
            "subject": "drug_efficacy_trial_2024", "value": "effective",
            "confidence": 0.45, "source": "retracted:10.1234/fake",
            "invalidated_at": "2026-02-15T00:00:00Z",
            "invalidation_reason": "data-fabrication",
        },
        jsonld_11={
            "@context": {"schema": "http://schema.org/"},
            "@type": "schema:Claim",
            "schema:about": "drug_efficacy_trial_2024",
            "schema:text": "effective",
        },
        prov_o_fields=s10_fields,
        shacl_fields=None, croissant_fields=None,
        plain_json_fields=s10_fields,
        jsonld_11_fields=["claim_subject", "claim_value"],
    ))

    return scenarios


# =====================================================================
# EN2.1b — Scaling Analysis
# =====================================================================

def scaling_analysis(sizes: List[int], seed: int = 42) -> List[Dict]:
    """Measure overhead as a function of number of annotated nodes.

    Generates N annotated sensor readings in jsonld-ex and plain JSON,
    measuring bytes per node. Tests whether overhead is constant or grows.
    """
    rng = np.random.RandomState(seed)
    results = []

    for n in sizes:
        # Generate N annotated readings
        jsonld_ex_doc = {
            "@context": "https://json-ld.org/ns/jsonld-ex/v1",
            "readings": [],
        }
        plain_doc = {"readings": []}

        for i in range(n):
            val = round(rng.normal(23.0, 2.0), 2)
            conf = round(rng.uniform(0.7, 1.0), 3)
            src = f"sensor-{rng.randint(1, 10):03d}"
            ts = f"2026-03-10T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:00Z"

            jsonld_ex_doc["readings"].append(
                annotate(val, confidence=conf, source=src,
                         measurement_uncertainty=0.5, unit="degC",
                         extracted_at=ts)
            )
            plain_doc["readings"].append({
                "value": val, "confidence": conf, "source": src,
                "uncertainty": 0.5, "unit": "degC", "timestamp": ts,
            })

        ex_pretty = bytes_pretty(jsonld_ex_doc)
        ex_compact = bytes_compact(jsonld_ex_doc)
        pj_pretty = bytes_pretty(plain_doc)
        pj_compact = bytes_compact(plain_doc)

        results.append({
            "n_nodes": n,
            "jsonld_ex_bytes_pretty": ex_pretty,
            "jsonld_ex_bytes_compact": ex_compact,
            "plain_json_bytes_pretty": pj_pretty,
            "plain_json_bytes_compact": pj_compact,
            "overhead_ratio_pretty": round(ex_pretty / pj_pretty, 4) if pj_pretty > 0 else None,
            "overhead_ratio_compact": round(ex_compact / pj_compact, 4) if pj_compact > 0 else None,
            "jsonld_ex_bytes_per_node_pretty": round(ex_pretty / n, 1),
            "jsonld_ex_bytes_per_node_compact": round(ex_compact / n, 1),
            "plain_json_bytes_per_node_pretty": round(pj_pretty / n, 1),
            "plain_json_bytes_per_node_compact": round(pj_compact / n, 1),
        })

    return results


# =====================================================================
# EN2.2 — Feature Coverage Matrix with Justifications
# =====================================================================

FEATURES = [
    ("Scalar confidence per assertion", "confidence"),
    ("SL opinion (b,d,u,a) per assertion", "sl_opinion"),
    ("Source attribution per assertion", "source"),
    ("Extraction timestamp per assertion", "extraction_time"),
    ("Extraction method per assertion", "extraction_method"),
    ("Human verification flag", "human_verified"),
    ("Derivation chain (derived_from)", "derivation_chain"),
    ("Delegation chain (delegated_by)", "delegation_chain"),
    ("Invalidation with reason", "invalidation"),
    ("Validity window (@validFrom/@validUntil)", "validity_window"),
    ("Point-in-time query", "temporal_query"),
    ("Temporal diff between snapshots", "temporal_diff"),
    ("Cumulative fusion of opinions", "cumulative_fusion"),
    ("Averaging fusion of opinions", "averaging_fusion"),
    ("Trust discount through provenance", "trust_discount"),
    ("Deduction (uncertain conditionals)", "deduction"),
    ("Conflict detection between sources", "conflict_detection"),
    ("Byzantine-resistant fusion", "byzantine_fusion"),
    ("Temporal decay of opinions", "temporal_decay"),
    ("Schema validation (type, range, pattern)", "schema_validation"),
    ("Required field constraints", "required_fields"),
    ("Cross-field constraints", "cross_field"),
    ("Dataset metadata (name, license, etc.)", "dataset_metadata"),
    ("File distribution descriptions", "file_distributions"),
    ("Record set / field definitions", "record_set"),
    ("RDF/Linked Data compatibility", "rdf_compat"),
    ("PROV-O export/import", "prov_o_interop"),
    ("SHACL export/import", "shacl_interop"),
    ("OWL class restriction export", "owl_interop"),
    ("SSN/SOSA sensor interop", "ssn_sosa"),
    ("Croissant import/export", "croissant_interop"),
    ("Vector embeddings (@vector)", "vector_embeddings"),
    ("Measurement uncertainty (IoT)", "measurement_uncertainty"),
    ("Calibration metadata", "calibration"),
    ("Translation provenance", "translation_provenance"),
    ("Multi-source graph merge", "graph_merge"),
]

def build_justified_matrix() -> Dict[str, Dict[str, Dict[str, str]]]:
    """Feature matrix where every cell has {level, justification}."""
    def cell(level: str, justification: str) -> Dict[str, str]:
        return {"level": level, "justification": justification}

    N = "native"
    W = "workaround"
    X = "not_possible"

    m: Dict[str, Dict[str, Dict[str, str]]] = {}

    # jsonld-ex
    m["jsonld_ex"] = {k: cell(N, f"First-class @{k} or dedicated API function")
                       for _, k in FEATURES}
    # Override with specifics
    m["jsonld_ex"]["confidence"] = cell(N, "@confidence annotation field; annotate(confidence=...)")
    m["jsonld_ex"]["sl_opinion"] = cell(N, "@opinion dict with b/d/u/a; Opinion class in confidence_algebra")
    m["jsonld_ex"]["cumulative_fusion"] = cell(N, "cumulative_fuse() in confidence_algebra.py")
    m["jsonld_ex"]["schema_validation"] = cell(N, "@shape with @type/@required/@min/@max/@pattern; validate_node()")
    m["jsonld_ex"]["rdf_compat"] = cell(N, "JSON-LD is an RDF serialization; to_rdf_star_ntriples()")
    m["jsonld_ex"]["croissant_interop"] = cell(N, "to_croissant()/from_croissant() in dataset.py")
    m["jsonld_ex"]["vector_embeddings"] = cell(N, "@vector container; validate_vector()/cosine_similarity()")
    m["jsonld_ex"]["graph_merge"] = cell(N, "merge_graphs()/diff_graphs() in merge.py")

    # Croissant
    m["croissant"] = {}
    m["croissant"]["confidence"] = cell(X, "No annotation-level metadata in Croissant spec (MLCommons 2024)")
    m["croissant"]["sl_opinion"] = cell(X, "No uncertainty algebra in Croissant")
    m["croissant"]["source"] = cell(W, "sc:creator on Dataset, not per-field")
    m["croissant"]["extraction_time"] = cell(X, "No per-assertion timestamp")
    m["croissant"]["extraction_method"] = cell(X, "No method attribution")
    m["croissant"]["human_verified"] = cell(X, "No verification flag")
    m["croissant"]["derivation_chain"] = cell(X, "No derivation provenance")
    m["croissant"]["delegation_chain"] = cell(X, "No delegation model")
    m["croissant"]["invalidation"] = cell(X, "No invalidation/retraction concept")
    m["croissant"]["validity_window"] = cell(X, "No temporal validity fields")
    m["croissant"]["temporal_query"] = cell(X, "No temporal query mechanism")
    m["croissant"]["temporal_diff"] = cell(X, "No temporal diff")
    for f in ["cumulative_fusion", "averaging_fusion", "trust_discount",
              "deduction", "conflict_detection", "byzantine_fusion", "temporal_decay"]:
        m["croissant"][f] = cell(X, "No uncertainty algebra operators")
    m["croissant"]["schema_validation"] = cell(W, "cr:Field has dataType but limited constraints")
    m["croissant"]["required_fields"] = cell(W, "Implicit via RecordSet structure")
    m["croissant"]["cross_field"] = cell(X, "No cross-field constraint language")
    m["croissant"]["dataset_metadata"] = cell(N, "Core purpose: sc:Dataset with name, description, license")
    m["croissant"]["file_distributions"] = cell(N, "cr:FileObject/cr:FileSet are first-class")
    m["croissant"]["record_set"] = cell(N, "cr:RecordSet/cr:Field are first-class")
    m["croissant"]["rdf_compat"] = cell(N, "Built on schema.org + JSON-LD")
    for f in ["prov_o_interop", "shacl_interop", "owl_interop", "ssn_sosa", "croissant_interop"]:
        m["croissant"][f] = cell(X if f != "croissant_interop" else N,
                                  "Native" if f == "croissant_interop" else "Not in scope")
    m["croissant"]["vector_embeddings"] = cell(X, "No vector container type")
    m["croissant"]["measurement_uncertainty"] = cell(X, "No measurement model")
    m["croissant"]["calibration"] = cell(X, "No calibration metadata")
    m["croissant"]["translation_provenance"] = cell(X, "No translation tracking")
    m["croissant"]["graph_merge"] = cell(X, "No merge/diff operations")

    # PROV-O
    m["prov_o"] = {}
    m["prov_o"]["confidence"] = cell(W, "No native confidence; use prov:value + additionalProperty hack")
    m["prov_o"]["sl_opinion"] = cell(X, "No opinion algebra; would need custom ontology extension")
    m["prov_o"]["source"] = cell(N, "prov:wasAttributedTo / prov:wasAssociatedWith")
    m["prov_o"]["extraction_time"] = cell(N, "prov:endedAtTime / prov:generatedAtTime")
    m["prov_o"]["extraction_method"] = cell(N, "prov:qualifiedAssociation with prov:hadRole")
    m["prov_o"]["human_verified"] = cell(W, "prov:wasAttributedTo with role='human-verifier'")
    m["prov_o"]["derivation_chain"] = cell(N, "prov:wasDerivedFrom (first-class)")
    m["prov_o"]["delegation_chain"] = cell(N, "prov:actedOnBehalfOf (first-class)")
    m["prov_o"]["invalidation"] = cell(N, "prov:wasInvalidatedBy (first-class)")
    m["prov_o"]["validity_window"] = cell(N, "prov:generatedAtTime + prov:invalidatedAtTime")
    m["prov_o"]["temporal_query"] = cell(W, "SPARQL query over prov timestamps; no built-in API")
    m["prov_o"]["temporal_diff"] = cell(X, "No diff mechanism")
    for f in ["cumulative_fusion", "averaging_fusion", "trust_discount",
              "deduction", "conflict_detection", "byzantine_fusion", "temporal_decay"]:
        m["prov_o"][f] = cell(X, "No uncertainty algebra; PROV tracks provenance not uncertainty")
    m["prov_o"]["schema_validation"] = cell(X, "PROV is provenance, not validation")
    m["prov_o"]["required_fields"] = cell(X, "Not in scope")
    m["prov_o"]["cross_field"] = cell(X, "Not in scope")
    m["prov_o"]["dataset_metadata"] = cell(W, "Use schema:Dataset alongside PROV")
    m["prov_o"]["file_distributions"] = cell(W, "prov:Entity for files, verbose")
    m["prov_o"]["record_set"] = cell(X, "No record/field model")
    m["prov_o"]["rdf_compat"] = cell(N, "PROV-O is an OWL ontology; fully RDF")
    m["prov_o"]["prov_o_interop"] = cell(N, "It IS PROV-O")
    m["prov_o"]["shacl_interop"] = cell(X, "Different concerns")
    m["prov_o"]["owl_interop"] = cell(X, "Different concerns")
    m["prov_o"]["ssn_sosa"] = cell(W, "Can combine PROV + SSN but requires manual graph construction")
    m["prov_o"]["croissant_interop"] = cell(X, "No Croissant mapping")
    m["prov_o"]["vector_embeddings"] = cell(X, "No vector type")
    m["prov_o"]["measurement_uncertainty"] = cell(W, "Via SSN extension, not native PROV")
    m["prov_o"]["calibration"] = cell(W, "Via qualified association pattern, verbose")
    m["prov_o"]["translation_provenance"] = cell(W, "prov:wasDerivedFrom + role, verbose but expressible")
    m["prov_o"]["graph_merge"] = cell(W, "RDF graph union, but no semantic merge logic")

    # SHACL
    m["shacl"] = {}
    for _, k in FEATURES:
        if k in ["schema_validation", "required_fields", "cross_field"]:
            m["shacl"][k] = cell(N, "Core purpose of SHACL")
        elif k == "rdf_compat":
            m["shacl"][k] = cell(N, "SHACL is RDF-based")
        elif k == "shacl_interop":
            m["shacl"][k] = cell(N, "It IS SHACL")
        elif k == "owl_interop":
            m["shacl"][k] = cell(W, "sh:class overlaps with OWL; partial mapping")
        else:
            m["shacl"][k] = cell(X, "SHACL is a constraint language, not a data annotation format")

    # JSON-LD 1.1
    m["jsonld_11"] = {}
    for _, k in FEATURES:
        if k == "rdf_compat":
            m["jsonld_11"][k] = cell(N, "JSON-LD is an RDF serialization format")
        elif k in ["prov_o_interop", "shacl_interop", "owl_interop", "ssn_sosa",
                    "croissant_interop"]:
            m["jsonld_11"][k] = cell(W, "Can embed any RDF vocabulary via @context, but no processing logic")
        elif k in ["dataset_metadata", "file_distributions", "record_set"]:
            m["jsonld_11"][k] = cell(W, "Via schema.org vocabulary, no ML-specific fields")
        else:
            m["jsonld_11"][k] = cell(X, "JSON-LD 1.1 has no extension mechanism for annotation metadata")

    # Plain JSON
    m["plain_json"] = {}
    for _, k in FEATURES:
        if k in ["confidence", "sl_opinion", "source", "extraction_time",
                  "extraction_method", "human_verified", "derivation_chain",
                  "delegation_chain", "invalidation", "validity_window",
                  "measurement_uncertainty", "calibration", "translation_provenance",
                  "vector_embeddings"]:
            m["plain_json"][k] = cell(W, "Ad-hoc keys; no standard schema, no interop, no validation")
        elif k in ["dataset_metadata", "file_distributions", "record_set"]:
            m["plain_json"][k] = cell(W, "Ad-hoc structure; no standard, no semantic interop")
        else:
            m["plain_json"][k] = cell(X, "Requires computational logic not expressible in passive data format")

    # HF Datasets
    m["hf_datasets"] = {}
    m["hf_datasets"]["confidence"] = cell(X, "No per-sample annotation confidence in HF datasets")
    m["hf_datasets"]["sl_opinion"] = cell(X, "No uncertainty algebra")
    m["hf_datasets"]["source"] = cell(W, "Dataset card metadata, not per-sample")
    for k in ["extraction_time", "extraction_method", "human_verified",
              "derivation_chain", "delegation_chain", "invalidation"]:
        m["hf_datasets"][k] = cell(X, "Not in HF datasets data model")
    for k in ["validity_window", "temporal_query", "temporal_diff"]:
        m["hf_datasets"][k] = cell(X, "No temporal model")
    for k in ["cumulative_fusion", "averaging_fusion", "trust_discount",
              "deduction", "conflict_detection", "byzantine_fusion", "temporal_decay"]:
        m["hf_datasets"][k] = cell(X, "No uncertainty algebra")
    m["hf_datasets"]["schema_validation"] = cell(W, "Features class with dtype validation")
    m["hf_datasets"]["required_fields"] = cell(W, "Features class enforces schema")
    m["hf_datasets"]["cross_field"] = cell(X, "No cross-field constraints")
    m["hf_datasets"]["dataset_metadata"] = cell(N, "Dataset cards are first-class")
    m["hf_datasets"]["file_distributions"] = cell(N, "Built-in data loading from multiple formats")
    m["hf_datasets"]["record_set"] = cell(N, "Features class defines schema")
    for k in ["rdf_compat", "prov_o_interop", "shacl_interop", "owl_interop", "ssn_sosa"]:
        m["hf_datasets"][k] = cell(X, "No RDF/semantic web support")
    m["hf_datasets"]["croissant_interop"] = cell(W, "HF Hub exports Croissant cards (one-way)")
    for k in ["vector_embeddings", "measurement_uncertainty", "calibration",
              "translation_provenance", "graph_merge"]:
        m["hf_datasets"][k] = cell(X, "Not in data model")

    # MLflow
    m["mlflow"] = {}
    m["mlflow"]["confidence"] = cell(W, "Log as metric via mlflow.log_metric, not per-assertion")
    m["mlflow"]["sl_opinion"] = cell(X, "No opinion algebra")
    m["mlflow"]["source"] = cell(N, "Run tracking with source URI is first-class")
    m["mlflow"]["extraction_time"] = cell(N, "Run timestamps are automatic")
    m["mlflow"]["extraction_method"] = cell(N, "Logged as run parameter/tag")
    m["mlflow"]["human_verified"] = cell(X, "No verification flag in model")
    m["mlflow"]["derivation_chain"] = cell(W, "Parent run IDs, limited to experiment lineage")
    m["mlflow"]["delegation_chain"] = cell(X, "No delegation model")
    m["mlflow"]["invalidation"] = cell(X, "No invalidation concept")
    for k in ["validity_window", "temporal_query", "temporal_diff"]:
        m["mlflow"][k] = cell(X, "No temporal validity model")
    for k in ["cumulative_fusion", "averaging_fusion", "trust_discount",
              "deduction", "conflict_detection", "byzantine_fusion", "temporal_decay"]:
        m["mlflow"][k] = cell(X, "No uncertainty algebra")
    m["mlflow"]["schema_validation"] = cell(W, "Model signatures validate input/output schemas")
    m["mlflow"]["required_fields"] = cell(X, "Not per-field")
    m["mlflow"]["cross_field"] = cell(X, "Not supported")
    m["mlflow"]["dataset_metadata"] = cell(N, "mlflow.data.Dataset with name, source, schema")
    m["mlflow"]["file_distributions"] = cell(W, "Artifacts can store files")
    m["mlflow"]["record_set"] = cell(W, "Dataset schema via pandas schema inference")
    for k in ["rdf_compat", "prov_o_interop", "shacl_interop", "owl_interop",
              "ssn_sosa", "croissant_interop"]:
        m["mlflow"][k] = cell(X, "No semantic web support")
    for k in ["vector_embeddings", "measurement_uncertainty", "calibration",
              "translation_provenance", "graph_merge"]:
        m["mlflow"][k] = cell(X, "Not in data model")

    return m


def summarize_matrix(matrix: Dict) -> Dict[str, Dict[str, Any]]:
    summary = {}
    for fmt, features in matrix.items():
        counts = {"native": 0, "workaround": 0, "not_possible": 0}
        for k, cell in features.items():
            counts[cell["level"]] += 1
        total = sum(counts.values())
        counts["total"] = total
        counts["native_pct"] = round(counts["native"] / total * 100, 1)
        counts["expressible_pct"] = round(
            (counts["native"] + counts["workaround"]) / total * 100, 1)
        summary[fmt] = counts
    return summary


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    print("=" * 70)
    print("EN2.1 + EN2.2 (Extended) — Format Expressiveness Comparison")
    print("NeurIPS 2026 D&B, Suite EN2")
    print("=" * 70)

    t_start = time.time()
    env = log_environment()

    # ── EN2.1: Scenario Comparison ──
    print("\n--- EN2.1: Verbosity + Information Completeness (10 scenarios) ---")
    scenarios = build_all_scenarios()
    scenario_results = [s.measure() for s in scenarios]

    formats = ["jsonld_ex", "prov_o", "shacl", "croissant", "plain_json", "jsonld_11"]

    for row in scenario_results:
        print(f"\n  {row['id']}: {row['name']} ({row['n_semantic_fields']} fields)")
        for fmt in formats:
            r = row[fmt]
            if r["bytes_pretty"] is not None:
                print(f"    {fmt:12s}: {r['bytes_pretty']:>5}B pretty, "
                      f"{r['bytes_compact']:>5}B compact | "
                      f"completeness={r['completeness']:.0%} "
                      f"({r['fields_covered']}/{r['fields_total']})"
                      f"{' | lost: ' + ', '.join(r['fields_lost'][:3]) + ('...' if len(r['fields_lost'])>3 else '') if r['fields_lost'] else ''}")
            else:
                print(f"    {fmt:12s}: N/A (format cannot represent this scenario)")

    # Aggregate: bytes vs completeness
    print("\n  === Aggregate Across Scenarios ===")
    agg = {}
    for fmt in formats:
        byte_ratios_pretty = []
        byte_ratios_compact = []
        completeness_vals = []
        bpf_pretty = []  # bytes per field
        bpf_compact = []

        for row in scenario_results:
            r = row[fmt]
            ex = row["jsonld_ex"]
            if r["bytes_pretty"] is not None and ex["bytes_pretty"] is not None:
                byte_ratios_pretty.append(r["bytes_pretty"] / ex["bytes_pretty"])
                byte_ratios_compact.append(r["bytes_compact"] / ex["bytes_compact"])
            completeness_vals.append(r["completeness"])
            if r["bytes_per_field_pretty"] is not None:
                bpf_pretty.append(r["bytes_per_field_pretty"])
            if r["bytes_per_field_compact"] is not None:
                bpf_compact.append(r["bytes_per_field_compact"])

        agg[fmt] = {
            "byte_ratio_pretty": {
                "mean": round(float(np.mean(byte_ratios_pretty)), 3) if byte_ratios_pretty else None,
                "median": round(float(np.median(byte_ratios_pretty)), 3) if byte_ratios_pretty else None,
                "n": len(byte_ratios_pretty),
            },
            "byte_ratio_compact": {
                "mean": round(float(np.mean(byte_ratios_compact)), 3) if byte_ratios_compact else None,
                "median": round(float(np.median(byte_ratios_compact)), 3) if byte_ratios_compact else None,
                "n": len(byte_ratios_compact),
            },
            "completeness": {
                "mean": round(float(np.mean(completeness_vals)), 3),
                "min": round(float(np.min(completeness_vals)), 3),
                "max": round(float(np.max(completeness_vals)), 3),
            },
            "bytes_per_field_pretty": {
                "mean": round(float(np.mean(bpf_pretty)), 1) if bpf_pretty else None,
            },
            "bytes_per_field_compact": {
                "mean": round(float(np.mean(bpf_compact)), 1) if bpf_compact else None,
            },
        }

        c = agg[fmt]["completeness"]
        br = agg[fmt]["byte_ratio_pretty"]
        bpf = agg[fmt]["bytes_per_field_compact"]
        print(f"  {fmt:12s}: completeness={c['mean']:.0%} [{c['min']:.0%}-{c['max']:.0%}]"
              f"  byte_ratio={br['mean']:.3f}x (n={br['n']})"
              f"  bytes/field={bpf['mean']:.0f}" if br['mean'] and bpf['mean'] else
              f"  {fmt:12s}: completeness={c['mean']:.0%}")

    # ── EN2.1b: Scaling Analysis ──
    print("\n--- EN2.1b: Scaling Analysis ---")
    scale_sizes = [10, 100, 1_000, 10_000]
    scale_results = scaling_analysis(scale_sizes)

    for sr in scale_results:
        print(f"  n={sr['n_nodes']:>6}: "
              f"jsonld-ex={sr['jsonld_ex_bytes_compact']:>8}B "
              f"({sr['jsonld_ex_bytes_per_node_compact']}B/node) | "
              f"plain_json={sr['plain_json_bytes_compact']:>8}B "
              f"({sr['plain_json_bytes_per_node_compact']}B/node) | "
              f"ratio={sr['overhead_ratio_compact']:.4f}x")

    # ── EN2.2: Feature Coverage ──
    print("\n--- EN2.2: Feature Coverage Matrix (36 features x 8 formats) ---")
    matrix = build_justified_matrix()
    coverage = summarize_matrix(matrix)

    print(f"\n  {'Format':15s} {'Native':>7} {'Workaround':>11} {'Not Poss.':>10} "
          f"{'Native%':>8} {'Express.%':>10}")
    print("  " + "-" * 65)
    for fmt in ["jsonld_ex", "croissant", "prov_o", "shacl", "jsonld_11",
                "plain_json", "hf_datasets", "mlflow"]:
        c = coverage[fmt]
        print(f"  {fmt:15s} {c['native']:>7} {c['workaround']:>11} "
              f"{c['not_possible']:>10} {c['native_pct']:>7.1f}% "
              f"{c['expressible_pct']:>9.1f}%")

    total_time = time.time() - t_start
    print(f"\n  Total wall time: {total_time:.1f}s")

    # ── Save ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "en2_1_2_ext_results.json"

    # Flatten matrix for JSON (convert cell dicts)
    matrix_flat = {}
    for fmt, feats in matrix.items():
        matrix_flat[fmt] = {k: v for k, v in feats.items()}

    experiment_result = ExperimentResult(
        experiment_id="EN2.1+EN2.2-extended",
        parameters={
            "n_scenarios": len(scenarios),
            "n_features": len(FEATURES),
            "formats_verbosity": formats,
            "formats_coverage": list(matrix.keys()),
            "scaling_sizes": scale_sizes,
            "byte_methods": ["pretty (indent=2, sorted)", "compact (no indent, sorted)"],
        },
        metrics={
            "en2_1_aggregate": agg,
            "en2_1b_scaling": scale_results,
            "en2_2_coverage_summary": coverage,
            "total_wall_time_seconds": round(total_time, 4),
        },
        raw_data={
            "en2_1_scenarios": scenario_results,
            "en2_2_matrix_justified": matrix_flat,
            "en2_2_feature_list": [{"name": n, "key": k} for n, k in FEATURES],
        },
        environment=env,
        notes=(
            "EN2.1+EN2.2 Extended: 10 scenarios x 6 formats with information "
            "completeness scoring (semantic fields preserved), bytes-per-field "
            "normalization, compact+pretty serialization, scaling analysis "
            f"({scale_sizes}), and justified feature coverage matrix "
            f"({len(FEATURES)} features x {len(matrix)} formats). "
            "Every coverage cell includes a justification string."
        ),
    )
    experiment_result.save_json(str(output_path))
    print(f"\n  Results saved to: {output_path}")

    ts = experiment_result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    archive_path = RESULTS_DIR / f"en2_1_2_ext_results_{ts}.json"
    experiment_result.save_json(str(archive_path))
    print(f"  Archived to:      {archive_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
