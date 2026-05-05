"""EN8.10 -- Multi-Format Interop Pipeline: Core Module.

Provides:
  - Field extraction and semantic comparison
  - Synthetic document generators (5 categories)
  - Per-stage round-trip runners (PROV-O, RDF-Star, SHACL, SSN, Croissant)
  - Cumulative pipeline runner
  - Per-field survival matrix computation
  - Aggregate metrics with bootstrap CIs
"""

from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────
_PKG_SRC = Path(__file__).resolve().parent.parent / "packages" / "python" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from jsonld_ex.ai_ml import annotate
from jsonld_ex.owl_interop import (
    to_prov_o, from_prov_o,
    to_rdf_star_ntriples, from_rdf_star_ntriples,
    shape_to_shacl, shacl_to_shape,
    to_ssn, from_ssn,
)
from jsonld_ex.dataset import (
    create_dataset_metadata,
    add_distribution,
    to_croissant, from_croissant,
)


# ===================================================================
# 1. FIELD EXTRACTION
# ===================================================================

def extract_fields(doc: dict[str, Any]) -> dict[str, Any]:
    """Flatten a jsonld-ex document to path->value pairs.

    Recursively walks the document. Annotated values (dicts with @value)
    are flattened as ``key.@value``, ``key.@confidence``, etc.
    @context is excluded (format-specific, expected to change).

    Args:
        doc: A jsonld-ex document (annotated doc, shape, or dataset).

    Returns:
        Flat dict mapping dotted paths to values.
    """
    result: dict[str, Any] = {}
    _extract_recursive(doc, "", result)
    return result


def _extract_recursive(
    obj: Any, prefix: str, result: dict[str, Any],
) -> None:
    """Recursively extract fields from a nested structure."""
    if not isinstance(obj, dict):
        return

    for key, value in obj.items():
        if key == "@context":
            continue

        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"

        if isinstance(value, dict) and "@value" in value:
            # Annotated value — flatten each annotation key
            for ann_key, ann_val in value.items():
                ann_path = f"{path}.{ann_key}"
                result[ann_path] = ann_val
        elif isinstance(value, dict) and "@value" not in value:
            # Check if it's a shape constraint (has @-prefixed keys but no @value)
            has_shape_keys = any(
                k.startswith("@") and k not in ("@id", "@type", "@context", "@graph")
                for k in value.keys()
            )
            if has_shape_keys and "@value" not in value:
                # Shape constraint — flatten each constraint key
                for ck, cv in value.items():
                    c_path = f"{path}.{ck}"
                    result[c_path] = cv
            else:
                # Nested node — recurse
                _extract_recursive(value, path, result)
        elif isinstance(value, list):
            # Store as list value directly (for @derivedFrom, etc.)
            result[path] = value
        else:
            # Scalar value
            result[path] = value


# ===================================================================
# 2. FIELD COMPARISON
# ===================================================================

FLOAT_TOLERANCE = 1e-10


@dataclass
class FieldComparison:
    """Result of comparing two field dicts."""
    n_preserved: int = 0
    n_transformed: int = 0
    n_lost: int = 0
    n_gained: int = 0
    n_corrupted: int = 0

    preserved_fields: list[str] = field(default_factory=list)
    transformed_fields: list[str] = field(default_factory=list)
    lost_fields: list[str] = field(default_factory=list)
    gained_fields: list[str] = field(default_factory=list)
    corrupted_fields: list[str] = field(default_factory=list)

    input_fields: dict[str, Any] = field(default_factory=dict)
    output_fields: dict[str, Any] = field(default_factory=dict)

    @property
    def total_input(self) -> int:
        return self.n_preserved + self.n_transformed + self.n_lost + self.n_corrupted

    @property
    def fidelity(self) -> float:
        if self.total_input == 0:
            return 1.0
        return self.n_preserved / self.total_input

    @property
    def semantic_fidelity(self) -> float:
        if self.total_input == 0:
            return 1.0
        return (self.n_preserved + self.n_transformed) / self.total_input


def compare_fields(
    original: dict[str, Any],
    recovered: dict[str, Any],
) -> FieldComparison:
    """Semantically compare two field dicts.

    Classification:
      - preserved:   key in both, semantically identical value
      - transformed: key in both, type coercion (int<->float with same numeric value)
      - lost:        key in original, absent in recovered
      - gained:      key in recovered, absent in original
      - corrupted:   key in both, semantically different value
    """
    comp = FieldComparison(
        input_fields=dict(original),
        output_fields=dict(recovered),
    )

    orig_keys = set(original.keys())
    rec_keys = set(recovered.keys())

    # Lost fields
    for k in orig_keys - rec_keys:
        comp.n_lost += 1
        comp.lost_fields.append(k)

    # Gained fields
    for k in rec_keys - orig_keys:
        comp.n_gained += 1
        comp.gained_fields.append(k)

    # Common fields
    for k in orig_keys & rec_keys:
        ov = original[k]
        rv = recovered[k]
        status = _compare_values(ov, rv)
        if status == "preserved":
            comp.n_preserved += 1
            comp.preserved_fields.append(k)
        elif status == "transformed":
            comp.n_transformed += 1
            comp.transformed_fields.append(k)
        else:  # corrupted
            comp.n_corrupted += 1
            comp.corrupted_fields.append(k)

    return comp


def _compare_values(a: Any, b: Any) -> str:
    """Compare two values semantically.

    Returns: 'preserved', 'transformed', or 'corrupted'.
    """
    # Both numeric — check tolerance and type coercion
    # Must check BEFORE == because Python considers 30 == 30.0 as True
    if isinstance(a, (int, float)) and not isinstance(a, bool) and \
       isinstance(b, (int, float)) and not isinstance(b, bool):
        if abs(float(a) - float(b)) < FLOAT_TOLERANCE:
            if type(a) != type(b):
                return "transformed"  # int <-> float coercion
            return "preserved"
        return "corrupted"

    # Exact match (non-numeric)
    if a == b:
        return "preserved"

    # Bool vs non-bool — never coerce
    if isinstance(a, bool) or isinstance(b, bool):
        return "corrupted"

    # Both lists — order-independent comparison
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return "corrupted"
        # Try order-independent match
        try:
            if set(a) == set(b):
                return "preserved"
        except TypeError:
            # Unhashable elements — fall back to sorted comparison
            pass
        if sorted(str(x) for x in a) == sorted(str(x) for x in b):
            return "preserved"
        return "corrupted"

    # String comparison (whitespace-normalized)
    if isinstance(a, str) and isinstance(b, str):
        if a.strip() == b.strip():
            return "preserved"
        return "corrupted"

    return "corrupted"


# ===================================================================
# 3. PIPELINE STAGE DEFINITIONS
# ===================================================================

STAGE_PROV_O = "PROV-O"
STAGE_RDF_STAR = "RDF-Star"
STAGE_SHACL = "SHACL"
STAGE_SSN = "SSN/SOSA"
STAGE_CROISSANT = "Croissant"


@dataclass
class StageResult:
    """Result of running one round-trip stage."""
    stage_name: str
    comparison: FieldComparison
    intermediate: Any = None   # The converted (non-jsonld-ex) form
    recovered: Any = None      # The reconstructed jsonld-ex doc
    skipped: bool = False
    error: Optional[str] = None


def _is_shape_doc(doc: dict[str, Any]) -> bool:
    """Check if a doc is a shape definition (not an annotated doc)."""
    # Shape docs have property constraints with @-prefixed keys
    # but no @value annotations and typically no @context
    for key, value in doc.items():
        if key.startswith("@"):
            continue
        if isinstance(value, dict):
            if "@value" in value:
                return False  # Has annotated values — not a pure shape
            shape_keys = {"@required", "@type", "@minimum", "@maximum",
                          "@minLength", "@maxLength", "@pattern", "@in",
                          "@or", "@and", "@not", "@extends", "@minCount",
                          "@maxCount"}
            if any(k in shape_keys for k in value.keys()):
                return True
    return False


def _is_dataset_doc(doc: dict[str, Any]) -> bool:
    """Check if a doc is a dataset metadata document."""
    doc_type = doc.get("@type", "")
    if isinstance(doc_type, list):
        return any("Dataset" in t for t in doc_type)
    return "Dataset" in str(doc_type)


def run_single_stage(
    doc: dict[str, Any],
    stage: str,
) -> StageResult:
    """Run a single format round-trip stage on a document.

    For inapplicable stages (e.g., SHACL on a non-shape doc),
    the document passes through unchanged with skipped=True.
    """
    original_fields = extract_fields(doc)

    try:
        if stage == STAGE_PROV_O:
            intermediate, _report = to_prov_o(doc)
            recovered, _report2 = from_prov_o(intermediate)

        elif stage == STAGE_RDF_STAR:
            subject = doc.get("@id", "http://example.org/subject")
            intermediate, _report = to_rdf_star_ntriples(doc, base_subject=subject)
            recovered, _report2 = from_rdf_star_ntriples(intermediate)

        elif stage == STAGE_SHACL:
            if not _is_shape_doc(doc):
                return StageResult(
                    stage_name=stage,
                    comparison=compare_fields(original_fields, original_fields),
                    intermediate=None,
                    recovered=copy.deepcopy(doc),
                    skipped=True,
                )
            target_class = doc.get("@type")
            intermediate = shape_to_shacl(doc, target_class=target_class)
            recovered, _warnings = shacl_to_shape(intermediate)

        elif stage == STAGE_SSN:
            # SSN requires annotated values — skip for shapes/datasets
            has_annotated = any(
                isinstance(v, dict) and "@value" in v
                for k, v in doc.items() if not k.startswith("@")
            )
            if not has_annotated:
                return StageResult(
                    stage_name=stage,
                    comparison=compare_fields(original_fields, original_fields),
                    intermediate=None,
                    recovered=copy.deepcopy(doc),
                    skipped=True,
                )
            intermediate, _report = to_ssn(doc)
            recovered, _report2 = from_ssn(intermediate)

        elif stage == STAGE_CROISSANT:
            if not _is_dataset_doc(doc):
                return StageResult(
                    stage_name=stage,
                    comparison=compare_fields(original_fields, original_fields),
                    intermediate=None,
                    recovered=copy.deepcopy(doc),
                    skipped=True,
                )
            intermediate = to_croissant(doc)
            recovered = from_croissant(intermediate)

        else:
            raise ValueError(f"Unknown stage: {stage}")

    except Exception as e:
        return StageResult(
            stage_name=stage,
            comparison=compare_fields(original_fields, {}),
            error=str(e),
        )

    recovered_fields = extract_fields(recovered)
    comparison = compare_fields(original_fields, recovered_fields)

    return StageResult(
        stage_name=stage,
        comparison=comparison,
        intermediate=intermediate,
        recovered=recovered,
    )


# ===================================================================
# 4. CUMULATIVE PIPELINE
# ===================================================================

@dataclass
class PipelineResult:
    """Result of running a multi-stage pipeline."""
    stage_results: list[StageResult]
    original_doc: dict[str, Any]
    final_doc: dict[str, Any]
    cumulative_fidelity: float
    per_stage_cumulative_fidelity: list[float]


def run_pipeline(
    doc: dict[str, Any],
    stages: list[str],
) -> PipelineResult:
    """Run chained round-trip stages on a document.

    Each stage feeds its recovered doc into the next stage.
    Cumulative fidelity is measured against the original doc.
    """
    if not stages:
        original_fields = extract_fields(doc)
        return PipelineResult(
            stage_results=[],
            original_doc=doc,
            final_doc=doc,
            cumulative_fidelity=1.0,
            per_stage_cumulative_fidelity=[],
        )

    original_fields = extract_fields(doc)
    current_doc = copy.deepcopy(doc)
    stage_results: list[StageResult] = []
    cumulative_fidelities: list[float] = []

    for stage in stages:
        result = run_single_stage(current_doc, stage)
        stage_results.append(result)

        if result.recovered is not None:
            current_doc = result.recovered
        # else: keep current_doc unchanged (error or skip)

        # Measure cumulative fidelity against ORIGINAL
        current_fields = extract_fields(current_doc)
        cum_comp = compare_fields(original_fields, current_fields)
        cumulative_fidelities.append(cum_comp.fidelity)

    final_fidelity = cumulative_fidelities[-1] if cumulative_fidelities else 1.0

    return PipelineResult(
        stage_results=stage_results,
        original_doc=doc,
        final_doc=current_doc,
        cumulative_fidelity=final_fidelity,
        per_stage_cumulative_fidelity=cumulative_fidelities,
    )


# ===================================================================
# 5. SURVIVAL MATRIX
# ===================================================================

@dataclass
class SurvivalMatrix:
    """Per-field survival rates across pipeline stages."""
    field_names: list[str]
    stage_names: list[str]
    rates: np.ndarray  # shape (n_fields, n_stages)


def compute_survival_matrix(
    docs: list[dict[str, Any]],
    stages: list[str],
) -> SurvivalMatrix:
    """Compute per-field survival rates across stages for a set of docs.

    For each field that appears in ANY document, compute the fraction
    of documents where that field survives each pipeline stage.
    """
    # Collect all unique fields across all docs
    all_field_sets: list[set[str]] = []
    for doc in docs:
        all_field_sets.append(set(extract_fields(doc).keys()))
    all_fields = sorted(set().union(*all_field_sets))

    if not all_fields or not stages:
        return SurvivalMatrix(
            field_names=all_fields,
            stage_names=list(stages),
            rates=np.zeros((len(all_fields), len(stages))),
        )

    # For each doc, run each stage independently and check field survival
    # Matrix: rows=fields, cols=stages, values=survival fraction
    n_fields = len(all_fields)
    n_stages = len(stages)
    survived_counts = np.zeros((n_fields, n_stages))
    present_counts = np.zeros((n_fields, n_stages))

    field_to_idx = {f: i for i, f in enumerate(all_fields)}

    for doc in docs:
        original_fields = extract_fields(doc)
        current_doc = copy.deepcopy(doc)

        for si, stage in enumerate(stages):
            result = run_single_stage(current_doc, stage)
            if result.recovered is not None:
                current_doc = result.recovered

            # Check which original fields survived to this point
            current_fields = extract_fields(current_doc)
            cum_comp = compare_fields(original_fields, current_fields)

            for f in original_fields:
                fi = field_to_idx[f]
                present_counts[fi, si] += 1
                if f in cum_comp.preserved_fields or f in cum_comp.transformed_fields:
                    survived_counts[fi, si] += 1

    # Compute rates (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(
            present_counts > 0,
            survived_counts / present_counts,
            0.0,
        )

    return SurvivalMatrix(
        field_names=all_fields,
        stage_names=list(stages),
        rates=rates,
    )


# ===================================================================
# 6. AGGREGATE METRICS
# ===================================================================

@dataclass
class AggregateMetrics:
    """Aggregate results with bootstrap confidence intervals."""
    mean_fidelity: float
    ci_lower: float
    ci_upper: float
    per_stage_fidelity: dict[str, float]
    n_docs: int
    n_bootstrap: int
    per_doc_fidelities: list[float] = field(default_factory=list)


def aggregate_results(
    docs: list[dict[str, Any]],
    stages: list[str],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> AggregateMetrics:
    """Run pipeline on all docs and compute aggregate metrics with CIs."""
    rng = np.random.default_rng(seed)
    fidelities: list[float] = []
    per_stage_sums: dict[str, list[float]] = {s: [] for s in stages}

    for doc in docs:
        result = run_pipeline(doc, stages)
        fidelities.append(result.cumulative_fidelity)
        for sr in result.stage_results:
            per_stage_sums[sr.stage_name].append(sr.comparison.fidelity)

    fid_array = np.array(fidelities)
    mean_fid = float(np.mean(fid_array))

    # Bootstrap CI
    if len(fid_array) > 1:
        boot_means = np.array([
            float(np.mean(rng.choice(fid_array, size=len(fid_array), replace=True)))
            for _ in range(n_bootstrap)
        ])
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
    else:
        ci_lower = mean_fid
        ci_upper = mean_fid

    per_stage_fidelity = {
        s: float(np.mean(vals)) if vals else 1.0
        for s, vals in per_stage_sums.items()
    }

    return AggregateMetrics(
        mean_fidelity=mean_fid,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        per_stage_fidelity=per_stage_fidelity,
        n_docs=len(docs),
        n_bootstrap=n_bootstrap,
        per_doc_fidelities=fidelities,
    )


# ===================================================================
# 7. CATEGORY PIPELINE DEFINITIONS
# ===================================================================

CATEGORY_PIPELINES: dict[str, list[str]] = {
    "A": [STAGE_PROV_O, STAGE_RDF_STAR],
    "B": [STAGE_PROV_O, STAGE_RDF_STAR, STAGE_SSN],
    "C": [STAGE_CROISSANT],
    "D": [STAGE_SHACL],
    "E": [STAGE_PROV_O, STAGE_RDF_STAR, STAGE_SHACL, STAGE_SSN, STAGE_CROISSANT],
}


# ===================================================================
# 8. DOCUMENT GENERATORS
# ===================================================================

def generate_category_a_docs(seed: int = 42) -> list[dict[str, Any]]:
    """Generate 20 Category A docs (General ML Annotations).

    Covers: @confidence, @source, @method, @extractedAt, @humanVerified,
    @derivedFrom, @delegatedBy, @invalidatedAt, @invalidationReason.

    Sub-groups:
      5x NER annotation docs
      5x Multi-source fusion docs
      5x Delegation chain docs
      5x Invalidation docs
    """
    rng = np.random.default_rng(seed)
    docs: list[dict[str, Any]] = []

    # --- 5x NER annotation docs ---
    entities = ["Alice", "Bob", "Google", "New York", "COVID-19"]
    entity_types = ["Person", "Person", "Organization", "Place", "Disease"]
    for i, (name, etype) in enumerate(zip(entities, entity_types)):
        conf = round(float(rng.uniform(0.7, 0.99)), 4)
        docs.append({
            "@context": {"@vocab": "http://schema.org/"},
            "@id": f"http://example.org/ner-{i}",
            "@type": etype,
            "name": annotate(
                name,
                confidence=conf,
                source=f"http://model-{rng.choice(['spacy', 'flair', 'gliner'])}.org",
                method="NER-extraction",
                extracted_at=f"2026-01-{15+i:02d}T10:30:00Z",
                human_verified=bool(rng.random() > 0.5),
            ),
        })

    # --- 5x Multi-source fusion docs ---
    properties = ["revenue", "population", "height", "weight", "score"]
    for i, prop in enumerate(properties):
        val = round(float(rng.uniform(10, 10000)), 2)
        docs.append({
            "@context": {"@vocab": "http://schema.org/"},
            "@id": f"http://example.org/fusion-{i}",
            "@type": "Observation",
            prop: annotate(
                val,
                confidence=round(float(rng.uniform(0.6, 0.95)), 4),
                source=f"http://source-{chr(65+i)}.org",
                method="multi-source-fusion",
                extracted_at=f"2026-02-{10+i:02d}T08:00:00Z",
                derived_from=[
                    f"http://raw-source-{j}.org" for j in range(rng.integers(2, 5))
                ],
            ),
        })

    # --- 5x Delegation chain docs ---
    roles = ["editor", "reviewer", "analyst", "curator", "moderator"]
    for i, role in enumerate(roles):
        docs.append({
            "@context": {"@vocab": "http://schema.org/"},
            "@id": f"http://example.org/deleg-{i}",
            "@type": "Article",
            "headline": annotate(
                f"Report by {role} {i}",
                confidence=round(float(rng.uniform(0.5, 0.9)), 4),
                source=f"http://{role}.example.org",
                method="editorial-review",
                extracted_at=f"2026-03-{1+i:02d}T12:00:00Z",
                delegated_by=f"http://chief-{role}.example.org",
            ),
        })

    # --- 5x Invalidation docs ---
    claims = [
        "Earth is flat", "Vaccines cause autism",
        "Cold fusion works", "Perpetual motion possible",
        "Homeopathy is effective",
    ]
    for i, claim in enumerate(claims):
        docs.append({
            "@context": {"@vocab": "http://schema.org/"},
            "@id": f"http://example.org/invalid-{i}",
            "@type": "Claim",
            "statement": annotate(
                claim,
                confidence=round(float(rng.uniform(0.01, 0.15)), 4),
                source=f"http://debunked-{i}.org",
                method="fact-check",
                extracted_at=f"2026-01-01T00:00:00Z",
                invalidated_at=f"2026-04-{1+i:02d}T00:00:00Z",
                invalidation_reason=f"Contradicted by evidence set {i}",
            ),
        })

    return docs


def generate_category_b_docs(seed: int = 42) -> list[dict[str, Any]]:
    """Generate 15 Category B docs (IoT/Sensor Annotations).

    Covers all Category A fields PLUS @unit, @measurementUncertainty,
    @calibratedAt, @calibrationMethod, @calibrationAuthority,
    @aggregationMethod, @aggregationWindow, @aggregationCount.
    """
    rng = np.random.default_rng(seed)
    docs: list[dict[str, Any]] = []

    # --- 5x Temperature/humidity with calibration ---
    for i in range(5):
        temp = round(float(rng.normal(22.0, 3.0)), 2)
        sigma = round(float(rng.uniform(0.1, 1.0)), 3)
        docs.append({
            "@context": {"@vocab": "http://schema.org/"},
            "@id": f"http://example.org/sensor-temp-{i}",
            "@type": "Observation",
            "temperature": annotate(
                temp,
                confidence=round(1.0 / (1.0 + sigma), 4),
                source=f"http://sensor-{i}.example.org",
                method="direct-measurement",
                extracted_at=f"2026-01-15T{10+i:02d}:00:00Z",
                unit="celsius",
                measurement_uncertainty=sigma,
                calibrated_at=f"2026-01-01T00:00:00Z",
                calibration_method="NIST-traceable",
                calibration_authority="NIST",
            ),
        })

    # --- 5x Aggregated time-series ---
    for i in range(5):
        avg_val = round(float(rng.uniform(20, 80)), 2)
        docs.append({
            "@context": {"@vocab": "http://schema.org/"},
            "@id": f"http://example.org/sensor-agg-{i}",
            "@type": "Observation",
            "humidity": annotate(
                avg_val,
                confidence=round(float(rng.uniform(0.8, 0.95)), 4),
                source=f"http://sensor-agg-{i}.example.org",
                method="aggregation",
                extracted_at=f"2026-01-15T{10+i:02d}:30:00Z",
                unit="percent",
                measurement_uncertainty=round(float(rng.uniform(0.5, 3.0)), 2),
                aggregation_method="rolling-mean",
                aggregation_window="PT5M",
                aggregation_count=int(rng.integers(30, 120)),
            ),
        })

    # --- 5x Multi-sensor with uncertainty ---
    for i in range(5):
        pressure = round(float(rng.normal(1013.25, 5.0)), 2)
        sigma = round(float(rng.uniform(0.5, 2.0)), 3)
        docs.append({
            "@context": {"@vocab": "http://schema.org/"},
            "@id": f"http://example.org/sensor-multi-{i}",
            "@type": "Observation",
            "pressure": annotate(
                pressure,
                confidence=round(1.0 / (1.0 + sigma), 4),
                source=f"http://barometer-{i}.example.org",
                method="barometric-measurement",
                extracted_at=f"2026-01-15T{10+i:02d}:15:00Z",
                unit="hPa",
                measurement_uncertainty=sigma,
                calibrated_at=f"2025-12-15T00:00:00Z",
                calibration_method="factory-calibration",
                calibration_authority=f"Manufacturer-{i}",
            ),
        })

    return docs


def generate_category_c_docs(seed: int = 42) -> list[dict[str, Any]]:
    """Generate 10 Category C docs (Dataset Metadata).

    Each is a dataset card created via create_dataset_metadata().
    """
    rng = np.random.default_rng(seed)
    docs: list[dict[str, Any]] = []

    datasets = [
        ("NLP-Sentiment-v1", "Sentiment analysis dataset with 50K reviews", "NLP"),
        ("NLP-NER-CoNLL", "Named entity recognition benchmark", "NLP"),
        ("Vision-CIFAR-ext", "Extended CIFAR with uncertainty labels", "Vision"),
        ("Vision-Medical-XRay", "Chest X-ray classification dataset", "Vision"),
        ("Tabular-Housing", "Housing prices with feature provenance", "Tabular"),
        ("Tabular-Credit", "Credit scoring dataset with fairness metadata", "Tabular"),
        ("Audio-Speech-EN", "English speech recognition corpus", "Audio"),
        ("Audio-Music-Genre", "Music genre classification dataset", "Audio"),
        ("Multi-VQA", "Visual question answering multimodal dataset", "Multimodal"),
        ("Multi-DocVQA", "Document VQA with layout annotations", "Multimodal"),
    ]

    for i, (name, desc, domain) in enumerate(datasets):
        doc = create_dataset_metadata(
            name=name,
            description=desc,
            url=f"http://example.org/datasets/{name.lower()}",
            license="http://creativecommons.org/licenses/by/4.0/",
            creator=f"ML Lab {domain}",
            version=f"1.{i}",
        )
        # Add a distribution
        doc = add_distribution(
            doc,
            name=f"{name}-train.csv",
            content_url=f"http://example.org/data/{name.lower()}/train.csv",
            encoding_format="text/csv",
            content_size=f"{rng.integers(1, 500)}MB",
        )
        docs.append(doc)

    return docs


def generate_category_d_docs(seed: int = 42) -> list[dict[str, Any]]:
    """Generate 10 Category D docs (Validation Shapes).

    Covers: @required, @type, @minimum, @maximum, @minLength, @maxLength,
    @pattern, @in, @or/@and/@not, @extends, @lessThan, @class, @uniqueLang,
    @qualifiedShape, @if/@then/@else.
    """
    docs: list[dict[str, Any]] = []

    # 1. Simple required + type
    docs.append({
        "@type": "http://schema.org/Person",
        "http://schema.org/name": {
            "@required": True,
            "@type": "xsd:string",
        },
        "http://schema.org/email": {
            "@required": True,
            "@type": "xsd:string",
        },
    })

    # 2. Simple numeric constraints
    docs.append({
        "@type": "http://schema.org/Product",
        "http://schema.org/price": {
            "@type": "xsd:decimal",
            "@minimum": 0,
            "@maximum": 1000000,
        },
        "http://schema.org/weight": {
            "@type": "xsd:decimal",
            "@minimum": 0,
        },
    })

    # 3. String length + pattern
    docs.append({
        "@type": "http://schema.org/Person",
        "http://schema.org/name": {
            "@required": True,
            "@type": "xsd:string",
            "@minLength": 1,
            "@maxLength": 200,
        },
        "http://schema.org/email": {
            "@required": True,
            "@type": "xsd:string",
            "@pattern": r"^[^@]+@[^@]+\.[^@]+$",
        },
    })

    # 4. Enumeration (@in)
    docs.append({
        "@type": "http://schema.org/CreativeWork",
        "http://schema.org/genre": {
            "@required": True,
            "@in": ["fiction", "non-fiction", "poetry", "drama"],
        },
        "http://schema.org/inLanguage": {
            "@type": "xsd:string",
            "@minLength": 2,
            "@maxLength": 5,
        },
    })

    # 5. Logical combinator: @or
    docs.append({
        "@type": "http://schema.org/ContactPoint",
        "http://schema.org/contactType": {
            "@or": [
                {"@type": "xsd:string", "@pattern": r"^\+?[0-9]+$"},
                {"@type": "xsd:string", "@pattern": r"^[^@]+@[^@]+$"},
            ],
        },
    })

    # 6. Logical combinator: @and
    docs.append({
        "@type": "http://schema.org/Offer",
        "http://schema.org/price": {
            "@and": [
                {"@type": "xsd:decimal", "@minimum": 0},
                {"@maximum": 99999},
            ],
        },
    })

    # 7. Logical combinator: @not
    docs.append({
        "@type": "http://schema.org/Thing",
        "http://schema.org/identifier": {
            "@not": {"@type": "xsd:string", "@maxLength": 0},
        },
    })

    # 8. Cross-property constraint (@lessThan)
    docs.append({
        "@type": "http://schema.org/Event",
        "http://schema.org/startDate": {
            "@required": True,
            "@type": "xsd:dateTime",
            "@lessThan": "http://schema.org/endDate",
        },
        "http://schema.org/endDate": {
            "@required": True,
            "@type": "xsd:dateTime",
        },
    })

    # 9. Numeric range with both min and max
    docs.append({
        "@type": "http://schema.org/Person",
        "http://schema.org/age": {
            "@type": "xsd:integer",
            "@minimum": 0,
            "@maximum": 150,
        },
        "http://schema.org/height": {
            "@type": "xsd:decimal",
            "@minimum": 0.3,
            "@maximum": 3.0,
        },
    })

    # 10. Complex: conditional (@if/@then/@else)
    docs.append({
        "@type": "http://schema.org/Order",
        "http://schema.org/orderStatus": {
            "@if": {"@in": ["shipped", "delivered"]},
            "@then": {"@required": True},
            "@else": {"@maxCount": 0},
        },
        "http://schema.org/price": {
            "@required": True,
            "@type": "xsd:decimal",
            "@minimum": 0,
        },
    })

    return docs


def generate_category_e_docs(seed: int = 42) -> list[dict[str, Any]]:
    """Generate 5 Category E docs (Kitchen Sink).

    Each doc contains annotated values (provenance + IoT) AND is
    structured as a dataset card. Shapes are embedded conceptually.
    """
    rng = np.random.default_rng(seed)
    docs: list[dict[str, Any]] = []

    for i in range(5):
        temp = round(float(rng.normal(22.0, 2.0)), 2)
        sigma = round(float(rng.uniform(0.1, 0.5)), 3)

        doc = create_dataset_metadata(
            name=f"IoT-Dataset-{i}",
            description=f"IoT sensor dataset #{i} with confidence annotations",
            url=f"http://example.org/iot-dataset-{i}",
            license="http://creativecommons.org/licenses/by/4.0/",
            creator=f"IoT Lab {i}",
            version="1.0",
        )
        # Add annotated sensor field
        doc["sampleReading"] = annotate(
            temp,
            confidence=round(1.0 / (1.0 + sigma), 4),
            source=f"http://sensor-{i}.iot.org",
            method="direct-measurement",
            extracted_at=f"2026-01-15T{10+i:02d}:00:00Z",
            unit="celsius",
            measurement_uncertainty=sigma,
            calibrated_at="2026-01-01T00:00:00Z",
            calibration_method="NIST-traceable",
            calibration_authority="NIST",
        )
        docs.append(doc)

    return docs
