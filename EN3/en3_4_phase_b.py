"""EN3.4 Phase B — FHIR Clinical Pipeline.

NeurIPS 2026 D&B, Suite EN3 (ML Pipeline Integration), Experiment 4.

Demonstrates that jsonld-ex enables clinical data exchange workflows
impossible with plain FHIR R4, using the pip-installed jsonld_ex package
for all FHIR and SL operations.

Covers:
  - FHIR bundle loading and resource extraction
  - Narrative text extraction for NER processing
  - Round-trip fidelity (FHIR → jsonld-ex → FHIR)
  - Annotation overhead measurement
  - Six query types impossible with plain FHIR (H3.4i)
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    conflict_metric,
)
from jsonld_ex.confidence_decay import decay_opinion
from jsonld_ex.fhir_interop import from_fhir, to_fhir


# =====================================================================
# JSON Serialization Helper
# =====================================================================


def _json_default(obj: Any) -> Any:
    """JSON fallback serializer for Opinion objects in jsonld-ex docs."""
    if isinstance(obj, Opinion):
        return {
            "belief": obj.belief,
            "disbelief": obj.disbelief,
            "uncertainty": obj.uncertainty,
            "base_rate": obj.base_rate,
            "projected_probability": obj.projected_probability(),
        }
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# =====================================================================
# 1. Bundle Loading
# =====================================================================


def load_bundle_resources(
    bundle: Dict[str, Any],
    target_types: Optional[Set[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract resources from a FHIR Bundle, grouped by type.

    Args:
        bundle:       FHIR R4 Bundle dict.
        target_types: If provided, only extract these resource types.

    Returns:
        Dict mapping resourceType → list of resource dicts.
    """
    entries = bundle.get("entry", [])
    resources: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for entry in entries:
        resource = entry.get("resource")
        if resource is None:
            continue
        rtype = resource.get("resourceType")
        if rtype is None:
            continue
        if target_types is not None and rtype not in target_types:
            continue
        resources[rtype].append(resource)

    return dict(resources)


# =====================================================================
# 2. Narrative Text Extraction
# =====================================================================

# Map of resourceType → list of JSON paths to text fields.
# Each path is a tuple of keys to traverse.
_TEXT_PATHS: Dict[str, List[tuple[str, ...]]] = {
    "Observation": [("code", "text")],
    "Condition": [("code", "text")],
    "MedicationRequest": [("medicationCodeableConcept", "text")],
    "DiagnosticReport": [("conclusion",)],
    "Procedure": [("code", "text")],
    "Immunization": [("vaccineCode", "text")],
    "AllergyIntolerance": [("code", "text")],
}


def extract_narrative_texts(
    resources: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract text fields from FHIR resources for NER processing.

    Args:
        resources: List of FHIR resource dicts.

    Returns:
        List of dicts with keys: text, resource_id, resource_type, field_path.
    """
    results: List[Dict[str, Any]] = []

    for resource in resources:
        rtype = resource.get("resourceType", "")
        rid = resource.get("id", "")
        paths = _TEXT_PATHS.get(rtype, [])

        for path in paths:
            # Traverse the path
            node = resource
            valid = True
            for key in path:
                if isinstance(node, dict) and key in node:
                    node = node[key]
                else:
                    valid = False
                    break

            if valid and isinstance(node, str) and node.strip():
                results.append({
                    "text": node,
                    "resource_id": rid,
                    "resource_type": rtype,
                    "field_path": ".".join(path),
                })

    return results


# =====================================================================
# 3. Round-Trip Fidelity
# =====================================================================


@dataclass
class RoundTripResult:
    """Result of a single FHIR round-trip test.

    Attributes:
        resource_type:    FHIR resource type.
        resource_id:      FHIR resource ID.
        fields_preserved: Number of key fields preserved.
        fields_total:     Total key fields checked.
        fidelity_rate:    fields_preserved / fields_total.
        lost_fields:      List of field names that were lost.
    """

    resource_type: str
    resource_id: str
    fields_preserved: int
    fields_total: int
    fidelity_rate: float
    lost_fields: List[str]


# Key fields to check per resource type (structural fields that
# must survive round-trip).
_ROUND_TRIP_FIELDS: Dict[str, List[str]] = {
    "Observation": ["id", "status", "resourceType"],
    "Condition": ["id", "resourceType"],
    "Patient": ["id", "resourceType", "active", "gender", "birthDate"],
    "MedicationRequest": ["id", "status", "intent", "resourceType"],
    "DiagnosticReport": ["id", "status", "resourceType"],
    "Procedure": ["id", "status", "resourceType"],
    "Immunization": ["id", "status", "resourceType"],
    "Encounter": ["id", "status", "resourceType"],
    "AllergyIntolerance": ["id", "resourceType"],
    "CarePlan": ["id", "status", "resourceType"],
}


def measure_round_trip_fidelity(
    resource: Dict[str, Any],
) -> RoundTripResult:
    """Measure field-level fidelity of FHIR → jsonld-ex → FHIR round-trip.

    Converts to jsonld-ex via from_fhir(), then back via to_fhir(),
    and compares key structural fields.

    Args:
        resource: Original FHIR R4 resource dict.

    Returns:
        RoundTripResult with preservation stats.
    """
    rtype = resource.get("resourceType", "Unknown")
    rid = resource.get("id", "")

    # Forward: FHIR → jsonld-ex
    jsonld_doc, _report = from_fhir(resource)

    # Backward: jsonld-ex → FHIR
    roundtripped, _rt_report = to_fhir(jsonld_doc)

    # Check key fields
    fields_to_check = _ROUND_TRIP_FIELDS.get(rtype, ["id", "resourceType"])
    preserved = 0
    lost: List[str] = []

    for field_name in fields_to_check:
        original_val = resource.get(field_name)
        roundtripped_val = roundtripped.get(field_name)
        if original_val == roundtripped_val:
            preserved += 1
        else:
            lost.append(field_name)

    total = len(fields_to_check)
    rate = preserved / total if total > 0 else 0.0

    return RoundTripResult(
        resource_type=rtype,
        resource_id=rid,
        fields_preserved=preserved,
        fields_total=total,
        fidelity_rate=rate,
        lost_fields=lost,
    )


# =====================================================================
# 4. Annotation Overhead
# =====================================================================


@dataclass
class OverheadResult:
    """Result of annotation overhead measurement.

    Attributes:
        resource_type:  FHIR resource type.
        resource_id:    FHIR resource ID.
        original_bytes: Size of original FHIR JSON in bytes.
        annotated_bytes: Size of jsonld-ex annotated doc in bytes.
        overhead_bytes: annotated - original.
        overhead_pct:   overhead / original × 100.
    """

    resource_type: str
    resource_id: str
    original_bytes: int
    annotated_bytes: int
    overhead_bytes: int
    overhead_pct: float


def measure_annotation_overhead(
    resource: Dict[str, Any],
) -> OverheadResult:
    """Measure byte overhead of jsonld-ex annotation on a FHIR resource.

    Args:
        resource: Original FHIR R4 resource dict.

    Returns:
        OverheadResult with size comparison.
    """
    rtype = resource.get("resourceType", "Unknown")
    rid = resource.get("id", "")

    original_json = json.dumps(resource).encode("utf-8")
    original_bytes = len(original_json)

    # Forward: FHIR → jsonld-ex (adds SL opinions)
    jsonld_doc, _report = from_fhir(resource)

    # Backward: jsonld-ex → FHIR (embeds opinions as FHIR extensions)
    # This is the actual artifact a FHIR system would receive.
    roundtripped, _rt_report = to_fhir(jsonld_doc)
    annotated_json = json.dumps(roundtripped).encode("utf-8")
    annotated_bytes = len(annotated_json)

    overhead = annotated_bytes - original_bytes
    overhead_pct = (overhead / original_bytes * 100) if original_bytes > 0 else 0.0

    return OverheadResult(
        resource_type=rtype,
        resource_id=rid,
        original_bytes=original_bytes,
        annotated_bytes=annotated_bytes,
        overhead_bytes=overhead,
        overhead_pct=overhead_pct,
    )


# =====================================================================
# 5. Query Expressiveness (6 query types)
# =====================================================================


def query_by_confidence(
    docs: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    """Query type 1: Filter annotated docs by projected probability.

    Returns docs where ANY opinion has P(ω) ≥ threshold.
    Impossible with plain FHIR — FHIR has no confidence fields.
    """
    results = []
    for doc in docs:
        opinions = doc.get("opinions", [])
        for op in opinions:
            pp = op.get("projected_probability", 0.0)
            if pp >= threshold:
                results.append(doc)
                break
    return results


def query_provenance_chain(
    docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Query type 2: Extract provenance (source model + method) per opinion.

    Impossible with plain FHIR — FHIR Provenance is a separate resource
    that cannot express per-assertion AI model attribution.
    """
    provenance_entries = []
    for doc in docs:
        rid = doc.get("resource_id", "")
        opinions = doc.get("opinions", [])
        for op in opinions:
            entry = {
                "resource_id": rid,
                "source": op.get("source", "unknown"),
                "method": op.get("method", "unknown"),
                "projected_probability": op.get("projected_probability"),
            }
            provenance_entries.append(entry)
    return provenance_entries


def query_fuse_multi_model(
    opinion_a: Opinion,
    opinion_b: Opinion,
) -> Dict[str, Any]:
    """Query type 3: Fuse predictions from two NLP models with conflict.

    Uses the pip-installed jsonld_ex cumulative_fuse and conflict_metric.
    Impossible with plain FHIR — FHIR cannot fuse probabilistic assertions.
    """
    fused = cumulative_fuse(opinion_a, opinion_b)
    conflict = conflict_metric(fused)

    return {
        "fused_probability": fused.projected_probability(),
        "fused_belief": fused.belief,
        "fused_disbelief": fused.disbelief,
        "fused_uncertainty": fused.uncertainty,
        "conflict": conflict,
    }


def query_temporal_decay(
    opinion: Opinion,
    age_days: float,
    half_life_days: float,
) -> Dict[str, Any]:
    """Query type 4: Apply temporal decay to old observations.

    Uses the pip-installed jsonld_ex decay_opinion.
    Impossible with plain FHIR — FHIR has no temporal decay model.
    """
    if age_days <= 0:
        return {
            "decayed_probability": opinion.projected_probability(),
            "original_probability": opinion.projected_probability(),
            "age_days": age_days,
            "decay_factor": 1.0,
        }

    decayed = decay_opinion(opinion, elapsed=age_days, half_life=half_life_days)

    return {
        "decayed_probability": decayed.projected_probability(),
        "original_probability": opinion.projected_probability(),
        "age_days": age_days,
        "decay_factor": decayed.belief / opinion.belief if opinion.belief > 0 else 0.0,
    }


def query_abstain_on_conflict(
    conflict_score: float,
    threshold: float,
) -> Dict[str, Any]:
    """Query type 5: Decide whether to abstain based on conflict.

    Impossible with plain FHIR — FHIR has no conflict metric.
    """
    return {
        "conflict_score": conflict_score,
        "threshold": threshold,
        "abstain": conflict_score > threshold,
    }


def query_by_uncertainty_component(
    docs: List[Dict[str, Any]],
    max_uncertainty: float,
) -> List[Dict[str, Any]]:
    """Query type 6: Filter by uncertainty component.

    Only returns docs where ALL opinions have uncertainty ≤ max_uncertainty.
    Impossible with plain FHIR — FHIR cannot distinguish "confident 50%"
    from "no evidence at all" (both map to the same scalar).
    """
    results = []
    for doc in docs:
        opinions = doc.get("opinions", [])
        if not opinions:
            continue
        if all(op.get("uncertainty", 1.0) <= max_uncertainty for op in opinions):
            results.append(doc)
    return results


# =====================================================================
# 6. Pipeline Report
# =====================================================================


@dataclass
class PipelineReport:
    """Summary report for Phase B pipeline execution.

    Attributes:
        n_bundles:                Number of FHIR bundles processed.
        n_resources_total:        Total resources extracted.
        n_resources_by_type:      Count per resource type.
        n_narrative_texts:        Total text snippets extracted for NER.
        round_trip_results:       Per-resource round-trip fidelity.
        overhead_results:         Per-resource annotation overhead.
        query_types_demonstrated: Number of query types shown (target ≥ 5).
    """

    n_bundles: int
    n_resources_total: int
    n_resources_by_type: Dict[str, int]
    n_narrative_texts: int
    round_trip_results: List[Any]
    overhead_results: List[Any]
    query_types_demonstrated: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)
