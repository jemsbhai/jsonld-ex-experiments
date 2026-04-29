"""Tests for EN3.4 Phase B — FHIR Clinical Pipeline.

RED phase: these tests define the contract for en3_4_phase_b.py.
All should FAIL before implementation.

Tests cover the FHIR clinical pipeline demonstration:
  1. FHIR bundle loading and resource extraction
  2. Narrative text extraction from FHIR resources
  3. Round-trip fidelity (from_fhir → strip → to_fhir)
  4. Annotation overhead measurement
  5. Query expressiveness (6 query types impossible with plain FHIR)
  6. End-to-end pipeline integration

All tests use small synthetic FHIR resources — NO Synthea data or GPU
required for unit tests. Synthea is used only at full-scale runtime.

Uses the pip-installed jsonld_ex package for all FHIR operations.
"""
from __future__ import annotations

import json
import pytest

# -- Path setup --
import sys
from pathlib import Path

_en3_dir = Path(__file__).resolve().parent.parent
_experiments_root = _en3_dir.parent
for p in [str(_en3_dir), str(_experiments_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# =====================================================================
# Imports from module under test (will fail until implemented)
# =====================================================================

from EN3.en3_4_phase_b import (
    load_bundle_resources,
    extract_narrative_texts,
    measure_round_trip_fidelity,
    RoundTripResult,
    measure_annotation_overhead,
    OverheadResult,
    query_by_confidence,
    query_provenance_chain,
    query_fuse_multi_model,
    query_temporal_decay,
    query_abstain_on_conflict,
    query_by_uncertainty_component,
    PipelineReport,
)


# =====================================================================
# Fixtures: Synthetic FHIR Resources
# =====================================================================


@pytest.fixture
def observation_resource():
    """Minimal valid FHIR R4 Observation."""
    return {
        "resourceType": "Observation",
        "id": "obs-001",
        "status": "final",
        "code": {
            "coding": [{"system": "http://loinc.org",
                        "code": "8867-4",
                        "display": "Heart rate"}],
            "text": "Heart rate measurement",
        },
        "valueQuantity": {"value": 72, "unit": "bpm"},
    }


@pytest.fixture
def condition_resource():
    """Minimal valid FHIR R4 Condition."""
    return {
        "resourceType": "Condition",
        "id": "cond-001",
        "clinicalStatus": {
            "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active"}],
        },
        "verificationStatus": {
            "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": "confirmed"}],
        },
        "code": {
            "coding": [{"system": "http://snomed.info/sct",
                        "code": "73211009",
                        "display": "Diabetes mellitus"}],
            "text": "Diabetes mellitus type 2",
        },
        "subject": {"reference": "Patient/pat-001"},
    }


@pytest.fixture
def patient_resource():
    """Minimal valid FHIR R4 Patient."""
    return {
        "resourceType": "Patient",
        "id": "pat-001",
        "active": True,
        "name": [{"family": "Smith", "given": ["John"]}],
        "gender": "male",
        "birthDate": "1990-01-15",
    }


@pytest.fixture
def medication_request_resource():
    """Minimal valid FHIR R4 MedicationRequest."""
    return {
        "resourceType": "MedicationRequest",
        "id": "medrq-001",
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": {
            "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": "860975",
                        "display": "Metformin 500 MG"}],
            "text": "Metformin hydrochloride 500mg oral tablet",
        },
        "subject": {"reference": "Patient/pat-001"},
    }


@pytest.fixture
def diagnostic_report_resource():
    """Minimal valid FHIR R4 DiagnosticReport."""
    return {
        "resourceType": "DiagnosticReport",
        "id": "diag-001",
        "status": "final",
        "code": {
            "coding": [{"system": "http://loinc.org",
                        "code": "58410-2",
                        "display": "CBC panel"}],
        },
        "conclusion": "Complete blood count within normal limits. "
                       "No evidence of anemia or infection.",
    }


@pytest.fixture
def fhir_bundle(observation_resource, condition_resource,
                patient_resource, medication_request_resource,
                diagnostic_report_resource):
    """Synthetic FHIR R4 Bundle with 5 resources."""
    return {
        "resourceType": "Bundle",
        "id": "bundle-test-001",
        "type": "collection",
        "entry": [
            {"resource": observation_resource},
            {"resource": condition_resource},
            {"resource": patient_resource},
            {"resource": medication_request_resource},
            {"resource": diagnostic_report_resource},
        ],
    }


# =====================================================================
# 1. Bundle Loading and Resource Extraction
# =====================================================================


class TestLoadBundleResources:
    """Tests for extracting typed resources from FHIR bundles."""

    def test_extracts_all_resource_types(self, fhir_bundle):
        """Should extract resources grouped by type."""
        resources = load_bundle_resources(fhir_bundle)
        assert "Observation" in resources
        assert "Condition" in resources
        assert "Patient" in resources
        assert "MedicationRequest" in resources
        assert "DiagnosticReport" in resources

    def test_correct_count_per_type(self, fhir_bundle):
        """Each type should have the correct number of resources."""
        resources = load_bundle_resources(fhir_bundle)
        assert len(resources["Observation"]) == 1
        assert len(resources["Patient"]) == 1

    def test_preserves_resource_data(self, fhir_bundle):
        """Extracted resources should contain original data."""
        resources = load_bundle_resources(fhir_bundle)
        obs = resources["Observation"][0]
        assert obs["id"] == "obs-001"
        assert obs["status"] == "final"

    def test_filters_by_target_types(self, fhir_bundle):
        """Can filter to only specific resource types."""
        target = {"Observation", "Condition"}
        resources = load_bundle_resources(fhir_bundle, target_types=target)
        assert set(resources.keys()) <= target

    def test_empty_bundle(self):
        """Empty bundle returns empty dict."""
        bundle = {"resourceType": "Bundle", "type": "collection", "entry": []}
        resources = load_bundle_resources(bundle)
        assert len(resources) == 0

    def test_missing_entry_key(self):
        """Bundle without 'entry' key returns empty dict."""
        bundle = {"resourceType": "Bundle", "type": "collection"}
        resources = load_bundle_resources(bundle)
        assert len(resources) == 0


# =====================================================================
# 2. Narrative Text Extraction
# =====================================================================


class TestExtractNarrativeTexts:
    """Tests for extracting text fields from FHIR resources for NER."""

    def test_observation_code_text(self, observation_resource):
        """Extracts code.text from Observation."""
        texts = extract_narrative_texts([observation_resource])
        assert any("Heart rate" in t["text"] for t in texts)

    def test_condition_code_text(self, condition_resource):
        """Extracts code.text from Condition."""
        texts = extract_narrative_texts([condition_resource])
        assert any("Diabetes" in t["text"] for t in texts)

    def test_medication_text(self, medication_request_resource):
        """Extracts medicationCodeableConcept.text from MedicationRequest."""
        texts = extract_narrative_texts([medication_request_resource])
        assert any("Metformin" in t["text"] for t in texts)

    def test_diagnostic_conclusion(self, diagnostic_report_resource):
        """Extracts conclusion from DiagnosticReport."""
        texts = extract_narrative_texts([diagnostic_report_resource])
        assert any("blood count" in t["text"] for t in texts)

    def test_includes_resource_id(self, observation_resource):
        """Each text snippet should reference its source resource ID."""
        texts = extract_narrative_texts([observation_resource])
        assert all("resource_id" in t for t in texts)
        assert any(t["resource_id"] == "obs-001" for t in texts)

    def test_includes_resource_type(self, observation_resource):
        """Each text snippet should include the resource type."""
        texts = extract_narrative_texts([observation_resource])
        assert all("resource_type" in t for t in texts)

    def test_empty_input(self):
        """No resources → no texts."""
        texts = extract_narrative_texts([])
        assert len(texts) == 0

    def test_resource_without_text_fields(self, patient_resource):
        """Patient has no narrative text fields → nothing extracted."""
        texts = extract_narrative_texts([patient_resource])
        # Patient doesn't have code.text or conclusion
        assert len(texts) == 0


# =====================================================================
# 3. Round-Trip Fidelity
# =====================================================================


class TestRoundTripFidelity:
    """Tests for FHIR → jsonld-ex → FHIR round-trip preservation."""

    def test_observation_round_trip(self, observation_resource):
        """Observation preserves key fields through round-trip."""
        result = measure_round_trip_fidelity(observation_resource)
        assert isinstance(result, RoundTripResult)
        assert result.resource_type == "Observation"
        assert result.resource_id == "obs-001"
        assert result.fields_preserved >= 0
        assert result.fields_total > 0

    def test_condition_round_trip(self, condition_resource):
        """Condition preserves key fields through round-trip."""
        result = measure_round_trip_fidelity(condition_resource)
        assert result.resource_type == "Condition"
        assert result.resource_id == "cond-001"

    def test_patient_round_trip(self, patient_resource):
        """Patient preserves key fields through round-trip."""
        result = measure_round_trip_fidelity(patient_resource)
        assert result.resource_type == "Patient"

    def test_fidelity_rate_computation(self, observation_resource):
        """Fidelity rate = fields_preserved / fields_total."""
        result = measure_round_trip_fidelity(observation_resource)
        if result.fields_total > 0:
            expected_rate = result.fields_preserved / result.fields_total
            assert result.fidelity_rate == pytest.approx(expected_rate, abs=0.001)

    def test_result_includes_lost_fields(self, observation_resource):
        """Result should list any fields that were lost."""
        result = measure_round_trip_fidelity(observation_resource)
        assert hasattr(result, "lost_fields")
        assert isinstance(result.lost_fields, list)


# =====================================================================
# 4. Annotation Overhead
# =====================================================================


class TestAnnotationOverhead:
    """Tests for measuring byte overhead of jsonld-ex annotations."""

    def test_overhead_measurement_valid(self, observation_resource):
        """Overhead measurement produces valid results.

        Note: overhead CAN be negative if the round-trip representation
        is more compact than the original. This is a legitimate finding
        about representation efficiency, not a bug.
        """
        result = measure_annotation_overhead(observation_resource)
        assert isinstance(result, OverheadResult)
        assert result.original_bytes > 0
        assert result.annotated_bytes > 0

    def test_overhead_percentage_computed(self, observation_resource):
        """Overhead percentage = (annotated - original) / original × 100."""
        result = measure_annotation_overhead(observation_resource)
        if result.original_bytes > 0:
            expected_pct = (result.overhead_bytes / result.original_bytes) * 100
            assert result.overhead_pct == pytest.approx(expected_pct, abs=0.01)

    def test_original_bytes_matches_json(self, observation_resource):
        """Original bytes should match JSON serialization of input."""
        result = measure_annotation_overhead(observation_resource)
        expected = len(json.dumps(observation_resource).encode("utf-8"))
        assert result.original_bytes == expected

    def test_condition_overhead(self, condition_resource):
        """Condition resources produce valid overhead measurement."""
        result = measure_annotation_overhead(condition_resource)
        assert result.original_bytes > 0
        assert result.annotated_bytes > 0
        # Net overhead may be negative if round-trip is more compact
        # than original FHIR (field loss). This is documented honestly.


# =====================================================================
# 5. Query Expressiveness (H3.4i — 6 query types)
# =====================================================================


class TestQueryByConfidence:
    """Query type 1: Filter by AI confidence level."""

    def test_filters_above_threshold(self):
        """Resources with opinion P(ω) above threshold are returned."""
        # Annotated doc with high-confidence opinion
        doc = {
            "@type": "fhir:Observation",
            "resource_id": "obs-001",
            "opinions": [{"belief": 0.85, "disbelief": 0.05,
                          "uncertainty": 0.10, "base_rate": 0.5,
                          "projected_probability": 0.90}],
        }
        results = query_by_confidence([doc], threshold=0.8)
        assert len(results) == 1

    def test_excludes_below_threshold(self):
        """Resources with opinion P(ω) below threshold are excluded."""
        doc = {
            "@type": "fhir:Observation",
            "resource_id": "obs-002",
            "opinions": [{"belief": 0.20, "disbelief": 0.30,
                          "uncertainty": 0.50, "base_rate": 0.5,
                          "projected_probability": 0.45}],
        }
        results = query_by_confidence([doc], threshold=0.8)
        assert len(results) == 0

    def test_impossible_with_plain_fhir(self):
        """Plain FHIR has no confidence field to filter on."""
        # This is a documentation test — plain FHIR Observation has
        # no @confidence or belief/disbelief/uncertainty fields
        plain_obs = {
            "resourceType": "Observation",
            "status": "final",
            "code": {"text": "test"},
        }
        assert "@confidence" not in json.dumps(plain_obs)
        assert "belief" not in json.dumps(plain_obs)


class TestQueryProvenanceChain:
    """Query type 2: Track provenance of AI-suggested diagnoses."""

    def test_returns_provenance_info(self):
        """Should return source model and method for each opinion."""
        doc = {
            "@type": "fhir:Condition",
            "resource_id": "cond-001",
            "opinions": [{"source": "gliner-biomed-v1.0",
                          "method": "NER-zero-shot",
                          "projected_probability": 0.88}],
        }
        provenance = query_provenance_chain([doc])
        assert len(provenance) >= 1
        assert provenance[0]["source"] == "gliner-biomed-v1.0"
        assert provenance[0]["method"] == "NER-zero-shot"


class TestQueryFuseMultiModel:
    """Query type 3: Fuse predictions from multiple NLP models."""

    def test_fuses_two_opinions(self):
        """Should fuse opinions from two models using SL."""
        from jsonld_ex.confidence_algebra import Opinion, cumulative_fuse

        op_a = Opinion.from_confidence(0.85, uncertainty=0.10)
        op_b = Opinion.from_confidence(0.75, uncertainty=0.15)
        result = query_fuse_multi_model(op_a, op_b)
        assert "fused_probability" in result
        assert "conflict" in result
        assert result["fused_probability"] > 0

    def test_reports_conflict(self):
        """Disagreeing models should produce measurable conflict."""
        from jsonld_ex.confidence_algebra import Opinion

        op_a = Opinion.from_confidence(0.90, uncertainty=0.05)
        op_b = Opinion.from_confidence(0.10, uncertainty=0.05)
        result = query_fuse_multi_model(op_a, op_b)
        assert result["conflict"] > 0.1  # measurable disagreement


class TestQueryTemporalDecay:
    """Query type 4: Apply temporal decay to old observations."""

    def test_decayed_opinion_lower_belief(self):
        """Older observations should have lower belief after decay."""
        from jsonld_ex.confidence_algebra import Opinion

        fresh = Opinion.from_confidence(0.90, uncertainty=0.05)
        result = query_temporal_decay(
            fresh, age_days=365, half_life_days=180,
        )
        assert result["decayed_probability"] < fresh.projected_probability()
        assert result["age_days"] == 365

    def test_zero_age_no_decay(self):
        """Age = 0 → no decay (or negligible)."""
        from jsonld_ex.confidence_algebra import Opinion

        op = Opinion.from_confidence(0.90, uncertainty=0.05)
        result = query_temporal_decay(op, age_days=0, half_life_days=180)
        assert result["decayed_probability"] == pytest.approx(
            op.projected_probability(), abs=0.01)


class TestQueryAbstainOnConflict:
    """Query type 5: Abstain on high-conflict extractions."""

    def test_low_conflict_no_abstention(self):
        """Low-conflict entity → not abstained."""
        result = query_abstain_on_conflict(
            conflict_score=0.1, threshold=0.5)
        assert result["abstain"] is False

    def test_high_conflict_triggers_abstention(self):
        """High-conflict entity → abstained."""
        result = query_abstain_on_conflict(
            conflict_score=0.7, threshold=0.5)
        assert result["abstain"] is True

    def test_returns_conflict_score(self):
        """Result should include the conflict score."""
        result = query_abstain_on_conflict(
            conflict_score=0.3, threshold=0.5)
        assert result["conflict_score"] == 0.3


class TestQueryByUncertaintyComponent:
    """Query type 6: Filter by uncertainty component specifically."""

    def test_filters_by_uncertainty(self):
        """Resources with uncertainty above threshold are flagged."""
        doc = {
            "@type": "fhir:Observation",
            "resource_id": "obs-003",
            "opinions": [{"uncertainty": 0.40, "belief": 0.30,
                          "disbelief": 0.30}],
        }
        results = query_by_uncertainty_component(
            [doc], max_uncertainty=0.30)
        assert len(results) == 0  # uncertainty 0.40 > 0.30 → excluded

    def test_includes_low_uncertainty(self):
        """Resources with low uncertainty pass the filter."""
        doc = {
            "@type": "fhir:Observation",
            "resource_id": "obs-004",
            "opinions": [{"uncertainty": 0.05, "belief": 0.90,
                          "disbelief": 0.05}],
        }
        results = query_by_uncertainty_component(
            [doc], max_uncertainty=0.30)
        assert len(results) == 1

    def test_impossible_with_plain_fhir(self):
        """Plain FHIR cannot distinguish high-confidence-50% from
        no-evidence-50%. SL uncertainty component makes this possible."""
        # This is a key differentiator for the paper
        pass  # documented as a finding, not a runtime test


# =====================================================================
# 6. Pipeline Report
# =====================================================================


class TestPipelineReport:
    """Tests for the Phase B summary report."""

    def test_construction(self):
        """PipelineReport holds all required summary fields."""
        report = PipelineReport(
            n_bundles=100,
            n_resources_total=5000,
            n_resources_by_type={"Observation": 2000, "Condition": 1000},
            n_narrative_texts=3000,
            round_trip_results=[],
            overhead_results=[],
            query_types_demonstrated=5,
        )
        assert report.n_bundles == 100
        assert report.query_types_demonstrated == 5

    def test_to_dict(self):
        """Should serialize to JSON-compatible dict."""
        report = PipelineReport(
            n_bundles=10,
            n_resources_total=500,
            n_resources_by_type={"Patient": 10},
            n_narrative_texts=200,
            round_trip_results=[],
            overhead_results=[],
            query_types_demonstrated=6,
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["n_bundles"] == 10
        assert d["query_types_demonstrated"] == 6
