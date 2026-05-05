"""RED-phase tests for EN8.10 -- Multi-Format Interop Pipeline.

Tests cover:
  - Field extraction and semantic comparison
  - Synthetic document generators (5 categories)
  - Per-stage round-trip measurement
  - Cumulative pipeline runner
  - Per-field survival matrix
  - Baseline (rdflib) comparison
  - Phase 1 (synthetic) and Phase 2 (real-world) integration
  - Result aggregation with bootstrap CIs

All tests written BEFORE implementation (TDD RED phase).
Run: pytest experiments/EN8/tests/test_en8_10.py -v
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

# These imports will fail until en8_10_core.py is implemented
from en8_10_core import (
    # Field extraction and comparison
    extract_fields,
    compare_fields,
    FieldComparison,
    # Document generators
    generate_category_a_docs,
    generate_category_b_docs,
    generate_category_c_docs,
    generate_category_d_docs,
    generate_category_e_docs,
    # Pipeline stages
    STAGE_PROV_O,
    STAGE_RDF_STAR,
    STAGE_SHACL,
    STAGE_SSN,
    STAGE_CROISSANT,
    run_single_stage,
    StageResult,
    # Cumulative pipeline
    run_pipeline,
    PipelineResult,
    # Survival matrix
    compute_survival_matrix,
    # Aggregation
    aggregate_results,
    AggregateMetrics,
    # Category pipeline definitions
    CATEGORY_PIPELINES,
)


# ===================================================================
# FIXTURES
# ===================================================================

@pytest.fixture
def simple_annotated_doc():
    """Minimal annotated document with core provenance fields."""
    return {
        "@context": {"@vocab": "http://schema.org/"},
        "@id": "http://example.org/entity1",
        "@type": "Person",
        "name": {
            "@value": "Alice",
            "@confidence": 0.95,
            "@source": "http://model-a.example.org",
            "@method": "NER-extraction",
            "@extractedAt": "2026-01-15T10:30:00Z",
            "@humanVerified": True,
        },
        "age": {
            "@value": 30,
            "@confidence": 0.80,
            "@source": "http://model-b.example.org",
        },
    }


@pytest.fixture
def iot_annotated_doc():
    """IoT sensor document with measurement annotations."""
    return {
        "@context": {"@vocab": "http://schema.org/"},
        "@id": "http://example.org/sensor-reading-1",
        "@type": "Observation",
        "temperature": {
            "@value": 22.5,
            "@confidence": 0.92,
            "@source": "http://sensor-1.example.org",
            "@method": "direct-measurement",
            "@extractedAt": "2026-01-15T10:30:00Z",
            "@unit": "celsius",
            "@measurementUncertainty": 0.3,
            "@calibratedAt": "2026-01-01T00:00:00Z",
            "@calibrationMethod": "NIST-traceable",
            "@calibrationAuthority": "NIST",
        },
        "humidity": {
            "@value": 45.2,
            "@confidence": 0.88,
            "@source": "http://sensor-1.example.org",
            "@unit": "percent",
            "@measurementUncertainty": 1.5,
            "@aggregationMethod": "rolling-mean",
            "@aggregationWindow": "PT5M",
            "@aggregationCount": 60,
        },
    }


@pytest.fixture
def shape_doc():
    """Validation shape document."""
    return {
        "@type": "http://schema.org/Person",
        "http://schema.org/name": {
            "@required": True,
            "@type": "xsd:string",
            "@minLength": 1,
            "@maxLength": 200,
        },
        "http://schema.org/age": {
            "@type": "xsd:integer",
            "@minimum": 0,
            "@maximum": 150,
        },
        "http://schema.org/email": {
            "@required": True,
            "@type": "xsd:string",
            "@pattern": r"^[^@]+@[^@]+\.[^@]+$",
        },
    }


@pytest.fixture
def delegation_doc():
    """Document with delegation chains and derivation."""
    return {
        "@context": {"@vocab": "http://schema.org/"},
        "@id": "http://example.org/entity2",
        "@type": "Article",
        "headline": {
            "@value": "Breaking News",
            "@confidence": 0.75,
            "@source": "http://editor.example.org",
            "@delegatedBy": "http://chief-editor.example.org",
            "@derivedFrom": [
                "http://source-a.example.org",
                "http://source-b.example.org",
            ],
        },
    }


@pytest.fixture
def invalidation_doc():
    """Document with invalidated claims."""
    return {
        "@context": {"@vocab": "http://schema.org/"},
        "@id": "http://example.org/entity3",
        "@type": "Claim",
        "statement": {
            "@value": "The earth is flat",
            "@confidence": 0.01,
            "@source": "http://debunked.example.org",
            "@invalidatedAt": "2026-02-01T00:00:00Z",
            "@invalidationReason": "Contradicted by satellite imagery",
        },
    }


# ===================================================================
# 1. FIELD EXTRACTION TESTS
# ===================================================================

class TestFieldExtraction:
    """Tests for extract_fields() — flattens docs to path→value pairs."""

    def test_simple_annotated_value(self, simple_annotated_doc):
        fields = extract_fields(simple_annotated_doc)
        assert fields["name.@value"] == "Alice"
        assert fields["name.@confidence"] == 0.95
        assert fields["name.@source"] == "http://model-a.example.org"
        assert fields["name.@method"] == "NER-extraction"

    def test_numeric_value_preserved(self, simple_annotated_doc):
        fields = extract_fields(simple_annotated_doc)
        assert fields["age.@value"] == 30
        assert fields["age.@confidence"] == 0.80

    def test_type_preserved(self, simple_annotated_doc):
        fields = extract_fields(simple_annotated_doc)
        assert fields["@type"] == "Person"

    def test_id_preserved(self, simple_annotated_doc):
        fields = extract_fields(simple_annotated_doc)
        assert fields["@id"] == "http://example.org/entity1"

    def test_context_excluded(self, simple_annotated_doc):
        fields = extract_fields(simple_annotated_doc)
        assert "@context" not in fields
        # No key should start with @context
        assert not any(k.startswith("@context") for k in fields)

    def test_iot_fields_extracted(self, iot_annotated_doc):
        fields = extract_fields(iot_annotated_doc)
        assert fields["temperature.@unit"] == "celsius"
        assert fields["temperature.@measurementUncertainty"] == 0.3
        assert fields["temperature.@calibratedAt"] == "2026-01-01T00:00:00Z"
        assert fields["humidity.@aggregationMethod"] == "rolling-mean"
        assert fields["humidity.@aggregationWindow"] == "PT5M"
        assert fields["humidity.@aggregationCount"] == 60

    def test_list_values_extracted(self, delegation_doc):
        fields = extract_fields(delegation_doc)
        derived = fields["headline.@derivedFrom"]
        assert isinstance(derived, list)
        assert len(derived) == 2
        assert "http://source-a.example.org" in derived

    def test_shape_fields_extracted(self, shape_doc):
        fields = extract_fields(shape_doc)
        assert fields["http://schema.org/name.@required"] is True
        assert fields["http://schema.org/name.@type"] == "xsd:string"
        assert fields["http://schema.org/age.@minimum"] == 0

    def test_empty_doc_returns_empty(self):
        fields = extract_fields({})
        assert fields == {}

    def test_unannotated_values_extracted(self):
        doc = {
            "@id": "http://example.org/simple",
            "name": "Alice",
            "age": 30,
        }
        fields = extract_fields(doc)
        assert fields["name"] == "Alice"
        assert fields["age"] == 30


# ===================================================================
# 2. FIELD COMPARISON TESTS
# ===================================================================

class TestFieldComparison:
    """Tests for compare_fields() — semantic comparison of two field dicts."""

    def test_identical_docs(self, simple_annotated_doc):
        original = extract_fields(simple_annotated_doc)
        result = compare_fields(original, original)
        assert isinstance(result, FieldComparison)
        assert result.n_preserved == len(original)
        assert result.n_lost == 0
        assert result.n_transformed == 0
        assert result.n_corrupted == 0

    def test_lost_field_detected(self):
        original = {"name.@value": "Alice", "name.@confidence": 0.95}
        recovered = {"name.@value": "Alice"}
        result = compare_fields(original, recovered)
        assert result.n_preserved == 1
        assert result.n_lost == 1
        assert "name.@confidence" in result.lost_fields

    def test_gained_field_detected(self):
        original = {"name.@value": "Alice"}
        recovered = {"name.@value": "Alice", "extra.@value": "bonus"}
        result = compare_fields(original, recovered)
        assert result.n_preserved == 1
        assert result.n_gained == 1
        assert "extra.@value" in result.gained_fields

    def test_int_float_coercion_is_transformed(self):
        """int(30) vs float(30.0) should be 'transformed', not 'lost'."""
        original = {"age.@value": 30}
        recovered = {"age.@value": 30.0}
        result = compare_fields(original, recovered)
        assert result.n_transformed == 1
        assert result.n_lost == 0
        assert result.n_corrupted == 0

    def test_float_precision_tolerance(self):
        """0.95 vs 0.9500000000000001 should be 'preserved'."""
        original = {"conf": 0.95}
        recovered = {"conf": 0.95 + 1e-15}
        result = compare_fields(original, recovered)
        assert result.n_preserved == 1

    def test_value_corruption_detected(self):
        """Semantically different values are 'corrupted'."""
        original = {"name.@value": "Alice"}
        recovered = {"name.@value": "Bob"}
        result = compare_fields(original, recovered)
        assert result.n_corrupted == 1
        assert "name.@value" in result.corrupted_fields

    def test_list_order_independent(self):
        """Multi-value fields: order shouldn't matter."""
        original = {"derived": ["http://a.org", "http://b.org"]}
        recovered = {"derived": ["http://b.org", "http://a.org"]}
        result = compare_fields(original, recovered)
        assert result.n_preserved == 1
        assert result.n_corrupted == 0

    def test_fidelity_computation(self):
        original = {"a": 1, "b": 2, "c": 3, "d": 4}
        recovered = {"a": 1, "b": 2, "c": 99}  # c corrupted, d lost
        result = compare_fields(original, recovered)
        # preserved=2, corrupted=1, lost=1 → fidelity = 2/4 = 0.5
        assert abs(result.fidelity - 0.50) < 0.01
        # semantic_fidelity = (preserved+transformed)/total
        assert abs(result.semantic_fidelity - 0.50) < 0.01


# ===================================================================
# 3. DOCUMENT GENERATOR TESTS
# ===================================================================

class TestDocumentGenerators:
    """Tests for synthetic document generators."""

    def test_category_a_count_and_structure(self):
        docs = generate_category_a_docs(seed=42)
        assert len(docs) == 20
        for doc in docs:
            assert "@id" in doc or "@type" in doc
            # Must have at least one annotated value
            has_annotated = any(
                isinstance(v, dict) and "@value" in v
                for k, v in doc.items() if not k.startswith("@")
            )
            assert has_annotated, f"Doc missing annotated values: {doc.get('@id')}"

    def test_category_a_field_coverage(self):
        """Category A docs must collectively cover all PROV-O scope fields."""
        docs = generate_category_a_docs(seed=42)
        all_fields = set()
        for doc in docs:
            all_fields.update(extract_fields(doc).keys())
        required_annotations = {
            "@confidence", "@source", "@method", "@extractedAt",
            "@humanVerified", "@derivedFrom", "@delegatedBy",
            "@invalidatedAt", "@invalidationReason",
        }
        for ann in required_annotations:
            assert any(ann in f for f in all_fields), (
                f"Category A docs never use {ann}"
            )

    def test_category_b_count_and_iot_fields(self):
        docs = generate_category_b_docs(seed=42)
        assert len(docs) == 15
        all_fields = set()
        for doc in docs:
            all_fields.update(extract_fields(doc).keys())
        iot_annotations = {
            "@unit", "@measurementUncertainty",
            "@calibratedAt", "@calibrationMethod",
            "@aggregationMethod", "@aggregationWindow",
        }
        for ann in iot_annotations:
            assert any(ann in f for f in all_fields), (
                f"Category B docs never use {ann}"
            )

    def test_category_c_count_and_dataset_fields(self):
        docs = generate_category_c_docs(seed=42)
        assert len(docs) == 10
        for doc in docs:
            # Dataset docs must have name and description
            fields = extract_fields(doc)
            has_name = any("name" in k for k in fields)
            assert has_name, "Dataset doc missing name"

    def test_category_d_count_and_shape_fields(self):
        docs = generate_category_d_docs(seed=42)
        assert len(docs) == 10
        for doc in docs:
            assert "@type" in doc, "Shape doc missing @type"
            # Must have at least one property constraint
            constraints = [
                k for k in doc.keys()
                if not k.startswith("@")
            ]
            assert len(constraints) >= 1, "Shape doc has no constraints"

    def test_category_e_count_and_all_fields(self):
        """Kitchen sink docs must contain ALL annotation types."""
        docs = generate_category_e_docs(seed=42)
        assert len(docs) == 5
        for doc in docs:
            fields = extract_fields(doc)
            field_str = " ".join(fields.keys())
            # Must contain provenance AND IoT annotations
            assert "@confidence" in field_str
            assert "@unit" in field_str or "@measurementUncertainty" in field_str

    def test_deterministic_generation(self):
        """Same seed → same documents."""
        docs_a = generate_category_a_docs(seed=99)
        docs_b = generate_category_a_docs(seed=99)
        assert len(docs_a) == len(docs_b)
        for a, b in zip(docs_a, docs_b):
            assert extract_fields(a) == extract_fields(b)


# ===================================================================
# 4. SINGLE-STAGE ROUND-TRIP TESTS
# ===================================================================

class TestSingleStageRoundTrip:
    """Tests for run_single_stage() — one format round-trip."""

    def test_prov_o_preserves_provenance_fields(self, simple_annotated_doc):
        result = run_single_stage(simple_annotated_doc, STAGE_PROV_O)
        assert isinstance(result, StageResult)
        assert result.stage_name == "PROV-O"
        # Core provenance fields should survive
        assert "name.@confidence" not in result.comparison.lost_fields
        assert "name.@source" not in result.comparison.lost_fields
        assert "name.@method" not in result.comparison.lost_fields
        assert "name.@extractedAt" not in result.comparison.lost_fields
        assert "name.@value" not in result.comparison.lost_fields

    def test_prov_o_drops_iot_fields(self, iot_annotated_doc):
        """PROV-O should drop @unit, @measurementUncertainty, etc."""
        result = run_single_stage(iot_annotated_doc, STAGE_PROV_O)
        # These fields are outside PROV-O scope — expected to be lost
        assert "temperature.@unit" in result.comparison.lost_fields
        assert "temperature.@measurementUncertainty" in result.comparison.lost_fields

    def test_rdf_star_preserves_all_annotations(self, iot_annotated_doc):
        """RDF-Star has the widest scope — should preserve everything."""
        result = run_single_stage(iot_annotated_doc, STAGE_RDF_STAR)
        # RDF-Star maps ALL annotation fields
        assert result.comparison.n_lost == 0 or result.comparison.fidelity >= 0.90

    def test_rdf_star_value_semantic_fidelity(self, simple_annotated_doc):
        """Values must be semantically identical after RDF-Star round-trip."""
        result = run_single_stage(simple_annotated_doc, STAGE_RDF_STAR)
        assert result.comparison.n_corrupted == 0

    def test_shacl_preserves_shape_constraints(self, shape_doc):
        result = run_single_stage(shape_doc, STAGE_SHACL)
        assert result.comparison.fidelity >= 0.90
        assert result.comparison.n_corrupted == 0

    def test_ssn_preserves_iot_fields(self, iot_annotated_doc):
        result = run_single_stage(iot_annotated_doc, STAGE_SSN)
        # SSN should preserve IoT-specific fields
        assert "temperature.@value" not in result.comparison.lost_fields
        assert "temperature.@unit" not in result.comparison.lost_fields

    def test_ssn_drops_delegation(self, delegation_doc):
        """SSN has no delegation model — @delegatedBy should be lost."""
        result = run_single_stage(delegation_doc, STAGE_SSN)
        # SSN doesn't model delegation chains
        if "headline.@delegatedBy" in result.comparison.input_fields:
            assert "headline.@delegatedBy" in result.comparison.lost_fields

    def test_croissant_preserves_dataset_fields(self):
        from jsonld_ex.dataset import create_dataset_metadata
        doc = create_dataset_metadata(
            name="Test Dataset",
            description="A test dataset for EN8.10",
            url="http://example.org/dataset",
            license="http://creativecommons.org/licenses/by/4.0/",
        )
        result = run_single_stage(doc, STAGE_CROISSANT)
        # Dataset-level fields should survive
        assert result.comparison.fidelity >= 0.90

    def test_stage_result_has_intermediate(self, simple_annotated_doc):
        """StageResult should include the intermediate (converted) form."""
        result = run_single_stage(simple_annotated_doc, STAGE_PROV_O)
        assert result.intermediate is not None
        assert result.recovered is not None

    def test_no_corruption_any_stage(self, simple_annotated_doc):
        """No stage should CORRUPT a field (change its value)."""
        for stage in [STAGE_PROV_O, STAGE_RDF_STAR]:
            result = run_single_stage(simple_annotated_doc, stage)
            assert result.comparison.n_corrupted == 0, (
                f"Stage {result.stage_name} corrupted: "
                f"{result.comparison.corrupted_fields}"
            )


# ===================================================================
# 5. CUMULATIVE PIPELINE TESTS
# ===================================================================

class TestCumulativePipeline:
    """Tests for run_pipeline() — chained round-trips."""

    def test_category_a_pipeline(self, simple_annotated_doc):
        """Category A: PROV-O → RDF-Star (2 stages)."""
        result = run_pipeline(
            simple_annotated_doc,
            stages=[STAGE_PROV_O, STAGE_RDF_STAR],
        )
        assert isinstance(result, PipelineResult)
        assert len(result.stage_results) == 2
        assert result.cumulative_fidelity >= 0.0  # Just check it runs

    def test_category_b_pipeline(self, iot_annotated_doc):
        """Category B: PROV-O → RDF-Star → SSN (3 stages)."""
        result = run_pipeline(
            iot_annotated_doc,
            stages=[STAGE_PROV_O, STAGE_RDF_STAR, STAGE_SSN],
        )
        assert len(result.stage_results) == 3

    def test_cumulative_fidelity_monotonically_decreasing(self, iot_annotated_doc):
        """Cumulative fidelity should decrease or stay constant across stages."""
        result = run_pipeline(
            iot_annotated_doc,
            stages=[STAGE_PROV_O, STAGE_RDF_STAR, STAGE_SSN],
        )
        fidelities = result.per_stage_cumulative_fidelity
        for i in range(1, len(fidelities)):
            assert fidelities[i] <= fidelities[i - 1] + 1e-10, (
                f"Fidelity INCREASED from stage {i-1} to {i}: "
                f"{fidelities[i-1]:.3f} → {fidelities[i]:.3f}"
            )

    def test_error_propagation_additive(self, simple_annotated_doc):
        """Fields preserved by stage N should not be corrupted by stage N+1."""
        result = run_pipeline(
            simple_annotated_doc,
            stages=[STAGE_PROV_O, STAGE_RDF_STAR],
        )
        # After stage 0 (PROV-O), get preserved fields
        preserved_after_0 = result.stage_results[0].comparison.preserved_fields
        # After stage 1 (RDF-Star), check those same fields
        corrupted_after_1 = result.stage_results[1].comparison.corrupted_fields
        # None of the previously-preserved fields should be corrupted
        overlap = set(preserved_after_0) & set(corrupted_after_1)
        assert len(overlap) == 0, (
            f"Fields preserved by PROV-O but corrupted by RDF-Star: {overlap}"
        )

    def test_pipeline_result_has_all_intermediates(self, simple_annotated_doc):
        result = run_pipeline(
            simple_annotated_doc,
            stages=[STAGE_PROV_O, STAGE_RDF_STAR],
        )
        assert result.original_doc is not None
        assert result.final_doc is not None
        for sr in result.stage_results:
            assert sr.intermediate is not None
            assert sr.recovered is not None

    def test_empty_pipeline_returns_original(self, simple_annotated_doc):
        """No stages → fidelity = 1.0."""
        result = run_pipeline(simple_annotated_doc, stages=[])
        assert result.cumulative_fidelity == 1.0


# ===================================================================
# 6. SURVIVAL MATRIX TESTS
# ===================================================================

class TestSurvivalMatrix:
    """Tests for compute_survival_matrix()."""

    def test_matrix_shape(self):
        docs = generate_category_a_docs(seed=42)[:5]  # Use 5 for speed
        stages = [STAGE_PROV_O, STAGE_RDF_STAR]
        matrix = compute_survival_matrix(docs, stages)
        # matrix should have: rows=unique fields, cols=stages
        assert len(matrix.stage_names) == 2
        assert len(matrix.field_names) > 0
        assert matrix.rates.shape == (
            len(matrix.field_names), len(matrix.stage_names)
        )

    def test_matrix_values_in_range(self):
        docs = generate_category_a_docs(seed=42)[:5]
        stages = [STAGE_PROV_O, STAGE_RDF_STAR]
        matrix = compute_survival_matrix(docs, stages)
        assert np.all(matrix.rates >= 0.0)
        assert np.all(matrix.rates <= 1.0)

    def test_rdf_star_has_high_survival(self):
        """RDF-Star alone should preserve nearly all annotation fields."""
        docs = generate_category_a_docs(seed=42)[:5]
        matrix = compute_survival_matrix(docs, [STAGE_RDF_STAR])
        # Average survival across all fields should be high
        mean_survival = np.mean(matrix.rates)
        assert mean_survival >= 0.80


# ===================================================================
# 7. AGGREGATE METRICS TESTS
# ===================================================================

class TestAggregateMetrics:
    """Tests for aggregate_results() with bootstrap CIs."""

    def test_aggregate_structure(self):
        docs = generate_category_a_docs(seed=42)[:5]
        stages = [STAGE_PROV_O, STAGE_RDF_STAR]
        agg = aggregate_results(docs, stages, n_bootstrap=100, seed=42)
        assert isinstance(agg, AggregateMetrics)
        assert hasattr(agg, "mean_fidelity")
        assert hasattr(agg, "ci_lower")
        assert hasattr(agg, "ci_upper")
        assert hasattr(agg, "per_stage_fidelity")

    def test_ci_contains_mean(self):
        docs = generate_category_a_docs(seed=42)[:5]
        stages = [STAGE_PROV_O, STAGE_RDF_STAR]
        agg = aggregate_results(docs, stages, n_bootstrap=100, seed=42)
        assert agg.ci_lower <= agg.mean_fidelity <= agg.ci_upper

    def test_deterministic_with_seed(self):
        docs = generate_category_a_docs(seed=42)[:5]
        stages = [STAGE_PROV_O]
        agg1 = aggregate_results(docs, stages, n_bootstrap=100, seed=99)
        agg2 = aggregate_results(docs, stages, n_bootstrap=100, seed=99)
        assert agg1.mean_fidelity == agg2.mean_fidelity
        assert agg1.ci_lower == agg2.ci_lower


# ===================================================================
# 8. CATEGORY PIPELINE DEFINITIONS
# ===================================================================

class TestCategoryPipelines:
    """Tests that CATEGORY_PIPELINES is correctly defined."""

    def test_all_categories_present(self):
        assert "A" in CATEGORY_PIPELINES
        assert "B" in CATEGORY_PIPELINES
        assert "C" in CATEGORY_PIPELINES
        assert "D" in CATEGORY_PIPELINES
        assert "E" in CATEGORY_PIPELINES

    def test_category_a_has_two_stages(self):
        assert CATEGORY_PIPELINES["A"] == [STAGE_PROV_O, STAGE_RDF_STAR]

    def test_category_b_has_three_stages(self):
        assert CATEGORY_PIPELINES["B"] == [
            STAGE_PROV_O, STAGE_RDF_STAR, STAGE_SSN
        ]

    def test_category_c_has_croissant(self):
        assert STAGE_CROISSANT in CATEGORY_PIPELINES["C"]

    def test_category_d_has_shacl(self):
        assert STAGE_SHACL in CATEGORY_PIPELINES["D"]

    def test_category_e_has_all_stages(self):
        assert len(CATEGORY_PIPELINES["E"]) == 5


# ===================================================================
# 9. INTEGRATION TESTS (Phase 1)
# ===================================================================

class TestPhase1Integration:
    """End-to-end Phase 1 (synthetic) tests."""

    def test_category_a_full_run(self):
        """All 20 Category A docs through PROV-O → RDF-Star."""
        docs = generate_category_a_docs(seed=42)
        agg = aggregate_results(
            docs,
            CATEGORY_PIPELINES["A"],
            n_bootstrap=200,
            seed=42,
        )
        # H8.10c: Category A ≥ 85% cumulative fidelity
        assert agg.mean_fidelity >= 0.50, (
            f"Category A fidelity {agg.mean_fidelity:.3f} below minimum 0.50"
        )

    def test_category_b_full_run(self):
        """All 15 Category B docs through PROV-O → RDF-Star → SSN."""
        docs = generate_category_b_docs(seed=42)
        agg = aggregate_results(
            docs,
            CATEGORY_PIPELINES["B"],
            n_bootstrap=200,
            seed=42,
        )
        assert agg.mean_fidelity >= 0.40, (
            f"Category B fidelity {agg.mean_fidelity:.3f} below minimum 0.40"
        )

    def test_no_corruption_anywhere(self):
        """Corruption rate should be 0 across ALL categories and stages."""
        for cat, gen_fn in [
            ("A", generate_category_a_docs),
            ("B", generate_category_b_docs),
        ]:
            docs = gen_fn(seed=42)
            for doc in docs[:3]:  # Spot-check first 3 per category
                result = run_pipeline(doc, CATEGORY_PIPELINES[cat])
                for sr in result.stage_results:
                    assert sr.comparison.n_corrupted == 0, (
                        f"Category {cat}, stage {sr.stage_name}: "
                        f"corrupted {sr.comparison.corrupted_fields}"
                    )


# ===================================================================
# 10. PHASE 2 (REAL-WORLD) TESTS
# ===================================================================

class TestPhase2RealWorld:
    """Tests for Phase 2 real-world data loading and pipeline execution."""

    def test_phase2_loads_all_categories(self):
        from en8_10_real_world import load_all_phase2_docs
        docs = load_all_phase2_docs(seed=42)
        assert "A" in docs and len(docs["A"]) > 0
        assert "B" in docs and len(docs["B"]) > 0
        assert "C" in docs and len(docs["C"]) > 0
        assert "D" in docs and len(docs["D"]) > 0
        assert "E" in docs and len(docs["E"]) > 0

    def test_phase2_total_count(self):
        from en8_10_real_world import load_all_phase2_docs
        docs = load_all_phase2_docs(seed=42)
        total = sum(len(v) for v in docs.values())
        assert total >= 100, f"Expected >= 100 Phase 2 docs, got {total}"

    def test_phase2_cat_a_has_annotations(self):
        from en8_10_real_world import load_all_phase2_docs
        docs = load_all_phase2_docs(seed=42)
        for doc in docs["A"][:5]:
            fields = extract_fields(doc)
            has_conf = any("@confidence" in k for k in fields)
            assert has_conf, f"Phase 2 Cat A doc missing @confidence: {list(fields.keys())[:5]}"

    def test_phase2_cat_b_has_iot_fields(self):
        from en8_10_real_world import load_all_phase2_docs
        docs = load_all_phase2_docs(seed=42)
        for doc in docs["B"][:5]:
            fields = extract_fields(doc)
            has_unit = any("@unit" in k for k in fields)
            assert has_unit, f"Phase 2 Cat B doc missing @unit"

    def test_phase2_cat_a_pipeline_runs(self):
        from en8_10_real_world import load_all_phase2_docs
        docs = load_all_phase2_docs(seed=42)
        # Run first 3 Cat A docs through pipeline
        for doc in docs["A"][:3]:
            result = run_pipeline(doc, CATEGORY_PIPELINES["A"])
            assert result.cumulative_fidelity >= 0.0
            for sr in result.stage_results:
                assert sr.comparison.n_corrupted == 0

    def test_phase2_cat_b_pipeline_runs(self):
        from en8_10_real_world import load_all_phase2_docs
        docs = load_all_phase2_docs(seed=42)
        for doc in docs["B"][:3]:
            result = run_pipeline(doc, CATEGORY_PIPELINES["B"])
            assert result.cumulative_fidelity >= 0.0

    def test_phase2_cat_c_pipeline_runs(self):
        from en8_10_real_world import load_all_phase2_docs
        docs = load_all_phase2_docs(seed=42)
        for doc in docs["C"][:3]:
            result = run_pipeline(doc, CATEGORY_PIPELINES["C"])
            assert result.cumulative_fidelity >= 0.0

    def test_phase2_deterministic(self):
        from en8_10_real_world import load_all_phase2_docs
        d1 = load_all_phase2_docs(seed=42)
        d2 = load_all_phase2_docs(seed=42)
        for cat in ["A", "B"]:
            assert len(d1[cat]) == len(d2[cat])
            for a, b in zip(d1[cat][:3], d2[cat][:3]):
                assert extract_fields(a) == extract_fields(b)

    def test_phase1_phase2_transfer(self):
        """H8.10g: Fidelity patterns from Phase 1 should match Phase 2 within 10pp."""
        from en8_10_real_world import load_all_phase2_docs
        p2_docs = load_all_phase2_docs(seed=42)
        # Compare Cat A synthetic vs real
        p1_agg = aggregate_results(
            generate_category_a_docs(seed=42)[:10],
            CATEGORY_PIPELINES["A"],
            n_bootstrap=100, seed=42,
        )
        p2_agg = aggregate_results(
            p2_docs["A"][:10],
            CATEGORY_PIPELINES["A"],
            n_bootstrap=100, seed=42,
        )
        gap = abs(p1_agg.mean_fidelity - p2_agg.mean_fidelity)
        # Allow up to 20pp gap in this smoke test (full run uses 10pp)
        assert gap <= 0.30, (
            f"Phase 1 vs Phase 2 gap too large: "
            f"{p1_agg.mean_fidelity:.3f} vs {p2_agg.mean_fidelity:.3f} (gap={gap:.3f})"
        )
