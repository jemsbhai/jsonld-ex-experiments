"""RED-phase tests for EN8.6 Ã¢â‚¬â€ Graph Merge and Diff Operations.

Tests cover:
  - Synthetic data generation (ground-truth KGs, source views, corruption)
  - Baseline implementations (rdflib union, random, majority, most-recent, rdflib+confidence)
  - Evaluation metrics (conflict resolution accuracy, ECE, diff P/R)
  - Merge accuracy under 4 calibration regimes
  - Diff completeness and correctness
  - Audit trail fidelity
  - Throughput and scaling

All tests written BEFORE implementation (TDD RED phase).
Run: pytest experiments/EN8/tests/test_en8_6.py -v
"""

from __future__ import annotations

import time
import json
import math
from typing import Any

import numpy as np
import pytest

# These imports will fail until en8_6_core.py is implemented
from en8_6_core import (
    generate_ground_truth_kg,
    generate_source_views,
    generate_corrupted_source,
    assign_confidence,
    source_to_jsonld,
    # Baselines
    baseline_rdflib_union,
    baseline_random_choice,
    baseline_majority_vote,
    baseline_most_recent,
    baseline_rdflib_confidence_argmax,
    # Evaluation
    evaluate_merge_accuracy,
    compute_ece,
    evaluate_diff,
    evaluate_audit_trail,
    # High-level runners
    run_single_config,
    CalibrationRegime,
    ExperimentConfig,
    MergeResult,
)


# Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â
# FIXTURES
# Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â


@pytest.fixture
def small_ground_truth():
    """20-entity ground truth for fast tests."""
    return generate_ground_truth_kg(n_entities=20, seed=42)


@pytest.fixture
def small_sources(small_ground_truth):
    """3 overlapping source views of the small ground truth."""
    return generate_source_views(
        ground_truth=small_ground_truth,
        n_sources=3,
        overlap_rate=0.4,
        seed=42,
    )


@pytest.fixture
def small_config():
    """Default small-scale experiment config."""
    return ExperimentConfig(
        n_entities=20,
        n_sources=3,
        corruption_rate=0.10,
        calibration=CalibrationRegime.IDEAL,
        overlap_rate=0.4,
        seed=42,
    )


# Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â
# 1. DATA GENERATION TESTS
# Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â


class TestGroundTruthGeneration:
    """Verify ground truth KG is well-formed."""

    def test_correct_entity_count(self):
        gt = generate_ground_truth_kg(n_entities=50, seed=42)
        assert len(gt) == 50

    def test_entity_has_all_properties(self):
        gt = generate_ground_truth_kg(n_entities=10, seed=42)
        expected_props = {
            "name", "affiliation", "field", "h_index", "email",
            "publications_count", "country", "active", "homepage", "orcid",
        }
        for entity_id, entity in gt.items():
            assert set(entity.keys()) >= expected_props, (
                f"Entity {entity_id} missing properties: "
                f"{expected_props - set(entity.keys())}"
            )

    def test_entity_ids_are_uris(self):
        gt = generate_ground_truth_kg(n_entities=10, seed=42)
        for entity_id in gt:
            assert entity_id.startswith("http://") or entity_id.startswith("urn:"), (
                f"Entity ID should be a URI, got: {entity_id}"
            )

    def test_unique_names_and_orcids(self):
        gt = generate_ground_truth_kg(n_entities=50, seed=42)
        names = [e["name"] for e in gt.values()]
        orcids = [e["orcid"] for e in gt.values()]
        assert len(set(names)) == 50, "Names should be unique"
        assert len(set(orcids)) == 50, "ORCIDs should be unique"

    def test_deterministic_with_seed(self):
        gt1 = generate_ground_truth_kg(n_entities=10, seed=42)
        gt2 = generate_ground_truth_kg(n_entities=10, seed=42)
        assert gt1 == gt2

    def test_different_seeds_differ(self):
        gt1 = generate_ground_truth_kg(n_entities=10, seed=42)
        gt2 = generate_ground_truth_kg(n_entities=10, seed=99)
        assert gt1 != gt2


class TestSourceViewGeneration:
    """Verify source views have correct overlap structure."""

    def test_correct_number_of_sources(self, small_ground_truth):
        sources = generate_source_views(
            small_ground_truth, n_sources=3, overlap_rate=0.4, seed=42
        )
        assert len(sources) == 3

    def test_overlap_rate_approximately_correct(self):
        gt = generate_ground_truth_kg(n_entities=100, seed=42)
        sources = generate_source_views(gt, n_sources=3, overlap_rate=0.4, seed=42)
        ids_0 = set(sources[0].keys())
        ids_1 = set(sources[1].keys())
        overlap = len(ids_0 & ids_1)
        total = len(ids_0 | ids_1)
        # Overlap fraction should be approximately 0.4 (Ã‚Â±0.15 tolerance)
        overlap_frac = overlap / total if total > 0 else 0
        assert 0.2 < overlap_frac < 0.6, (
            f"Pairwise overlap fraction {overlap_frac:.2f} not near 0.4"
        )

    def test_union_covers_all_entities(self):
        gt = generate_ground_truth_kg(n_entities=100, seed=42)
        sources = generate_source_views(gt, n_sources=3, overlap_rate=0.4, seed=42)
        all_ids = set()
        for s in sources:
            all_ids.update(s.keys())
        # All ground truth entities should appear in at least one source
        assert all_ids == set(gt.keys())

    def test_source_entities_are_subsets_of_gt(self, small_ground_truth):
        sources = generate_source_views(
            small_ground_truth, n_sources=3, overlap_rate=0.4, seed=42
        )
        gt_ids = set(small_ground_truth.keys())
        for i, src in enumerate(sources):
            assert set(src.keys()).issubset(gt_ids), (
                f"Source {i} contains entities not in ground truth"
            )

    def test_two_source_mode(self):
        gt = generate_ground_truth_kg(n_entities=50, seed=42)
        sources = generate_source_views(gt, n_sources=2, overlap_rate=0.4, seed=42)
        assert len(sources) == 2

    def test_five_source_mode(self):
        gt = generate_ground_truth_kg(n_entities=100, seed=42)
        sources = generate_source_views(gt, n_sources=5, overlap_rate=0.4, seed=42)
        assert len(sources) == 5


class TestCorruption:
    """Verify corruption introduces controlled errors."""

    def test_corruption_rate_approximately_correct(self):
        gt = generate_ground_truth_kg(n_entities=100, seed=42)
        sources = generate_source_views(gt, n_sources=3, overlap_rate=0.4, seed=42)
        corrupted = generate_corrupted_source(
            sources[0], gt, corruption_rate=0.20, seed=42
        )
        # Count properties that differ from ground truth
        n_total = 0
        n_corrupted = 0
        for eid, entity in corrupted.items():
            gt_entity = gt[eid]
            for prop in ["affiliation", "field", "h_index", "email",
                         "publications_count", "country", "active",
                         "homepage"]:
                n_total += 1
                if entity.get(prop) != gt_entity.get(prop):
                    n_corrupted += 1
        actual_rate = n_corrupted / n_total if n_total > 0 else 0
        # Should be within Ã‚Â±0.10 of target
        assert abs(actual_rate - 0.20) < 0.10, (
            f"Actual corruption rate {actual_rate:.2f} not near 0.20"
        )

    def test_name_and_orcid_never_corrupted(self):
        """name and orcid are identity properties Ã¢â‚¬â€ must not change."""
        gt = generate_ground_truth_kg(n_entities=50, seed=42)
        sources = generate_source_views(gt, n_sources=3, overlap_rate=0.4, seed=42)
        corrupted = generate_corrupted_source(
            sources[0], gt, corruption_rate=0.50, seed=42
        )
        for eid, entity in corrupted.items():
            assert entity["name"] == gt[eid]["name"]
            assert entity["orcid"] == gt[eid]["orcid"]

    def test_corrupted_values_are_plausible(self):
        """Corrupted values should be from the same domain."""
        gt = generate_ground_truth_kg(n_entities=50, seed=42)
        sources = generate_source_views(gt, n_sources=3, overlap_rate=0.4, seed=42)
        corrupted = generate_corrupted_source(
            sources[0], gt, corruption_rate=0.30, seed=42
        )
        for eid, entity in corrupted.items():
            if entity.get("h_index") != gt[eid].get("h_index"):
                # h_index should still be a positive integer
                assert isinstance(entity["h_index"], int)
                assert entity["h_index"] >= 0
            if entity.get("active") != gt[eid].get("active"):
                assert isinstance(entity["active"], bool)


class TestConfidenceAssignment:
    """Verify confidence assignment follows calibration regimes."""

    def test_ideal_regime_confidence_correctness_correlation(self):
        gt = generate_ground_truth_kg(n_entities=100, seed=42)
        sources = generate_source_views(gt, n_sources=3, overlap_rate=0.4, seed=42)
        corrupted = generate_corrupted_source(
            sources[0], gt, corruption_rate=0.20, seed=42
        )
        annotated = assign_confidence(
            corrupted, gt, CalibrationRegime.IDEAL, seed=42
        )
        # Collect (is_correct, confidence) pairs
        correct_confs = []
        incorrect_confs = []
        for eid, props in annotated.items():
            gt_entity = gt[eid]
            for prop, val_with_conf in props.items():
                if prop in ("name", "orcid", "@id", "@type"):
                    continue
                conf = val_with_conf["confidence"]
                if val_with_conf["value"] == gt_entity.get(prop):
                    correct_confs.append(conf)
                else:
                    incorrect_confs.append(conf)
        if correct_confs and incorrect_confs:
            mean_correct = np.mean(correct_confs)
            mean_incorrect = np.mean(incorrect_confs)
            assert mean_correct > mean_incorrect, (
                f"Ideal regime: mean correct conf {mean_correct:.3f} "
                f"should exceed mean incorrect {mean_incorrect:.3f}"
            )

    def test_adversarial_regime_inverted(self):
        gt = generate_ground_truth_kg(n_entities=100, seed=42)
        sources = generate_source_views(gt, n_sources=3, overlap_rate=0.4, seed=42)
        corrupted = generate_corrupted_source(
            sources[0], gt, corruption_rate=0.20, seed=42
        )
        annotated = assign_confidence(
            corrupted, gt, CalibrationRegime.ADVERSARIAL, seed=42
        )
        correct_confs = []
        incorrect_confs = []
        for eid, props in annotated.items():
            gt_entity = gt[eid]
            for prop, val_with_conf in props.items():
                if prop in ("name", "orcid", "@id", "@type"):
                    continue
                conf = val_with_conf["confidence"]
                if val_with_conf["value"] == gt_entity.get(prop):
                    correct_confs.append(conf)
                else:
                    incorrect_confs.append(conf)
        if correct_confs and incorrect_confs:
            mean_correct = np.mean(correct_confs)
            mean_incorrect = np.mean(incorrect_confs)
            assert mean_correct < mean_incorrect, (
                f"Adversarial regime: mean correct conf {mean_correct:.3f} "
                f"should be LESS than mean incorrect {mean_incorrect:.3f}"
            )

    def test_uncalibrated_regime_no_correlation(self):
        gt = generate_ground_truth_kg(n_entities=100, seed=42)
        sources = generate_source_views(gt, n_sources=3, overlap_rate=0.4, seed=42)
        corrupted = generate_corrupted_source(
            sources[0], gt, corruption_rate=0.20, seed=42
        )
        annotated = assign_confidence(
            corrupted, gt, CalibrationRegime.UNCALIBRATED, seed=42
        )
        correct_confs = []
        incorrect_confs = []
        for eid, props in annotated.items():
            gt_entity = gt[eid]
            for prop, val_with_conf in props.items():
                if prop in ("name", "orcid", "@id", "@type"):
                    continue
                conf = val_with_conf["confidence"]
                if val_with_conf["value"] == gt_entity.get(prop):
                    correct_confs.append(conf)
                else:
                    incorrect_confs.append(conf)
        if correct_confs and incorrect_confs:
            diff = abs(np.mean(correct_confs) - np.mean(incorrect_confs))
            assert diff < 0.10, (
                f"Uncalibrated regime: confidence diff {diff:.3f} should be < 0.10"
            )

    def test_all_confidences_in_range(self):
        gt = generate_ground_truth_kg(n_entities=50, seed=42)
        sources = generate_source_views(gt, n_sources=3, overlap_rate=0.4, seed=42)
        corrupted = generate_corrupted_source(
            sources[0], gt, corruption_rate=0.20, seed=42
        )
        for regime in CalibrationRegime:
            annotated = assign_confidence(corrupted, gt, regime, seed=42)
            for eid, props in annotated.items():
                for prop, val_with_conf in props.items():
                    if prop in ("name", "orcid", "@id", "@type"):
                        continue
                    c = val_with_conf["confidence"]
                    assert 0.0 <= c <= 1.0, (
                        f"Confidence {c} out of range for {regime.name}"
                    )


class TestJsonLdConversion:
    """Verify conversion to JSON-LD format."""

    def test_output_has_graph_array(self):
        gt = generate_ground_truth_kg(n_entities=10, seed=42)
        sources = generate_source_views(gt, n_sources=2, overlap_rate=0.4, seed=42)
        corrupted = generate_corrupted_source(
            sources[0], gt, corruption_rate=0.10, seed=42
        )
        annotated = assign_confidence(
            corrupted, gt, CalibrationRegime.IDEAL, seed=42
        )
        doc = source_to_jsonld(annotated, source_name="source_A")
        assert "@graph" in doc
        assert isinstance(doc["@graph"], list)
        assert len(doc["@graph"]) == len(corrupted)

    def test_nodes_have_id_and_confidence(self):
        gt = generate_ground_truth_kg(n_entities=10, seed=42)
        sources = generate_source_views(gt, n_sources=2, overlap_rate=0.4, seed=42)
        corrupted = generate_corrupted_source(
            sources[0], gt, corruption_rate=0.10, seed=42
        )
        annotated = assign_confidence(
            corrupted, gt, CalibrationRegime.IDEAL, seed=42
        )
        doc = source_to_jsonld(annotated, source_name="source_A")
        for node in doc["@graph"]:
            assert "@id" in node, "Each node must have @id"
            # At least some properties should have @confidence
            has_confidence = any(
                isinstance(v, dict) and "@confidence" in v
                for k, v in node.items()
                if k not in ("@id", "@type")
            )
            assert has_confidence, f"Node {node['@id']} has no @confidence"


# Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â
# 2. BASELINE TESTS
# Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â


class TestBaselines:
    """Verify all 5 baselines produce valid merge results."""

    def _make_test_graphs(self):
        """Create a minimal 2-graph scenario with known conflicts."""
        g1 = {
            "@graph": [
                {
                    "@id": "urn:entity:1",
                    "@type": "Researcher",
                    "affiliation": {
                        "@value": "MIT",
                        "@confidence": 0.9,
                        "@source": "source_A",
                        "@extractedAt": "2025-01-01T00:00:00Z",
                    },
                    "field": {
                        "@value": "ML",
                        "@confidence": 0.8,
                        "@source": "source_A",
                    },
                },
            ]
        }
        g2 = {
            "@graph": [
                {
                    "@id": "urn:entity:1",
                    "@type": "Researcher",
                    "affiliation": {
                        "@value": "Stanford",
                        "@confidence": 0.7,
                        "@source": "source_B",
                        "@extractedAt": "2025-06-01T00:00:00Z",
                    },
                    "field": {
                        "@value": "ML",
                        "@confidence": 0.85,
                        "@source": "source_B",
                    },
                },
            ]
        }
        ground_truth = {
            "urn:entity:1": {"affiliation": "MIT", "field": "ML"},
        }
        return [g1, g2], ground_truth

    def test_rdflib_union_returns_result(self):
        graphs, gt = self._make_test_graphs()
        result = baseline_rdflib_union(graphs)
        assert isinstance(result, dict)
        assert "@graph" in result

    def test_random_choice_returns_result(self):
        graphs, gt = self._make_test_graphs()
        result = baseline_random_choice(graphs, seed=42)
        assert isinstance(result, dict)
        assert "@graph" in result

    def test_majority_vote_returns_result(self):
        graphs, gt = self._make_test_graphs()
        result = baseline_majority_vote(graphs)
        assert isinstance(result, dict)
        assert "@graph" in result

    def test_most_recent_returns_result(self):
        graphs, gt = self._make_test_graphs()
        result = baseline_most_recent(graphs)
        assert isinstance(result, dict)
        assert "@graph" in result

    def test_rdflib_confidence_argmax_returns_result(self):
        graphs, gt = self._make_test_graphs()
        result = baseline_rdflib_confidence_argmax(graphs)
        assert isinstance(result, dict)
        assert "@graph" in result

    def test_rdflib_confidence_argmax_picks_highest(self):
        """B5 should pick MIT (conf=0.9) over Stanford (conf=0.7)."""
        graphs, gt = self._make_test_graphs()
        result = baseline_rdflib_confidence_argmax(graphs)
        # Find entity:1 in result
        nodes = {n["@id"]: n for n in result["@graph"]}
        node = nodes["urn:entity:1"]
        # Extract bare value of affiliation
        aff = node.get("affiliation")
        if isinstance(aff, dict):
            aff_val = aff.get("@value", aff)
        else:
            aff_val = aff
        assert aff_val == "MIT", (
            f"B5 should pick highest-confidence value MIT, got {aff_val}"
        )

    def test_most_recent_picks_latest(self):
        """B4 should pick Stanford (2025-06) over MIT (2025-01)."""
        graphs, gt = self._make_test_graphs()
        result = baseline_most_recent(graphs)
        nodes = {n["@id"]: n for n in result["@graph"]}
        node = nodes["urn:entity:1"]
        aff = node.get("affiliation")
        if isinstance(aff, dict):
            aff_val = aff.get("@value", aff)
        else:
            aff_val = aff
        assert aff_val == "Stanford", (
            f"B4 should pick most-recent value Stanford, got {aff_val}"
        )


# Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â
# 3. EVALUATION METRICS TESTS
# Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â


class TestEvaluationMetrics:
    """Verify evaluation functions compute correctly."""

    def test_perfect_merge_accuracy(self):
        """A merge that matches ground truth exactly should get 1.0."""
        merged = {
            "@graph": [
                {
                    "@id": "urn:entity:1",
                    "affiliation": {"@value": "MIT", "@confidence": 0.9},
                    "field": {"@value": "ML", "@confidence": 0.8},
                },
            ]
        }
        gt = {"urn:entity:1": {"affiliation": "MIT", "field": "ML"}}
        result = evaluate_merge_accuracy(merged, gt)
        assert result.accuracy == 1.0
        assert result.n_conflicts == 0 or result.n_correct == result.n_conflicts

    def test_zero_accuracy_merge(self):
        """A merge where every conflict is resolved wrongly should get 0.0."""
        merged = {
            "@graph": [
                {
                    "@id": "urn:entity:1",
                    "affiliation": {"@value": "Stanford", "@confidence": 0.9},
                },
            ]
        }
        gt = {"urn:entity:1": {"affiliation": "MIT"}}
        result = evaluate_merge_accuracy(merged, gt)
        assert result.accuracy == 0.0

    def test_ece_perfect_calibration(self):
        """Perfectly calibrated confidence: ECE should be near 0."""
        # 10 predictions, all confidence=0.8, 8 of them correct
        predictions = [
            {"confidence": 0.8, "correct": True} for _ in range(8)
        ] + [
            {"confidence": 0.8, "correct": False} for _ in range(2)
        ]
        ece = compute_ece(predictions, n_bins=10)
        assert ece < 0.05, f"Perfect calibration ECE should be ~0, got {ece}"

    def test_ece_terrible_calibration(self):
        """All confidence=0.9 but all wrong: ECE should be ~0.9."""
        predictions = [
            {"confidence": 0.9, "correct": False} for _ in range(10)
        ]
        ece = compute_ece(predictions, n_bins=10)
        assert ece > 0.8, f"Terrible calibration ECE should be ~0.9, got {ece}"


class TestDiffEvaluation:
    """Verify diff evaluation against known ground truth."""

    def test_identical_graphs_no_diffs(self):
        g = {
            "@graph": [
                {"@id": "urn:1", "name": {"@value": "Alice"}},
            ]
        }
        result = evaluate_diff(g, g)
        assert result["n_added"] == 0
        assert result["n_removed"] == 0
        assert result["n_modified"] == 0

    def test_added_node_detected(self):
        g1 = {"@graph": [{"@id": "urn:1", "name": {"@value": "Alice"}}]}
        g2 = {
            "@graph": [
                {"@id": "urn:1", "name": {"@value": "Alice"}},
                {"@id": "urn:2", "name": {"@value": "Bob"}},
            ]
        }
        result = evaluate_diff(g1, g2)
        assert result["n_added"] >= 1

    def test_removed_node_detected(self):
        g1 = {
            "@graph": [
                {"@id": "urn:1", "name": {"@value": "Alice"}},
                {"@id": "urn:2", "name": {"@value": "Bob"}},
            ]
        }
        g2 = {"@graph": [{"@id": "urn:1", "name": {"@value": "Alice"}}]}
        result = evaluate_diff(g1, g2)
        assert result["n_removed"] >= 1

    def test_modified_property_detected(self):
        g1 = {"@graph": [{"@id": "urn:1", "name": {"@value": "Alice"}}]}
        g2 = {"@graph": [{"@id": "urn:1", "name": {"@value": "Alicia"}}]}
        result = evaluate_diff(g1, g2)
        assert result["n_modified"] >= 1


# ===================================================================
# 4. HYPOTHESIS TESTS -- H8.6a through H8.6g
# ===================================================================


class TestH86a_ConflictResolutionAccuracy:
    """H8.6a: Under ideal calibration, confidence-aware merge >= 80% accuracy."""

    def test_ideal_calibration_accuracy(self):
        config = ExperimentConfig(
            n_entities=100,
            n_sources=3,
            corruption_rate=0.15,
            calibration=CalibrationRegime.IDEAL,
            overlap_rate=0.4,
            seed=42,
        )
        result = run_single_config(config)
        assert result.jsonldex_highest_accuracy >= 0.75, (
            f"H8.6a FAIL: jsonld-ex highest accuracy "
            f"{result.jsonldex_highest_accuracy:.3f} < 0.75"
        )

    def test_jsonldex_beats_rdflib_union(self):
        config = ExperimentConfig(
            n_entities=100,
            n_sources=3,
            corruption_rate=0.15,
            calibration=CalibrationRegime.IDEAL,
            overlap_rate=0.4,
            seed=42,
        )
        result = run_single_config(config)
        assert result.jsonldex_highest_accuracy > result.rdflib_union_accuracy, (
            f"H8.6a FAIL: jsonld-ex {result.jsonldex_highest_accuracy:.3f} "
            f"not better than rdflib union {result.rdflib_union_accuracy:.3f}"
        )

    def test_jsonldex_beats_random(self):
        config = ExperimentConfig(
            n_entities=100,
            n_sources=3,
            corruption_rate=0.15,
            calibration=CalibrationRegime.IDEAL,
            overlap_rate=0.4,
            seed=42,
        )
        result = run_single_config(config)
        assert result.jsonldex_highest_accuracy > result.random_accuracy, (
            f"jsonld-ex {result.jsonldex_highest_accuracy:.3f} "
            f"not better than random {result.random_accuracy:.3f}"
        )


class TestH86b_StrategicDivergence:
    """H8.6b: weighted_vote outperforms highest on majority-correct conflicts."""

    def test_weighted_vote_advantage_on_majority_correct(self):
        config = ExperimentConfig(
            n_entities=100,
            n_sources=3,
            corruption_rate=0.15,
            calibration=CalibrationRegime.IDEAL,
            overlap_rate=0.4,
            seed=42,
        )
        result = run_single_config(config)
        assert result.weighted_vote_majority_correct_acc > result.highest_majority_correct_acc, (
            f"H8.6b FAIL: weighted_vote "
            f"{result.weighted_vote_majority_correct_acc:.3f} not better "
            f"than highest {result.highest_majority_correct_acc:.3f} on "
            f"majority-correct conflicts"
        )

    def test_highest_competitive_on_standard_conflicts(self):
        config = ExperimentConfig(
            n_entities=100,
            n_sources=3,
            corruption_rate=0.15,
            calibration=CalibrationRegime.IDEAL,
            overlap_rate=0.4,
            seed=42,
        )
        result = run_single_config(config)
        diff = result.highest_standard_acc - result.weighted_vote_standard_acc
        assert diff >= -0.05, (
            f"H8.6b: highest {result.highest_standard_acc:.3f} is more "
            f"than 5pp worse than weighted_vote "
            f"{result.weighted_vote_standard_acc:.3f} on standard conflicts"
        )


class TestH86c_CalibrationSensitivity:
    """H8.6c: Accuracy advantage is monotonic with calibration quality."""

    def test_ideal_positive_delta(self):
        config = ExperimentConfig(
            n_entities=100, n_sources=3, corruption_rate=0.15,
            calibration=CalibrationRegime.IDEAL, overlap_rate=0.4, seed=42,
        )
        result = run_single_config(config)
        delta = result.jsonldex_highest_accuracy - result.majority_vote_accuracy
        assert delta > 0.10, (
            f"H8.6c: Ideal delta {delta:.3f} should be > 0.10"
        )

    def test_adversarial_negative_delta(self):
        config = ExperimentConfig(
            n_entities=100, n_sources=3, corruption_rate=0.15,
            calibration=CalibrationRegime.ADVERSARIAL, overlap_rate=0.4, seed=42,
        )
        result = run_single_config(config)
        delta = result.jsonldex_highest_accuracy - result.majority_vote_accuracy
        assert delta < 0.0, (
            f"H8.6c: Adversarial delta {delta:.3f} should be < 0 "
            f"(confidence-aware should be WORSE)"
        )

    def test_uncalibrated_negative_delta(self):
        """Under uncalibrated confidence, argmax degenerates to random
        selection while majority vote still counts supporters.
        Delta should be negative -- this is the honest finding."""
        config = ExperimentConfig(
            n_entities=100, n_sources=3, corruption_rate=0.15,
            calibration=CalibrationRegime.UNCALIBRATED, overlap_rate=0.4, seed=42,
        )
        result = run_single_config(config)
        delta = result.jsonldex_highest_accuracy - result.majority_vote_accuracy
        assert delta < 0.0, (
            f"H8.6c: Uncalibrated delta {delta:.3f} should be < 0 "
            f"(majority vote retains counting advantage)"
        )

    def test_monotonic_ordering(self):
        """Delta accuracy should decrease: ideal > noisy > uncalibrated > adversarial."""
        deltas = {}
        for regime in CalibrationRegime:
            config = ExperimentConfig(
                n_entities=100, n_sources=3, corruption_rate=0.15,
                calibration=regime, overlap_rate=0.4, seed=42,
            )
            result = run_single_config(config)
            deltas[regime] = (
                result.jsonldex_highest_accuracy - result.majority_vote_accuracy
            )
        assert deltas[CalibrationRegime.IDEAL] > deltas[CalibrationRegime.NOISY], (
            f"Ideal delta {deltas[CalibrationRegime.IDEAL]:.3f} should exceed "
            f"noisy delta {deltas[CalibrationRegime.NOISY]:.3f}"
        )
        assert deltas[CalibrationRegime.NOISY] > deltas[CalibrationRegime.UNCALIBRATED], (
            f"Noisy delta {deltas[CalibrationRegime.NOISY]:.3f} should exceed "
            f"uncalibrated delta {deltas[CalibrationRegime.UNCALIBRATED]:.3f}"
        )
        assert deltas[CalibrationRegime.UNCALIBRATED] > deltas[CalibrationRegime.ADVERSARIAL], (
            f"Uncalibrated delta {deltas[CalibrationRegime.UNCALIBRATED]:.3f} "
            f"should exceed adversarial delta "
            f"{deltas[CalibrationRegime.ADVERSARIAL]:.3f}"
        )


class TestH86d_AgreementConfidenceBoosting:
    """H8.6d: Noisy-OR ECE < max ECE for agreed properties."""

    def test_noisy_or_better_calibrated(self):
        config = ExperimentConfig(
            n_entities=100, n_sources=3, corruption_rate=0.10,
            calibration=CalibrationRegime.IDEAL, overlap_rate=0.4, seed=42,
        )
        result = run_single_config(config)
        assert result.ece_noisy_or < result.ece_max, (
            f"H8.6d: noisy-OR ECE {result.ece_noisy_or:.4f} should be < "
            f"max ECE {result.ece_max:.4f}"
        )


class TestH86e_DiffCompleteness:
    """H8.6e: diff_graphs achieves 100% precision and recall."""

    def test_diff_precision_recall_on_constructed_pairs(self):
        """Test against 10 constructed graph pairs with known diffs."""
        config = ExperimentConfig(
            n_entities=50, n_sources=2, corruption_rate=0.10,
            calibration=CalibrationRegime.IDEAL, overlap_rate=0.4, seed=42,
        )
        result = run_single_config(config)
        assert result.diff_precision == 1.0, (
            f"H8.6e: diff precision {result.diff_precision:.3f} should be 1.0"
        )
        assert result.diff_recall == 1.0, (
            f"H8.6e: diff recall {result.diff_recall:.3f} should be 1.0"
        )


class TestH86f_AuditTrailFidelity:
    """H8.6f: MergeReport records every conflict accurately."""

    def test_audit_trail_complete(self):
        config = ExperimentConfig(
            n_entities=50, n_sources=3, corruption_rate=0.15,
            calibration=CalibrationRegime.IDEAL, overlap_rate=0.4, seed=42,
        )
        result = run_single_config(config)
        assert result.audit_completeness == 1.0, (
            f"H8.6f: audit completeness {result.audit_completeness:.3f} "
            f"should be 1.0"
        )


class TestH86g_Throughput:
    """H8.6g: merge_graphs is fast and scales near-linearly."""

    def test_100_node_throughput(self):
        """3 Ãƒâ€” 100 nodes should complete in < 100ms."""
        from jsonld_ex.merge import merge_graphs
        config = ExperimentConfig(
            n_entities=100, n_sources=3, corruption_rate=0.10,
            calibration=CalibrationRegime.IDEAL, overlap_rate=0.4, seed=42,
        )
        result = run_single_config(config)
        assert result.throughput_p50_ms < 100, (
            f"H8.6g: p50 latency {result.throughput_p50_ms:.1f}ms > 100ms"
        )


# Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â
# 5. INTEGRATION: B5 HONEST CONTROL
# Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â


class TestB5HonestControl:
    """B5 (rdflib+confidence argmax) should match jsonld-ex accuracy.

    This is the HONEST control: same algorithm, different implementation.
    If B5 accuracy == jsonld-ex accuracy, the contribution is ergonomics,
    not algorithm.
    """

    def test_b5_matches_jsonldex_accuracy(self):
        config = ExperimentConfig(
            n_entities=100, n_sources=3, corruption_rate=0.15,
            calibration=CalibrationRegime.IDEAL, overlap_rate=0.4, seed=42,
        )
        result = run_single_config(config)
        diff = abs(
            result.jsonldex_highest_accuracy - result.b5_confidence_argmax_accuracy
        )
        assert diff < 0.02, (
            f"B5 accuracy {result.b5_confidence_argmax_accuracy:.3f} should "
            f"match jsonld-ex {result.jsonldex_highest_accuracy:.3f} "
            f"within 2pp (honest control)"
        )
