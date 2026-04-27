"""Tests for the Meridian knowledge base generator.

TDD: These tests are written BEFORE the implementation.
They define the contract that knowledge_base.py must satisfy.
"""

import pytest
import math
from collections import Counter


class TestFactStructure:
    """Each fact must have the required fields with valid values."""

    def test_fact_has_required_fields(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        required_fields = {
            "id", "entity_id", "entity_name", "entity_type",
            "relation", "value", "tier", "opinion", "provenance",
        }
        for fact in kb["facts"]:
            missing = required_fields - set(fact.keys())
            assert not missing, f"Fact {fact.get('id', '?')} missing fields: {missing}"

    def test_fact_ids_are_unique(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        ids = [f["id"] for f in kb["facts"]]
        assert len(ids) == len(set(ids)), "Duplicate fact IDs found"

    def test_entity_ids_are_namespaced(self, small_kb_config):
        """Entity IDs should be URN-style: urn:meridian:<type>:<slug>"""
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            assert fact["entity_id"].startswith("urn:meridian:"), (
                f"Entity ID {fact['entity_id']} not URN-namespaced"
            )

    def test_fact_count_matches_config(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        assert len(kb["facts"]) == small_kb_config["total_facts"]


class TestSLOpinionValidity:
    """Every SL opinion must satisfy the mathematical constraints."""

    def test_opinion_components_sum_to_one(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            op = fact["opinion"]
            total = op["belief"] + op["disbelief"] + op["uncertainty"]
            assert abs(total - 1.0) < 1e-9, (
                f"Fact {fact['id']}: b+d+u = {total}, not 1.0"
            )

    def test_opinion_components_non_negative(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            op = fact["opinion"]
            for key in ("belief", "disbelief", "uncertainty"):
                assert op[key] >= 0.0, (
                    f"Fact {fact['id']}: {key} = {op[key]} is negative"
                )

    def test_opinion_components_at_most_one(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            op = fact["opinion"]
            for key in ("belief", "disbelief", "uncertainty"):
                assert op[key] <= 1.0, (
                    f"Fact {fact['id']}: {key} = {op[key]} exceeds 1.0"
                )

    def test_base_rate_in_unit_interval(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            a = fact["opinion"]["base_rate"]
            assert 0.0 <= a <= 1.0, (
                f"Fact {fact['id']}: base_rate = {a} outside [0, 1]"
            )

    def test_opinion_has_all_four_fields(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            op = fact["opinion"]
            for key in ("belief", "disbelief", "uncertainty", "base_rate"):
                assert key in op, f"Fact {fact['id']} opinion missing '{key}'"


class TestConfidenceTierAssignment:
    """Facts must be assigned tiers with correct opinion ranges."""

    def test_valid_tiers_only(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        valid_tiers = {"T1_established", "T2_probable", "T3_uncertain",
                       "T4_speculative", "T5_contested"}
        for fact in kb["facts"]:
            assert fact["tier"] in valid_tiers, (
                f"Fact {fact['id']} has invalid tier: {fact['tier']}"
            )

    def test_tier_distribution_matches_config(self, small_kb_config):
        """Tier fractions should be within ±10% of configured fractions."""
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        total = len(kb["facts"])
        tier_counts = Counter(f["tier"] for f in kb["facts"])

        for tier_name, tier_cfg in small_kb_config["confidence_tiers"].items():
            expected_frac = tier_cfg["fraction"]
            actual_frac = tier_counts.get(tier_name, 0) / total
            tolerance = 0.10  # Allow ±10% deviation for small sample sizes
            assert abs(actual_frac - expected_frac) <= tolerance, (
                f"Tier {tier_name}: expected ~{expected_frac:.0%}, "
                f"got {actual_frac:.0%}"
            )

    def test_t1_belief_in_range(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            if fact["tier"] == "T1_established":
                b = fact["opinion"]["belief"]
                assert 0.90 <= b <= 0.99, (
                    f"T1 fact {fact['id']} has belief={b}, outside [0.90, 0.99]"
                )

    def test_t2_belief_in_range(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            if fact["tier"] == "T2_probable":
                b = fact["opinion"]["belief"]
                assert 0.70 <= b <= 0.89, (
                    f"T2 fact {fact['id']} has belief={b}, outside [0.70, 0.89]"
                )

    def test_t3_belief_in_range(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            if fact["tier"] == "T3_uncertain":
                b = fact["opinion"]["belief"]
                assert 0.40 <= b <= 0.69, (
                    f"T3 fact {fact['id']} has belief={b}, outside [0.40, 0.69]"
                )

    def test_t4_belief_in_range(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            if fact["tier"] == "T4_speculative":
                b = fact["opinion"]["belief"]
                assert 0.15 <= b <= 0.39, (
                    f"T4 fact {fact['id']} has belief={b}, outside [0.15, 0.39]"
                )

    def test_t5_has_significant_disbelief(self, small_kb_config):
        """T5 (contested) facts must have both belief and disbelief."""
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            if fact["tier"] == "T5_contested":
                b = fact["opinion"]["belief"]
                d = fact["opinion"]["disbelief"]
                assert 0.30 <= b <= 0.60, (
                    f"T5 fact {fact['id']} belief={b}, outside [0.30, 0.60]"
                )
                assert 0.20 <= d <= 0.50, (
                    f"T5 fact {fact['id']} disbelief={d}, outside [0.20, 0.50]"
                )

    def test_t5_has_low_uncertainty(self, small_kb_config):
        """T5 facts represent CONFLICT, not ignorance — uncertainty should be low."""
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            if fact["tier"] == "T5_contested":
                u = fact["opinion"]["uncertainty"]
                assert u <= 0.30, (
                    f"T5 fact {fact['id']} has u={u}, too high for contested "
                    f"(conflict = low u, not high u)"
                )


class TestProvenanceMetadata:
    """Each fact must have provenance with source identifiers."""

    def test_provenance_has_sources(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            prov = fact["provenance"]
            assert "sources" in prov, f"Fact {fact['id']} missing provenance.sources"
            assert len(prov["sources"]) >= 1, (
                f"Fact {fact['id']} has no sources"
            )

    def test_source_ids_are_urns(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            for src in fact["provenance"]["sources"]:
                assert src["id"].startswith("urn:meridian:source:"), (
                    f"Source ID {src['id']} not URN-namespaced"
                )

    def test_provenance_has_method(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            prov = fact["provenance"]
            assert "method" in prov, f"Fact {fact['id']} missing provenance.method"

    def test_high_confidence_facts_have_multiple_sources(self, small_kb_config):
        """T1 facts should have >=2 corroborating sources."""
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            if fact["tier"] == "T1_established":
                n_sources = len(fact["provenance"]["sources"])
                assert n_sources >= 2, (
                    f"T1 fact {fact['id']} has only {n_sources} source(s), "
                    f"expected >=2 for high confidence"
                )


class TestTemporalMetadata:
    """Some facts should have temporal validity windows."""

    def test_some_facts_have_temporal(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        temporal_count = sum(
            1 for f in kb["facts"] if f.get("valid_from") is not None
        )
        # Protocol says ~40% should have temporal metadata
        total = len(kb["facts"])
        fraction = temporal_count / total
        assert 0.25 <= fraction <= 0.55, (
            f"Expected ~40% temporal facts, got {fraction:.0%}"
        )

    def test_valid_from_is_iso_date(self, small_kb_config):
        """Temporal dates must be ISO 8601 format."""
        from src.knowledge_base import generate_knowledge_base
        from datetime import date

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            if fact.get("valid_from") is not None:
                # Should not raise
                date.fromisoformat(fact["valid_from"])

    def test_valid_until_after_valid_from(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base
        from datetime import date

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            if fact.get("valid_from") and fact.get("valid_until"):
                vf = date.fromisoformat(fact["valid_from"])
                vu = date.fromisoformat(fact["valid_until"])
                assert vu > vf, (
                    f"Fact {fact['id']}: valid_until ({vu}) not after "
                    f"valid_from ({vf})"
                )


class TestDeterminism:
    """KB generation must be deterministic given the same seed."""

    def test_same_seed_same_output(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb1 = generate_knowledge_base(small_kb_config)
        kb2 = generate_knowledge_base(small_kb_config)

        assert len(kb1["facts"]) == len(kb2["facts"])
        for f1, f2 in zip(kb1["facts"], kb2["facts"]):
            assert f1["id"] == f2["id"]
            assert f1["entity_name"] == f2["entity_name"]
            assert f1["opinion"] == f2["opinion"]

    def test_different_seed_different_output(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        cfg1 = {**small_kb_config, "seed": 42}
        cfg2 = {**small_kb_config, "seed": 999}
        kb1 = generate_knowledge_base(cfg1)
        kb2 = generate_knowledge_base(cfg2)

        # At least some entity names should differ
        names1 = {f["entity_name"] for f in kb1["facts"]}
        names2 = {f["entity_name"] for f in kb2["facts"]}
        assert names1 != names2, "Different seeds produced identical entity names"


class TestDataSplits:
    """The KB must split into train/val/test_id/test_ood correctly."""

    def test_split_produces_correct_keys(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base, split_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        splits = split_knowledge_base(kb, small_kb_config)
        assert set(splits.keys()) == {"train", "val", "test_id", "test_ood"}

    def test_split_sizes_match_config(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base, split_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        splits = split_knowledge_base(kb, small_kb_config)

        for split_name, expected_size in small_kb_config["splits"].items():
            actual_size = len(splits[split_name])
            assert actual_size == expected_size, (
                f"Split '{split_name}': expected {expected_size}, got {actual_size}"
            )

    def test_splits_are_disjoint(self, small_kb_config):
        """No fact should appear in more than one split."""
        from src.knowledge_base import generate_knowledge_base, split_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        splits = split_knowledge_base(kb, small_kb_config)

        all_ids = []
        for split_facts in splits.values():
            all_ids.extend(f["id"] for f in split_facts)
        assert len(all_ids) == len(set(all_ids)), "Facts appear in multiple splits"

    def test_splits_cover_all_facts(self, small_kb_config):
        """All facts must appear in exactly one split."""
        from src.knowledge_base import generate_knowledge_base, split_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        splits = split_knowledge_base(kb, small_kb_config)

        total_in_splits = sum(len(s) for s in splits.values())
        assert total_in_splits == len(kb["facts"])

    def test_test_ood_has_different_entity_types(self, small_kb_config):
        """OOD test set must have entity types NOT in training set."""
        from src.knowledge_base import generate_knowledge_base, split_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        splits = split_knowledge_base(kb, small_kb_config)

        train_types = {f["entity_type"] for f in splits["train"]}
        ood_types = {f["entity_type"] for f in splits["test_ood"]}

        # OOD types should have NO overlap with train types
        overlap = train_types & ood_types
        assert len(overlap) == 0, (
            f"OOD test set shares entity types with train: {overlap}"
        )

    def test_all_tiers_represented_in_train(self, small_kb_config):
        """Training set must contain facts from all 5 tiers."""
        from src.knowledge_base import generate_knowledge_base, split_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        splits = split_knowledge_base(kb, small_kb_config)

        train_tiers = {f["tier"] for f in splits["train"]}
        expected = {"T1_established", "T2_probable", "T3_uncertain",
                    "T4_speculative", "T5_contested"}
        missing = expected - train_tiers
        assert not missing, f"Training set missing tiers: {missing}"

    def test_all_tiers_represented_in_test(self, small_kb_config):
        """Both test sets must contain facts from all 5 tiers for evaluation."""
        from src.knowledge_base import generate_knowledge_base, split_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        splits = split_knowledge_base(kb, small_kb_config)

        for test_split in ("test_id", "test_ood"):
            test_tiers = {f["tier"] for f in splits[test_split]}
            expected = {"T1_established", "T2_probable", "T3_uncertain",
                        "T4_speculative", "T5_contested"}
            missing = expected - test_tiers
            assert not missing, (
                f"Test split '{test_split}' missing tiers: {missing}"
            )


class TestEntityGeneration:
    """Entity names and types must be well-formed."""

    def test_entity_types_are_valid(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        valid_types = {
            "Organization", "Person", "Location", "ResearchFinding",
            "Product", "Event",
            # OOD types (only in test_ood)
            "SportsTeam", "GeologicalFeature", "ArtWork",
        }
        for fact in kb["facts"]:
            assert fact["entity_type"] in valid_types, (
                f"Unknown entity type: {fact['entity_type']}"
            )

    def test_entity_names_are_nonempty_strings(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            assert isinstance(fact["entity_name"], str)
            assert len(fact["entity_name"].strip()) > 0

    def test_relations_are_nonempty_strings(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            assert isinstance(fact["relation"], str)
            assert len(fact["relation"].strip()) > 0

    def test_values_are_nonempty_strings(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            assert isinstance(fact["value"], str)
            assert len(fact["value"].strip()) > 0


class TestQuestionGeneration:
    """Each fact should map to at least one natural-language question."""

    def test_facts_have_questions(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            assert "questions" in fact, f"Fact {fact['id']} has no questions"
            assert len(fact["questions"]) >= 1, (
                f"Fact {fact['id']} has empty questions list"
            )

    def test_questions_are_strings(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            for q in fact["questions"]:
                assert isinstance(q, str)
                assert q.strip().endswith("?"), (
                    f"Question doesn't end with '?': {q}"
                )

    def test_questions_contain_entity_reference(self, small_kb_config):
        """Each question should reference the entity by name."""
        from src.knowledge_base import generate_knowledge_base

        kb = generate_knowledge_base(small_kb_config)
        for fact in kb["facts"]:
            for q in fact["questions"]:
                assert fact["entity_name"].lower() in q.lower() or \
                       any(word in q.lower()
                           for word in fact["entity_name"].lower().split()), (
                    f"Question '{q}' doesn't reference entity "
                    f"'{fact['entity_name']}'"
                )


class TestKBSerialization:
    """KB must be serializable to and from JSON."""

    def test_roundtrip_json(self, small_kb_config, tmp_path):
        import json
        from src.knowledge_base import (
            generate_knowledge_base, save_knowledge_base, load_knowledge_base,
        )

        kb = generate_knowledge_base(small_kb_config)
        path = tmp_path / "test_kb.json"
        save_knowledge_base(kb, path)
        kb_loaded = load_knowledge_base(path)

        assert len(kb_loaded["facts"]) == len(kb["facts"])
        for f1, f2 in zip(kb["facts"], kb_loaded["facts"]):
            assert f1["id"] == f2["id"]
            assert f1["opinion"] == f2["opinion"]
