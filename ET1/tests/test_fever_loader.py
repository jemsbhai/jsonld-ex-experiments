"""Tests for the FEVER dataset loader and SL opinion derivation.

TDD: Written BEFORE fever_loader.py implementation.
Unit tests use mock data — no FEVER download required.
Integration tests (marked @pytest.mark.integration) use real data.
"""

import pytest
import math


# ---------------------------------------------------------------------------
# Mock FEVER data for unit tests (no download required)
# ---------------------------------------------------------------------------

def _mock_fever_examples():
    """Minimal mock of FEVER HuggingFace dataset rows."""
    return [
        {
            "id": 75397,
            "label": "SUPPORTS",
            "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
            "evidence_annotation_id": 92206,
            "evidence_id": 104971,
            "evidence_wiki_url": "Nikolaj_Coster-Waldau",
            "evidence_sentence_id": 7,
        },
        {
            "id": 75397,
            "label": "SUPPORTS",
            "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
            "evidence_annotation_id": 92207,
            "evidence_id": 104972,
            "evidence_wiki_url": "Fox_Broadcasting_Company",
            "evidence_sentence_id": 3,
        },
        {
            "id": 150000,
            "label": "REFUTES",
            "claim": "The Colosseum is in Paris.",
            "evidence_annotation_id": 200001,
            "evidence_id": 300001,
            "evidence_wiki_url": "Colosseum",
            "evidence_sentence_id": 0,
        },
        {
            "id": 200000,
            "label": "NOT ENOUGH INFO",
            "claim": "Tilda Swinton is a vegan.",
            "evidence_annotation_id": -1,
            "evidence_id": -1,
            "evidence_wiki_url": "",
            "evidence_sentence_id": -1,
        },
        {
            "id": 200001,
            "label": "NOT ENOUGH INFO",
            "claim": "The population of Mars exceeds 1 million.",
            "evidence_annotation_id": -1,
            "evidence_id": -1,
            "evidence_wiki_url": "",
            "evidence_sentence_id": -1,
        },
        {
            "id": 300000,
            "label": "SUPPORTS",
            "claim": "Python is a programming language.",
            "evidence_annotation_id": 400001,
            "evidence_id": 500001,
            "evidence_wiki_url": "Python_(programming_language)",
            "evidence_sentence_id": 0,
        },
        {
            "id": 300001,
            "label": "REFUTES",
            "claim": "The Earth is flat.",
            "evidence_annotation_id": 400002,
            "evidence_id": 500002,
            "evidence_wiki_url": "Earth",
            "evidence_sentence_id": 0,
        },
        {
            "id": 300001,
            "label": "REFUTES",
            "claim": "The Earth is flat.",
            "evidence_annotation_id": 400003,
            "evidence_id": 500003,
            "evidence_wiki_url": "Flat_Earth",
            "evidence_sentence_id": 1,
        },
    ]


# ---------------------------------------------------------------------------
# SL Opinion Derivation Tests
# ---------------------------------------------------------------------------

class TestSLDerivationFromLabel:
    """derive_fever_opinion must produce valid SL opinions for each label."""

    def test_supports_has_high_belief(self):
        from src.fever_loader import derive_fever_opinion

        op = derive_fever_opinion(label="SUPPORTS", evidence_count=1)
        assert op.belief >= 0.80, f"SUPPORTS should have high belief, got {op.belief}"
        assert op.disbelief < 0.10

    def test_refutes_has_high_disbelief(self):
        from src.fever_loader import derive_fever_opinion

        op = derive_fever_opinion(label="REFUTES", evidence_count=1)
        assert op.disbelief >= 0.70, f"REFUTES should have high disbelief, got {op.disbelief}"
        assert op.belief < 0.15

    def test_nei_has_high_uncertainty(self):
        from src.fever_loader import derive_fever_opinion

        op = derive_fever_opinion(label="NOT ENOUGH INFO", evidence_count=0)
        assert op.uncertainty >= 0.50, (
            f"NEI should have high uncertainty, got {op.uncertainty}"
        )

    def test_all_labels_produce_valid_opinions(self):
        from src.fever_loader import derive_fever_opinion

        for label, ev_count in [
            ("SUPPORTS", 1), ("SUPPORTS", 3),
            ("REFUTES", 1), ("REFUTES", 2),
            ("NOT ENOUGH INFO", 0),
        ]:
            op = derive_fever_opinion(label=label, evidence_count=ev_count)
            total = op.belief + op.disbelief + op.uncertainty
            assert abs(total - 1.0) < 1e-9, (
                f"label={label}, ev={ev_count}: b+d+u={total}"
            )
            for attr in ("belief", "disbelief", "uncertainty"):
                val = getattr(op, attr)
                assert 0.0 <= val <= 1.0, (
                    f"label={label}: {attr}={val} out of range"
                )


class TestEvidenceModulation:
    """More evidence sentences should increase confidence for SUPPORTS/REFUTES."""

    def test_more_evidence_increases_supports_belief(self):
        from src.fever_loader import derive_fever_opinion

        op1 = derive_fever_opinion(label="SUPPORTS", evidence_count=1)
        op3 = derive_fever_opinion(label="SUPPORTS", evidence_count=3)
        assert op3.belief >= op1.belief, (
            f"3 evidence sentences should have >= belief than 1: "
            f"{op3.belief} vs {op1.belief}"
        )

    def test_more_evidence_increases_refutes_disbelief(self):
        from src.fever_loader import derive_fever_opinion

        op1 = derive_fever_opinion(label="REFUTES", evidence_count=1)
        op2 = derive_fever_opinion(label="REFUTES", evidence_count=2)
        assert op2.disbelief >= op1.disbelief, (
            f"2 evidence sentences should have >= disbelief than 1: "
            f"{op2.disbelief} vs {op1.disbelief}"
        )

    def test_evidence_capped(self):
        """Belief/disbelief should not exceed 0.99 regardless of evidence count."""
        from src.fever_loader import derive_fever_opinion

        op = derive_fever_opinion(label="SUPPORTS", evidence_count=100)
        assert op.belief <= 0.99
        assert op.uncertainty >= 0.0

        op2 = derive_fever_opinion(label="REFUTES", evidence_count=100)
        assert op2.disbelief <= 0.99


class TestFEVERTierAssignment:
    """Each FEVER fact should receive a meaningful confidence tier."""

    def test_supports_gets_high_tier(self):
        from src.fever_loader import derive_fever_tier

        tier = derive_fever_tier(label="SUPPORTS", evidence_count=2)
        assert tier in ("T1_established", "T2_probable")

    def test_refutes_gets_contested_tier(self):
        from src.fever_loader import derive_fever_tier

        tier = derive_fever_tier(label="REFUTES", evidence_count=1)
        assert tier == "T5_contested"

    def test_nei_gets_uncertain_tier(self):
        from src.fever_loader import derive_fever_tier

        tier = derive_fever_tier(label="NOT ENOUGH INFO", evidence_count=0)
        assert tier in ("T3_uncertain", "T4_speculative")


# ---------------------------------------------------------------------------
# Claim Grouping Tests
# ---------------------------------------------------------------------------

class TestClaimGrouping:
    """FEVER rows with same claim ID must be grouped to count evidence."""

    def test_group_by_claim_id(self):
        from src.fever_loader import group_fever_rows

        rows = _mock_fever_examples()
        grouped = group_fever_rows(rows)

        # Claim 75397 has 2 evidence rows
        assert 75397 in grouped
        assert grouped[75397]["evidence_count"] == 2
        assert len(grouped[75397]["evidence_wiki_urls"]) == 2

    def test_nei_has_zero_evidence(self):
        from src.fever_loader import group_fever_rows

        rows = _mock_fever_examples()
        grouped = group_fever_rows(rows)

        assert grouped[200000]["evidence_count"] == 0

    def test_group_preserves_label(self):
        from src.fever_loader import group_fever_rows

        rows = _mock_fever_examples()
        grouped = group_fever_rows(rows)

        assert grouped[75397]["label"] == "SUPPORTS"
        assert grouped[150000]["label"] == "REFUTES"
        assert grouped[200000]["label"] == "NOT ENOUGH INFO"


# ---------------------------------------------------------------------------
# Fact Conversion Tests
# ---------------------------------------------------------------------------

class TestFEVERToFacts:
    """Grouped FEVER claims must convert to valid Fact objects."""

    def test_converts_to_fact_objects(self):
        from src.fever_loader import fever_rows_to_facts
        from src.fact import Fact

        rows = _mock_fever_examples()
        facts = fever_rows_to_facts(rows, max_facts=10, seed=42)

        assert len(facts) > 0
        for f in facts:
            assert isinstance(f, Fact)
            assert f.dataset == "fever"

    def test_fact_question_is_claim_verification(self):
        from src.fever_loader import fever_rows_to_facts

        rows = _mock_fever_examples()
        facts = fever_rows_to_facts(rows, max_facts=10, seed=42)

        for f in facts:
            # Question should ask about the claim
            assert "?" in f.question
            assert len(f.question) > 10

    def test_fact_answer_reflects_label(self):
        from src.fever_loader import fever_rows_to_facts

        rows = _mock_fever_examples()
        facts = fever_rows_to_facts(rows, max_facts=10, seed=42)

        for f in facts:
            assert len(f.answer) > 0

    def test_fact_provenance_has_wikipedia_sources(self):
        from src.fever_loader import fever_rows_to_facts

        rows = _mock_fever_examples()
        facts = fever_rows_to_facts(rows, max_facts=10, seed=42)

        for f in facts:
            if f.tier != "T3_uncertain" and f.tier != "T4_speculative":
                # SUPPORTS and REFUTES should have Wikipedia sources
                assert len(f.provenance.sources) >= 1
                for s in f.provenance.sources:
                    assert "wikipedia" in s.id.lower() or "urn:" in s.id

    def test_fact_opinion_valid(self):
        from src.fever_loader import fever_rows_to_facts

        rows = _mock_fever_examples()
        facts = fever_rows_to_facts(rows, max_facts=10, seed=42)

        for f in facts:
            total = f.opinion.belief + f.opinion.disbelief + f.opinion.uncertainty
            assert abs(total - 1.0) < 1e-9

    def test_max_facts_respected(self):
        from src.fever_loader import fever_rows_to_facts

        rows = _mock_fever_examples()
        facts = fever_rows_to_facts(rows, max_facts=3, seed=42)
        assert len(facts) <= 3

    def test_deterministic_output(self):
        from src.fever_loader import fever_rows_to_facts

        rows = _mock_fever_examples()
        facts1 = fever_rows_to_facts(rows, max_facts=5, seed=42)
        facts2 = fever_rows_to_facts(rows, max_facts=5, seed=42)

        assert len(facts1) == len(facts2)
        for f1, f2 in zip(facts1, facts2):
            assert f1.id == f2.id
            assert f1.opinion.belief == f2.opinion.belief


class TestLabelBalance:
    """When subsampling, we should maintain rough label balance."""

    def test_all_labels_represented(self):
        from src.fever_loader import fever_rows_to_facts

        rows = _mock_fever_examples()
        facts = fever_rows_to_facts(rows, max_facts=6, seed=42)

        tiers = {f.tier for f in facts}
        # Should have at least 2 distinct tiers from our small mock data
        assert len(tiers) >= 2, f"Only got tiers: {tiers}"
