"""Tests for the common Fact dataclass.

TDD: Written BEFORE fact.py implementation.
The Fact is the bridge between data sources (Meridian, FEVER, SQuAD)
and the 7-condition formatter. Its invariants MUST hold regardless of source.
"""

import pytest
import math


class TestFactCreation:
    """Fact must be constructable with all required fields."""

    def test_create_minimal_fact(self):
        from src.fact import Fact, SLOpinion, Provenance, Source

        opinion = SLOpinion(belief=0.9, disbelief=0.02, uncertainty=0.08)
        source = Source(id="urn:test:src1", name="Test Source", reliability=0.9)
        provenance = Provenance(sources=[source], method="test_method")

        fact = Fact(
            id="F00001",
            question="Where is NovaTech headquartered?",
            answer="Cedarpoint, Valoria",
            entity_name="NovaTech Industries",
            entity_type="Organization",
            relation="headquartered_in",
            opinion=opinion,
            provenance=provenance,
            dataset="meridian",
        )
        assert fact.id == "F00001"
        assert fact.answer == "Cedarpoint, Valoria"
        assert fact.dataset == "meridian"

    def test_fact_optional_temporal_fields(self):
        from src.fact import Fact, SLOpinion, Provenance, Source

        opinion = SLOpinion(belief=0.9, disbelief=0.02, uncertainty=0.08)
        source = Source(id="urn:test:src1", name="Test", reliability=0.9)
        provenance = Provenance(sources=[source], method="test")

        # Without temporal
        fact = Fact(
            id="F1", question="Q?", answer="A",
            entity_name="E", entity_type="Organization",
            relation="rel", opinion=opinion,
            provenance=provenance, dataset="meridian",
        )
        assert fact.valid_from is None
        assert fact.valid_until is None

        # With temporal
        fact_t = Fact(
            id="F2", question="Q?", answer="A",
            entity_name="E", entity_type="Organization",
            relation="rel", opinion=opinion,
            provenance=provenance, dataset="fever",
            valid_from="2020-01-01", valid_until="2025-12-31",
        )
        assert fact_t.valid_from == "2020-01-01"

    def test_fact_optional_context_field(self):
        """SQuAD facts need a context paragraph; others don't."""
        from src.fact import Fact, SLOpinion, Provenance, Source

        opinion = SLOpinion(belief=0.9, disbelief=0.02, uncertainty=0.08)
        source = Source(id="urn:test:src1", name="Test", reliability=0.9)
        provenance = Provenance(sources=[source], method="test")

        fact = Fact(
            id="F1", question="Q?", answer="A",
            entity_name="E", entity_type="Answer",
            relation="extractive_qa", opinion=opinion,
            provenance=provenance, dataset="squad",
            context="Normandy is a region in France. Its capital is Rouen.",
        )
        assert fact.context is not None
        assert "Normandy" in fact.context

    def test_fact_dataset_must_be_valid(self):
        from src.fact import Fact, SLOpinion, Provenance, Source

        opinion = SLOpinion(belief=0.9, disbelief=0.02, uncertainty=0.08)
        source = Source(id="urn:test:src1", name="Test", reliability=0.9)
        provenance = Provenance(sources=[source], method="test")

        with pytest.raises(ValueError, match="dataset"):
            Fact(
                id="F1", question="Q?", answer="A",
                entity_name="E", entity_type="Organization",
                relation="rel", opinion=opinion,
                provenance=provenance, dataset="imagenet",
            )


class TestSLOpinionInvariants:
    """SL opinion must satisfy mathematical constraints regardless of source."""

    def test_components_sum_to_one(self):
        from src.fact import SLOpinion

        op = SLOpinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        total = op.belief + op.disbelief + op.uncertainty
        assert abs(total - 1.0) < 1e-9

    def test_rejects_components_not_summing_to_one(self):
        from src.fact import SLOpinion

        with pytest.raises(ValueError, match="sum to 1"):
            SLOpinion(belief=0.7, disbelief=0.1, uncertainty=0.3)

    def test_rejects_negative_components(self):
        from src.fact import SLOpinion

        with pytest.raises(ValueError, match="non-negative"):
            SLOpinion(belief=-0.1, disbelief=0.5, uncertainty=0.6)

    def test_rejects_components_above_one(self):
        from src.fact import SLOpinion

        with pytest.raises(ValueError, match="non-negative"):
            SLOpinion(belief=1.1, disbelief=0.0, uncertainty=-0.1)

    def test_base_rate_default(self):
        from src.fact import SLOpinion

        op = SLOpinion(belief=0.5, disbelief=0.2, uncertainty=0.3)
        assert op.base_rate == 0.5

    def test_base_rate_custom(self):
        from src.fact import SLOpinion

        op = SLOpinion(belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.3)
        assert op.base_rate == 0.3

    def test_projected_probability(self):
        """P(ω) = b + a·u per Subjective Logic."""
        from src.fact import SLOpinion

        op = SLOpinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
        expected = 0.7 + 0.5 * 0.2  # = 0.8
        assert abs(op.projected_probability - expected) < 1e-9

    def test_to_dict_roundtrip(self):
        from src.fact import SLOpinion

        op = SLOpinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.4)
        d = op.to_dict()
        assert d == {
            "belief": 0.7,
            "disbelief": 0.1,
            "uncertainty": 0.2,
            "base_rate": 0.4,
        }
        op2 = SLOpinion.from_dict(d)
        assert op2.belief == op.belief
        assert op2.projected_probability == op.projected_probability

    def test_near_boundary_opinion(self):
        """Edge case: vacuous opinion (total ignorance)."""
        from src.fact import SLOpinion

        op = SLOpinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        assert op.projected_probability == 0.5  # = 0 + 0.5 * 1.0

    def test_dogmatic_opinion(self):
        """Edge case: dogmatic belief (zero uncertainty)."""
        from src.fact import SLOpinion

        op = SLOpinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        assert op.projected_probability == 1.0


class TestSourceAndProvenance:
    """Source and Provenance structures must be well-formed."""

    def test_source_fields(self):
        from src.fact import Source

        s = Source(id="urn:test:s1", name="Test Source", reliability=0.85)
        assert s.id == "urn:test:s1"
        assert s.reliability == 0.85

    def test_source_reliability_bounds(self):
        from src.fact import Source

        with pytest.raises(ValueError, match="reliability"):
            Source(id="urn:test:s1", name="Test", reliability=1.5)
        with pytest.raises(ValueError, match="reliability"):
            Source(id="urn:test:s1", name="Test", reliability=-0.1)

    def test_provenance_requires_at_least_one_source(self):
        from src.fact import Provenance

        with pytest.raises(ValueError, match="source"):
            Provenance(sources=[], method="test")

    def test_provenance_method_nonempty(self):
        from src.fact import Provenance, Source

        s = Source(id="urn:test:s1", name="Test", reliability=0.9)
        with pytest.raises(ValueError, match="method"):
            Provenance(sources=[s], method="")


class TestFactSerialization:
    """Facts must round-trip to/from dict for JSON serialization."""

    def test_fact_to_dict(self):
        from src.fact import Fact, SLOpinion, Provenance, Source

        opinion = SLOpinion(belief=0.9, disbelief=0.02, uncertainty=0.08)
        source = Source(id="urn:test:src1", name="Test", reliability=0.9)
        provenance = Provenance(sources=[source], method="test")

        fact = Fact(
            id="F1", question="Q?", answer="A",
            entity_name="E", entity_type="Organization",
            relation="rel", opinion=opinion,
            provenance=provenance, dataset="meridian",
        )
        d = fact.to_dict()
        assert d["id"] == "F1"
        assert d["opinion"]["belief"] == 0.9
        assert d["provenance"]["sources"][0]["id"] == "urn:test:src1"
        assert d["dataset"] == "meridian"

    def test_fact_from_dict_roundtrip(self):
        from src.fact import Fact, SLOpinion, Provenance, Source

        opinion = SLOpinion(belief=0.9, disbelief=0.02, uncertainty=0.08)
        source = Source(id="urn:test:src1", name="Test", reliability=0.9)
        provenance = Provenance(sources=[source], method="test")

        original = Fact(
            id="F1", question="Q?", answer="A",
            entity_name="E", entity_type="Organization",
            relation="rel", opinion=opinion,
            provenance=provenance, dataset="fever",
            valid_from="2020-01-01",
        )
        d = original.to_dict()
        restored = Fact.from_dict(d)
        assert restored.id == original.id
        assert restored.opinion.belief == original.opinion.belief
        assert restored.valid_from == original.valid_from
        assert restored.dataset == original.dataset


class TestFactFromMeridianKB:
    """The Meridian KB generator output must convert to Facts cleanly."""

    def test_meridian_kb_fact_converts(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base
        from src.fact import Fact

        kb = generate_knowledge_base(small_kb_config)
        raw = kb["facts"][0]
        fact = Fact.from_meridian(raw)

        assert fact.dataset == "meridian"
        assert fact.id == raw["id"]
        assert abs(
            fact.opinion.belief + fact.opinion.disbelief + fact.opinion.uncertainty - 1.0
        ) < 1e-9

    def test_all_meridian_facts_convert(self, small_kb_config):
        from src.knowledge_base import generate_knowledge_base
        from src.fact import Fact

        kb = generate_knowledge_base(small_kb_config)
        for raw in kb["facts"]:
            fact = Fact.from_meridian(raw)
            assert fact.question.endswith("?")
            assert len(fact.answer) > 0
