"""Tests for the 7-condition data formatter.

TDD: Written BEFORE data_formatter.py implementation.
The formatter takes Fact objects and produces (prompt, response) pairs
in each of the 7 experimental conditions. The prompt is IDENTICAL across
conditions; only the response format differs.
"""

import pytest
import json
import random

from src.fact import Fact, SLOpinion, Provenance, Source


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_fact():
    """A representative Meridian fact for testing."""
    opinion = SLOpinion(belief=0.94, disbelief=0.02, uncertainty=0.04)
    source1 = Source(id="urn:meridian:source:vcr-2024", name="Valorian Corporate Registry", reliability=0.95)
    source2 = Source(id="urn:meridian:source:mbi-2024", name="Meridian Business Index", reliability=0.90)
    provenance = Provenance(sources=[source1, source2], method="cross_reference_verification")
    return Fact(
        id="F00001",
        dataset="meridian",
        question="Where is NovaTech Industries headquartered?",
        answer="Cedarpoint, Valoria",
        entity_name="NovaTech Industries",
        entity_type="Organization",
        relation="headquartered_in",
        opinion=opinion,
        provenance=provenance,
        tier="T1_established",
        valid_from="2019-06-01",
    )


@pytest.fixture
def squad_fact():
    """A representative SQuAD fact with context."""
    opinion = SLOpinion(belief=0.95, disbelief=0.02, uncertainty=0.03)
    source = Source(id="urn:squad:56be4db0", name="Normans", reliability=0.85)
    provenance = Provenance(sources=[source], method="extractive_qa_annotation")
    return Fact(
        id="SQUAD-56be4db0",
        dataset="squad",
        question="When were the Normans in Normandy?",
        answer="10th and 11th centuries",
        entity_name="Normans",
        entity_type="Answer",
        relation="extractive_qa",
        opinion=opinion,
        provenance=provenance,
        tier="T1_established",
        context="The Normans were the people who in the 10th and 11th centuries gave their name to Normandy.",
    )


@pytest.fixture
def nei_fact():
    """A high-uncertainty fact (NOT ENOUGH INFO / unanswerable)."""
    opinion = SLOpinion(belief=0.15, disbelief=0.10, uncertainty=0.75)
    source = Source(id="urn:fever:no-evidence", name="No evidence", reliability=0.0)
    provenance = Provenance(sources=[source], method="insufficient_evidence")
    return Fact(
        id="FEVER-200000",
        dataset="fever",
        question='Is the following claim true? "Tilda Swinton is a vegan."?',
        answer="There is not enough information to verify this claim.",
        entity_name="Tilda Swinton is a vegan.",
        entity_type="ClaimReview",
        relation="verification",
        opinion=opinion,
        provenance=provenance,
        tier="T4_speculative",
    )


ALL_CONDITIONS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


# ---------------------------------------------------------------------------
# Core Contract: format_fact returns (prompt, response) for each condition
# ---------------------------------------------------------------------------

class TestFormatterInterface:
    """format_fact must return a dict with prompt and response strings."""

    def test_returns_dict_with_required_keys(self, sample_fact):
        from src.data_formatter import format_fact

        for cond in ALL_CONDITIONS:
            result = format_fact(sample_fact, condition=cond, seed=42)
            assert "prompt" in result, f"{cond}: missing 'prompt'"
            assert "response" in result, f"{cond}: missing 'response'"
            assert isinstance(result["prompt"], str)
            assert isinstance(result["response"], str)

    def test_prompt_identical_across_conditions(self, sample_fact):
        """The independent variable is ONLY the response format."""
        from src.data_formatter import format_fact

        prompts = set()
        for cond in ALL_CONDITIONS:
            result = format_fact(sample_fact, condition=cond, seed=42)
            prompts.add(result["prompt"])

        assert len(prompts) == 1, (
            f"Prompts differ across conditions: {prompts}"
        )

    def test_prompt_contains_question(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C1", seed=42)
        assert "NovaTech" in result["prompt"]

    def test_response_nonempty(self, sample_fact):
        from src.data_formatter import format_fact

        for cond in ALL_CONDITIONS:
            result = format_fact(sample_fact, condition=cond, seed=42)
            assert len(result["response"].strip()) > 0, (
                f"{cond}: empty response"
            )

    def test_squad_prompt_includes_context(self, squad_fact):
        """SQuAD prompts must include the context paragraph."""
        from src.data_formatter import format_fact

        result = format_fact(squad_fact, condition="C1", seed=42)
        assert "Normans" in result["prompt"]
        assert "10th and 11th centuries" in result["prompt"]

    def test_invalid_condition_raises(self, sample_fact):
        from src.data_formatter import format_fact

        with pytest.raises(ValueError, match="condition"):
            format_fact(sample_fact, condition="C99", seed=42)


# ---------------------------------------------------------------------------
# C1: Plain Text
# ---------------------------------------------------------------------------

class TestC1PlainText:
    """C1 response is a natural-language sentence."""

    def test_response_is_plain_text(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C1", seed=42)
        resp = result["response"]
        # Should NOT contain JSON syntax
        assert "{" not in resp
        assert "@context" not in resp

    def test_response_contains_answer(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C1", seed=42)
        assert "Cedarpoint" in result["response"]


# ---------------------------------------------------------------------------
# C2: Plain JSON
# ---------------------------------------------------------------------------

class TestC2PlainJSON:
    """C2 response is a flat key-value JSON object."""

    def test_response_is_valid_json(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C2", seed=42)
        parsed = json.loads(result["response"])
        assert isinstance(parsed, dict)

    def test_json_contains_answer(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C2", seed=42)
        parsed = json.loads(result["response"])
        # Should have entity, relation, value fields
        values = " ".join(str(v) for v in parsed.values())
        assert "Cedarpoint" in values

    def test_json_has_no_ld_fields(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C2", seed=42)
        parsed = json.loads(result["response"])
        assert "@context" not in parsed
        assert "@type" not in parsed
        assert "@opinion" not in parsed


# ---------------------------------------------------------------------------
# C3: JSON-LD (no SL annotations)
# ---------------------------------------------------------------------------

class TestC3JSONLD:
    """C3 response is JSON-LD with @context, @type, @id but NO @opinion."""

    def test_response_is_valid_json(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C3", seed=42)
        parsed = json.loads(result["response"])
        assert isinstance(parsed, dict)

    def test_has_ld_fields(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C3", seed=42)
        parsed = json.loads(result["response"])
        assert "@context" in parsed
        assert "@type" in parsed

    def test_no_opinion_field(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C3", seed=42)
        raw = result["response"]
        assert "@opinion" not in raw


# ---------------------------------------------------------------------------
# C4: jsonld-ex Full (PRIMARY TREATMENT)
# ---------------------------------------------------------------------------

class TestC4JsonldExFull:
    """C4 is the full jsonld-ex format with SL opinions and provenance."""

    def test_response_is_valid_json(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C4", seed=42)
        parsed = json.loads(result["response"])
        assert isinstance(parsed, dict)

    def test_has_jsonldex_context(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C4", seed=42)
        parsed = json.loads(result["response"])
        ctx = parsed.get("@context", "")
        # Should reference jsonld-ex context
        ctx_str = json.dumps(ctx)
        assert "jsonld-ex" in ctx_str or "jsonld_ex" in ctx_str

    def test_has_opinion_field(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C4", seed=42)
        raw = result["response"]
        assert "@opinion" in raw

    def test_opinion_matches_fact(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C4", seed=42)
        parsed = json.loads(result["response"])
        # Find @opinion somewhere in the nested structure
        opinion = _find_nested(parsed, "@opinion")
        assert opinion is not None, "No @opinion found in C4 response"
        assert abs(opinion["belief"] - 0.94) < 1e-4
        assert abs(opinion["disbelief"] - 0.02) < 1e-4

    def test_has_source_field(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C4", seed=42)
        raw = result["response"]
        assert "@source" in raw

    def test_has_method_field(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C4", seed=42)
        raw = result["response"]
        assert "@method" in raw


# ---------------------------------------------------------------------------
# C5: Verbose Text (token-matched to C4)
# ---------------------------------------------------------------------------

class TestC5VerboseText:
    """C5 is plain text padded with metadata prose to match C4 token count."""

    def test_response_is_plain_text(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C5", seed=42)
        resp = result["response"]
        assert "{" not in resp or "@context" not in resp

    def test_contains_answer(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C5", seed=42)
        assert "Cedarpoint" in result["response"]

    def test_longer_than_c1(self, sample_fact):
        """C5 should have more tokens than C1 (it's padded)."""
        from src.data_formatter import format_fact

        c1 = format_fact(sample_fact, condition="C1", seed=42)
        c5 = format_fact(sample_fact, condition="C5", seed=42)
        assert len(c5["response"]) > len(c1["response"])

    def test_roughly_matches_c4_length(self, sample_fact):
        """C5 token count should be within ±30% of C4."""
        from src.data_formatter import format_fact

        c4 = format_fact(sample_fact, condition="C4", seed=42)
        c5 = format_fact(sample_fact, condition="C5", seed=42)

        c4_tokens = len(c4["response"].split())
        c5_tokens = len(c5["response"].split())

        if c4_tokens > 0:
            ratio = c5_tokens / c4_tokens
            assert 0.7 <= ratio <= 1.3, (
                f"C5 ({c5_tokens} tokens) not within ±30% of "
                f"C4 ({c4_tokens} tokens), ratio={ratio:.2f}"
            )


# ---------------------------------------------------------------------------
# C6: jsonld-ex with RANDOMIZED SL values
# ---------------------------------------------------------------------------

class TestC6RandomizedOpinion:
    """C6 has jsonld-ex structure but randomized, meaningless SL values."""

    def test_response_is_valid_json(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C6", seed=42)
        parsed = json.loads(result["response"])
        assert isinstance(parsed, dict)

    def test_has_opinion_field(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C6", seed=42)
        assert "@opinion" in result["response"]

    def test_opinion_is_valid_sl(self, sample_fact):
        """Even randomized, b+d+u must equal 1.0."""
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C6", seed=42)
        parsed = json.loads(result["response"])
        opinion = _find_nested(parsed, "@opinion")
        assert opinion is not None
        total = opinion["belief"] + opinion["disbelief"] + opinion["uncertainty"]
        assert abs(total - 1.0) < 1e-6

    def test_opinion_differs_from_c4(self, sample_fact):
        """C6 opinions must NOT match the real C4 opinions."""
        from src.data_formatter import format_fact

        c4 = format_fact(sample_fact, condition="C4", seed=42)
        c6 = format_fact(sample_fact, condition="C6", seed=42)

        c4_op = _find_nested(json.loads(c4["response"]), "@opinion")
        c6_op = _find_nested(json.loads(c6["response"]), "@opinion")

        # With overwhelming probability, random values won't match real ones
        assert abs(c4_op["belief"] - c6_op["belief"]) > 0.01 or \
               abs(c4_op["disbelief"] - c6_op["disbelief"]) > 0.01

    def test_structure_matches_c4(self, sample_fact):
        """C6 should have the same JSON keys as C4 (just different values)."""
        from src.data_formatter import format_fact

        c4 = format_fact(sample_fact, condition="C4", seed=42)
        c6 = format_fact(sample_fact, condition="C6", seed=42)

        c4_keys = set(json.loads(c4["response"]).keys())
        c6_keys = set(json.loads(c6["response"]).keys())
        assert c4_keys == c6_keys


# ---------------------------------------------------------------------------
# C7: Scalar Confidence
# ---------------------------------------------------------------------------

class TestC7ScalarConfidence:
    """C7 has a single numeric confidence field, no SL opinion."""

    def test_response_is_valid_json(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C7", seed=42)
        parsed = json.loads(result["response"])
        assert isinstance(parsed, dict)

    def test_has_confidence_field(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C7", seed=42)
        parsed = json.loads(result["response"])
        assert "confidence" in parsed
        assert isinstance(parsed["confidence"], float)

    def test_confidence_matches_projected_probability(self, sample_fact):
        """C7 confidence should be P(ω) = b + a·u from the real opinion."""
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C7", seed=42)
        parsed = json.loads(result["response"])

        expected = sample_fact.opinion.projected_probability
        assert abs(parsed["confidence"] - expected) < 1e-4

    def test_no_opinion_field(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C7", seed=42)
        assert "@opinion" not in result["response"]

    def test_confidence_in_unit_interval(self, sample_fact):
        from src.data_formatter import format_fact

        result = format_fact(sample_fact, condition="C7", seed=42)
        parsed = json.loads(result["response"])
        assert 0.0 <= parsed["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Batch formatting
# ---------------------------------------------------------------------------

class TestBatchFormatting:
    """format_facts should process a list and return instruction pairs."""

    def test_format_multiple_facts(self, sample_fact, squad_fact, nei_fact):
        from src.data_formatter import format_facts

        facts = [sample_fact, squad_fact, nei_fact]
        results = format_facts(facts, condition="C4", seed=42)

        assert len(results) == 3
        for r in results:
            assert "prompt" in r
            assert "response" in r

    def test_each_fact_formatted_independently(self, sample_fact, nei_fact):
        from src.data_formatter import format_facts

        results = format_facts([sample_fact, nei_fact], condition="C4", seed=42)
        # Different facts should produce different responses
        assert results[0]["response"] != results[1]["response"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same seed must produce identical output."""

    def test_c6_deterministic_with_seed(self, sample_fact):
        from src.data_formatter import format_fact

        r1 = format_fact(sample_fact, condition="C6", seed=42)
        r2 = format_fact(sample_fact, condition="C6", seed=42)
        assert r1["response"] == r2["response"]

    def test_c6_different_seed_different_output(self, sample_fact):
        from src.data_formatter import format_fact

        r1 = format_fact(sample_fact, condition="C6", seed=42)
        r2 = format_fact(sample_fact, condition="C6", seed=999)
        assert r1["response"] != r2["response"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_nested(obj: dict, key: str):
    """Recursively find a key in a nested dict/list structure."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_nested(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_nested(item, key)
            if found is not None:
                return found
    return None
