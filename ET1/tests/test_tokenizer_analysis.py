"""Tests for tokenizer analysis utility.

Validates that the tokenizer analysis correctly computes token counts,
efficiency ratios, and fragmentation metrics across conditions.
"""

from __future__ import annotations

import pytest

from src.fact import Fact, SLOpinion, Source, Provenance


# ---- Helpers ----

def _make_fact(fact_id: str = "F0001", belief: float = 0.9) -> Fact:
    """Create a minimal Fact for tokenizer testing."""
    return Fact(
        id=fact_id,
        dataset="meridian",
        question="What is the headquarters of Acme Corp?",
        answer="New York City",
        entity_name="Acme Corp",
        entity_type="Organization",
        relation="headquartered_in",
        opinion=SLOpinion(belief=belief, disbelief=0.05,
                          uncertainty=round(1.0 - belief - 0.05, 4)),
        provenance=Provenance(
            sources=[Source(id="s1", name="TestSource", reliability=0.9)],
            method="cross_reference_verification",
        ),
        tier="T1_established",
    )


# ====================================================================
# Test: compute_token_stats
# ====================================================================

class TestComputeTokenStats:
    """Tests for per-condition token counting."""

    def test_returns_dict_with_all_conditions(self):
        """Should return stats for all 7 conditions."""
        from src.tokenizer_analysis import compute_token_stats

        facts = [_make_fact(f"F{i:04d}") for i in range(5)]
        stats = compute_token_stats(facts, model_name="HuggingFaceTB/SmolLM2-135M-Instruct")

        assert set(stats.keys()) == {"C1", "C2", "C3", "C4", "C5", "C6", "C7"}

    def test_each_condition_has_required_fields(self):
        """Each condition's stats should have mean, std, min, max, total."""
        from src.tokenizer_analysis import compute_token_stats

        facts = [_make_fact(f"F{i:04d}") for i in range(5)]
        stats = compute_token_stats(facts, model_name="HuggingFaceTB/SmolLM2-135M-Instruct")

        for cond, cond_stats in stats.items():
            assert "mean_tokens" in cond_stats, f"{cond} missing mean_tokens"
            assert "std_tokens" in cond_stats, f"{cond} missing std_tokens"
            assert "min_tokens" in cond_stats, f"{cond} missing min_tokens"
            assert "max_tokens" in cond_stats, f"{cond} missing max_tokens"
            assert "total_tokens" in cond_stats, f"{cond} missing total_tokens"

    def test_c1_shorter_than_c4(self):
        """C1 (plain text) should use fewer tokens than C4 (jsonld-ex full)."""
        from src.tokenizer_analysis import compute_token_stats

        facts = [_make_fact(f"F{i:04d}") for i in range(10)]
        stats = compute_token_stats(facts, model_name="HuggingFaceTB/SmolLM2-135M-Instruct")

        assert stats["C1"]["mean_tokens"] < stats["C4"]["mean_tokens"]

    def test_c5_word_matched_but_not_token_matched_to_c4(self):
        """C5 pads by word count, NOT token count.

        FINDING: C5's word-based padding does not achieve token parity with C4
        because JSON structures tokenize into many more tokens per whitespace-
        delimited 'word'. This means C5 is currently a weak length control.
        The tokenizer analysis report quantifies this gap.

        TODO: Consider a protocol amendment to pad C5 by token count instead
        of word count, or document this gap in the paper's limitations.
        """
        from src.tokenizer_analysis import compute_token_stats

        facts = [_make_fact(f"F{i:04d}") for i in range(10)]
        stats = compute_token_stats(facts, model_name="HuggingFaceTB/SmolLM2-135M-Instruct")

        c4_mean = stats["C4"]["mean_tokens"]
        c5_mean = stats["C5"]["mean_tokens"]
        ratio = c5_mean / c4_mean
        # C5 uses far fewer tokens than C4 because word-count matching
        # doesn't account for JSON tokenization overhead.
        # This is a documented gap, not a passing assertion.
        assert ratio < 1.0, (
            f"C5/C4 token ratio = {ratio:.2f}; expected <1.0 because "
            f"JSON structures inflate token count relative to prose"
        )
        assert c5_mean > stats["C1"]["mean_tokens"], (
            "C5 should still be longer than C1 (plain text)"
        )

    def test_positive_token_counts(self):
        """All token counts should be positive."""
        from src.tokenizer_analysis import compute_token_stats

        facts = [_make_fact(f"F{i:04d}") for i in range(5)]
        stats = compute_token_stats(facts, model_name="HuggingFaceTB/SmolLM2-135M-Instruct")

        for cond, cond_stats in stats.items():
            assert cond_stats["mean_tokens"] > 0, f"{cond} has non-positive mean"
            assert cond_stats["min_tokens"] > 0, f"{cond} has non-positive min"


# ====================================================================
# Test: analyze_token_fragmentation
# ====================================================================

class TestAnalyzeTokenFragmentation:
    """Tests for JSON syntax token fragmentation analysis."""

    def test_returns_fragmentation_dict(self):
        """Should return fragmentation data for key JSON-LD tokens."""
        from src.tokenizer_analysis import analyze_token_fragmentation

        result = analyze_token_fragmentation(
            model_name="HuggingFaceTB/SmolLM2-135M-Instruct"
        )

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_includes_jsonld_keywords(self):
        """Should include analysis of @context, @type, @id."""
        from src.tokenizer_analysis import analyze_token_fragmentation

        result = analyze_token_fragmentation(
            model_name="HuggingFaceTB/SmolLM2-135M-Instruct"
        )

        # Check that key JSON-LD terms are analyzed
        analyzed_strings = set(result.keys())
        assert '"@context"' in analyzed_strings or '@context' in analyzed_strings

    def test_each_entry_has_token_count_and_tokens(self):
        """Each entry should report n_tokens and the actual token strings."""
        from src.tokenizer_analysis import analyze_token_fragmentation

        result = analyze_token_fragmentation(
            model_name="HuggingFaceTB/SmolLM2-135M-Instruct"
        )

        for text, info in result.items():
            assert "n_tokens" in info, f"Missing n_tokens for '{text}'"
            assert "tokens" in info, f"Missing tokens for '{text}'"
            assert info["n_tokens"] >= 1
            assert isinstance(info["tokens"], list)


# ====================================================================
# Test: compute_efficiency_ratios
# ====================================================================

class TestComputeEfficiencyRatios:
    """Tests for token efficiency relative to C1 baseline."""

    def test_c1_ratio_is_one(self):
        """C1's ratio relative to itself should be 1.0."""
        from src.tokenizer_analysis import compute_efficiency_ratios

        token_stats = {
            "C1": {"mean_tokens": 20.0},
            "C2": {"mean_tokens": 30.0},
            "C3": {"mean_tokens": 40.0},
            "C4": {"mean_tokens": 60.0},
            "C5": {"mean_tokens": 55.0},
            "C6": {"mean_tokens": 60.0},
            "C7": {"mean_tokens": 35.0},
        }

        ratios = compute_efficiency_ratios(token_stats, baseline="C1")
        assert ratios["C1"] == pytest.approx(1.0)

    def test_ratios_are_positive(self):
        """All ratios should be positive."""
        from src.tokenizer_analysis import compute_efficiency_ratios

        token_stats = {
            "C1": {"mean_tokens": 20.0},
            "C4": {"mean_tokens": 60.0},
        }

        ratios = compute_efficiency_ratios(token_stats, baseline="C1")
        for cond, ratio in ratios.items():
            assert ratio > 0, f"{cond} has non-positive ratio"

    def test_c4_ratio_greater_than_one(self):
        """C4 should use more tokens than C1 (ratio > 1)."""
        from src.tokenizer_analysis import compute_efficiency_ratios

        token_stats = {
            "C1": {"mean_tokens": 20.0},
            "C4": {"mean_tokens": 60.0},
        }

        ratios = compute_efficiency_ratios(token_stats, baseline="C1")
        assert ratios["C4"] > 1.0
