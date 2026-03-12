"""Tests for EN3.2-H3 — Metadata-Enriched Prompting.

Defines contracts for prompt builders (including ANSWERS-ONLY ablation),
difficulty classification, SL metadata computation, and statistical testing.

Run from: E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest experiments/EN3/tests/test_en3_2_h3.py -v

The conftest.py in this directory adds experiments/ to sys.path so
that ``from EN3.en3_2_h3_core import ...`` resolves correctly.
"""
from __future__ import annotations

import math
import re

import pytest

from EN3.en3_2_h3_core import (
    build_prompt_plain,
    build_prompt_scalar,
    build_prompt_answers_only,
    build_prompt_jsonldex,
    classify_question_difficulty,
    compute_passage_sl_metadata,
    compute_group_metadata,
    mcnemars_test,
)


# ═══════════════════════════════════════════════════════════════════
# Test fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_passages():
    """Minimal passage list matching checkpoint format."""
    return [
        {"passage_id": "p_gold", "score": 0.92},
        {"passage_id": "p_relevant", "score": 0.78},
        {"passage_id": "p_poison", "score": 0.85},
        {"passage_id": "p_noise", "score": 0.45},
    ]


@pytest.fixture
def sample_pid_to_text():
    return {
        "p_gold": "Paris is the capital of France and its largest city.",
        "p_relevant": "France is a country in Western Europe with Paris as capital.",
        "p_poison": "Lyon is the capital of France, located in the southeast.",
        "p_noise": "The Eiffel Tower was built in 1889 for the World's Fair.",
    }


@pytest.fixture
def sample_extractions():
    """QA extractions keyed by passage_id (per-question dict)."""
    return {
        "p_gold": {"answer": "Paris", "qa_score": 0.95},
        "p_relevant": {"answer": "Paris", "qa_score": 0.80},
        "p_poison": {"answer": "Lyon", "qa_score": 0.88},
        "p_noise": {"answer": "1889", "qa_score": 0.60},
    }


@pytest.fixture
def sample_question():
    return "What is the capital of France?"


# ═══════════════════════════════════════════════════════════════════
# 1. Prompt builder tests
# ═══════════════════════════════════════════════════════════════════

class TestBuildPromptPlain:
    """PLAIN condition: passages only, no metadata."""

    def test_contains_all_passages(self, sample_question, sample_passages, sample_pid_to_text):
        prompt = build_prompt_plain(sample_question, sample_passages, sample_pid_to_text)
        assert "Paris is the capital of France" in prompt
        assert "Lyon is the capital of France" in prompt
        assert "Eiffel Tower" in prompt

    def test_contains_question(self, sample_question, sample_passages, sample_pid_to_text):
        prompt = build_prompt_plain(sample_question, sample_passages, sample_pid_to_text)
        assert sample_question in prompt

    def test_no_scores_in_output(self, sample_question, sample_passages, sample_pid_to_text):
        prompt = build_prompt_plain(sample_question, sample_passages, sample_pid_to_text)
        # Should NOT contain metadata keywords
        assert "belief" not in prompt.lower()
        assert "uncertainty" not in prompt.lower()
        assert "conflict" not in prompt.lower()

    def test_passages_numbered(self, sample_question, sample_passages, sample_pid_to_text):
        prompt = build_prompt_plain(sample_question, sample_passages, sample_pid_to_text)
        assert "[Passage 1]" in prompt
        assert f"[Passage {len(sample_passages)}]" in prompt

    def test_asks_for_short_answer(self, sample_question, sample_passages, sample_pid_to_text):
        prompt = build_prompt_plain(sample_question, sample_passages, sample_pid_to_text)
        # Must instruct LLM to give short direct answer
        assert "short" in prompt.lower() or "direct" in prompt.lower()


class TestBuildPromptScalar:
    """SCALAR condition: passages + cosine similarity scores."""

    def test_contains_scores(self, sample_question, sample_passages, sample_pid_to_text):
        prompt = build_prompt_scalar(sample_question, sample_passages, sample_pid_to_text)
        # Each passage should have its cosine similarity score
        assert "0.92" in prompt  # p_gold score
        assert "0.45" in prompt  # p_noise score

    def test_contains_all_passages(self, sample_question, sample_passages, sample_pid_to_text):
        prompt = build_prompt_scalar(sample_question, sample_passages, sample_pid_to_text)
        assert "Paris is the capital of France" in prompt
        assert "Lyon is the capital of France" in prompt

    def test_score_labeled(self, sample_question, sample_passages, sample_pid_to_text):
        prompt = build_prompt_scalar(sample_question, sample_passages, sample_pid_to_text)
        # Score should be clearly labeled
        assert "relevance" in prompt.lower() or "similarity" in prompt.lower()

    def test_no_sl_metadata(self, sample_question, sample_passages, sample_pid_to_text):
        prompt = build_prompt_scalar(sample_question, sample_passages, sample_pid_to_text)
        assert "belief" not in prompt.lower()
        assert "disbelief" not in prompt.lower()
        assert "uncertainty" not in prompt.lower()


class TestBuildPromptAnswersOnly:
    """ANSWERS-ONLY condition: passages + extracted answers, no SL metadata.

    This is the ablation condition that isolates whether the extracted
    answers (the confound) or the SL triples drive the improvement.
    """

    def test_contains_extracted_answers(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        prompt = build_prompt_answers_only(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        # Must show extracted answers per passage
        assert "Paris" in prompt
        assert "Lyon" in prompt
        assert "1889" in prompt

    def test_contains_all_passages(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        prompt = build_prompt_answers_only(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        assert "Paris is the capital of France" in prompt
        assert "Lyon is the capital of France" in prompt
        assert "Eiffel Tower" in prompt

    def test_no_sl_metadata(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        """Must NOT contain any SL-specific metadata."""
        prompt = build_prompt_answers_only(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        assert "belief" not in prompt.lower()
        assert "disbelief" not in prompt.lower()
        assert "uncertainty" not in prompt.lower()

    def test_no_conflict_or_agreement(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        """Must NOT contain conflict detection or source agreement counts."""
        prompt = build_prompt_answers_only(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        assert "conflict" not in prompt.lower()
        assert "assessment" not in prompt.lower()
        assert "fused" not in prompt.lower()

    def test_no_cosine_scores(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        """Must NOT contain cosine similarity scores."""
        prompt = build_prompt_answers_only(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        assert "relevance" not in prompt.lower()
        assert "similarity" not in prompt.lower()
        # Should not contain the raw score values in a labeled context
        assert "0.92" not in prompt
        assert "0.45" not in prompt

    def test_passages_numbered(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        prompt = build_prompt_answers_only(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        assert "[Passage 1]" in prompt
        assert f"[Passage {len(sample_passages)}]" in prompt

    def test_ends_with_answer(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        prompt = build_prompt_answers_only(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        assert "Answer:" in prompt

    def test_contains_question(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        prompt = build_prompt_answers_only(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        assert sample_question in prompt


class TestBuildPromptJsonldEx:
    """JSONLD-EX condition: passages + full SL metadata."""

    def test_contains_sl_triple(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        prompt = build_prompt_jsonldex(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        # Must contain belief, disbelief, uncertainty for at least one passage
        assert "belief" in prompt.lower()
        assert "uncertainty" in prompt.lower()

    def test_contains_conflict_info(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        prompt = build_prompt_jsonldex(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        # Poison passage should trigger conflict detection
        assert "conflict" in prompt.lower()

    def test_contains_agreement_info(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        prompt = build_prompt_jsonldex(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        # Should mention source agreement
        assert "agree" in prompt.lower() or "sources" in prompt.lower()

    def test_contains_fused_assessment(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        prompt = build_prompt_jsonldex(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        # Must contain a fused/overall assessment section
        assert "fused" in prompt.lower() or "overall" in prompt.lower() or "assessment" in prompt.lower()

    def test_contains_all_passages(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        prompt = build_prompt_jsonldex(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        assert "Paris is the capital of France" in prompt
        assert "Lyon is the capital of France" in prompt

    def test_prompt_is_valid_string(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        prompt = build_prompt_jsonldex(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ═══════════════════════════════════════════════════════════════════
# 2. Difficulty classification tests
# ═══════════════════════════════════════════════════════════════════

class TestDifficultyClassification:
    """Classify questions as easy/medium/hard for stratified analysis.

    Difficulty criteria:
      - easy:   gold passage in top-3 AND 0 poison passages
      - medium: (gold in top-3 with 1 poison) OR (gold rank 4-7, <=1 poison)
      - hard:   gold rank >= 8 OR gold missing OR >= 2 poison passages
    """

    def test_easy_gold_top3_no_poison(self):
        retrieved = [
            {"passage_id": "p_gold", "score": 0.95},
            {"passage_id": "p_2", "score": 0.80},
            {"passage_id": "p_3", "score": 0.70},
        ]
        diff = classify_question_difficulty(
            gold_passage_id="p_gold",
            retrieved=retrieved,
            poison_pids=set(),
        )
        assert diff == "easy"

    def test_easy_gold_rank3_no_poison(self):
        retrieved = [
            {"passage_id": "p_1", "score": 0.95},
            {"passage_id": "p_2", "score": 0.90},
            {"passage_id": "p_gold", "score": 0.85},
            {"passage_id": "p_4", "score": 0.60},
        ]
        diff = classify_question_difficulty(
            gold_passage_id="p_gold",
            retrieved=retrieved,
            poison_pids=set(),
        )
        assert diff == "easy"

    def test_medium_gold_top3_with_one_poison(self):
        retrieved = [
            {"passage_id": "p_gold", "score": 0.95},
            {"passage_id": "p_poison1", "score": 0.85},
            {"passage_id": "p_3", "score": 0.70},
        ]
        diff = classify_question_difficulty(
            gold_passage_id="p_gold",
            retrieved=retrieved,
            poison_pids={"p_poison1"},
        )
        assert diff == "medium"

    def test_medium_gold_mid_rank_no_poison(self):
        retrieved = [
            {"passage_id": f"p_{i}", "score": 0.9 - i * 0.05}
            for i in range(10)
        ]
        retrieved[4]["passage_id"] = "p_gold"
        diff = classify_question_difficulty(
            gold_passage_id="p_gold",
            retrieved=retrieved,
            poison_pids=set(),
        )
        assert diff == "medium"

    def test_hard_gold_low_rank(self):
        retrieved = [
            {"passage_id": f"p_{i}", "score": 0.9 - i * 0.05}
            for i in range(10)
        ]
        retrieved[8]["passage_id"] = "p_gold"
        diff = classify_question_difficulty(
            gold_passage_id="p_gold",
            retrieved=retrieved,
            poison_pids=set(),
        )
        assert diff == "hard"

    def test_hard_multiple_poison(self):
        retrieved = [
            {"passage_id": "p_gold", "score": 0.95},
            {"passage_id": "p_poison1", "score": 0.90},
            {"passage_id": "p_poison2", "score": 0.85},
            {"passage_id": "p_4", "score": 0.60},
        ]
        diff = classify_question_difficulty(
            gold_passage_id="p_gold",
            retrieved=retrieved,
            poison_pids={"p_poison1", "p_poison2"},
        )
        assert diff == "hard"

    def test_hard_gold_missing(self):
        retrieved = [
            {"passage_id": f"p_{i}", "score": 0.9 - i * 0.05}
            for i in range(10)
        ]
        diff = classify_question_difficulty(
            gold_passage_id="p_gold",
            retrieved=retrieved,
            poison_pids=set(),
        )
        assert diff == "hard"


# ═══════════════════════════════════════════════════════════════════
# 3. SL metadata computation tests
# ═══════════════════════════════════════════════════════════════════

class TestPassageSLMetadata:
    """Test per-passage SL opinion construction."""

    def test_returns_opinion_fields(self):
        meta = compute_passage_sl_metadata(
            cosine_score=0.85, qa_score=0.90,
            evidence_weight=10, prior_weight=2,
        )
        assert "belief" in meta
        assert "disbelief" in meta
        assert "uncertainty" in meta
        assert "projected_probability" in meta

    def test_bdu_sum_to_one(self):
        meta = compute_passage_sl_metadata(
            cosine_score=0.85, qa_score=0.90,
            evidence_weight=10, prior_weight=2,
        )
        total = meta["belief"] + meta["disbelief"] + meta["uncertainty"]
        assert abs(total - 1.0) < 1e-9

    def test_high_scores_yield_high_belief(self):
        meta = compute_passage_sl_metadata(
            cosine_score=0.95, qa_score=0.95,
            evidence_weight=10, prior_weight=2,
        )
        assert meta["belief"] > meta["disbelief"]

    def test_low_scores_yield_high_disbelief(self):
        meta = compute_passage_sl_metadata(
            cosine_score=0.10, qa_score=0.10,
            evidence_weight=10, prior_weight=2,
        )
        assert meta["disbelief"] > meta["belief"]

    def test_evidence_weight_affects_uncertainty(self):
        meta_low_w = compute_passage_sl_metadata(
            cosine_score=0.85, qa_score=0.90,
            evidence_weight=2, prior_weight=2,
        )
        meta_high_w = compute_passage_sl_metadata(
            cosine_score=0.85, qa_score=0.90,
            evidence_weight=50, prior_weight=2,
        )
        # More evidence -> lower uncertainty
        assert meta_high_w["uncertainty"] < meta_low_w["uncertainty"]


class TestGroupMetadata:
    """Test group-level SL metadata (conflict, agreement, fused assessment)."""

    def test_agreeing_passages_low_conflict(self):
        passage_metas = [
            {"belief": 0.80, "disbelief": 0.05, "uncertainty": 0.15,
             "projected_probability": 0.87, "answer": "Paris"},
            {"belief": 0.75, "disbelief": 0.08, "uncertainty": 0.17,
             "projected_probability": 0.84, "answer": "Paris"},
        ]
        group_meta = compute_group_metadata(passage_metas)
        assert group_meta["max_pairwise_conflict"] < 0.3

    def test_conflicting_passages_high_conflict(self):
        passage_metas = [
            {"belief": 0.85, "disbelief": 0.05, "uncertainty": 0.10,
             "projected_probability": 0.90, "answer": "Paris"},
            {"belief": 0.80, "disbelief": 0.08, "uncertainty": 0.12,
             "projected_probability": 0.86, "answer": "Lyon"},
        ]
        group_meta = compute_group_metadata(passage_metas)
        assert group_meta["has_conflict"] is True

    def test_agreement_count(self):
        passage_metas = [
            {"belief": 0.8, "disbelief": 0.05, "uncertainty": 0.15,
             "projected_probability": 0.87, "answer": "Paris"},
            {"belief": 0.7, "disbelief": 0.1, "uncertainty": 0.2,
             "projected_probability": 0.80, "answer": "Paris"},
            {"belief": 0.6, "disbelief": 0.2, "uncertainty": 0.2,
             "projected_probability": 0.70, "answer": "Lyon"},
        ]
        group_meta = compute_group_metadata(passage_metas)
        assert group_meta["best_answer"] == "Paris"
        assert group_meta["best_answer_source_count"] == 2

    def test_returns_fused_opinion(self):
        passage_metas = [
            {"belief": 0.8, "disbelief": 0.05, "uncertainty": 0.15,
             "projected_probability": 0.87, "answer": "Paris"},
        ]
        group_meta = compute_group_metadata(passage_metas)
        assert "fused_belief" in group_meta
        assert "fused_uncertainty" in group_meta

    def test_natural_language_assessment(self):
        passage_metas = [
            {"belief": 0.8, "disbelief": 0.05, "uncertainty": 0.15,
             "projected_probability": 0.87, "answer": "Paris"},
            {"belief": 0.6, "disbelief": 0.2, "uncertainty": 0.2,
             "projected_probability": 0.70, "answer": "Lyon"},
        ]
        group_meta = compute_group_metadata(passage_metas)
        assert "assessment_text" in group_meta
        assert isinstance(group_meta["assessment_text"], str)
        assert len(group_meta["assessment_text"]) > 0


# ═══════════════════════════════════════════════════════════════════
# 4. McNemar's test
# ═══════════════════════════════════════════════════════════════════

class TestMcNemarsTest:
    """Paired McNemar's test for comparing two conditions on same questions."""

    def test_identical_predictions_not_significant(self):
        pred_a = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        pred_b = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        result = mcnemars_test(pred_a, pred_b)
        assert result["p_value"] == 1.0 or result["p_value"] > 0.05
        assert result["significant"] is False

    def test_clearly_different_predictions(self):
        # B gets many right that A gets wrong, A gets none right that B misses
        pred_a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        pred_b = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        result = mcnemars_test(pred_a, pred_b)
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_returns_contingency_counts(self):
        pred_a = [1, 0, 1, 0]
        pred_b = [1, 1, 0, 0]
        result = mcnemars_test(pred_a, pred_b)
        assert "n_01" in result
        assert "n_10" in result
        assert "n_11" in result
        assert "n_00" in result

    def test_specific_contingency(self):
        # idx 0: both right (n_11)
        # idx 1: A wrong B right (n_01)
        # idx 2: A right B wrong (n_10)
        # idx 3: both wrong (n_00)
        pred_a = [1, 0, 1, 0]
        pred_b = [1, 1, 0, 0]
        result = mcnemars_test(pred_a, pred_b)
        assert result["n_11"] == 1
        assert result["n_00"] == 1
        assert result["n_10"] == 1
        assert result["n_01"] == 1

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError):
            mcnemars_test([1, 0, 1], [1, 0])


# ═══════════════════════════════════════════════════════════════════
# 5. Integration-level prompt tests
# ═══════════════════════════════════════════════════════════════════

class TestPromptDifferences:
    """Verify the four prompts are structurally different for same input."""

    def test_answers_only_longer_than_plain(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        plain = build_prompt_plain(sample_question, sample_passages, sample_pid_to_text)
        answers = build_prompt_answers_only(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        assert len(answers) > len(plain)

    def test_jsonldex_longer_than_answers_only(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        answers = build_prompt_answers_only(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        jsonldex = build_prompt_jsonldex(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        # JSONLD-EX has SL triples + conflict + assessment on top of answers
        assert len(jsonldex) > len(answers)

    def test_jsonldex_longer_than_plain(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        plain = build_prompt_plain(sample_question, sample_passages, sample_pid_to_text)
        jsonldex = build_prompt_jsonldex(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        assert len(jsonldex) > len(plain)

    def test_scalar_longer_than_plain(
        self, sample_question, sample_passages, sample_pid_to_text
    ):
        plain = build_prompt_plain(sample_question, sample_passages, sample_pid_to_text)
        scalar = build_prompt_scalar(sample_question, sample_passages, sample_pid_to_text)
        assert len(scalar) > len(plain)

    def test_jsonldex_longer_than_scalar(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        scalar = build_prompt_scalar(sample_question, sample_passages, sample_pid_to_text)
        jsonldex = build_prompt_jsonldex(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        assert len(jsonldex) > len(scalar)

    def test_all_prompts_end_with_answer_prompt(
        self, sample_question, sample_passages, sample_pid_to_text, sample_extractions
    ):
        """All four prompts should end with the same answer elicitation."""
        plain = build_prompt_plain(sample_question, sample_passages, sample_pid_to_text)
        scalar = build_prompt_scalar(sample_question, sample_passages, sample_pid_to_text)
        answers = build_prompt_answers_only(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        jsonldex = build_prompt_jsonldex(
            sample_question, sample_passages, sample_pid_to_text, sample_extractions
        )
        for prompt in [plain, scalar, answers, jsonldex]:
            assert "Answer:" in prompt
