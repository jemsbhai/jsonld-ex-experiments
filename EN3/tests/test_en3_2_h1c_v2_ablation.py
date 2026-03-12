"""Tests for EN3.2-H1c v2 Ablation — Per-Difficulty, Model Subset, Param Sweep, Precision-at-Coverage.

RED phase — all tests should FAIL until the ablation core is implemented.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest experiments/EN3/tests/test_en3_2_h1c_v2_ablation.py -v
"""
from __future__ import annotations

import pytest

from EN3.en3_2_h1c_v2_ablation_core import (
    # Difficulty classification (adapted for H1c data structures)
    classify_question_difficulty_h1c,
    # Model subset evaluation
    evaluate_model_subset,
    MODEL_SUBSETS,
    # Parameter sweep
    evaluate_param_combo,
    # Precision-at-coverage
    compute_precision_at_coverage,
    # Aggregation helpers
    compute_per_difficulty_breakdown,
    compute_mcnemar_contingency,
)
from EN3.en3_2_h1c_v2_core import (
    STRATEGY_NAMES,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

def _make_passage_extractions(entries):
    """Build per-passage extractions. entries: [(model, answer, qa_score), ...]"""
    return {
        model: {"answer": answer, "qa_score": qa_score}
        for model, answer, qa_score in entries
    }


def _make_question_passages(passage_list):
    """Build question_passages.

    passage_list: [(cosine, passage_id, [(model, answer, qa_score), ...])]
    """
    return [
        {
            "cosine": cosine,
            "passage_id": pid,
            "extractions": _make_passage_extractions(entries),
        }
        for cosine, pid, entries in passage_list
    ]


def _make_retrieval_entry(passage_ids_and_scores):
    """Build retrieval data. [(passage_id, score), ...]"""
    return [
        {"passage_id": pid, "score": score}
        for pid, score in passage_ids_and_scores
    ]


# ═══════════════════════════════════════════════════════════════════
# 1. Difficulty classification
# ═══════════════════════════════════════════════════════════════════

class TestClassifyDifficultyH1c:
    """classify_question_difficulty_h1c adapts H3's classifier for H1c data."""

    def test_easy_gold_top3_no_poison(self):
        retrieved = _make_retrieval_entry([
            ("gold_p", 0.9), ("p2", 0.8), ("p3", 0.7),
            ("p4", 0.6), ("p5", 0.5),
        ])
        result = classify_question_difficulty_h1c(
            gold_passage_id="gold_p",
            retrieved=retrieved,
            poison_pids=set(),
        )
        assert result == "easy"

    def test_easy_gold_rank2_no_poison(self):
        retrieved = _make_retrieval_entry([
            ("p1", 0.9), ("p2", 0.85), ("gold_p", 0.8),
            ("p4", 0.6), ("p5", 0.5),
        ])
        result = classify_question_difficulty_h1c(
            gold_passage_id="gold_p",
            retrieved=retrieved,
            poison_pids=set(),
        )
        assert result == "easy"

    def test_medium_gold_top3_with_1_poison(self):
        retrieved = _make_retrieval_entry([
            ("gold_p", 0.9), ("poison1", 0.85), ("p3", 0.7),
        ])
        result = classify_question_difficulty_h1c(
            gold_passage_id="gold_p",
            retrieved=retrieved,
            poison_pids={"poison1"},
        )
        assert result == "medium"

    def test_medium_gold_rank4_no_poison(self):
        retrieved = _make_retrieval_entry([
            ("p1", 0.9), ("p2", 0.85), ("p3", 0.8),
            ("gold_p", 0.75), ("p5", 0.7),
        ])
        result = classify_question_difficulty_h1c(
            gold_passage_id="gold_p",
            retrieved=retrieved,
            poison_pids=set(),
        )
        assert result == "medium"

    def test_hard_2_or_more_poison(self):
        retrieved = _make_retrieval_entry([
            ("gold_p", 0.9), ("poison1", 0.85), ("poison2", 0.8),
            ("p4", 0.6),
        ])
        result = classify_question_difficulty_h1c(
            gold_passage_id="gold_p",
            retrieved=retrieved,
            poison_pids={"poison1", "poison2"},
        )
        assert result == "hard"

    def test_hard_gold_missing(self):
        retrieved = _make_retrieval_entry([
            ("p1", 0.9), ("p2", 0.85), ("p3", 0.8),
        ])
        result = classify_question_difficulty_h1c(
            gold_passage_id="gold_p",
            retrieved=retrieved,
            poison_pids=set(),
        )
        assert result == "hard"

    def test_hard_gold_rank_8_plus(self):
        retrieved = _make_retrieval_entry([
            (f"p{i}", 0.9 - 0.05 * i)
            for i in range(10)
        ])
        # Place gold_p at rank 8 (0-indexed = index 8, which is > _MEDIUM_MAX_RANK=6)
        retrieved[8] = {"passage_id": "gold_p", "score": 0.45}
        result = classify_question_difficulty_h1c(
            gold_passage_id="gold_p",
            retrieved=retrieved,
            poison_pids=set(),
        )
        assert result == "hard"


# ═══════════════════════════════════════════════════════════════════
# 2. Model subset ablation
# ═══════════════════════════════════════════════════════════════════

class TestModelSubsets:

    def test_model_subsets_defined(self):
        """MODEL_SUBSETS should define at least 5 named subsets."""
        assert len(MODEL_SUBSETS) >= 5
        subset_names = set(MODEL_SUBSETS.keys())
        assert "all_4" in subset_names
        assert "drop_bert_tiny" in subset_names
        assert "drop_roberta" in subset_names

    def test_all_4_includes_all_models(self):
        assert set(MODEL_SUBSETS["all_4"]) == {
            "distilbert", "roberta", "electra", "bert_tiny"
        }

    def test_drop_subsets_remove_correct_model(self):
        assert "bert_tiny" not in MODEL_SUBSETS["drop_bert_tiny"]
        assert "roberta" not in MODEL_SUBSETS["drop_roberta"]

    def test_evaluate_model_subset_returns_strategies(self):
        """evaluate_model_subset should produce strategy results for a subset."""
        q_passages = _make_question_passages([
            (0.9, "p1", [
                ("roberta", "Paris", 0.9), ("electra", "Paris", 0.85),
                ("distilbert", "Paris", 0.7), ("bert_tiny", "London", 0.6),
            ]),
            (0.5, "p2", [
                ("roberta", "Tokyo", 0.4), ("electra", "Berlin", 0.3),
                ("distilbert", "Tokyo", 0.5), ("bert_tiny", "Tokyo", 0.45),
            ]),
        ])
        model_trust = {
            "roberta": 0.85, "electra": 0.81,
            "distilbert": 0.78, "bert_tiny": 0.24,
        }
        models = ["roberta", "electra"]

        results = evaluate_model_subset(
            question_passages=q_passages,
            models=models,
            model_trust=model_trust,
        )
        assert isinstance(results, dict)
        assert "sl_fusion" in results
        assert "sl_trust_discount" in results
        for name, res in results.items():
            assert "answer" in res
            assert "confidence" in res

    def test_evaluate_model_subset_filters_extractions(self):
        """Only models in the subset should be used."""
        q_passages = _make_question_passages([
            (0.9, "p1", [
                ("roberta", "Paris", 0.9), ("bert_tiny", "London", 0.99),
            ]),
        ])
        model_trust = {"roberta": 0.85, "bert_tiny": 0.24}
        results = evaluate_model_subset(
            question_passages=q_passages,
            models=["roberta"],
            model_trust=model_trust,
        )
        assert results["sl_fusion"]["answer"].lower() == "paris"


# ═══════════════════════════════════════════════════════════════════
# 3. Parameter sweep
# ═══════════════════════════════════════════════════════════════════

class TestParameterSweep:

    def test_evaluate_param_combo_returns_sl_strategies(self):
        """Parameter sweep function returns SL strategies with given ew/pw."""
        q_passages = _make_question_passages([
            (0.9, "p1", [
                ("roberta", "Paris", 0.9), ("electra", "Paris", 0.85),
                ("distilbert", "Paris", 0.7), ("bert_tiny", "London", 0.6),
            ]),
        ])
        model_trust = {
            "roberta": 0.85, "electra": 0.81,
            "distilbert": 0.78, "bert_tiny": 0.24,
        }

        results = evaluate_param_combo(
            question_passages=q_passages,
            model_trust=model_trust,
            evidence_weight=20.0,
            prior_weight=5.0,
        )
        assert "sl_fusion" in results
        assert "sl_trust_discount" in results
        # Non-SL strategies should NOT be included (they don't depend on ew/pw)
        assert "scalar_majority" not in results

    def test_different_params_give_different_confidences(self):
        """Varying evidence_weight should change the fused opinion confidence."""
        q_passages = _make_question_passages([
            (0.9, "p1", [
                ("roberta", "Paris", 0.9), ("electra", "Paris", 0.85),
            ]),
        ])
        model_trust = {"roberta": 0.85, "electra": 0.81}

        results_low = evaluate_param_combo(
            q_passages, model_trust, evidence_weight=5.0, prior_weight=1.0,
        )
        results_high = evaluate_param_combo(
            q_passages, model_trust, evidence_weight=50.0, prior_weight=1.0,
        )
        conf_low = results_low["sl_fusion"]["confidence"]
        conf_high = results_high["sl_fusion"]["confidence"]
        assert conf_low != pytest.approx(conf_high, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════
# 4. Precision-at-coverage
# ═══════════════════════════════════════════════════════════════════

class TestPrecisionAtCoverage:

    def test_full_coverage_equals_overall_accuracy(self):
        """At coverage=1.0, precision should equal overall accuracy."""
        per_question = [
            {"confidence": 0.9, "correct": True},
            {"confidence": 0.7, "correct": True},
            {"confidence": 0.5, "correct": False},
            {"confidence": 0.3, "correct": False},
        ]
        curve = compute_precision_at_coverage(per_question)
        full_cov = [pt for pt in curve if pt["coverage"] == pytest.approx(1.0)]
        assert len(full_cov) == 1
        assert full_cov[0]["precision"] == pytest.approx(0.5)

    def test_high_confidence_subset_higher_precision(self):
        """Top-confidence questions should have higher precision."""
        per_question = [
            {"confidence": 0.95, "correct": True},
            {"confidence": 0.90, "correct": True},
            {"confidence": 0.40, "correct": False},
            {"confidence": 0.20, "correct": False},
        ]
        curve = compute_precision_at_coverage(per_question)
        half_cov = [pt for pt in curve if pt["coverage"] == pytest.approx(0.5)]
        assert len(half_cov) == 1
        assert half_cov[0]["precision"] == pytest.approx(1.0)

    def test_returns_sorted_coverage_levels(self):
        """Curve should have monotonically increasing coverage levels."""
        per_question = [
            {"confidence": 0.9 - 0.1 * i, "correct": i % 2 == 0}
            for i in range(10)
        ]
        curve = compute_precision_at_coverage(per_question)
        coverages = [pt["coverage"] for pt in curve]
        assert coverages == sorted(coverages)

    def test_empty_input(self):
        curve = compute_precision_at_coverage([])
        assert curve == []

    def test_single_question(self):
        curve = compute_precision_at_coverage(
            [{"confidence": 0.9, "correct": True}],
        )
        assert len(curve) >= 1
        assert curve[-1]["precision"] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════
# 5. Per-difficulty breakdown aggregation
# ═══════════════════════════════════════════════════════════════════

class TestPerDifficultyBreakdown:

    def test_returns_easy_medium_hard(self):
        """compute_per_difficulty_breakdown returns all 3 levels."""
        per_question = [
            {"difficulty": "easy", "strategies": {
                "sl_fusion": True, "scalar_majority": True}},
            {"difficulty": "easy", "strategies": {
                "sl_fusion": True, "scalar_majority": False}},
            {"difficulty": "medium", "strategies": {
                "sl_fusion": False, "scalar_majority": True}},
            {"difficulty": "hard", "strategies": {
                "sl_fusion": False, "scalar_majority": False}},
        ]
        breakdown = compute_per_difficulty_breakdown(per_question)
        assert "easy" in breakdown
        assert "medium" in breakdown
        assert "hard" in breakdown

    def test_accuracy_computed_correctly(self):
        per_question = [
            {"difficulty": "easy", "strategies": {
                "sl_fusion": True, "scalar_majority": True}},
            {"difficulty": "easy", "strategies": {
                "sl_fusion": True, "scalar_majority": False}},
            {"difficulty": "easy", "strategies": {
                "sl_fusion": False, "scalar_majority": False}},
        ]
        breakdown = compute_per_difficulty_breakdown(per_question)
        assert breakdown["easy"]["sl_fusion"]["accuracy"] == pytest.approx(
            2 / 3)
        assert breakdown["easy"]["sl_fusion"]["n"] == 3
        assert breakdown["easy"]["scalar_majority"]["accuracy"] == pytest.approx(
            1 / 3)

    def test_empty_difficulty_level_handled(self):
        """If no questions at a difficulty, it should appear with n=0."""
        per_question = [
            {"difficulty": "easy", "strategies": {"sl_fusion": True}},
        ]
        breakdown = compute_per_difficulty_breakdown(per_question)
        assert breakdown["medium"]["sl_fusion"]["n"] == 0
        assert breakdown["hard"]["sl_fusion"]["n"] == 0


# ═══════════════════════════════════════════════════════════════════
# 6. McNemar contingency table
# ═══════════════════════════════════════════════════════════════════

class TestMcNemarContingency:

    def test_counts_correct(self):
        """Compute McNemar contingency between two strategies."""
        per_question = [
            {"sl_fusion": True, "scalar_qa_weighted": True},
            {"sl_fusion": True, "scalar_qa_weighted": False},
            {"sl_fusion": True, "scalar_qa_weighted": False},
            {"sl_fusion": False, "scalar_qa_weighted": True},
            {"sl_fusion": False, "scalar_qa_weighted": False},
        ]
        result = compute_mcnemar_contingency(
            per_question, "sl_fusion", "scalar_qa_weighted")
        assert result["both_correct"] == 1
        assert result["a_only"] == 2
        assert result["b_only"] == 1
        assert result["both_wrong"] == 1

    def test_pvalue_computed(self):
        per_question = [
            {"sl_fusion": True, "scalar_qa_weighted": False}
        ] * 20 + [
            {"sl_fusion": False, "scalar_qa_weighted": True}
        ] * 5
        result = compute_mcnemar_contingency(
            per_question, "sl_fusion", "scalar_qa_weighted")
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_identical_strategies_pvalue_1(self):
        """If both strategies always agree, p=1."""
        per_question = [
            {"sl_fusion": True, "scalar_qa_weighted": True},
            {"sl_fusion": False, "scalar_qa_weighted": False},
        ]
        result = compute_mcnemar_contingency(
            per_question, "sl_fusion", "scalar_qa_weighted")
        assert result["a_only"] == 0
        assert result["b_only"] == 0
        assert result["p_value"] == pytest.approx(1.0)
