"""Tests for EN3.2-H1c v2 Core — Per-Passage Multi-Model Fusion.

Two-level architecture:
  Level 1: Per-passage — fuse 4 models' extractions for one passage
  Level 2: Cross-passage — rank passages and select best answer

RED phase.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest experiments/EN3/tests/test_en3_2_h1c_v2.py -v
"""
from __future__ import annotations

import pytest

from EN3.en3_2_h1c_v2_core import (
    # Level 1: Per-passage fusion
    fuse_passage_scalar_majority,
    fuse_passage_scalar_qa_weighted,
    fuse_passage_sl_fusion,
    fuse_passage_sl_trust_discount,
    # Level 2: Cross-passage ranking
    rank_passages_by_confidence,
    rank_passages_by_confidence_x_cosine,
    select_answer_from_ranking,
    # Full pipeline strategies
    evaluate_all_strategies,
    STRATEGY_NAMES,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

def _make_passage_extractions(entries):
    """Build per-passage extractions for one passage.

    entries: list of (model_name, answer, qa_score)
    Returns: {model_name: {answer, qa_score}}
    """
    return {
        model: {"answer": answer, "qa_score": qa_score}
        for model, answer, qa_score in entries
    }


def _make_question_data(passage_list):
    """Build full question data.

    passage_list: list of (cosine, [(model, answer, qa_score), ...])
    Returns: list of {cosine, extractions: {model: {answer, qa_score}}}
    """
    return [
        {
            "cosine": cosine,
            "passage_id": f"p_{i}",
            "extractions": _make_passage_extractions(entries),
        }
        for i, (cosine, entries) in enumerate(passage_list)
    ]


# ═══════════════════════════════════════════════════════════════════
# 1. Level 1 — Per-passage fusion
# ═══════════════════════════════════════════════════════════════════

class TestFusePassageScalarMajority:

    def test_unanimous_returns_that_answer(self):
        ext = _make_passage_extractions([
            ("m1", "Paris", 0.9), ("m2", "Paris", 0.8),
            ("m3", "Paris", 0.7), ("m4", "Paris", 0.6),
        ])
        answer, confidence = fuse_passage_scalar_majority(ext)
        assert answer.lower() == "paris"
        assert confidence == pytest.approx(1.0)  # 4/4 agree

    def test_split_vote_returns_majority(self):
        ext = _make_passage_extractions([
            ("m1", "Paris", 0.9), ("m2", "Paris", 0.8),
            ("m3", "London", 0.95), ("m4", "Paris", 0.3),
        ])
        answer, confidence = fuse_passage_scalar_majority(ext)
        assert answer.lower() == "paris"
        assert confidence == pytest.approx(0.75)  # 3/4 agree

    def test_confidence_is_agreement_fraction(self):
        ext = _make_passage_extractions([
            ("m1", "Paris", 0.9), ("m2", "London", 0.8),
        ])
        _, confidence = fuse_passage_scalar_majority(ext)
        assert confidence == pytest.approx(0.5)  # 1/2 agree

    def test_single_model(self):
        ext = _make_passage_extractions([("m1", "Paris", 0.9)])
        answer, confidence = fuse_passage_scalar_majority(ext)
        assert answer.lower() == "paris"
        assert confidence == pytest.approx(1.0)


class TestFusePassageScalarQAWeighted:

    def test_high_qa_wins_over_count(self):
        ext = _make_passage_extractions([
            ("m1", "Paris", 0.3), ("m2", "Paris", 0.2),
            ("m3", "London", 0.95), ("m4", "Paris", 0.1),
        ])
        answer, confidence = fuse_passage_scalar_qa_weighted(ext)
        # Paris total qa = 0.6, London = 0.95 → London wins by qa sum
        assert answer.lower() == "london"

    def test_agreement_plus_qa(self):
        ext = _make_passage_extractions([
            ("m1", "Paris", 0.8), ("m2", "Paris", 0.7),
            ("m3", "London", 0.6), ("m4", "Paris", 0.5),
        ])
        answer, confidence = fuse_passage_scalar_qa_weighted(ext)
        # Paris qa sum = 2.0, London = 0.6 → Paris wins
        assert answer.lower() == "paris"

    def test_confidence_is_normalized_qa_sum(self):
        ext = _make_passage_extractions([
            ("m1", "Paris", 0.8), ("m2", "Paris", 0.6),
        ])
        _, confidence = fuse_passage_scalar_qa_weighted(ext)
        # Total qa = 1.4, Paris qa = 1.4 → 1.0
        assert confidence == pytest.approx(1.0)
        
    def test_confidence_between_0_and_1(self):
        ext = _make_passage_extractions([
            ("m1", "Paris", 0.8), ("m2", "London", 0.6),
        ])
        _, confidence = fuse_passage_scalar_qa_weighted(ext)
        assert 0.0 <= confidence <= 1.0


class TestFusePassageSLFusion:

    def test_returns_answer_and_opinion(self):
        ext = _make_passage_extractions([
            ("m1", "Paris", 0.9), ("m2", "Paris", 0.8),
        ])
        answer, opinion = fuse_passage_sl_fusion(ext)
        assert answer.lower() == "paris"
        assert hasattr(opinion, 'belief')
        assert hasattr(opinion, 'uncertainty')
        assert opinion.belief + opinion.disbelief + opinion.uncertainty == pytest.approx(1.0)

    def test_unanimous_high_confidence_high_belief(self):
        ext = _make_passage_extractions([
            ("m1", "Paris", 0.95), ("m2", "Paris", 0.9),
            ("m3", "Paris", 0.88), ("m4", "Paris", 0.85),
        ])
        _, opinion = fuse_passage_sl_fusion(ext)
        assert opinion.belief > 0.7
        assert opinion.uncertainty < 0.1

    def test_agreement_reduces_uncertainty(self):
        ext_agree = _make_passage_extractions([
            ("m1", "Paris", 0.9), ("m2", "Paris", 0.8),
            ("m3", "Paris", 0.7), ("m4", "Paris", 0.6),
        ])
        ext_disagree = _make_passage_extractions([
            ("m1", "Paris", 0.9), ("m2", "London", 0.8),
            ("m3", "Berlin", 0.7), ("m4", "Tokyo", 0.6),
        ])
        _, op_agree = fuse_passage_sl_fusion(ext_agree)
        _, op_disagree = fuse_passage_sl_fusion(ext_disagree)
        # Agreement: 4 opinions fused → much lower uncertainty than
        # disagreement where winning group has only 1 opinion
        assert op_agree.uncertainty < op_disagree.uncertainty

    def test_single_model_returns_base_opinion(self):
        ext = _make_passage_extractions([("m1", "Paris", 0.8)])
        answer, opinion = fuse_passage_sl_fusion(ext)
        assert answer.lower() == "paris"
        assert 0.0 < opinion.belief < 1.0


class TestFusePassageSLTrustDiscount:

    def test_weak_model_downweighted(self):
        ext = _make_passage_extractions([
            ("strong", "Paris", 0.7),
            ("weak", "London", 0.99),
        ])
        trust = {"strong": 0.9, "weak": 0.2}
        answer, opinion = fuse_passage_sl_trust_discount(ext, trust)
        # Strong model's answer should win despite weak model's higher qa_score
        assert answer.lower() == "paris"

    def test_equal_trust_similar_to_vanilla(self):
        ext = _make_passage_extractions([
            ("m1", "Paris", 0.8), ("m2", "Paris", 0.7),
        ])
        trust = {"m1": 0.9, "m2": 0.9}
        answer_td, op_td = fuse_passage_sl_trust_discount(ext, trust)
        answer_sl, op_sl = fuse_passage_sl_fusion(ext)
        assert answer_td.lower() == answer_sl.lower()

    def test_no_trust_dict_uses_defaults(self):
        ext = _make_passage_extractions([
            ("m1", "Paris", 0.8), ("m2", "London", 0.7),
        ])
        answer, opinion = fuse_passage_sl_trust_discount(ext, model_trust=None)
        assert isinstance(answer, str)
        assert len(answer) > 0


# ═══════════════════════════════════════════════════════════════════
# 2. Level 2 — Cross-passage ranking
# ═══════════════════════════════════════════════════════════════════

class TestRankPassages:

    def test_rank_by_confidence_selects_highest(self):
        passage_results = [
            {"passage_id": "p0", "answer": "Paris", "confidence": 0.9, "cosine": 0.5},
            {"passage_id": "p1", "answer": "London", "confidence": 0.3, "cosine": 0.9},
        ]
        ranked = rank_passages_by_confidence(passage_results)
        assert ranked[0]["passage_id"] == "p0"

    def test_rank_by_conf_x_cosine_balances(self):
        passage_results = [
            {"passage_id": "p0", "answer": "Paris", "confidence": 0.9, "cosine": 0.3},
            {"passage_id": "p1", "answer": "London", "confidence": 0.5, "cosine": 0.9},
        ]
        ranked = rank_passages_by_confidence_x_cosine(passage_results)
        # p0: 0.9×0.3=0.27, p1: 0.5×0.9=0.45 → p1 wins
        assert ranked[0]["passage_id"] == "p1"

    def test_select_answer_from_ranking(self):
        passage_results = [
            {"passage_id": "p0", "answer": "Paris", "confidence": 0.9, "cosine": 0.9},
            {"passage_id": "p1", "answer": "London", "confidence": 0.5, "cosine": 0.5},
        ]
        ranked = rank_passages_by_confidence(passage_results)
        answer = select_answer_from_ranking(ranked)
        assert answer == "Paris"


# ═══════════════════════════════════════════════════════════════════
# 3. Full pipeline
# ═══════════════════════════════════════════════════════════════════

class TestEvaluateAllStrategies:

    def test_returns_all_strategy_names(self):
        q_data = _make_question_data([
            (0.8, [("m1", "Paris", 0.9), ("m2", "Paris", 0.8)]),
            (0.6, [("m1", "London", 0.7), ("m2", "London", 0.6)]),
        ])
        trust = {"m1": 0.9, "m2": 0.8}
        results = evaluate_all_strategies(q_data, trust)
        for name in STRATEGY_NAMES:
            assert name in results, f"Missing strategy: {name}"
            assert "answer" in results[name]

    def test_all_strategies_return_valid_answers(self):
        q_data = _make_question_data([
            (0.9, [("roberta", "Paris", 0.9), ("electra", "Paris", 0.85),
                    ("distilbert", "Paris", 0.7), ("bert_tiny", "London", 0.6)]),
            (0.5, [("roberta", "Tokyo", 0.4), ("electra", "Berlin", 0.3),
                    ("distilbert", "Tokyo", 0.5), ("bert_tiny", "Tokyo", 0.45)]),
        ])
        trust = {"roberta": 0.85, "electra": 0.8, "distilbert": 0.7, "bert_tiny": 0.3}
        results = evaluate_all_strategies(q_data, trust)
        for name, res in results.items():
            assert isinstance(res["answer"], str), f"{name} answer not a string"
            assert len(res["answer"]) > 0, f"{name} answer is empty"

    def test_single_passage_single_model(self):
        q_data = _make_question_data([
            (0.8, [("roberta", "Paris", 0.9)]),
        ])
        trust = {"roberta": 0.9}
        results = evaluate_all_strategies(q_data, trust)
        for name, res in results.items():
            assert res["answer"].lower() == "paris", f"{name}: got '{res['answer']}'"


# ═══════════════════════════════════════════════════════════════════
# 4. Edge cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_empty_extractions_handled(self):
        ext = _make_passage_extractions([])
        # Should return empty/fallback
        answer, conf = fuse_passage_scalar_majority(ext)
        assert answer == ""
        assert conf == 0.0

    def test_all_zero_qa_scores(self):
        ext = _make_passage_extractions([
            ("m1", "Paris", 0.0), ("m2", "London", 0.0),
        ])
        answer, conf = fuse_passage_scalar_qa_weighted(ext)
        assert isinstance(answer, str)

    def test_identical_qa_scores_uses_count(self):
        ext = _make_passage_extractions([
            ("m1", "Paris", 0.5), ("m2", "Paris", 0.5),
            ("m3", "London", 0.5),
        ])
        answer, _ = fuse_passage_scalar_qa_weighted(ext)
        # Paris: qa_sum=1.0, London: qa_sum=0.5 → Paris
        assert answer.lower() == "paris"

    def test_many_passages(self):
        q_data = _make_question_data([
            (0.9 - 0.05*i, [
                ("m1", f"ans_{i%3}", 0.5 + 0.1*(i%3)),
                ("m2", f"ans_{i%3}", 0.4 + 0.1*(i%3)),
            ])
            for i in range(10)
        ])
        trust = {"m1": 0.9, "m2": 0.8}
        results = evaluate_all_strategies(q_data, trust)
        for name, res in results.items():
            assert isinstance(res["answer"], str)
