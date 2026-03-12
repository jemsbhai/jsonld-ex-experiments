"""Tests for EN3.2-H1c Core — Multi-Extractor RAG Fusion.

Tests multi-model answer fusion strategies (scalar and SL),
answer grouping, abstention, and comparison metrics.

RED phase: tests before implementation.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest experiments/EN3/tests/test_en3_2_h1c.py -v
"""
from __future__ import annotations

import pytest

from EN3.en3_2_h1c_core import (
    group_answers_fuzzy,
    fuse_answer_groups_scalar,
    fuse_answer_groups_sl,
    select_answer_majority,
    select_answer_weighted_qa,
    select_answer_model_agreement,
    select_answer_sl_fusion,
    select_answer_sl_trust_discount,
    compute_abstention_signal,
    SCALAR_STRATEGY_NAMES,
    SL_STRATEGY_NAMES,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

def _make_extractions(entries):
    """Build multi-model extraction data.

    entries: list of (model, passage_id, answer, qa_score)
    Returns: dict model → {pid: {answer, qa_score}}
    """
    result = {}
    for model, pid, answer, qa_score in entries:
        if model not in result:
            result[model] = {}
        result[model][pid] = {"answer": answer, "qa_score": qa_score}
    return result


def _make_passages(pids_scores):
    """Build passage list from (pid, score) pairs."""
    return [
        {"passage_id": pid, "score": score, "is_poison": False, "is_gold": False}
        for pid, score in pids_scores
    ]


# ═══════════════════════════════════════════════════════════════════
# 1. Answer grouping
# ═══════════════════════════════════════════════════════════════════

class TestGroupAnswersFuzzy:

    def test_identical_answers_single_group(self):
        entries = [
            ("m1", "p0", "Paris", 0.9),
            ("m2", "p0", "Paris", 0.8),
            ("m1", "p1", "paris", 0.7),  # case difference
        ]
        groups = group_answers_fuzzy(_make_extractions(entries))
        assert len(groups) == 1

    def test_different_answers_separate_groups(self):
        entries = [
            ("m1", "p0", "Paris", 0.9),
            ("m2", "p0", "London", 0.8),
            ("m3", "p0", "Berlin", 0.7),
        ]
        groups = group_answers_fuzzy(_make_extractions(entries))
        assert len(groups) == 3

    def test_group_members_preserved(self):
        entries = [
            ("m1", "p0", "Denver Broncos", 0.9),
            ("m2", "p0", "Broncos", 0.85),  # substring match
            ("m1", "p1", "Denver Broncos", 0.7),
        ]
        groups = group_answers_fuzzy(_make_extractions(entries))
        # "Broncos" and "Denver Broncos" should fuzzy-match
        assert len(groups) == 1
        # Group should have 3 members
        canonical = list(groups.keys())[0]
        assert len(groups[canonical]) == 3

    def test_each_member_has_required_fields(self):
        entries = [("m1", "p0", "Paris", 0.9)]
        groups = group_answers_fuzzy(_make_extractions(entries))
        canonical = list(groups.keys())[0]
        member = groups[canonical][0]
        assert "model" in member
        assert "passage_id" in member
        assert "answer" in member
        assert "qa_score" in member

    def test_empty_answers_excluded(self):
        entries = [
            ("m1", "p0", "Paris", 0.9),
            ("m2", "p0", "", 0.5),
            ("m3", "p0", "Paris", 0.8),
        ]
        groups = group_answers_fuzzy(_make_extractions(entries))
        total_members = sum(len(v) for v in groups.values())
        assert total_members == 2  # empty answer excluded


# ═══════════════════════════════════════════════════════════════════
# 2. Scalar fusion strategies
# ═══════════════════════════════════════════════════════════════════

class TestScalarStrategies:

    def _simple_groups(self):
        """3 models, 2 passages: 'Paris' from 3 extractions, 'London' from 1."""
        entries = [
            ("m1", "p0", "Paris", 0.9),
            ("m2", "p0", "Paris", 0.85),
            ("m1", "p1", "Paris", 0.7),
            ("m2", "p1", "London", 0.95),
        ]
        return group_answers_fuzzy(_make_extractions(entries))

    def test_majority_selects_most_common(self):
        groups = self._simple_groups()
        answer = select_answer_majority(groups)
        assert answer.lower() == "paris"

    def test_weighted_qa_selects_highest_qa_sum(self):
        # London has highest single qa_score (0.95) but Paris has higher sum
        groups = self._simple_groups()
        answer = select_answer_weighted_qa(groups)
        assert answer.lower() == "paris"

    def test_model_agreement_counts_distinct_models(self):
        entries = [
            ("m1", "p0", "Paris", 0.9),
            ("m2", "p0", "Paris", 0.8),
            ("m3", "p0", "London", 0.95),
            ("m1", "p1", "Paris", 0.3),  # same model, doesn't add new model
        ]
        groups = group_answers_fuzzy(_make_extractions(entries))
        answer = select_answer_model_agreement(groups)
        # Paris: 2 distinct models (m1, m2). London: 1 (m3).
        assert answer.lower() == "paris"

    def test_single_answer_group_returns_it(self):
        entries = [
            ("m1", "p0", "Paris", 0.9),
            ("m2", "p0", "Paris", 0.8),
        ]
        groups = group_answers_fuzzy(_make_extractions(entries))
        assert select_answer_majority(groups).lower() == "paris"
        assert select_answer_weighted_qa(groups).lower() == "paris"
        assert select_answer_model_agreement(groups).lower() == "paris"


# ═══════════════════════════════════════════════════════════════════
# 3. SL fusion strategies
# ═══════════════════════════════════════════════════════════════════

class TestSLStrategies:

    def test_sl_fusion_returns_answer(self):
        entries = [
            ("m1", "p0", "Paris", 0.9),
            ("m2", "p0", "Paris", 0.85),
            ("m1", "p1", "London", 0.6),
        ]
        groups = group_answers_fuzzy(_make_extractions(entries))
        answer = select_answer_sl_fusion(groups)
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_sl_fusion_prefers_well_supported_answer(self):
        entries = [
            ("m1", "p0", "Paris", 0.9),
            ("m2", "p0", "Paris", 0.85),
            ("m3", "p0", "Paris", 0.8),
            ("m4", "p0", "London", 0.6),
        ]
        groups = group_answers_fuzzy(_make_extractions(entries))
        answer = select_answer_sl_fusion(groups)
        assert answer.lower() == "paris"

    def test_sl_trust_discount_downweights_weak_model(self):
        """With trust discount, a weak model's lone answer shouldn't win
        even if it has high qa_score."""
        entries = [
            ("strong1", "p0", "Paris", 0.7),
            ("strong2", "p0", "Paris", 0.65),
            ("weak", "p0", "London", 0.99),  # high qa but weak model
        ]
        groups = group_answers_fuzzy(_make_extractions(entries))
        model_trust = {"strong1": 0.9, "strong2": 0.85, "weak": 0.2}
        answer = select_answer_sl_trust_discount(groups, model_trust)
        assert answer.lower() == "paris"

    def test_sl_trust_discount_without_trust_dict_falls_back(self):
        entries = [
            ("m1", "p0", "Paris", 0.9),
            ("m2", "p0", "London", 0.6),
        ]
        groups = group_answers_fuzzy(_make_extractions(entries))
        # No trust dict → equal trust → should still work
        answer = select_answer_sl_trust_discount(groups, model_trust=None)
        assert isinstance(answer, str)


# ═══════════════════════════════════════════════════════════════════
# 4. Abstention signal
# ═══════════════════════════════════════════════════════════════════

class TestAbstentionSignal:

    def test_high_agreement_low_abstention(self):
        """All models agree → low abstention signal (high confidence)."""
        entries = [
            ("m1", "p0", "Paris", 0.95),
            ("m2", "p0", "Paris", 0.90),
            ("m3", "p0", "Paris", 0.88),
        ]
        groups = group_answers_fuzzy(_make_extractions(entries))
        signal = compute_abstention_signal(groups)
        assert 0.0 <= signal <= 1.0

    def test_high_disagreement_high_abstention(self):
        """All models disagree → high abstention signal."""
        entries = [
            ("m1", "p0", "Paris", 0.9),
            ("m2", "p0", "London", 0.9),
            ("m3", "p0", "Berlin", 0.9),
        ]
        groups = group_answers_fuzzy(_make_extractions(entries))
        signal_disagree = compute_abstention_signal(groups)

        # Compare with all-agree
        entries_agree = [
            ("m1", "p0", "Paris", 0.9),
            ("m2", "p0", "Paris", 0.9),
            ("m3", "p0", "Paris", 0.9),
        ]
        groups_agree = group_answers_fuzzy(_make_extractions(entries_agree))
        signal_agree = compute_abstention_signal(groups_agree)

        assert signal_disagree > signal_agree

    def test_abstention_in_range(self):
        entries = [
            ("m1", "p0", "Paris", 0.5),
            ("m2", "p0", "London", 0.5),
        ]
        groups = group_answers_fuzzy(_make_extractions(entries))
        signal = compute_abstention_signal(groups)
        assert 0.0 <= signal <= 1.0


# ═══════════════════════════════════════════════════════════════════
# 5. Edge cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_single_extraction(self):
        entries = [("m1", "p0", "Paris", 0.9)]
        groups = group_answers_fuzzy(_make_extractions(entries))
        assert select_answer_majority(groups).lower() == "paris"
        assert select_answer_sl_fusion(groups).lower() == "paris"

    def test_all_low_qa_scores(self):
        entries = [
            ("m1", "p0", "Paris", 0.01),
            ("m2", "p0", "London", 0.02),
        ]
        groups = group_answers_fuzzy(_make_extractions(entries))
        answer = select_answer_sl_fusion(groups)
        assert isinstance(answer, str)

    def test_many_models_many_passages(self):
        entries = [
            (f"m{m}", f"p{p}", f"answer_{(m+p) % 3}", 0.5 + 0.1 * m)
            for m in range(4) for p in range(10)
        ]
        groups = group_answers_fuzzy(_make_extractions(entries))
        answer = select_answer_sl_fusion(groups)
        assert isinstance(answer, str)
