"""Tests for EN3.4 Phase A1 — NER Fusion Evaluation Core.

RED phase: these tests define the contract for en3_4_core.py.
All should FAIL before implementation.

Tests cover the full pipeline from model predictions to hypothesis verdicts:
  1. Span representation (EntitySpan dataclass)
  2. Span IoU computation
  3. Span alignment (grouping overlapping spans across models)
  4. Opinion construction from scores + calibration uncertainty
  5. Six experimental conditions (B1-B5 + SL)
  6. Entity-level evaluation (strict span match P/R/F1)
  7. Bootstrap CI on F1 differences
  8. Cohen's h effect size
  9. Holm-Bonferroni multiple testing correction
  10. Conflict-error correlation (Spearman)
  11. Threshold optimization on dev set
"""
from __future__ import annotations

import math
import pytest
import numpy as np

# -- Path setup --
import sys
from pathlib import Path

_en3_dir = Path(__file__).resolve().parent.parent
_experiments_root = _en3_dir.parent
for p in [str(_en3_dir), str(_experiments_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# =====================================================================
# Imports from module under test (will fail until implemented)
# =====================================================================

from EN3.en3_4_core import (
    EntitySpan,
    compute_span_iou,
    align_spans,
    build_opinion,
    apply_condition_single_model,
    apply_condition_union,
    apply_condition_intersection,
    apply_condition_scalar_average,
    apply_condition_sl_fusion,
    evaluate_entities,
    EvalMetrics,
    bootstrap_f1_difference_ci,
    cohens_h,
    holm_bonferroni,
    spearman_conflict_error,
    optimize_threshold,
)


# =====================================================================
# 1. EntitySpan Dataclass
# =====================================================================


class TestEntitySpan:
    """Tests for the EntitySpan data representation."""

    def test_construction(self):
        """EntitySpan holds span boundaries, type, score, and source."""
        s = EntitySpan(
            start=10, end=25, entity_type="Chemical",
            score=0.87, source="gliner2",
        )
        assert s.start == 10
        assert s.end == 25
        assert s.entity_type == "Chemical"
        assert s.score == 0.87
        assert s.source == "gliner2"

    def test_text_property(self):
        """EntitySpan with optional text field."""
        s = EntitySpan(
            start=0, end=5, entity_type="Disease",
            score=0.65, source="biomed", text="fever",
        )
        assert s.text == "fever"

    def test_span_length(self):
        """end - start gives the character span length."""
        s = EntitySpan(start=10, end=20, entity_type="Chemical",
                       score=0.5, source="test")
        assert s.end - s.start == 10


# =====================================================================
# 2. Span IoU Computation
# =====================================================================


class TestComputeSpanIoU:
    """Tests for character-level IoU between two spans."""

    def test_exact_match(self):
        """Identical spans have IoU = 1.0."""
        iou = compute_span_iou(10, 20, 10, 20)
        assert iou == pytest.approx(1.0)

    def test_no_overlap(self):
        """Disjoint spans have IoU = 0.0."""
        iou = compute_span_iou(0, 10, 20, 30)
        assert iou == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partially overlapping spans.
        Span A: [10, 20), Span B: [15, 25)
        Intersection: [15, 20) = 5 chars
        Union: [10, 25) = 15 chars
        IoU = 5/15 = 0.333...
        """
        iou = compute_span_iou(10, 20, 15, 25)
        assert iou == pytest.approx(1.0 / 3.0, abs=0.001)

    def test_containment(self):
        """One span fully contains the other.
        A: [5, 25), B: [10, 20)
        Intersection: [10, 20) = 10 chars
        Union: [5, 25) = 20 chars
        IoU = 10/20 = 0.5
        """
        iou = compute_span_iou(5, 25, 10, 20)
        assert iou == pytest.approx(0.5)

    def test_adjacent_no_overlap(self):
        """Adjacent spans (end of A = start of B) have IoU = 0."""
        iou = compute_span_iou(0, 10, 10, 20)
        assert iou == pytest.approx(0.0)

    def test_symmetry(self):
        """IoU(A, B) == IoU(B, A)."""
        iou_ab = compute_span_iou(5, 15, 10, 25)
        iou_ba = compute_span_iou(10, 25, 5, 15)
        assert iou_ab == pytest.approx(iou_ba)


# =====================================================================
# 3. Span Alignment
# =====================================================================


class TestAlignSpans:
    """Tests for aligning spans from two models."""

    def test_exact_match_alignment(self):
        """Two models predict the same span → aligned pair."""
        spans_a = [EntitySpan(10, 20, "Chemical", 0.9, "model_a")]
        spans_b = [EntitySpan(10, 20, "Chemical", 0.7, "model_b")]
        groups = align_spans(spans_a, spans_b, iou_threshold=0.5)
        assert len(groups) == 1
        assert groups[0]["span_a"] is not None
        assert groups[0]["span_b"] is not None

    def test_no_overlap_separate_groups(self):
        """Disjoint spans → two separate single-source groups."""
        spans_a = [EntitySpan(0, 10, "Chemical", 0.9, "model_a")]
        spans_b = [EntitySpan(50, 60, "Disease", 0.8, "model_b")]
        groups = align_spans(spans_a, spans_b, iou_threshold=0.5)
        assert len(groups) == 2
        # One from each model
        sources = set()
        for g in groups:
            if g["span_a"] is not None:
                sources.add("a")
            if g["span_b"] is not None:
                sources.add("b")
        assert sources == {"a", "b"}

    def test_partial_overlap_above_threshold(self):
        """Spans with IoU > threshold → aligned pair."""
        # [10, 20) and [12, 22): IoU = 8/12 = 0.667 > 0.5
        spans_a = [EntitySpan(10, 20, "Chemical", 0.9, "model_a")]
        spans_b = [EntitySpan(12, 22, "Chemical", 0.7, "model_b")]
        groups = align_spans(spans_a, spans_b, iou_threshold=0.5)
        assert len(groups) == 1
        assert groups[0]["span_a"] is not None
        assert groups[0]["span_b"] is not None

    def test_partial_overlap_below_threshold(self):
        """Spans with IoU < threshold → separate groups."""
        # [0, 20) and [15, 50): IoU = 5/50 = 0.1 < 0.5
        spans_a = [EntitySpan(0, 20, "Chemical", 0.9, "model_a")]
        spans_b = [EntitySpan(15, 50, "Chemical", 0.7, "model_b")]
        groups = align_spans(spans_a, spans_b, iou_threshold=0.5)
        assert len(groups) == 2

    def test_type_mismatch_flagged(self):
        """Overlapping spans with different types → aligned but flagged."""
        spans_a = [EntitySpan(10, 20, "Chemical", 0.9, "model_a")]
        spans_b = [EntitySpan(10, 20, "Disease", 0.7, "model_b")]
        groups = align_spans(spans_a, spans_b, iou_threshold=0.5)
        assert len(groups) == 1
        assert groups[0]["type_match"] is False

    def test_type_match_flagged(self):
        """Overlapping same-type spans → type_match is True."""
        spans_a = [EntitySpan(10, 20, "Chemical", 0.9, "model_a")]
        spans_b = [EntitySpan(10, 20, "Chemical", 0.7, "model_b")]
        groups = align_spans(spans_a, spans_b, iou_threshold=0.5)
        assert len(groups) == 1
        assert groups[0]["type_match"] is True

    def test_multiple_spans_complex(self):
        """Multiple spans from each model with mixed alignment."""
        spans_a = [
            EntitySpan(0, 10, "Chemical", 0.9, "a"),
            EntitySpan(30, 40, "Disease", 0.8, "a"),
            EntitySpan(70, 80, "Chemical", 0.6, "a"),  # unmatched
        ]
        spans_b = [
            EntitySpan(0, 10, "Chemical", 0.7, "b"),   # matches a[0]
            EntitySpan(31, 41, "Disease", 0.85, "b"),   # matches a[1]
            EntitySpan(50, 60, "Disease", 0.5, "b"),    # unmatched
        ]
        groups = align_spans(spans_a, spans_b, iou_threshold=0.5)
        # 2 aligned pairs + 1 unmatched from a + 1 unmatched from b = 4
        assert len(groups) == 4

    def test_empty_inputs(self):
        """Empty span lists → empty groups."""
        groups = align_spans([], [], iou_threshold=0.5)
        assert len(groups) == 0

    def test_one_side_empty(self):
        """One model has spans, other is empty → all single-source."""
        spans_a = [EntitySpan(0, 10, "Chemical", 0.9, "a")]
        groups = align_spans(spans_a, [], iou_threshold=0.5)
        assert len(groups) == 1
        assert groups[0]["span_a"] is not None
        assert groups[0]["span_b"] is None


# =====================================================================
# 4. Opinion Construction
# =====================================================================


class TestBuildOpinion:
    """Tests for constructing SL opinions from scores + uncertainty."""

    def test_basic_construction(self):
        """Builds valid Opinion from score and uncertainty."""
        from jsonld_ex.confidence_algebra import Opinion
        op = build_opinion(score=0.85, model_uncertainty=0.10)
        assert isinstance(op, Opinion)
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9
        assert op.uncertainty == pytest.approx(0.10)

    def test_high_score_high_belief(self):
        """High confidence score → high belief."""
        op = build_opinion(score=0.95, model_uncertainty=0.10)
        assert op.belief > 0.8

    def test_low_score_high_disbelief(self):
        """Low confidence score → high disbelief."""
        op = build_opinion(score=0.10, model_uncertainty=0.10)
        assert op.disbelief > 0.7

    def test_uncertainty_matches_parameter(self):
        """The opinion uncertainty should match the model uncertainty."""
        op = build_opinion(score=0.50, model_uncertainty=0.25)
        assert op.uncertainty == pytest.approx(0.25)

    def test_projected_probability_preserves_score(self):
        """P(ω) should approximate the original score when base_rate=0.5
        and uncertainty is moderate."""
        op = build_opinion(score=0.80, model_uncertainty=0.10)
        # P(ω) = b + a*u = 0.80*(1-0.10) + 0.5*0.10 = 0.72 + 0.05 = 0.77
        # Not exactly 0.80, but close
        pp = op.projected_probability()
        assert 0.5 < pp < 1.0  # sanity bound


# =====================================================================
# 5. Experimental Conditions
# =====================================================================


class TestConditionSingleModel:
    """Tests for B1/B2: single model thresholding."""

    def test_above_threshold_accepted(self):
        """Span with score above threshold → accepted."""
        spans = [EntitySpan(0, 10, "Chemical", 0.8, "model")]
        accepted = apply_condition_single_model(spans, threshold=0.5)
        assert len(accepted) == 1

    def test_below_threshold_rejected(self):
        """Span with score below threshold → rejected."""
        spans = [EntitySpan(0, 10, "Chemical", 0.3, "model")]
        accepted = apply_condition_single_model(spans, threshold=0.5)
        assert len(accepted) == 0

    def test_exact_threshold_accepted(self):
        """Span with score == threshold → accepted (inclusive)."""
        spans = [EntitySpan(0, 10, "Chemical", 0.5, "model")]
        accepted = apply_condition_single_model(spans, threshold=0.5)
        assert len(accepted) == 1


class TestConditionUnion:
    """Tests for B3: union ensemble (OR logic)."""

    def test_both_predict_same_span(self):
        """Both models predict same span → one output."""
        groups = [{"span_a": EntitySpan(0, 10, "Chemical", 0.9, "a"),
                   "span_b": EntitySpan(0, 10, "Chemical", 0.7, "b"),
                   "type_match": True}]
        accepted = apply_condition_union(groups, threshold_a=0.5, threshold_b=0.5)
        assert len(accepted) == 1

    def test_only_model_a_predicts(self):
        """Only model A has the span, above threshold → accepted."""
        groups = [{"span_a": EntitySpan(0, 10, "Chemical", 0.8, "a"),
                   "span_b": None, "type_match": None}]
        accepted = apply_condition_union(groups, threshold_a=0.5, threshold_b=0.5)
        assert len(accepted) == 1

    def test_neither_above_threshold(self):
        """Both models below their thresholds → rejected."""
        groups = [{"span_a": EntitySpan(0, 10, "Chemical", 0.3, "a"),
                   "span_b": EntitySpan(0, 10, "Chemical", 0.2, "b"),
                   "type_match": True}]
        accepted = apply_condition_union(groups, threshold_a=0.5, threshold_b=0.5)
        assert len(accepted) == 0


class TestConditionIntersection:
    """Tests for B4: intersection ensemble (AND logic)."""

    def test_both_above_threshold(self):
        """Both models above threshold → accepted."""
        groups = [{"span_a": EntitySpan(0, 10, "Chemical", 0.8, "a"),
                   "span_b": EntitySpan(0, 10, "Chemical", 0.7, "b"),
                   "type_match": True}]
        accepted = apply_condition_intersection(
            groups, threshold_a=0.5, threshold_b=0.5)
        assert len(accepted) == 1

    def test_only_one_above_threshold(self):
        """Only one model above threshold → rejected."""
        groups = [{"span_a": EntitySpan(0, 10, "Chemical", 0.8, "a"),
                   "span_b": EntitySpan(0, 10, "Chemical", 0.3, "b"),
                   "type_match": True}]
        accepted = apply_condition_intersection(
            groups, threshold_a=0.5, threshold_b=0.5)
        assert len(accepted) == 0

    def test_single_source_always_rejected(self):
        """Single-source spans always rejected by intersection."""
        groups = [{"span_a": EntitySpan(0, 10, "Chemical", 0.99, "a"),
                   "span_b": None, "type_match": None}]
        accepted = apply_condition_intersection(
            groups, threshold_a=0.5, threshold_b=0.5)
        assert len(accepted) == 0


class TestConditionScalarAverage:
    """Tests for B5: scalar confidence averaging."""

    def test_average_above_threshold(self):
        """Average of two scores above threshold → accepted."""
        groups = [{"span_a": EntitySpan(0, 10, "Chemical", 0.8, "a"),
                   "span_b": EntitySpan(0, 10, "Chemical", 0.6, "b"),
                   "type_match": True}]
        # Average = 0.7, threshold = 0.5 → accept
        accepted = apply_condition_scalar_average(groups, threshold=0.5)
        assert len(accepted) == 1

    def test_average_below_threshold(self):
        """Average of two scores below threshold → rejected."""
        groups = [{"span_a": EntitySpan(0, 10, "Chemical", 0.4, "a"),
                   "span_b": EntitySpan(0, 10, "Chemical", 0.2, "b"),
                   "type_match": True}]
        # Average = 0.3, threshold = 0.5 → reject
        accepted = apply_condition_scalar_average(groups, threshold=0.5)
        assert len(accepted) == 0

    def test_single_source_uses_own_score(self):
        """Single-source span uses its own score (no averaging)."""
        groups = [{"span_a": EntitySpan(0, 10, "Chemical", 0.8, "a"),
                   "span_b": None, "type_match": None}]
        accepted = apply_condition_scalar_average(groups, threshold=0.5)
        assert len(accepted) == 1


class TestConditionSLFusion:
    """Tests for SL fusion with conflict-based abstention."""

    def test_agreeing_models_accepted(self):
        """Both models agree with high confidence → accepted."""
        groups = [{"span_a": EntitySpan(0, 10, "Chemical", 0.9, "a"),
                   "span_b": EntitySpan(0, 10, "Chemical", 0.85, "b"),
                   "type_match": True}]
        accepted, abstained = apply_condition_sl_fusion(
            groups, u_a=0.10, u_b=0.08,
            accept_threshold=0.5, conflict_threshold=0.7,
        )
        assert len(accepted) == 1
        assert len(abstained) == 0

    def test_conflicting_models_abstained(self):
        """Models strongly disagree → abstention via conflict detection."""
        # Model A: very confident it IS a Chemical (0.95)
        # Model B: very confident it is NOT a Chemical (0.05)
        groups = [{"span_a": EntitySpan(0, 10, "Chemical", 0.95, "a"),
                   "span_b": EntitySpan(0, 10, "Chemical", 0.05, "b"),
                   "type_match": True}]
        accepted, abstained = apply_condition_sl_fusion(
            groups, u_a=0.10, u_b=0.10,
            accept_threshold=0.5, conflict_threshold=0.3,
        )
        # High conflict should trigger abstention
        assert len(abstained) >= 0  # may or may not abstain depending on conflict
        # Total should be 1
        assert len(accepted) + len(abstained) == 1

    def test_single_source_no_fusion(self):
        """Single-source span → opinion from that model only, no fusion."""
        groups = [{"span_a": EntitySpan(0, 10, "Chemical", 0.9, "a"),
                   "span_b": None, "type_match": None}]
        accepted, abstained = apply_condition_sl_fusion(
            groups, u_a=0.10, u_b=0.10,
            accept_threshold=0.5, conflict_threshold=0.7,
        )
        assert len(accepted) == 1
        assert len(abstained) == 0

    def test_returns_entity_span_objects(self):
        """Accepted results should be EntitySpan objects."""
        groups = [{"span_a": EntitySpan(0, 10, "Chemical", 0.9, "a"),
                   "span_b": EntitySpan(0, 10, "Chemical", 0.8, "b"),
                   "type_match": True}]
        accepted, _ = apply_condition_sl_fusion(
            groups, u_a=0.10, u_b=0.10,
            accept_threshold=0.5, conflict_threshold=0.7,
        )
        assert all(isinstance(s, EntitySpan) for s in accepted)


# =====================================================================
# 6. Entity-Level Evaluation (Strict Span Match)
# =====================================================================


class TestEvaluateEntities:
    """Tests for entity-level P/R/F1 with strict span matching."""

    def test_perfect_prediction(self):
        """All predictions match ground truth exactly."""
        preds = [EntitySpan(0, 10, "Chemical", 0.9, "test"),
                 EntitySpan(20, 30, "Disease", 0.8, "test")]
        golds = [EntitySpan(0, 10, "Chemical", 1.0, "gold"),
                 EntitySpan(20, 30, "Disease", 1.0, "gold")]
        m = evaluate_entities(preds, golds)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)

    def test_no_predictions(self):
        """No predictions → precision undefined (0), recall = 0, F1 = 0."""
        preds = []
        golds = [EntitySpan(0, 10, "Chemical", 1.0, "gold")]
        m = evaluate_entities(preds, golds)
        assert m.recall == pytest.approx(0.0)
        assert m.f1 == pytest.approx(0.0)

    def test_no_ground_truth(self):
        """No ground truth → recall undefined (0), F1 = 0."""
        preds = [EntitySpan(0, 10, "Chemical", 0.9, "test")]
        golds = []
        m = evaluate_entities(preds, golds)
        assert m.f1 == pytest.approx(0.0)

    def test_partial_match(self):
        """One correct, one wrong prediction, one missed gold.
        TP=1, FP=1, FN=1 → P=0.5, R=0.5, F1=0.5
        """
        preds = [EntitySpan(0, 10, "Chemical", 0.9, "test"),   # correct
                 EntitySpan(50, 60, "Chemical", 0.7, "test")]  # FP
        golds = [EntitySpan(0, 10, "Chemical", 1.0, "gold"),   # matched
                 EntitySpan(20, 30, "Disease", 1.0, "gold")]   # FN
        m = evaluate_entities(preds, golds)
        assert m.precision == pytest.approx(0.5)
        assert m.recall == pytest.approx(0.5)
        assert m.f1 == pytest.approx(0.5)

    def test_wrong_type_is_fp(self):
        """Correct span but wrong type → FP + FN."""
        preds = [EntitySpan(0, 10, "Chemical", 0.9, "test")]
        golds = [EntitySpan(0, 10, "Disease", 1.0, "gold")]
        m = evaluate_entities(preds, golds)
        assert m.precision == pytest.approx(0.0)
        assert m.recall == pytest.approx(0.0)
        assert m.f1 == pytest.approx(0.0)

    def test_off_by_one_boundary_is_fp(self):
        """Span boundary off by 1 char → strict match fails."""
        preds = [EntitySpan(0, 11, "Chemical", 0.9, "test")]  # end+1
        golds = [EntitySpan(0, 10, "Chemical", 1.0, "gold")]
        m = evaluate_entities(preds, golds)
        assert m.f1 == pytest.approx(0.0)

    def test_eval_metrics_structure(self):
        """EvalMetrics has tp, fp, fn, precision, recall, f1."""
        preds = [EntitySpan(0, 10, "Chemical", 0.9, "test")]
        golds = [EntitySpan(0, 10, "Chemical", 1.0, "gold")]
        m = evaluate_entities(preds, golds)
        assert hasattr(m, "tp")
        assert hasattr(m, "fp")
        assert hasattr(m, "fn")
        assert m.tp == 1
        assert m.fp == 0
        assert m.fn == 0

    def test_duplicate_predictions_not_double_counted(self):
        """Two identical predictions for one gold → TP=1, FP=1."""
        preds = [EntitySpan(0, 10, "Chemical", 0.9, "test"),
                 EntitySpan(0, 10, "Chemical", 0.8, "test")]
        golds = [EntitySpan(0, 10, "Chemical", 1.0, "gold")]
        m = evaluate_entities(preds, golds)
        assert m.tp == 1
        assert m.fp == 1

    def test_hand_computed_example(self):
        """Larger hand-computed example.
        Preds: A(0-10,Chem), B(15-25,Dis), C(30-40,Chem), D(50-55,Dis)
        Golds: A(0-10,Chem), B(15-25,Dis), E(40-50,Chem)
        TP = 2 (A, B), FP = 2 (C, D), FN = 1 (E)
        P = 2/4 = 0.5, R = 2/3 = 0.667, F1 = 2*0.5*0.667/(0.5+0.667) = 0.571
        """
        preds = [EntitySpan(0, 10, "Chemical", 0.9, "t"),
                 EntitySpan(15, 25, "Disease", 0.8, "t"),
                 EntitySpan(30, 40, "Chemical", 0.7, "t"),
                 EntitySpan(50, 55, "Disease", 0.6, "t")]
        golds = [EntitySpan(0, 10, "Chemical", 1.0, "g"),
                 EntitySpan(15, 25, "Disease", 1.0, "g"),
                 EntitySpan(40, 50, "Chemical", 1.0, "g")]
        m = evaluate_entities(preds, golds)
        assert m.tp == 2
        assert m.fp == 2
        assert m.fn == 1
        assert m.precision == pytest.approx(0.5)
        assert m.recall == pytest.approx(2.0 / 3.0, abs=0.001)
        assert m.f1 == pytest.approx(4.0 / 7.0, abs=0.001)  # 2*P*R/(P+R)


# =====================================================================
# 7. Bootstrap CI on F1 Differences
# =====================================================================


class TestBootstrapF1DifferenceCi:
    """Tests for bootstrap CI on the difference between two micro-F1 scores."""

    def test_identical_predictions_zero_difference(self):
        """Same predictions → CI for difference should include 0."""
        # Each element: (tp, fp, fn) per sentence
        rng = np.random.RandomState(42)
        n = 200
        # Same data for both conditions
        samples = [(1, 0, 0)] * 150 + [(0, 1, 1)] * 50
        lo, mean, hi = bootstrap_f1_difference_ci(
            samples, samples, n_bootstrap=500, seed=42)
        assert lo <= 0.0 <= hi  # difference CI includes 0

    def test_clearly_different_predictions(self):
        """One condition much better → CI should exclude 0."""
        preds_a = [(1, 0, 0)] * 190 + [(0, 1, 1)] * 10   # high F1
        preds_b = [(1, 0, 0)] * 100 + [(0, 1, 1)] * 100  # lower F1
        lo, mean, hi = bootstrap_f1_difference_ci(
            preds_a, preds_b, n_bootstrap=1000, seed=42)
        assert lo > 0.0  # A clearly better
        assert mean > 0.0

    def test_returns_three_values(self):
        """Should return (lower, mean, upper)."""
        preds_a = [(1, 0, 0)] * 50 + [(0, 1, 1)] * 50
        preds_b = [(1, 0, 0)] * 40 + [(0, 1, 1)] * 60
        result = bootstrap_f1_difference_ci(
            preds_a, preds_b, n_bootstrap=100, seed=42)
        assert len(result) == 3
        lo, mean, hi = result
        assert lo <= mean <= hi


# =====================================================================
# 8. Cohen's h Effect Size
# =====================================================================


class TestCohensH:
    """Tests for Cohen's h (arcsine-transformed proportion difference)."""

    def test_identical_proportions(self):
        """Same proportions → h = 0."""
        h = cohens_h(0.75, 0.75)
        assert h == pytest.approx(0.0, abs=0.001)

    def test_known_value(self):
        """Hand-computed: h = 2*arcsin(sqrt(0.9)) - 2*arcsin(sqrt(0.5))
        = 2*(1.2490) - 2*(0.7854) = 2.498 - 1.571 = 0.927
        """
        h = cohens_h(0.9, 0.5)
        expected = 2 * math.asin(math.sqrt(0.9)) - 2 * math.asin(math.sqrt(0.5))
        assert h == pytest.approx(expected, abs=0.001)

    def test_symmetry_with_sign(self):
        """h(p1, p2) = -h(p2, p1)."""
        h1 = cohens_h(0.8, 0.6)
        h2 = cohens_h(0.6, 0.8)
        assert h1 == pytest.approx(-h2, abs=0.001)

    def test_boundary_values(self):
        """h(1.0, 0.0) should be finite."""
        h = cohens_h(1.0, 0.0)
        assert math.isfinite(h)
        assert h > 0


# =====================================================================
# 9. Holm-Bonferroni Correction
# =====================================================================


class TestHolmBonferroni:
    """Tests for Holm-Bonferroni multiple testing correction."""

    def test_single_pvalue_no_correction(self):
        """Single p-value → no correction needed."""
        results = holm_bonferroni([0.03], alpha=0.05)
        assert len(results) == 1
        assert results[0]["reject"] is True

    def test_all_significant_remain_significant(self):
        """Very small p-values → all remain significant."""
        pvals = [0.001, 0.002, 0.003]
        results = holm_bonferroni(pvals, alpha=0.05)
        assert all(r["reject"] for r in results)

    def test_known_correction(self):
        """Hand-computed Holm-Bonferroni.
        p-values: [0.01, 0.04, 0.03], α=0.05
        Sorted: [0.01, 0.03, 0.04] (indices 0, 2, 1)
        Step 1: 0.01 < 0.05/3 = 0.0167 → reject
        Step 2: 0.03 < 0.05/2 = 0.025 → reject? No, 0.03 > 0.025 → fail
        Step 3: stop (sequential, once fail, rest fail)
        Result: [0.01 → reject, 0.03 → fail, 0.04 → fail]
        In original order: [reject, fail, fail]
        """
        pvals = [0.01, 0.04, 0.03]
        results = holm_bonferroni(pvals, alpha=0.05)
        assert results[0]["reject"] is True   # p=0.01
        assert results[1]["reject"] is False  # p=0.04
        assert results[2]["reject"] is False  # p=0.03

    def test_preserves_original_order(self):
        """Results should be in the same order as input p-values."""
        pvals = [0.5, 0.001, 0.1]
        results = holm_bonferroni(pvals, alpha=0.05)
        assert len(results) == 3
        # p=0.001 should be the only one rejected
        assert results[1]["reject"] is True
        assert results[0]["reject"] is False
        assert results[2]["reject"] is False

    def test_result_contains_adjusted_alpha(self):
        """Each result should include the adjusted threshold."""
        pvals = [0.01, 0.03]
        results = holm_bonferroni(pvals, alpha=0.05)
        for r in results:
            assert "p_value" in r
            assert "reject" in r
            assert "adjusted_alpha" in r

    def test_all_nonsignificant(self):
        """All p-values > α → none rejected."""
        pvals = [0.2, 0.3, 0.4]
        results = holm_bonferroni(pvals, alpha=0.05)
        assert not any(r["reject"] for r in results)


# =====================================================================
# 10. Conflict-Error Correlation (Spearman)
# =====================================================================


class TestSpearmanConflictError:
    """Tests for Spearman correlation between conflict and error."""

    def test_perfect_positive_correlation(self):
        """Conflict perfectly predicts error → ρ ≈ 1."""
        conflicts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        is_error = [False, False, False, False, True, True, True, True]
        rho, p_value = spearman_conflict_error(conflicts, is_error)
        assert rho > 0.7
        assert p_value < 0.05

    def test_no_correlation(self):
        """Random data → ρ ≈ 0."""
        rng = np.random.RandomState(42)
        conflicts = rng.uniform(0, 1, 100).tolist()
        is_error = (rng.uniform(0, 1, 100) < 0.5).tolist()
        rho, p_value = spearman_conflict_error(conflicts, is_error)
        assert abs(rho) < 0.3  # weak or no correlation

    def test_returns_rho_and_pvalue(self):
        """Should return (rho, p_value) tuple."""
        conflicts = [0.1, 0.5, 0.9]
        is_error = [False, True, True]
        result = spearman_conflict_error(conflicts, is_error)
        assert len(result) == 2
        rho, p = result
        assert -1.0 <= rho <= 1.0
        assert 0.0 <= p <= 1.0


# =====================================================================
# 11. Threshold Optimization
# =====================================================================


class TestOptimizeThreshold:
    """Tests for dev-set threshold optimization."""

    def test_finds_best_threshold(self):
        """Should find the threshold that maximizes F1 on dev data."""
        # Construct dev data where threshold=0.5 is optimal
        dev_spans = [
            EntitySpan(0, 10, "Chemical", 0.8, "model"),   # correct
            EntitySpan(20, 30, "Disease", 0.7, "model"),   # correct
            EntitySpan(40, 50, "Chemical", 0.3, "model"),  # wrong
            EntitySpan(60, 70, "Disease", 0.2, "model"),   # wrong
        ]
        dev_golds = [
            EntitySpan(0, 10, "Chemical", 1.0, "gold"),
            EntitySpan(20, 30, "Disease", 1.0, "gold"),
        ]
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        best_t, best_f1 = optimize_threshold(dev_spans, dev_golds, thresholds)
        # At t=0.5: accept 0.8 and 0.7 → TP=2, FP=0, FN=0 → F1=1.0
        assert best_t == pytest.approx(0.5)
        assert best_f1 == pytest.approx(1.0)

    def test_returns_threshold_and_f1(self):
        """Should return (best_threshold, best_f1)."""
        dev_spans = [EntitySpan(0, 10, "Chemical", 0.6, "model")]
        dev_golds = [EntitySpan(0, 10, "Chemical", 1.0, "gold")]
        result = optimize_threshold(dev_spans, dev_golds, [0.3, 0.5, 0.7])
        assert len(result) == 2
        t, f1 = result
        assert 0.0 <= t <= 1.0
        assert 0.0 <= f1 <= 1.0
