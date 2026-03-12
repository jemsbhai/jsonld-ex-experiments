"""Tests for EN3.2-H1b — Poison Passage Detection (Binary Classification).

Tests signal computation for poison detection, AUROC evaluation,
and precision-at-recall operating points.

RED phase: all tests written before implementation.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest experiments/EN3/tests/test_en3_2_h1b.py -v
"""
from __future__ import annotations

import math
import pytest

from EN3.en3_2_h1b_core import (
    compute_poison_detection_signals,
    compute_auroc,
    precision_at_recall,
    SCALAR_DETECTION_SIGNALS,
    SL_DETECTION_SIGNALS,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

def _make_passages(scores, poison_flags=None):
    """Create synthetic passage list."""
    if poison_flags is None:
        poison_flags = [False] * len(scores)
    return [
        {"passage_id": f"p_{i}", "score": s, "is_poison": pf, "is_gold": i == 0}
        for i, (s, pf) in enumerate(zip(scores, poison_flags))
    ]


def _make_extractions(n, qa_scores=None, answers=None):
    """Create synthetic extraction dict."""
    if qa_scores is None:
        qa_scores = [0.5] * n
    if answers is None:
        answers = [f"answer_{i}" for i in range(n)]
    return {
        f"p_{i}": {"answer": answers[i], "qa_score": qa_scores[i]}
        for i in range(n)
    }


# ═══════════════════════════════════════════════════════════════════
# 1. Signal computation
# ═══════════════════════════════════════════════════════════════════

class TestComputePoisonDetectionSignals:
    """Tests for compute_poison_detection_signals."""

    def test_returns_all_expected_keys(self):
        passages = _make_passages([0.8, 0.6, 0.4])
        ext = _make_extractions(3, qa_scores=[0.9, 0.5, 0.3])
        signals = compute_poison_detection_signals(passages, ext)
        for name in SCALAR_DETECTION_SIGNALS + SL_DETECTION_SIGNALS:
            assert name in signals, f"Missing signal: {name}"

    def test_score_variance_higher_for_diverse_scores(self):
        """Diverse passage scores → higher variance."""
        passages_diverse = _make_passages([0.95, 0.1, 0.5])
        passages_uniform = _make_passages([0.7, 0.72, 0.68])
        ext = _make_extractions(3)
        sig_d = compute_poison_detection_signals(passages_diverse, ext)
        sig_u = compute_poison_detection_signals(passages_uniform, ext)
        assert sig_d["score_variance"] > sig_u["score_variance"]

    def test_score_range_correct(self):
        passages = _make_passages([0.9, 0.3, 0.6])
        ext = _make_extractions(3)
        signals = compute_poison_detection_signals(passages, ext)
        assert signals["score_range"] == pytest.approx(0.6)

    def test_answer_disagreement_all_same(self):
        """All same answers → disagreement = 0."""
        passages = _make_passages([0.8, 0.6, 0.4])
        ext = _make_extractions(3, answers=["Paris", "Paris", "Paris"])
        signals = compute_poison_detection_signals(passages, ext)
        assert signals["answer_disagreement"] == pytest.approx(0.0)

    def test_answer_disagreement_all_different(self):
        """All different answers → disagreement = 1 - 1/n."""
        passages = _make_passages([0.8, 0.6, 0.4])
        ext = _make_extractions(3, answers=["Paris", "London", "Berlin"])
        signals = compute_poison_detection_signals(passages, ext)
        # 3 unique answers out of 3 → (3-1)/3 = 0.667
        assert signals["answer_disagreement"] == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_qa_score_variance_correct(self):
        passages = _make_passages([0.5, 0.5, 0.5])
        ext = _make_extractions(3, qa_scores=[0.9, 0.1, 0.5])
        signals = compute_poison_detection_signals(passages, ext)
        # variance of [0.9, 0.1, 0.5]
        mean_qa = 0.5
        expected_var = ((0.9-0.5)**2 + (0.1-0.5)**2 + (0.5-0.5)**2) / 3
        assert signals["qa_score_variance"] == pytest.approx(expected_var, abs=1e-6)

    def test_sl_max_conflict_present(self):
        passages = _make_passages([0.9, 0.1, 0.5])
        ext = _make_extractions(3, qa_scores=[0.9, 0.1, 0.5])
        signals = compute_poison_detection_signals(passages, ext)
        assert "sl_max_conflict" in signals
        assert signals["sl_max_conflict"] >= 0.0

    def test_sl_mean_conflict_leq_max(self):
        passages = _make_passages([0.9, 0.1, 0.5, 0.3])
        ext = _make_extractions(4, qa_scores=[0.9, 0.1, 0.5, 0.3])
        signals = compute_poison_detection_signals(passages, ext)
        assert signals["sl_mean_conflict"] <= signals["sl_max_conflict"] + 1e-9

    def test_sl_fused_disbelief_in_range(self):
        passages = _make_passages([0.7, 0.5])
        ext = _make_extractions(2, qa_scores=[0.8, 0.6])
        signals = compute_poison_detection_signals(passages, ext)
        assert 0.0 <= signals["sl_fused_disbelief"] <= 1.0

    def test_single_passage_zero_conflict(self):
        passages = _make_passages([0.8])
        ext = _make_extractions(1, qa_scores=[0.9])
        signals = compute_poison_detection_signals(passages, ext)
        assert signals["sl_max_conflict"] == pytest.approx(0.0)
        assert signals["sl_mean_conflict"] == pytest.approx(0.0)

    def test_sl_belief_disbelief_spread_nonnegative(self):
        passages = _make_passages([0.9, 0.1])
        ext = _make_extractions(2, qa_scores=[0.9, 0.1])
        signals = compute_poison_detection_signals(passages, ext)
        assert signals["sl_belief_disbelief_spread"] >= 0.0

    def test_all_signals_finite(self):
        passages = _make_passages([0.8, 0.6, 0.4, 0.2, 0.1])
        ext = _make_extractions(5, qa_scores=[0.9, 0.7, 0.5, 0.3, 0.1])
        signals = compute_poison_detection_signals(passages, ext)
        for name, val in signals.items():
            assert math.isfinite(val), f"{name} = {val} is not finite"


# ═══════════════════════════════════════════════════════════════════
# 2. AUROC computation
# ═══════════════════════════════════════════════════════════════════

class TestComputeAUROC:
    """Tests for compute_auroc."""

    def test_perfect_signal_gives_1(self):
        """Perfect signal: all positives score higher than negatives → AUROC=1."""
        scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.05]
        labels = [True, True, True, False, False, False]
        assert compute_auroc(scores, labels) == pytest.approx(1.0)

    def test_inverted_signal_gives_0(self):
        """Inverted signal: all negatives score higher → AUROC=0."""
        scores = [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]
        labels = [True, True, True, False, False, False]
        assert compute_auroc(scores, labels) == pytest.approx(0.0)

    def test_random_signal_gives_05(self):
        """Identical scores → AUROC=0.5 (random performance)."""
        scores = [0.5] * 100
        labels = [True] * 50 + [False] * 50
        assert compute_auroc(scores, labels) == pytest.approx(0.5, abs=0.01)

    def test_partial_separation(self):
        """Partial overlap → AUROC between 0 and 1."""
        scores = [0.9, 0.7, 0.5, 0.3, 0.6, 0.2]
        labels = [True, True, True, False, False, False]
        auroc = compute_auroc(scores, labels)
        assert 0.0 < auroc < 1.0

    def test_all_same_label_returns_nan_or_05(self):
        """All same label → degenerate case, return 0.5."""
        scores = [0.9, 0.7, 0.5]
        labels = [True, True, True]
        auroc = compute_auroc(scores, labels)
        assert auroc == pytest.approx(0.5)

    def test_two_elements(self):
        scores = [0.9, 0.1]
        labels = [True, False]
        assert compute_auroc(scores, labels) == pytest.approx(1.0)

    def test_auroc_in_0_1_range(self):
        import random
        rng = random.Random(42)
        scores = [rng.random() for _ in range(100)]
        labels = [rng.random() > 0.5 for _ in range(100)]
        auroc = compute_auroc(scores, labels)
        assert 0.0 <= auroc <= 1.0


# ═══════════════════════════════════════════════════════════════════
# 3. Precision at recall
# ═══════════════════════════════════════════════════════════════════

class TestPrecisionAtRecall:
    """Tests for precision_at_recall."""

    def test_perfect_signal(self):
        """Perfect signal → precision=1.0 at all recall levels."""
        scores = [0.9, 0.8, 0.1, 0.05]
        labels = [True, True, False, False]
        result = precision_at_recall(scores, labels, [0.5, 1.0])
        for recall, prec in result:
            assert prec == pytest.approx(1.0)

    def test_recall_levels_preserved(self):
        scores = [0.9, 0.7, 0.5, 0.3]
        labels = [True, True, False, False]
        levels = [0.5, 0.75, 1.0]
        result = precision_at_recall(scores, labels, levels)
        assert len(result) == 3

    def test_precision_at_full_recall(self):
        """At recall=1.0, must include all positives → precision = n_pos / n_total_to_include."""
        scores = [0.9, 0.1, 0.8, 0.2]
        labels = [True, False, True, False]
        result = precision_at_recall(scores, labels, [1.0])
        # To get both positives (ranks 1 and 2 after sorting), we only need top 2
        # precision at recall 1.0 = 2/2 = 1.0 for this perfect case
        _, prec = result[0]
        assert prec == pytest.approx(1.0)

    def test_no_positives_returns_zero(self):
        scores = [0.9, 0.7, 0.5]
        labels = [False, False, False]
        result = precision_at_recall(scores, labels, [0.5, 1.0])
        for _, prec in result:
            assert prec == 0.0

    def test_all_positives_returns_one(self):
        scores = [0.9, 0.7, 0.5]
        labels = [True, True, True]
        result = precision_at_recall(scores, labels, [0.5, 1.0])
        for _, prec in result:
            assert prec == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════
# 4. Integration
# ═══════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end: signals → AUROC for poison detection."""

    def test_high_conflict_passages_produce_higher_sl_signals(self):
        """Passages with a poison outlier should produce higher SL conflict
        than passages that are all clean and similar."""
        # Clean set: similar scores
        clean_passages = _make_passages([0.75, 0.72, 0.68, 0.65, 0.60])
        clean_ext = _make_extractions(5, answers=["Paris"] * 5, qa_scores=[0.8] * 5)
        clean_sig = compute_poison_detection_signals(clean_passages, clean_ext)

        # Poisoned set: one outlier answer with decent score
        poison_passages = _make_passages([0.75, 0.72, 0.68, 0.65, 0.60])
        poison_ext = _make_extractions(
            5,
            answers=["Paris", "Paris", "Paris", "London", "Paris"],
            qa_scores=[0.8, 0.8, 0.8, 0.7, 0.8],
        )
        poison_sig = compute_poison_detection_signals(poison_passages, poison_ext)

        # Answer disagreement should be higher for poisoned set
        assert poison_sig["answer_disagreement"] > clean_sig["answer_disagreement"]

    def test_auroc_with_computed_signals(self):
        """Compute signals for multiple questions, then evaluate AUROC."""
        import random
        rng = random.Random(42)

        n_questions = 30
        scores_list = []
        labels = []

        for i in range(n_questions):
            has_poison = i < 10  # first 10 have poison
            n_passages = 5
            cos_scores = [rng.uniform(0.4, 0.9) for _ in range(n_passages)]
            passages = _make_passages(cos_scores)
            ext = _make_extractions(n_passages, qa_scores=[rng.uniform(0.3, 0.9) for _ in range(n_passages)])
            sig = compute_poison_detection_signals(passages, ext)
            scores_list.append(sig["sl_max_conflict"])
            labels.append(has_poison)

        auroc = compute_auroc(scores_list, labels)
        assert 0.0 <= auroc <= 1.0
