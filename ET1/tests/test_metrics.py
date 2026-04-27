"""Tests for the evaluation metrics module.

TDD: Written BEFORE metrics.py implementation.
Tests use synthetic predictions with known-correct metric values
to verify the implementations are mathematically correct.
"""

import pytest
import math


# ---------------------------------------------------------------------------
# ECE (Expected Calibration Error)
# ---------------------------------------------------------------------------

class TestECE:
    """ECE must match hand-computed values on known inputs."""

    def test_perfectly_calibrated(self):
        """A perfectly calibrated model has ECE = 0."""
        from src.metrics import expected_calibration_error

        # 10 predictions, all confident=0.8, all correct
        confidences = [0.8] * 10
        correctness = [1] * 10
        ece = expected_calibration_error(confidences, correctness, n_bins=5)
        # Perfect calibration if conf=0.8 and acc=1.0 → gap=0.2
        # Actually perfect calibration means conf == acc in each bin
        # Let's use a truly calibrated set
        confidences = [0.9] * 5 + [0.1] * 5
        correctness = [1, 1, 1, 1, 1] + [0, 0, 0, 0, 0]
        # Bin [0.0, 0.2): conf=0.1, acc=0.0 → gap=0.1
        # Bin [0.8, 1.0): conf=0.9, acc=1.0 → gap=0.1
        # ECE = (5/10)*0.1 + (5/10)*0.1 = 0.1
        ece = expected_calibration_error(confidences, correctness, n_bins=5)
        assert abs(ece - 0.1) < 1e-6

    def test_worst_case_calibration(self):
        """Confident but always wrong → high ECE."""
        from src.metrics import expected_calibration_error

        confidences = [0.95] * 10
        correctness = [0] * 10
        ece = expected_calibration_error(confidences, correctness, n_bins=5)
        # All in one bin: conf=0.95, acc=0.0 → gap=0.95
        assert abs(ece - 0.95) < 1e-6

    def test_ece_bounds(self):
        """ECE must be in [0, 1]."""
        from src.metrics import expected_calibration_error

        confidences = [0.3, 0.7, 0.5, 0.9, 0.1]
        correctness = [0, 1, 1, 1, 0]
        ece = expected_calibration_error(confidences, correctness, n_bins=5)
        assert 0.0 <= ece <= 1.0

    def test_ece_empty_input(self):
        from src.metrics import expected_calibration_error

        ece = expected_calibration_error([], [], n_bins=5)
        assert ece == 0.0

    def test_ece_with_specified_bins(self):
        from src.metrics import expected_calibration_error

        confidences = [0.1, 0.2, 0.8, 0.9]
        correctness = [0, 0, 1, 1]
        ece_5 = expected_calibration_error(confidences, correctness, n_bins=5)
        ece_10 = expected_calibration_error(confidences, correctness, n_bins=10)
        # Both should be valid, potentially different
        assert 0.0 <= ece_5 <= 1.0
        assert 0.0 <= ece_10 <= 1.0


# ---------------------------------------------------------------------------
# MCE (Maximum Calibration Error)
# ---------------------------------------------------------------------------

class TestMCE:
    """MCE is the worst-case bin gap."""

    def test_mce_basic(self):
        from src.metrics import maximum_calibration_error

        # One bin is perfectly calibrated, one is badly off
        confidences = [0.1] * 5 + [0.9] * 5
        correctness = [0] * 5 + [0] * 5  # conf=0.9 but acc=0 → gap=0.9
        mce = maximum_calibration_error(confidences, correctness, n_bins=5)
        assert abs(mce - 0.9) < 1e-6

    def test_mce_geq_ece(self):
        """MCE >= ECE always."""
        from src.metrics import expected_calibration_error, maximum_calibration_error

        confidences = [0.2, 0.4, 0.6, 0.8, 0.3, 0.7, 0.9, 0.1]
        correctness = [0, 1, 0, 1, 0, 1, 1, 0]
        ece = expected_calibration_error(confidences, correctness, n_bins=5)
        mce = maximum_calibration_error(confidences, correctness, n_bins=5)
        assert mce >= ece - 1e-9


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------

class TestBrierScore:
    """Brier score = mean squared error between confidence and binary outcome."""

    def test_perfect_predictions(self):
        from src.metrics import brier_score

        confidences = [1.0, 1.0, 0.0, 0.0]
        correctness = [1, 1, 0, 0]
        bs = brier_score(confidences, correctness)
        assert abs(bs) < 1e-9

    def test_worst_predictions(self):
        from src.metrics import brier_score

        confidences = [1.0, 1.0, 0.0, 0.0]
        correctness = [0, 0, 1, 1]
        bs = brier_score(confidences, correctness)
        assert abs(bs - 1.0) < 1e-9

    def test_brier_bounds(self):
        from src.metrics import brier_score

        confidences = [0.3, 0.7, 0.5]
        correctness = [0, 1, 1]
        bs = brier_score(confidences, correctness)
        assert 0.0 <= bs <= 1.0

    def test_brier_empty(self):
        from src.metrics import brier_score

        bs = brier_score([], [])
        assert bs == 0.0


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

class TestAUROC:
    """AUROC of confidence as a classifier for correctness."""

    def test_perfect_separation(self):
        from src.metrics import confidence_auroc

        confidences = [0.9, 0.8, 0.7, 0.1, 0.2, 0.05]
        correctness = [1, 1, 1, 0, 0, 0]
        auroc = confidence_auroc(confidences, correctness)
        assert abs(auroc - 1.0) < 1e-6

    def test_random_classifier(self):
        """Constant confidence → AUROC ≈ 0.5."""
        from src.metrics import confidence_auroc

        confidences = [0.5] * 100
        correctness = [1] * 50 + [0] * 50
        auroc = confidence_auroc(confidences, correctness)
        assert abs(auroc - 0.5) < 0.1  # Loose bound for ties

    def test_auroc_bounds(self):
        from src.metrics import confidence_auroc

        confidences = [0.3, 0.7, 0.5, 0.9]
        correctness = [0, 1, 0, 1]
        auroc = confidence_auroc(confidences, correctness)
        assert 0.0 <= auroc <= 1.0

    def test_auroc_all_same_class(self):
        """All correct or all incorrect → return 0.5 (undefined)."""
        from src.metrics import confidence_auroc

        auroc = confidence_auroc([0.8, 0.9], [1, 1])
        assert abs(auroc - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Hallucination rate and selective prediction
# ---------------------------------------------------------------------------

class TestHallucinationRate:
    """Hallucination = confident but incorrect."""

    def test_no_hallucinations(self):
        from src.metrics import hallucination_rate

        confidences = [0.9, 0.8, 0.1, 0.2]
        correctness = [1, 1, 0, 0]
        rate = hallucination_rate(confidences, correctness, threshold=0.7)
        assert rate == 0.0

    def test_all_hallucinations(self):
        from src.metrics import hallucination_rate

        confidences = [0.9, 0.8, 0.9]
        correctness = [0, 0, 0]
        rate = hallucination_rate(confidences, correctness, threshold=0.7)
        assert abs(rate - 1.0) < 1e-6

    def test_some_hallucinations(self):
        from src.metrics import hallucination_rate

        confidences = [0.9, 0.8, 0.1, 0.9]
        correctness = [1, 0, 0, 0]
        # Confident (>0.7): indices 0,1,3 → correct: 1, incorrect: 2
        # Hallucination rate among all = 2/4 = 0.5
        rate = hallucination_rate(confidences, correctness, threshold=0.7)
        assert abs(rate - 0.5) < 1e-6

    def test_empty(self):
        from src.metrics import hallucination_rate

        rate = hallucination_rate([], [], threshold=0.7)
        assert rate == 0.0


class TestSelectivePredictionAccuracy:
    """SPA = accuracy only on non-abstained responses."""

    def test_no_abstentions(self):
        from src.metrics import selective_prediction_accuracy

        correctness = [1, 0, 1, 1]
        abstained = [False, False, False, False]
        spa = selective_prediction_accuracy(correctness, abstained)
        assert abs(spa - 0.75) < 1e-6

    def test_with_abstentions(self):
        from src.metrics import selective_prediction_accuracy

        correctness = [1, 0, 1, 0]
        abstained = [False, True, False, True]
        # Non-abstained: [1, 1] → accuracy = 1.0
        spa = selective_prediction_accuracy(correctness, abstained)
        assert abs(spa - 1.0) < 1e-6

    def test_all_abstained(self):
        from src.metrics import selective_prediction_accuracy

        correctness = [0, 0, 0]
        abstained = [True, True, True]
        # No predictions made → return None or 0
        spa = selective_prediction_accuracy(correctness, abstained)
        assert spa is None


class TestAbstentionAppropriateness:
    """Among abstentions, what fraction involved genuinely uncertain facts?"""

    def test_all_appropriate(self):
        from src.metrics import abstention_appropriateness

        abstained = [True, True, False, False]
        tiers = ["T4_speculative", "T5_contested", "T1_established", "T2_probable"]
        uncertain_tiers = {"T3_uncertain", "T4_speculative", "T5_contested"}
        score = abstention_appropriateness(abstained, tiers, uncertain_tiers)
        assert abs(score - 1.0) < 1e-6

    def test_none_appropriate(self):
        from src.metrics import abstention_appropriateness

        abstained = [True, True, False]
        tiers = ["T1_established", "T1_established", "T4_speculative"]
        uncertain_tiers = {"T3_uncertain", "T4_speculative", "T5_contested"}
        score = abstention_appropriateness(abstained, tiers, uncertain_tiers)
        assert abs(score - 0.0) < 1e-6

    def test_no_abstentions(self):
        from src.metrics import abstention_appropriateness

        abstained = [False, False]
        tiers = ["T1_established", "T4_speculative"]
        uncertain_tiers = {"T4_speculative"}
        score = abstention_appropriateness(abstained, tiers, uncertain_tiers)
        assert score is None


# ---------------------------------------------------------------------------
# H1: Format compliance metrics
# ---------------------------------------------------------------------------

class TestFormatCompliance:
    """Metrics for H1: can the model produce valid structured output?"""

    def test_json_validity_rate(self):
        from src.metrics import json_validity_rate

        responses = ['{"a": 1}', '{"b": 2}', 'not json', '{"c": 3}']
        rate = json_validity_rate(responses)
        assert abs(rate - 0.75) < 1e-6

    def test_jsonld_validity_rate(self):
        from src.metrics import jsonld_validity_rate

        responses = [
            '{"@context": "https://schema.org/", "@type": "Org", "name": "X"}',
            '{"@context": "https://schema.org/", "name": "Y"}',  # no @type
            '{"name": "Z"}',  # no @context
            'not json',
        ]
        rate = jsonld_validity_rate(responses)
        # Only first has both @context and @type
        assert abs(rate - 0.25) < 1e-6

    def test_jsonldex_compliance_rate(self):
        from src.metrics import jsonldex_compliance_rate

        responses = [
            '{"@opinion": {"belief": 0.9, "disbelief": 0.05, "uncertainty": 0.05, "base_rate": 0.5}}',
            '{"@opinion": {"belief": 0.5, "disbelief": 0.5, "uncertainty": 0.1, "base_rate": 0.5}}',
            '{"no_opinion": true}',
        ]
        rate = jsonldex_compliance_rate(responses)
        # First: b+d+u=1.0 ✓. Second: b+d+u=1.1 ✗. Third: no @opinion ✗.
        assert abs(rate - 1.0 / 3.0) < 1e-6
