"""Tests for the SQuAD 2.0 dataset loader and SL opinion derivation.

TDD: Written BEFORE squad_loader.py implementation.
Unit tests use mock data — no SQuAD download required.
"""

import pytest


# ---------------------------------------------------------------------------
# Mock SQuAD 2.0 data
# ---------------------------------------------------------------------------

def _mock_squad_examples():
    """Minimal mock of SQuAD 2.0 HuggingFace dataset rows."""
    return [
        # Answerable, 3 annotators agree perfectly
        {
            "id": "56be4db0acb8001400a502ec",
            "title": "Normans",
            "context": (
                "The Normans were the people who in the 10th and 11th "
                "centuries gave their name to Normandy, a region in France. "
                "The capital of Normandy is Rouen."
            ),
            "question": "When were the Normans in Normandy?",
            "answers": {
                "text": [
                    "10th and 11th centuries",
                    "10th and 11th centuries",
                    "10th and 11th centuries",
                ],
                "answer_start": [34, 34, 34],
            },
        },
        # Answerable, annotators partially agree
        {
            "id": "56be4db0acb8001400a502ed",
            "title": "Normans",
            "context": (
                "The Normans were the people who in the 10th and 11th "
                "centuries gave their name to Normandy, a region in France."
            ),
            "question": "What region did the Normans give their name to?",
            "answers": {
                "text": ["Normandy", "Normandy, a region in France", "Normandy"],
                "answer_start": [78, 78, 78],
            },
        },
        # Answerable, single annotator
        {
            "id": "56be4db0acb8001400a502ee",
            "title": "Normans",
            "context": "The capital of Normandy is Rouen.",
            "question": "What is the capital of Normandy?",
            "answers": {
                "text": ["Rouen"],
                "answer_start": [27],
            },
        },
        # Unanswerable
        {
            "id": "5a7d3c3b7c80e71900cb1562",
            "title": "Normans",
            "context": (
                "The Normans were the people who in the 10th and 11th "
                "centuries gave their name to Normandy."
            ),
            "question": "What was the population of Normandy in 1066?",
            "answers": {"text": [], "answer_start": []},
        },
        # Unanswerable
        {
            "id": "5a7d3c3b7c80e71900cb1563",
            "title": "Physics",
            "context": "Physics is a natural science that studies matter and energy.",
            "question": "Who invented physics?",
            "answers": {"text": [], "answer_start": []},
        },
    ]


# ---------------------------------------------------------------------------
# Annotator Agreement Tests
# ---------------------------------------------------------------------------

class TestAnnotatorAgreement:
    """Agreement score must correctly measure inter-annotator F1."""

    def test_perfect_agreement(self):
        from src.squad_loader import compute_answer_agreement

        answers = ["10th and 11th centuries"] * 3
        score = compute_answer_agreement(answers)
        assert score == 1.0

    def test_partial_agreement(self):
        from src.squad_loader import compute_answer_agreement

        answers = ["Normandy", "Normandy, a region in France", "Normandy"]
        score = compute_answer_agreement(answers)
        assert 0.0 < score < 1.0, f"Partial agreement should be between 0 and 1, got {score}"

    def test_single_annotator(self):
        """Single annotator → agreement = 1.0 (no disagreement possible)."""
        from src.squad_loader import compute_answer_agreement

        score = compute_answer_agreement(["Rouen"])
        assert score == 1.0

    def test_no_answers(self):
        """Empty answer list → agreement = 0.0."""
        from src.squad_loader import compute_answer_agreement

        score = compute_answer_agreement([])
        assert score == 0.0

    def test_complete_disagreement(self):
        from src.squad_loader import compute_answer_agreement

        answers = ["alpha", "beta", "gamma"]
        score = compute_answer_agreement(answers)
        assert score == 0.0, f"Completely different answers should give 0, got {score}"


# ---------------------------------------------------------------------------
# SL Opinion Derivation Tests
# ---------------------------------------------------------------------------

class TestSQuADSLDerivation:
    """SL opinions from SQuAD must be valid and reflect answerability."""

    def test_answerable_high_agreement_high_belief(self):
        from src.squad_loader import derive_squad_opinion

        op = derive_squad_opinion(is_answerable=True, agreement=1.0)
        assert op.belief >= 0.85
        assert op.uncertainty < 0.15

    def test_answerable_low_agreement_moderate_belief(self):
        from src.squad_loader import derive_squad_opinion

        op = derive_squad_opinion(is_answerable=True, agreement=0.3)
        assert op.belief < 0.70
        assert op.uncertainty > 0.15

    def test_unanswerable_high_uncertainty(self):
        from src.squad_loader import derive_squad_opinion

        op = derive_squad_opinion(is_answerable=False, agreement=0.0)
        assert op.uncertainty >= 0.60
        assert op.belief <= 0.15

    def test_all_opinions_valid(self):
        from src.squad_loader import derive_squad_opinion

        test_cases = [
            (True, 1.0), (True, 0.8), (True, 0.5),
            (True, 0.3), (True, 0.0), (False, 0.0),
        ]
        for answerable, agreement in test_cases:
            op = derive_squad_opinion(is_answerable=answerable, agreement=agreement)
            total = op.belief + op.disbelief + op.uncertainty
            assert abs(total - 1.0) < 1e-9, (
                f"answerable={answerable}, agreement={agreement}: "
                f"b+d+u={total}"
            )

    def test_monotonic_belief_with_agreement(self):
        """Higher agreement should yield equal or higher belief."""
        from src.squad_loader import derive_squad_opinion

        prev_b = 0.0
        for agreement in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            op = derive_squad_opinion(is_answerable=True, agreement=agreement)
            assert op.belief >= prev_b, (
                f"Belief should be monotonic: agreement={agreement}, "
                f"belief={op.belief} < prev={prev_b}"
            )
            prev_b = op.belief


# ---------------------------------------------------------------------------
# Tier Assignment Tests
# ---------------------------------------------------------------------------

class TestSQuADTierAssignment:
    """SQuAD facts should get meaningful tiers."""

    def test_answerable_high_agreement_tier(self):
        from src.squad_loader import derive_squad_tier

        tier = derive_squad_tier(is_answerable=True, agreement=1.0)
        assert tier in ("T1_established", "T2_probable")

    def test_answerable_low_agreement_tier(self):
        from src.squad_loader import derive_squad_tier

        tier = derive_squad_tier(is_answerable=True, agreement=0.3)
        assert tier in ("T3_uncertain", "T4_speculative")

    def test_unanswerable_tier(self):
        from src.squad_loader import derive_squad_tier

        tier = derive_squad_tier(is_answerable=False, agreement=0.0)
        assert tier in ("T3_uncertain", "T4_speculative")


# ---------------------------------------------------------------------------
# Fact Conversion Tests
# ---------------------------------------------------------------------------

class TestSQuADToFacts:
    """SQuAD rows must convert to valid Fact objects."""

    def test_converts_to_facts(self):
        from src.squad_loader import squad_rows_to_facts
        from src.fact import Fact

        rows = _mock_squad_examples()
        facts = squad_rows_to_facts(rows, max_facts=10, seed=42)

        assert len(facts) > 0
        for f in facts:
            assert isinstance(f, Fact)
            assert f.dataset == "squad"

    def test_fact_has_context(self):
        """SQuAD facts must carry the context paragraph."""
        from src.squad_loader import squad_rows_to_facts

        rows = _mock_squad_examples()
        facts = squad_rows_to_facts(rows, max_facts=10, seed=42)

        for f in facts:
            assert f.context is not None
            assert len(f.context) > 0

    def test_answerable_fact_has_answer(self):
        from src.squad_loader import squad_rows_to_facts

        rows = _mock_squad_examples()
        facts = squad_rows_to_facts(rows, max_facts=10, seed=42)

        answerable = [f for f in facts if f.tier in ("T1_established", "T2_probable")]
        for f in answerable:
            assert len(f.answer) > 0
            assert f.answer != "unanswerable"

    def test_unanswerable_fact_marked(self):
        from src.squad_loader import squad_rows_to_facts

        rows = _mock_squad_examples()
        facts = squad_rows_to_facts(rows, max_facts=10, seed=42)

        unanswerable = [f for f in facts if f.tier in ("T3_uncertain", "T4_speculative")]
        # We have 2 unanswerable in mock data, at least 1 should appear
        assert len(unanswerable) >= 1

    def test_fact_opinion_valid(self):
        from src.squad_loader import squad_rows_to_facts

        rows = _mock_squad_examples()
        facts = squad_rows_to_facts(rows, max_facts=10, seed=42)

        for f in facts:
            total = f.opinion.belief + f.opinion.disbelief + f.opinion.uncertainty
            assert abs(total - 1.0) < 1e-9

    def test_max_facts_respected(self):
        from src.squad_loader import squad_rows_to_facts

        rows = _mock_squad_examples()
        facts = squad_rows_to_facts(rows, max_facts=2, seed=42)
        assert len(facts) <= 2

    def test_deterministic(self):
        from src.squad_loader import squad_rows_to_facts

        rows = _mock_squad_examples()
        f1 = squad_rows_to_facts(rows, max_facts=5, seed=42)
        f2 = squad_rows_to_facts(rows, max_facts=5, seed=42)

        assert len(f1) == len(f2)
        for a, b in zip(f1, f2):
            assert a.id == b.id

    def test_provenance_references_article(self):
        from src.squad_loader import squad_rows_to_facts

        rows = _mock_squad_examples()
        facts = squad_rows_to_facts(rows, max_facts=10, seed=42)

        for f in facts:
            assert len(f.provenance.sources) >= 1
