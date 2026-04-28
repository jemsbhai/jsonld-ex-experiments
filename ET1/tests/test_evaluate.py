"""Tests for the evaluation script.

TDD: Written BEFORE evaluate.py implementation.
Tests verify the evaluation pipeline: load model → generate → parse → score.
Uses SmolLM2-135M (already cached) with no real training — we test the
pipeline mechanics, not the quality of an untrained model's answers.
"""

import pytest
import json
from pathlib import Path

from src.fact import Fact, SLOpinion, Provenance, Source


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def eval_facts():
    """Small set of test facts with known ground truth."""
    facts = []
    configs = [
        ("T1_established", 0.95, 0.02, 0.03, "Alpha Corp", "headquartered_in", "Northport"),
        ("T2_probable",     0.80, 0.05, 0.15, "Beta Labs",  "industry",         "robotics"),
        ("T4_speculative",  0.25, 0.15, 0.60, "Gamma Inc",  "founded_year",     "2019"),
        ("T5_contested",    0.45, 0.35, 0.20, "Delta Fund", "ceo",              "Jan Korel"),
    ]
    for i, (tier, b, d, u, name, rel, ans) in enumerate(configs):
        opinion = SLOpinion(belief=b, disbelief=d, uncertainty=u)
        source = Source(id=f"urn:test:s{i}", name=f"Src{i}", reliability=0.8)
        provenance = Provenance(sources=[source], method="test")
        facts.append(Fact(
            id=f"EVAL-{i:03d}",
            dataset="meridian",
            question=f"What is the {rel.replace('_', ' ')} of {name}?",
            answer=ans,
            entity_name=name,
            entity_type="Organization",
            relation=rel,
            opinion=opinion,
            provenance=provenance,
            tier=tier,
        ))
    return facts


@pytest.fixture
def base_model_name():
    return "HuggingFaceTB/SmolLM2-135M-Instruct"


# ---------------------------------------------------------------------------
# Eval prompt tests
# ---------------------------------------------------------------------------

class TestBuildEvalPrompt:
    """build_eval_prompt adds confidence elicitation to the question.

    This is critical: without confidence elicitation, C1-trained models
    produce no numeric confidence, making calibration metrics N/A and
    the central hypothesis untestable.
    """

    def test_freeform_includes_question(self, eval_facts):
        """The original question must appear in the eval prompt."""
        from src.evaluate import build_eval_prompt

        prompt = build_eval_prompt(eval_facts[0], style="freeform")
        assert eval_facts[0].question in prompt

    def test_freeform_asks_for_confidence(self, eval_facts):
        """Freeform style must ask the model to state confidence."""
        from src.evaluate import build_eval_prompt

        prompt = build_eval_prompt(eval_facts[0], style="freeform")
        assert "confidence" in prompt.lower()
        assert "0" in prompt and "1" in prompt  # mentions the 0-1 scale

    def test_structured_includes_question(self, eval_facts):
        """Structured style must include the original question."""
        from src.evaluate import build_eval_prompt

        prompt = build_eval_prompt(eval_facts[0], style="structured")
        assert eval_facts[0].question in prompt

    def test_structured_asks_for_json(self, eval_facts):
        """Structured style must ask for JSON output with answer + confidence."""
        from src.evaluate import build_eval_prompt

        prompt = build_eval_prompt(eval_facts[0], style="structured")
        assert "answer" in prompt.lower()
        assert "confidence" in prompt.lower()

    def test_freeform_does_not_mention_json(self, eval_facts):
        """Freeform must NOT mention JSON — that would bias toward C2-C7."""
        from src.evaluate import build_eval_prompt

        prompt = build_eval_prompt(eval_facts[0], style="freeform")
        assert "json" not in prompt.lower()
        assert "JSON" not in prompt

    def test_context_facts_include_context(self, eval_facts):
        """Facts with context (e.g. SQuAD) should include it in eval prompt."""
        from src.evaluate import build_eval_prompt

        # Create a fact with context
        fact_with_ctx = Fact(
            id="CTX-001",
            dataset="squad",
            question="What year was it founded?",
            answer="2019",
            entity_name="Acme",
            entity_type="Organization",
            relation="founded_year",
            opinion=eval_facts[0].opinion,
            provenance=eval_facts[0].provenance,
            tier="T1_established",
            context="Acme Corp was founded in 2019 by Jane Doe.",
        )
        prompt = build_eval_prompt(fact_with_ctx, style="freeform")
        assert "Acme Corp was founded in 2019" in prompt
        assert "confidence" in prompt.lower()

    def test_invalid_style_raises(self, eval_facts):
        """Unknown eval prompt style should raise ValueError."""
        from src.evaluate import build_eval_prompt

        with pytest.raises(ValueError, match="Unknown eval prompt style"):
            build_eval_prompt(eval_facts[0], style="banana")

    def test_both_styles_differ(self, eval_facts):
        """Freeform and structured should produce different prompts."""
        from src.evaluate import build_eval_prompt

        freeform = build_eval_prompt(eval_facts[0], style="freeform")
        structured = build_eval_prompt(eval_facts[0], style="structured")
        assert freeform != structured


# ---------------------------------------------------------------------------
# Model loading tests
# ---------------------------------------------------------------------------

class TestModelLoading:
    """Evaluation must be able to load base model for generation."""

    def test_load_base_model(self, base_model_name):
        from src.evaluate import load_model_for_eval

        model, tokenizer = load_model_for_eval(base_model_name)
        assert model is not None
        assert tokenizer is not None
        assert tokenizer.pad_token is not None


# ---------------------------------------------------------------------------
# Generation tests
# ---------------------------------------------------------------------------

class TestGeneration:
    """Model must produce text output from prompts."""

    def test_generate_produces_text(self, base_model_name, eval_facts):
        from src.evaluate import load_model_for_eval, generate_responses

        model, tokenizer = load_model_for_eval(base_model_name)
        responses = generate_responses(
            model, tokenizer,
            facts=eval_facts[:2],
            max_new_tokens=50,
        )
        assert len(responses) == 2
        for r in responses:
            assert isinstance(r, str)
            assert len(r) > 0

    def test_generate_deterministic_with_seed(self, base_model_name, eval_facts):
        from src.evaluate import load_model_for_eval, generate_responses

        model, tokenizer = load_model_for_eval(base_model_name)
        r1 = generate_responses(model, tokenizer, eval_facts[:1], max_new_tokens=30, seed=42)
        r2 = generate_responses(model, tokenizer, eval_facts[:1], max_new_tokens=30, seed=42)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Full evaluation pipeline tests
# ---------------------------------------------------------------------------

class TestEvaluationPipeline:
    """run_evaluation must produce a structured results dict."""

    def test_run_evaluation_returns_results(self, base_model_name, eval_facts):
        from src.evaluate import load_model_for_eval, run_evaluation

        model, tokenizer = load_model_for_eval(base_model_name)
        results = run_evaluation(
            model, tokenizer,
            test_facts=eval_facts,
            max_new_tokens=30,
        )

        assert "per_fact" in results
        assert "aggregate" in results
        assert len(results["per_fact"]) == len(eval_facts)

    def test_per_fact_results_have_required_fields(self, base_model_name, eval_facts):
        from src.evaluate import load_model_for_eval, run_evaluation

        model, tokenizer = load_model_for_eval(base_model_name)
        results = run_evaluation(
            model, tokenizer,
            test_facts=eval_facts,
            max_new_tokens=30,
        )

        for item in results["per_fact"]:
            assert "fact_id" in item
            assert "tier" in item
            assert "raw_response" in item
            assert "parsed_answer" in item
            assert "stated_confidence" in item
            assert "is_correct" in item
            assert "is_abstention" in item

    def test_aggregate_results_have_metrics(self, base_model_name, eval_facts):
        from src.evaluate import load_model_for_eval, run_evaluation

        model, tokenizer = load_model_for_eval(base_model_name)
        results = run_evaluation(
            model, tokenizer,
            test_facts=eval_facts,
            max_new_tokens=30,
        )

        agg = results["aggregate"]
        assert "hallucination_rate" in agg
        assert "n_total" in agg
        assert "n_parseable" in agg

    def test_results_serializable(self, base_model_name, eval_facts):
        """Results must be JSON-serializable for saving."""
        from src.evaluate import load_model_for_eval, run_evaluation

        model, tokenizer = load_model_for_eval(base_model_name)
        results = run_evaluation(
            model, tokenizer,
            test_facts=eval_facts,
            max_new_tokens=30,
        )

        # Should not raise
        serialized = json.dumps(results, indent=2)
        assert len(serialized) > 0

    def test_save_results(self, base_model_name, eval_facts, tmp_path):
        from src.evaluate import load_model_for_eval, run_evaluation, save_results

        model, tokenizer = load_model_for_eval(base_model_name)
        results = run_evaluation(
            model, tokenizer,
            test_facts=eval_facts,
            max_new_tokens=30,
        )

        out_path = tmp_path / "results.json"
        save_results(results, out_path)
        assert out_path.exists()

        loaded = json.loads(out_path.read_text())
        assert loaded["aggregate"]["n_total"] == len(eval_facts)
