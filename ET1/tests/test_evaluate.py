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
