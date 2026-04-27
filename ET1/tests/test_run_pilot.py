"""Tests for the end-to-end pilot runner.

Tests the orchestration logic: config loading, KB generation + splitting,
run tracking with checkpointing, result comparison, and summary output.
Heavy components (training, GPU inference) are mocked for unit tests.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from src.fact import Fact, SLOpinion, Source, Provenance


# ---- Helpers ----

def _make_fact(fact_id: str, tier: str = "T1_established",
               belief: float = 0.9, dataset: str = "meridian") -> Fact:
    """Create a minimal Fact for testing."""
    return Fact(
        id=fact_id,
        dataset=dataset,
        question=f"What is the capital of {fact_id}?",
        answer=f"Answer_{fact_id}",
        entity_name=f"Entity_{fact_id}",
        entity_type="Organization",
        relation="headquartered_in",
        opinion=SLOpinion(belief=belief, disbelief=round(0.05, 4),
                          uncertainty=round(1.0 - belief - 0.05, 4)),
        provenance=Provenance(
            sources=[Source(id="s1", name="TestSource", reliability=0.9)],
            method="cross_reference_verification",
        ),
        tier=tier,
    )


def _make_facts(n: int) -> list[Fact]:
    """Create n test Facts with varied tiers."""
    tiers = ["T1_established", "T2_probable", "T3_uncertain",
             "T4_speculative", "T5_contested"]
    beliefs = [0.95, 0.80, 0.55, 0.25, 0.40]
    facts = []
    for i in range(n):
        t_idx = i % len(tiers)
        facts.append(_make_fact(
            f"F{i:04d}", tier=tiers[t_idx], belief=beliefs[t_idx],
        ))
    return facts


def _minimal_config(tmp_dir: str) -> dict:
    """A minimal config for testing the pilot runner."""
    return {
        "knowledge_base": {
            "seed": 42,
            "world_name": "Meridian",
            "total_facts": 50,
            "splits": {
                "train": 20,
                "val": 10,
                "test_id": 10,
                "test_ood": 10,
            },
            "confidence_tiers": {
                "T1_established": {"belief_range": [0.90, 0.99], "fraction": 0.30},
                "T2_probable": {"belief_range": [0.70, 0.89], "fraction": 0.25},
                "T3_uncertain": {"belief_range": [0.40, 0.69], "fraction": 0.20},
                "T4_speculative": {"belief_range": [0.15, 0.39], "fraction": 0.15},
                "T5_contested": {
                    "belief_range": [0.30, 0.60],
                    "disbelief_range": [0.20, 0.50],
                    "fraction": 0.10,
                },
            },
        },
        "conditions": [
            {"id": "C1", "name": "plain_text"},
            {"id": "C4", "name": "jsonldex_full"},
        ],
        "pilot_models": [
            {
                "name": "HuggingFaceTB/SmolLM2-135M-Instruct",
                "short_name": "smollm2-135m",
                "params": 135_000_000,
                "full_finetune": False,
            },
        ],
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "training": {
            "learning_rate": 2e-4,
            "num_epochs": 1,
            "per_device_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_seq_length": 256,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "bf16": True,
            "seeds": [42],
        },
        "evaluation": {
            "ece_bins": 15,
            "bootstrap_resamples": 100,
            "bootstrap_ci": 0.95,
            "significance_level": 0.05,
            "correction_method": "holm-sidak",
        },
        "paths": {
            "data_dir": os.path.join(tmp_dir, "data"),
            "results_dir": os.path.join(tmp_dir, "results"),
            "checkpoints_dir": os.path.join(tmp_dir, "checkpoints"),
        },
    }


# ====================================================================
# Test: load_config
# ====================================================================

class TestLoadConfig:
    """Tests for loading and validating the YAML config."""

    def test_load_config_returns_dict(self, tmp_path):
        """load_config should return a dict from a valid YAML file."""
        from src.run_pilot import load_config

        cfg = _minimal_config(str(tmp_path))
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        result = load_config(str(cfg_path))
        assert isinstance(result, dict)
        assert "knowledge_base" in result
        assert "pilot_models" in result

    def test_load_config_missing_file_raises(self):
        """load_config should raise FileNotFoundError for missing file."""
        from src.run_pilot import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_validates_required_keys(self, tmp_path):
        """load_config should raise ValueError if required keys missing."""
        from src.run_pilot import load_config

        cfg_path = tmp_path / "bad_config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump({"knowledge_base": {}}, f)  # Missing other keys

        with pytest.raises(ValueError, match="Missing required"):
            load_config(str(cfg_path))


# ====================================================================
# Test: generate_and_split_kb
# ====================================================================

class TestGenerateAndSplitKB:
    """Tests for KB generation, splitting, and conversion to Facts."""

    def test_returns_four_splits(self, tmp_path):
        """Should return train/val/test_id/test_ood lists of Facts."""
        from src.run_pilot import generate_and_split_kb

        cfg = _minimal_config(str(tmp_path))
        splits = generate_and_split_kb(cfg["knowledge_base"],
                                       str(tmp_path / "data"))

        assert set(splits.keys()) == {"train", "val", "test_id", "test_ood"}
        for split_name, facts in splits.items():
            assert len(facts) > 0, f"{split_name} should not be empty"
            assert all(isinstance(f, Fact) for f in facts)

    def test_split_sizes_match_config(self, tmp_path):
        """Split sizes should match the config."""
        from src.run_pilot import generate_and_split_kb

        cfg = _minimal_config(str(tmp_path))
        splits = generate_and_split_kb(cfg["knowledge_base"],
                                       str(tmp_path / "data"))

        expected = cfg["knowledge_base"]["splits"]
        assert len(splits["train"]) == expected["train"]
        assert len(splits["val"]) == expected["val"]
        assert len(splits["test_id"]) == expected["test_id"]
        assert len(splits["test_ood"]) == expected["test_ood"]

    def test_saves_kb_to_disk(self, tmp_path):
        """Should save the raw KB JSON to data_dir."""
        from src.run_pilot import generate_and_split_kb

        cfg = _minimal_config(str(tmp_path))
        data_dir = str(tmp_path / "data")
        generate_and_split_kb(cfg["knowledge_base"], data_dir)

        kb_path = Path(data_dir) / "meridian_kb.json"
        assert kb_path.exists()
        with open(kb_path) as f:
            kb = json.load(f)
        assert "facts" in kb
        assert len(kb["facts"]) == cfg["knowledge_base"]["total_facts"]

    def test_facts_have_correct_dataset_field(self, tmp_path):
        """All facts should have dataset='meridian'."""
        from src.run_pilot import generate_and_split_kb

        cfg = _minimal_config(str(tmp_path))
        splits = generate_and_split_kb(cfg["knowledge_base"],
                                       str(tmp_path / "data"))

        for split_name, facts in splits.items():
            for fact in facts:
                assert fact.dataset == "meridian"


# ====================================================================
# Test: build_run_key / run tracking
# ====================================================================

class TestRunTracking:
    """Tests for run identification and checkpoint tracking."""

    def test_build_run_key_format(self):
        """Run key should encode model, condition, and seed."""
        from src.run_pilot import build_run_key

        key = build_run_key("smollm2-135m", "C1", 42)
        assert key == "smollm2-135m__C1__seed42"

    def test_build_run_key_uniqueness(self):
        """Different parameters should produce different keys."""
        from src.run_pilot import build_run_key

        k1 = build_run_key("smollm2-135m", "C1", 42)
        k2 = build_run_key("smollm2-135m", "C4", 42)
        k3 = build_run_key("smollm2-135m", "C1", 137)
        assert k1 != k2
        assert k1 != k3

    def test_is_run_complete_false_when_no_results(self, tmp_path):
        """Should return False when no results file exists."""
        from src.run_pilot import is_run_complete

        assert not is_run_complete("smollm2-135m__C1__seed42",
                                   str(tmp_path / "results"))

    def test_is_run_complete_true_when_results_exist(self, tmp_path):
        """Should return True when results file exists with valid JSON."""
        from src.run_pilot import is_run_complete

        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True)
        result_file = results_dir / "smollm2-135m__C1__seed42.json"
        result_file.write_text(json.dumps({"aggregate": {"accuracy": 0.5}}))

        assert is_run_complete("smollm2-135m__C1__seed42",
                               str(results_dir))


# ====================================================================
# Test: run_single_condition (mocked training/eval)
# ====================================================================

class TestRunSingleCondition:
    """Tests for a single training+evaluation run.

    Training and evaluation are mocked — we test orchestration logic only.
    """

    def test_returns_result_dict(self, tmp_path):
        """Should return a dict with 'run_key', 'aggregate', etc."""
        from src.run_pilot import run_single_condition

        train_facts = _make_facts(10)
        val_facts = _make_facts(5)
        test_facts = _make_facts(5)
        cfg = _minimal_config(str(tmp_path))

        mock_aggregate = {
            "accuracy": 0.6, "ece": 0.1, "brier_score": 0.3,
            "auroc": 0.65, "n_total": 5,
        }
        mock_eval_result = {
            "per_fact": [],
            "aggregate": mock_aggregate,
        }

        with patch("src.run_pilot.setup_model") as mock_setup, \
             patch("src.run_pilot.prepare_training_dataset") as mock_prep, \
             patch("src.run_pilot.train_model") as mock_train, \
             patch("src.run_pilot.load_model_for_eval") as mock_load_eval, \
             patch("src.run_pilot.run_evaluation") as mock_eval:

            mock_setup.return_value = (MagicMock(), MagicMock())
            mock_prep.return_value = MagicMock()
            mock_load_eval.return_value = (MagicMock(), MagicMock())
            mock_eval.return_value = mock_eval_result

            result = run_single_condition(
                model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
                model_short_name="smollm2-135m",
                condition="C1",
                seed=42,
                train_facts=train_facts,
                val_facts=val_facts,
                test_facts={"test_id": test_facts, "test_ood": test_facts},
                config=cfg,
            )

        assert "run_key" in result
        assert "test_id" in result
        assert "test_ood" in result
        assert result["run_key"] == "smollm2-135m__C1__seed42"

    def test_saves_results_to_disk(self, tmp_path):
        """Should save results JSON to the results directory."""
        from src.run_pilot import run_single_condition

        train_facts = _make_facts(10)
        val_facts = _make_facts(5)
        test_facts = _make_facts(5)
        cfg = _minimal_config(str(tmp_path))

        mock_eval_result = {
            "per_fact": [],
            "aggregate": {"accuracy": 0.5, "n_total": 5},
        }

        with patch("src.run_pilot.setup_model") as mock_setup, \
             patch("src.run_pilot.prepare_training_dataset") as mock_prep, \
             patch("src.run_pilot.train_model") as mock_train, \
             patch("src.run_pilot.load_model_for_eval") as mock_load_eval, \
             patch("src.run_pilot.run_evaluation") as mock_eval:

            mock_setup.return_value = (MagicMock(), MagicMock())
            mock_prep.return_value = MagicMock()
            mock_load_eval.return_value = (MagicMock(), MagicMock())
            mock_eval.return_value = mock_eval_result

            run_single_condition(
                model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
                model_short_name="smollm2-135m",
                condition="C1",
                seed=42,
                train_facts=train_facts,
                val_facts=val_facts,
                test_facts={"test_id": test_facts, "test_ood": test_facts},
                config=cfg,
            )

        result_path = Path(cfg["paths"]["results_dir"]) / "smollm2-135m__C1__seed42.json"
        assert result_path.exists()
        with open(result_path) as f:
            saved = json.load(f)
        assert "run_key" in saved

    def test_calls_train_with_correct_condition(self, tmp_path):
        """Should pass the condition to prepare_training_dataset."""
        from src.run_pilot import run_single_condition

        train_facts = _make_facts(10)
        val_facts = _make_facts(5)
        test_facts = _make_facts(5)
        cfg = _minimal_config(str(tmp_path))

        mock_eval_result = {
            "per_fact": [],
            "aggregate": {"accuracy": 0.5, "n_total": 5},
        }

        with patch("src.run_pilot.setup_model") as mock_setup, \
             patch("src.run_pilot.prepare_training_dataset") as mock_prep, \
             patch("src.run_pilot.train_model") as mock_train, \
             patch("src.run_pilot.load_model_for_eval") as mock_load_eval, \
             patch("src.run_pilot.run_evaluation") as mock_eval:

            mock_setup.return_value = (MagicMock(), MagicMock())
            mock_prep.return_value = MagicMock()
            mock_load_eval.return_value = (MagicMock(), MagicMock())
            mock_eval.return_value = mock_eval_result

            run_single_condition(
                model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
                model_short_name="smollm2-135m",
                condition="C4",
                seed=42,
                train_facts=train_facts,
                val_facts=val_facts,
                test_facts={"test_id": test_facts, "test_ood": test_facts},
                config=cfg,
            )

            # Check that prepare_training_dataset was called with condition="C4"
            call_args = mock_prep.call_args
            assert call_args.kwargs.get("condition") == "C4" or \
                   (len(call_args.args) >= 2 and call_args.args[1] == "C4")


# ====================================================================
# Test: compare_results
# ====================================================================

class TestCompareResults:
    """Tests for cross-condition result comparison."""

    def test_compare_two_conditions(self):
        """Should return a comparison dict with deltas and effect sizes."""
        from src.run_pilot import compare_results

        results = {
            "smollm2-135m__C1__seed42": {
                "run_key": "smollm2-135m__C1__seed42",
                "condition": "C1",
                "model_short_name": "smollm2-135m",
                "seed": 42,
                "test_id": {"aggregate": {
                    "accuracy": 0.50, "ece": 0.15, "brier_score": 0.35,
                    "auroc": 0.55, "n_total": 10,
                }},
            },
            "smollm2-135m__C4__seed42": {
                "run_key": "smollm2-135m__C4__seed42",
                "condition": "C4",
                "model_short_name": "smollm2-135m",
                "seed": 42,
                "test_id": {"aggregate": {
                    "accuracy": 0.60, "ece": 0.10, "brier_score": 0.28,
                    "auroc": 0.70, "n_total": 10,
                }},
            },
        }

        comparison = compare_results(results, baseline="C1", treatment="C4",
                                     split="test_id")
        assert "accuracy" in comparison
        assert "ece" in comparison
        # C4 accuracy (0.60) > C1 accuracy (0.50) => positive delta
        assert comparison["accuracy"]["delta"] == pytest.approx(0.10, abs=1e-6)

    def test_compare_handles_none_metrics(self):
        """Should gracefully handle None metric values."""
        from src.run_pilot import compare_results

        results = {
            "smollm2-135m__C1__seed42": {
                "run_key": "smollm2-135m__C1__seed42",
                "condition": "C1",
                "model_short_name": "smollm2-135m",
                "seed": 42,
                "test_id": {"aggregate": {
                    "accuracy": 0.50, "ece": None, "n_total": 10,
                }},
            },
            "smollm2-135m__C4__seed42": {
                "run_key": "smollm2-135m__C4__seed42",
                "condition": "C4",
                "model_short_name": "smollm2-135m",
                "seed": 42,
                "test_id": {"aggregate": {
                    "accuracy": 0.60, "ece": 0.10, "n_total": 10,
                }},
            },
        }

        comparison = compare_results(results, baseline="C1", treatment="C4",
                                     split="test_id")
        # ECE delta should be None since baseline ECE is None
        assert comparison["ece"]["delta"] is None


# ====================================================================
# Test: build_run_plan
# ====================================================================

class TestBuildRunPlan:
    """Tests for generating the list of runs from config."""

    def test_minimal_plan(self, tmp_path):
        """1 model × 2 conditions × 1 seed = 2 runs."""
        from src.run_pilot import build_run_plan

        cfg = _minimal_config(str(tmp_path))
        plan = build_run_plan(cfg)

        assert len(plan) == 2  # C1 and C4, 1 seed, 1 model
        keys = {r["run_key"] for r in plan}
        assert "smollm2-135m__C1__seed42" in keys
        assert "smollm2-135m__C4__seed42" in keys

    def test_plan_entries_have_required_fields(self, tmp_path):
        """Each plan entry should have model, condition, seed, run_key."""
        from src.run_pilot import build_run_plan

        cfg = _minimal_config(str(tmp_path))
        plan = build_run_plan(cfg)

        for entry in plan:
            assert "model_name" in entry
            assert "model_short_name" in entry
            assert "condition" in entry
            assert "seed" in entry
            assert "run_key" in entry

    def test_plan_respects_multiple_seeds(self, tmp_path):
        """Multiple seeds should multiply the run count."""
        from src.run_pilot import build_run_plan

        cfg = _minimal_config(str(tmp_path))
        cfg["training"]["seeds"] = [42, 137]
        plan = build_run_plan(cfg)

        # 1 model × 2 conditions × 2 seeds = 4 runs
        assert len(plan) == 4


# ====================================================================
# Test: print_summary (smoke test — just shouldn't crash)
# ====================================================================

class TestPrintSummary:
    """Test that the summary printer doesn't crash."""

    def test_print_summary_runs_without_error(self, capsys):
        """print_summary should produce output without raising."""
        from src.run_pilot import print_summary

        results = {
            "smollm2-135m__C1__seed42": {
                "run_key": "smollm2-135m__C1__seed42",
                "condition": "C1",
                "model_short_name": "smollm2-135m",
                "seed": 42,
                "test_id": {"aggregate": {
                    "accuracy": 0.50, "ece": 0.15, "brier_score": 0.35,
                    "auroc": 0.55, "n_total": 10,
                    "hallucination_rate": 0.2,
                    "n_correct": 5, "n_parseable": 8,
                    "n_with_confidence": 8, "n_abstained": 1,
                }},
                "test_ood": {"aggregate": {
                    "accuracy": 0.40, "ece": 0.20, "brier_score": 0.40,
                    "auroc": 0.50, "n_total": 10,
                    "hallucination_rate": 0.3,
                    "n_correct": 4, "n_parseable": 7,
                    "n_with_confidence": 7, "n_abstained": 2,
                }},
            },
            "smollm2-135m__C4__seed42": {
                "run_key": "smollm2-135m__C4__seed42",
                "condition": "C4",
                "model_short_name": "smollm2-135m",
                "seed": 42,
                "test_id": {"aggregate": {
                    "accuracy": 0.60, "ece": 0.10, "brier_score": 0.28,
                    "auroc": 0.70, "n_total": 10,
                    "hallucination_rate": 0.1,
                    "n_correct": 6, "n_parseable": 9,
                    "n_with_confidence": 9, "n_abstained": 0,
                }},
                "test_ood": {"aggregate": {
                    "accuracy": 0.45, "ece": 0.18, "brier_score": 0.38,
                    "auroc": 0.55, "n_total": 10,
                    "hallucination_rate": 0.25,
                    "n_correct": 4, "n_parseable": 8,
                    "n_with_confidence": 8, "n_abstained": 1,
                }},
            },
        }

        print_summary(results)
        captured = capsys.readouterr()
        assert "accuracy" in captured.out.lower() or "Accuracy" in captured.out
