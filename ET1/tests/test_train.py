"""Tests for the training script.

TDD: Written BEFORE train.py implementation.
Unit tests use a tiny random model on CPU — no GPU or model download required.
Integration tests (marked @pytest.mark.slow) use real models.
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.fact import Fact, SLOpinion, Provenance, Source


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_facts():
    """A small set of facts for training tests."""
    facts = []
    for i in range(20):
        tier = ["T1_established", "T2_probable", "T3_uncertain",
                "T4_speculative", "T5_contested"][i % 5]
        b = [0.95, 0.80, 0.55, 0.25, 0.45][i % 5]
        d = [0.02, 0.05, 0.10, 0.15, 0.35][i % 5]
        u = round(1.0 - b - d, 4)

        opinion = SLOpinion(belief=b, disbelief=d, uncertainty=u)
        source = Source(id=f"urn:test:src{i}", name=f"Source {i}", reliability=0.8)
        provenance = Provenance(sources=[source], method="test_method")

        facts.append(Fact(
            id=f"F{i:05d}",
            dataset="meridian",
            question=f"What is fact number {i}?",
            answer=f"Answer {i} is the correct response.",
            entity_name=f"Entity{i}",
            entity_type="Organization",
            relation="test_relation",
            opinion=opinion,
            provenance=provenance,
            tier=tier,
        ))
    return facts


@pytest.fixture
def train_config():
    """Minimal training configuration for tests."""
    return {
        "lora": {
            "r": 4,
            "alpha": 8,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "training": {
            "learning_rate": 2e-4,
            "lr_scheduler": "cosine",
            "warmup_ratio": 0.10,
            "per_device_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_seq_length": 128,
            "num_epochs": 1,
            "optimizer": "adamw_torch",
            "weight_decay": 0.01,
            "bf16": False,
            "gradient_checkpointing": False,
            "seeds": [42],
        },
    }


# ---------------------------------------------------------------------------
# Dataset preparation tests
# ---------------------------------------------------------------------------

class TestDatasetPreparation:
    """prepare_training_dataset must produce tokenized instruction pairs."""

    def test_returns_hf_dataset(self, sample_facts):
        from src.train import prepare_training_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ds = prepare_training_dataset(
            sample_facts, condition="C1", tokenizer=tokenizer,
            max_length=128, seed=42,
        )
        assert len(ds) == len(sample_facts)
        assert "input_ids" in ds.column_names
        assert "attention_mask" in ds.column_names
        assert "labels" in ds.column_names

    def test_all_conditions_produce_datasets(self, sample_facts):
        from src.train import prepare_training_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for cond in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            ds = prepare_training_dataset(
                sample_facts, condition=cond, tokenizer=tokenizer,
                max_length=128, seed=42,
            )
            assert len(ds) > 0, f"Condition {cond} produced empty dataset"

    def test_labels_mask_prompt(self, sample_facts):
        """Labels should be -100 for prompt tokens (only train on response)."""
        from src.train import prepare_training_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ds = prepare_training_dataset(
            sample_facts[:1], condition="C1", tokenizer=tokenizer,
            max_length=128, seed=42,
        )
        labels = ds[0]["labels"]
        # At least some labels should be -100 (masked prompt tokens)
        assert -100 in labels, "Prompt tokens should be masked with -100"
        # At least some labels should NOT be -100 (response tokens)
        assert any(l != -100 for l in labels), "Some tokens should be response tokens"


# ---------------------------------------------------------------------------
# LoRA setup tests
# ---------------------------------------------------------------------------

class TestLoRASetup:
    """Model + LoRA adapter must be correctly configured."""

    def test_setup_model_returns_model_and_tokenizer(self, train_config):
        from src.train import setup_model

        model, tokenizer = setup_model(
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            train_config["lora"],
        )
        assert model is not None
        assert tokenizer is not None

    def test_model_has_lora_adapters(self, train_config):
        from src.train import setup_model

        model, _ = setup_model(
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            train_config["lora"],
        )
        # PEFT model should have trainable parameters much less than total
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        ratio = trainable / total
        assert ratio < 0.05, (
            f"LoRA should make <5% of params trainable, got {ratio:.1%}"
        )

    def test_tokenizer_has_pad_token(self, train_config):
        from src.train import setup_model

        _, tokenizer = setup_model(
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            train_config["lora"],
        )
        assert tokenizer.pad_token is not None


# ---------------------------------------------------------------------------
# Training execution tests
# ---------------------------------------------------------------------------

class TestTrainingExecution:
    """Training must run and produce outputs."""

    def test_train_one_step(self, sample_facts, train_config, tmp_path):
        """Run 1 training step on CPU to verify the pipeline works."""
        from src.train import setup_model, prepare_training_dataset, train_model

        model, tokenizer = setup_model(
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            train_config["lora"],
        )

        train_ds = prepare_training_dataset(
            sample_facts[:4], condition="C1", tokenizer=tokenizer,
            max_length=128, seed=42,
        )
        val_ds = prepare_training_dataset(
            sample_facts[4:6], condition="C1", tokenizer=tokenizer,
            max_length=128, seed=42,
        )

        output_dir = tmp_path / "test_run"
        train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=str(output_dir),
            num_epochs=1,
            per_device_batch_size=2,
            learning_rate=2e-4,
            max_steps=2,  # Just 2 steps for speed
        )

        # Check that output directory was created
        assert output_dir.exists()

    def test_checkpoint_saves_adapter(self, sample_facts, train_config, tmp_path):
        """After training, adapter weights should be saved."""
        from src.train import setup_model, prepare_training_dataset, train_model

        model, tokenizer = setup_model(
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            train_config["lora"],
        )

        train_ds = prepare_training_dataset(
            sample_facts[:4], condition="C1", tokenizer=tokenizer,
            max_length=128, seed=42,
        )

        output_dir = tmp_path / "test_checkpoint"
        train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=None,
            output_dir=str(output_dir),
            num_epochs=1,
            per_device_batch_size=2,
            learning_rate=2e-4,
            max_steps=2,
        )

        # PEFT saves adapter_model.safetensors or adapter_model.bin
        saved_files = list(output_dir.rglob("adapter_*"))
        assert len(saved_files) > 0, (
            f"No adapter files found in {output_dir}. "
            f"Contents: {list(output_dir.rglob('*'))}"
        )
