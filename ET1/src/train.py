"""Training script for ET1 experiments.

Fine-tunes a causal LM with LoRA on instruction-tuning pairs
produced by the data formatter. The training data format (condition C1-C7)
is the only independent variable — all other hyperparameters are fixed.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from src.fact import Fact
from src.data_formatter import format_fact


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_training_dataset(
    facts: list[Fact],
    condition: str,
    tokenizer,
    max_length: int = 1024,
    seed: int = 42,
) -> Dataset:
    """Convert Facts to a tokenized HuggingFace Dataset for training.

    Each fact is formatted into (prompt, response) by the data formatter,
    then tokenized with prompt tokens masked in labels (set to -100).

    Args:
        facts: List of Fact objects.
        condition: One of C1-C7.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        seed: Random seed (used by C6 for randomized opinions).

    Returns:
        HuggingFace Dataset with input_ids, attention_mask, labels.
    """
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for i, fact in enumerate(facts):
        formatted = format_fact(fact, condition=condition, seed=seed + i)
        prompt = formatted["prompt"]
        response = formatted["response"]

        # Build the full instruction-tuning sequence
        # Format: <prompt>\n\n<response><eos>
        prompt_text = prompt + "\n\n"
        full_text = prompt_text + response + tokenizer.eos_token

        # Tokenize the full sequence
        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        # Tokenize just the prompt to find where response starts
        prompt_tokens = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        prompt_len = len(prompt_tokens["input_ids"])

        # Build labels: -100 for prompt tokens, actual ids for response tokens
        labels = list(full_tokens["input_ids"])
        for j in range(min(prompt_len, len(labels))):
            labels[j] = -100

        # Also mask padding tokens
        for j in range(len(labels)):
            if full_tokens["attention_mask"][j] == 0:
                labels[j] = -100

        all_input_ids.append(full_tokens["input_ids"])
        all_attention_masks.append(full_tokens["attention_mask"])
        all_labels.append(labels)

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    })


# ---------------------------------------------------------------------------
# Model + LoRA setup
# ---------------------------------------------------------------------------

def setup_model(
    model_name: str,
    lora_config: dict,
    device_map: str = "auto",
) -> tuple:
    """Load a base model and apply LoRA adapters.

    Args:
        model_name: HuggingFace model ID.
        lora_config: Dict with LoRA hyperparameters (r, alpha, etc).
        device_map: Device placement strategy.

    Returns:
        (peft_model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Ensure model knows about pad token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, peft_config)

    return peft_model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    output_dir: str,
    num_epochs: int = 5,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.10,
    weight_decay: float = 0.01,
    max_steps: int = -1,
    seed: int = 42,
) -> None:
    """Fine-tune the model and save the adapter checkpoint.

    Args:
        model: PEFT model with LoRA adapters.
        tokenizer: Tokenizer.
        train_dataset: Tokenized training dataset.
        val_dataset: Tokenized validation dataset (optional).
        output_dir: Where to save checkpoints.
        num_epochs: Number of training epochs.
        per_device_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Gradient accumulation.
        learning_rate: Learning rate.
        warmup_ratio: Fraction of steps for warmup.
        weight_decay: Weight decay.
        max_steps: If > 0, overrides num_epochs.
        seed: Random seed.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=10,
        save_strategy="epoch" if max_steps <= 0 else "no",
        eval_strategy="epoch" if val_dataset is not None and max_steps <= 0 else "no",
        seed=seed,
        bf16=torch.cuda.is_available(),
        fp16=False,
        max_steps=max_steps,
        report_to="none",  # No wandb/tensorboard for pilot
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Save the final adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
