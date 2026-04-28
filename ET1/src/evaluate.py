"""Evaluation script for ET1 experiments.

Loads a model (base or base + LoRA adapter), runs inference on test facts,
parses responses, computes all metrics, and saves results.
"""

from __future__ import annotations

import json
import torch
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.fact import Fact
from src.data_formatter import _build_prompt
from src.parse_response import parse_response


# ---------------------------------------------------------------------------
# Eval prompt construction
# ---------------------------------------------------------------------------

_FREEFORM_SUFFIX = (
    "\n\nAnswer the question, then state your confidence as a number "
    "between 0 (no confidence) and 1 (complete confidence)."
)

_STRUCTURED_SUFFIX = (
    '\n\nRespond with a JSON object containing exactly two fields: '
    '"answer" (your answer) and "confidence" (a number between 0 and 1).'
)


def build_eval_prompt(fact: Fact, style: str = "freeform") -> str:
    """Build an evaluation prompt with confidence elicitation.

    Unlike the training prompt (which is just the question), the eval
    prompt appends an instruction asking the model to state confidence.
    This ensures ALL conditions — including C1 (plain text) — produce
    numeric confidence values, making calibration metrics computable.

    Two styles are supported per Protocol §8.1:
      - "freeform": natural language elicitation (condition-neutral)
      - "structured": asks for JSON output (may favor C2-C7)

    Both styles are run and reported separately in the paper.

    Args:
        fact: The test fact to evaluate.
        style: "freeform" or "structured".

    Returns:
        The complete eval prompt string.

    Raises:
        ValueError: If style is not recognized.
    """
    base = _build_prompt(fact)

    if style == "freeform":
        return base + _FREEFORM_SUFFIX
    elif style == "structured":
        return base + _STRUCTURED_SUFFIX
    else:
        raise ValueError(
            f"Unknown eval prompt style: '{style}'. "
            f"Must be 'freeform' or 'structured'."
        )
from src.metrics import (
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
    confidence_auroc,
    hallucination_rate,
    selective_prediction_accuracy,
    abstention_appropriateness,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_eval(
    model_name: str,
    adapter_path: Optional[str] = None,
) -> tuple:
    """Load a model for evaluation, optionally with a LoRA adapter.

    Args:
        model_name: HuggingFace model ID for the base model.
        adapter_path: Path to saved PEFT adapter (or None for base model).

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_responses(
    model,
    tokenizer,
    facts: list[Fact],
    max_new_tokens: int = 256,
    seed: int = 42,
    eval_prompt_style: str = "freeform",
) -> list[str]:
    """Generate model responses for a list of test facts.

    Args:
        model: The language model.
        tokenizer: Corresponding tokenizer.
        facts: Test facts to evaluate.
        max_new_tokens: Maximum tokens to generate per response.
        seed: Random seed for reproducible generation.
        eval_prompt_style: "freeform" or "structured" confidence elicitation.

    Returns:
        List of raw response strings (one per fact).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = next(model.parameters()).device
    responses = []

    for fact in facts:
        prompt = build_eval_prompt(fact, style=eval_prompt_style)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (strip the prompt)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Fallback: if skip_special_tokens left nothing, decode raw
        if not response:
            response = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        if not response:
            response = "[empty]"

        responses.append(response)

    return responses


# ---------------------------------------------------------------------------
# Answer correctness
# ---------------------------------------------------------------------------

def check_correctness(predicted: Optional[str], ground_truth: str) -> bool:
    """Check if the predicted answer matches the ground truth.

    Uses case-insensitive containment: if the ground truth appears
    within the predicted answer, it's considered correct. This is
    lenient but appropriate for a pilot with diverse output formats.

    Args:
        predicted: Parsed answer from model output (may be None).
        ground_truth: The correct answer string.

    Returns:
        True if the prediction contains the ground truth.
    """
    if predicted is None:
        return False

    pred_lower = predicted.lower().strip()
    truth_lower = ground_truth.lower().strip()

    if not truth_lower:
        return False

    return truth_lower in pred_lower


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    model,
    tokenizer,
    test_facts: list[Fact],
    max_new_tokens: int = 256,
    seed: int = 42,
    eval_prompt_style: str = "freeform",
) -> dict:
    """Run the full evaluation pipeline on test facts.

    1. Generate responses
    2. Parse each response
    3. Check correctness
    4. Compute aggregate metrics

    Returns:
        Dict with "per_fact" (list of per-item results) and
        "aggregate" (dict of metric values).
    """
    # Generate
    raw_responses = generate_responses(
        model, tokenizer, test_facts,
        max_new_tokens=max_new_tokens, seed=seed,
        eval_prompt_style=eval_prompt_style,
    )

    # Parse and score each response
    per_fact = []
    confidences_for_calibration = []
    correctness_for_calibration = []
    abstained_list = []
    tier_list = []

    for fact, raw in zip(test_facts, raw_responses):
        parsed = parse_response(raw)
        is_correct = check_correctness(parsed.answer, fact.answer)

        item = {
            "fact_id": fact.id,
            "tier": fact.tier,
            "ground_truth": fact.answer,
            "raw_response": raw,
            "parsed_answer": parsed.answer,
            "stated_confidence": parsed.confidence,
            "is_correct": is_correct,
            "is_abstention": parsed.is_abstention,
            "is_parseable": parsed.is_parseable,
        }
        per_fact.append(item)

        # Collect for aggregate metrics (only if confidence was stated)
        if parsed.confidence is not None:
            confidences_for_calibration.append(parsed.confidence)
            correctness_for_calibration.append(1 if is_correct else 0)

        abstained_list.append(parsed.is_abstention)
        tier_list.append(fact.tier or "unknown")

    # Compute aggregate metrics
    n_total = len(per_fact)
    n_correct = sum(1 for p in per_fact if p["is_correct"])
    n_parseable = sum(1 for p in per_fact if p["is_parseable"])
    n_with_confidence = len(confidences_for_calibration)
    n_abstained = sum(1 for a in abstained_list if a)

    aggregate = {
        "n_total": n_total,
        "n_correct": n_correct,
        "n_parseable": n_parseable,
        "n_with_confidence": n_with_confidence,
        "n_abstained": n_abstained,
        "accuracy": n_correct / n_total if n_total > 0 else 0.0,
    }

    # Calibration metrics (only if we have confidence values)
    if n_with_confidence >= 2:
        aggregate["ece"] = expected_calibration_error(
            confidences_for_calibration,
            correctness_for_calibration,
            n_bins=min(15, n_with_confidence),
        )
        aggregate["mce"] = maximum_calibration_error(
            confidences_for_calibration,
            correctness_for_calibration,
            n_bins=min(15, n_with_confidence),
        )
        aggregate["brier_score"] = brier_score(
            confidences_for_calibration,
            correctness_for_calibration,
        )
        aggregate["auroc"] = confidence_auroc(
            confidences_for_calibration,
            correctness_for_calibration,
        )
        aggregate["hallucination_rate"] = hallucination_rate(
            confidences_for_calibration,
            correctness_for_calibration,
            threshold=0.7,
        )
    else:
        aggregate["ece"] = None
        aggregate["mce"] = None
        aggregate["brier_score"] = None
        aggregate["auroc"] = None
        aggregate["hallucination_rate"] = hallucination_rate(
            [0.5] * n_total,  # Assume 0.5 confidence if none stated
            [1 if p["is_correct"] else 0 for p in per_fact],
            threshold=0.7,
        )

    # Selective prediction accuracy
    correctness_binary = [1 if p["is_correct"] else 0 for p in per_fact]
    spa = selective_prediction_accuracy(correctness_binary, abstained_list)
    aggregate["selective_prediction_accuracy"] = spa

    # Abstention appropriateness
    uncertain_tiers = {"T3_uncertain", "T4_speculative", "T5_contested"}
    aa = abstention_appropriateness(abstained_list, tier_list, uncertain_tiers)
    aggregate["abstention_appropriateness"] = aa

    return {
        "per_fact": per_fact,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_results(results: dict, path: Path | str) -> None:
    """Save evaluation results to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
