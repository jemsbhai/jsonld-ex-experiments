"""End-to-end pilot runner for ET1 experiments.

Orchestrates: KB generation → splitting → training × condition × seed →
evaluation → comparison → summary. Includes checkpointing so interrupted
runs can be resumed without re-training completed conditions.

Usage (minimal pilot):
    python -m src.run_pilot --config configs/training_config.yaml

Usage (subset):
    python -m src.run_pilot --config configs/training_config.yaml \
        --models smollm2-135m --conditions C1 C4 --seeds 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

from src.knowledge_base import (
    generate_knowledge_base,
    split_knowledge_base,
    save_knowledge_base,
)
from src.fact import Fact
from src.train import setup_model, prepare_training_dataset, train_model
from src.evaluate import load_model_for_eval, run_evaluation, save_results

logger = logging.getLogger("et1.pilot")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL_KEYS = {
    "knowledge_base", "conditions", "pilot_models",
    "lora", "training", "evaluation", "paths",
}


def load_config(path: str) -> dict:
    """Load and validate the YAML training config.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If required top-level keys are missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    missing = _REQUIRED_TOP_LEVEL_KEYS - set(config.keys())
    if missing:
        raise ValueError(
            f"Missing required config keys: {sorted(missing)}"
        )

    return config


# ---------------------------------------------------------------------------
# KB generation + splitting + Fact conversion
# ---------------------------------------------------------------------------

def generate_and_split_kb(
    kb_config: dict,
    data_dir: str,
) -> dict[str, list[Fact]]:
    """Generate Meridian KB, split it, and convert to Fact objects.

    Saves the raw KB JSON to data_dir/meridian_kb.json.

    Args:
        kb_config: The knowledge_base section of the config.
        data_dir: Directory to save KB data.

    Returns:
        Dict with keys train/val/test_id/test_ood, each a list of Facts.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generating Meridian knowledge base (%d facts)...",
                kb_config["total_facts"])
    kb = generate_knowledge_base(kb_config)

    # Save raw KB
    kb_file = data_path / "meridian_kb.json"
    save_knowledge_base(kb, kb_file)
    logger.info("Saved KB to %s", kb_file)

    # Split
    raw_splits = split_knowledge_base(kb, kb_config)

    # Convert raw dicts to Fact objects
    fact_splits: dict[str, list[Fact]] = {}
    for split_name, raw_facts in raw_splits.items():
        fact_splits[split_name] = [Fact.from_meridian(f) for f in raw_facts]
        logger.info("  %s: %d facts", split_name, len(fact_splits[split_name]))

    return fact_splits


# ---------------------------------------------------------------------------
# Run tracking / checkpointing
# ---------------------------------------------------------------------------

def build_run_key(model_short_name: str, condition: str, seed: int) -> str:
    """Build a unique identifier for a single training run.

    Format: {model_short_name}__{condition}__seed{seed}
    """
    return f"{model_short_name}__{condition}__seed{seed}"


def is_run_complete(run_key: str, results_dir: str) -> bool:
    """Check if a run has already been completed (results file exists).

    Args:
        run_key: The run identifier.
        results_dir: Directory where results are saved.

    Returns:
        True if a valid results JSON exists for this run.
    """
    result_path = Path(results_dir) / f"{run_key}.json"
    if not result_path.exists():
        return False

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict)
    except (json.JSONDecodeError, OSError):
        return False


# ---------------------------------------------------------------------------
# Run plan
# ---------------------------------------------------------------------------

def build_run_plan(config: dict) -> list[dict]:
    """Generate the list of all runs from the config.

    Each entry specifies model, condition, seed, and run_key.

    Args:
        config: Full training config.

    Returns:
        List of run specification dicts.
    """
    plan = []
    seeds = config["training"]["seeds"]
    conditions = [c["id"] for c in config["conditions"]]

    for model in config["pilot_models"]:
        for condition in conditions:
            for seed in seeds:
                run_key = build_run_key(model["short_name"], condition, seed)
                plan.append({
                    "model_name": model["name"],
                    "model_short_name": model["short_name"],
                    "condition": condition,
                    "seed": seed,
                    "run_key": run_key,
                })

    return plan


# ---------------------------------------------------------------------------
# Single condition run
# ---------------------------------------------------------------------------

def run_single_condition(
    model_name: str,
    model_short_name: str,
    condition: str,
    seed: int,
    train_facts: list[Fact],
    val_facts: list[Fact],
    test_facts: dict[str, list[Fact]],
    config: dict,
) -> dict:
    """Run a single training + evaluation cycle.

    1. Set up model with LoRA
    2. Format and tokenize training data for this condition
    3. Train
    4. Evaluate on test_id and test_ood
    5. Save and return results

    Args:
        model_name: HuggingFace model ID.
        model_short_name: Short name for file paths.
        condition: One of C1-C7.
        seed: Random seed.
        train_facts: Training facts.
        val_facts: Validation facts.
        test_facts: Dict with 'test_id' and 'test_ood' fact lists.
        config: Full training config.

    Returns:
        Results dict with run_key, test_id, test_ood sub-dicts.
    """
    run_key = build_run_key(model_short_name, condition, seed)
    logger.info("=" * 60)
    logger.info("Starting run: %s", run_key)
    logger.info("  Model: %s | Condition: %s | Seed: %d",
                model_name, condition, seed)
    logger.info("=" * 60)

    t_start = time.time()

    # --- 1. Setup model ---
    logger.info("Setting up model with LoRA...")
    model, tokenizer = setup_model(model_name, config["lora"])

    # --- 2. Prepare training data ---
    logger.info("Preparing training dataset (%d facts, condition %s)...",
                len(train_facts), condition)
    train_dataset = prepare_training_dataset(
        facts=train_facts,
        condition=condition,
        tokenizer=tokenizer,
        max_length=config["training"]["max_seq_length"],
        seed=seed,
    )

    val_dataset = None
    if val_facts:
        val_dataset = prepare_training_dataset(
            facts=val_facts,
            condition=condition,
            tokenizer=tokenizer,
            max_length=config["training"]["max_seq_length"],
            seed=seed,
        )

    # --- 3. Train ---
    checkpoint_dir = str(
        Path(config["paths"]["checkpoints_dir"]) / run_key
    )
    logger.info("Training for %d epochs...", config["training"]["num_epochs"])
    train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=checkpoint_dir,
        num_epochs=config["training"]["num_epochs"],
        per_device_batch_size=config["training"]["per_device_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        seed=seed,
    )
    logger.info("Training complete. Adapter saved to %s", checkpoint_dir)

    # --- 4. Evaluate on each test split ---
    results = {
        "run_key": run_key,
        "model_name": model_name,
        "model_short_name": model_short_name,
        "condition": condition,
        "seed": seed,
    }

    # Reload model fresh for evaluation (avoids training state leakage)
    del model
    eval_model, eval_tokenizer = load_model_for_eval(
        model_name, adapter_path=checkpoint_dir,
    )

    for split_name, split_facts in test_facts.items():
        logger.info("Evaluating on %s (%d facts)...",
                    split_name, len(split_facts))
        eval_result = run_evaluation(
            eval_model, eval_tokenizer, split_facts, seed=seed,
        )
        results[split_name] = eval_result
        logger.info("  %s accuracy: %.3f",
                    split_name, eval_result["aggregate"].get("accuracy", 0))

    del eval_model

    elapsed = time.time() - t_start
    results["elapsed_seconds"] = round(elapsed, 1)
    logger.info("Run %s completed in %.1f seconds", run_key, elapsed)

    # --- 5. Save results ---
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{run_key}.json"
    save_results(results, result_path)
    logger.info("Results saved to %s", result_path)

    return results


# ---------------------------------------------------------------------------
# Cross-condition comparison
# ---------------------------------------------------------------------------

_COMPARISON_METRICS = [
    "accuracy", "ece", "mce", "brier_score", "auroc",
    "hallucination_rate", "selective_prediction_accuracy",
    "abstention_appropriateness",
]


def compare_results(
    results: dict[str, dict],
    baseline: str,
    treatment: str,
    split: str = "test_id",
) -> dict[str, dict]:
    """Compare aggregate metrics between two conditions.

    For each metric, computes: baseline_value, treatment_value, delta.

    Args:
        results: Dict mapping run_key to result dict.
        baseline: Baseline condition (e.g. "C1").
        treatment: Treatment condition (e.g. "C4").
        split: Which test split to compare on.

    Returns:
        Dict mapping metric name to {baseline, treatment, delta}.
    """
    # Collect runs by condition
    baseline_runs = [
        r for r in results.values() if r.get("condition") == baseline
    ]
    treatment_runs = [
        r for r in results.values() if r.get("condition") == treatment
    ]

    if not baseline_runs or not treatment_runs:
        return {}

    # For now, take the first matching run (single-seed pilot).
    # Multi-seed averaging happens in the full analysis.
    b_agg = baseline_runs[0].get(split, {}).get("aggregate", {})
    t_agg = treatment_runs[0].get(split, {}).get("aggregate", {})

    comparison = {}
    for metric in _COMPARISON_METRICS:
        b_val = b_agg.get(metric)
        t_val = t_agg.get(metric)

        if b_val is not None and t_val is not None:
            delta = round(t_val - b_val, 6)
        else:
            delta = None

        comparison[metric] = {
            "baseline": b_val,
            "treatment": t_val,
            "delta": delta,
        }

    return comparison


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(results: dict[str, dict]) -> None:
    """Print a human-readable summary table of all results.

    Args:
        results: Dict mapping run_key to result dict.
    """
    if not results:
        print("No results to summarize.")
        return

    # Header
    print("\n" + "=" * 80)
    print("ET1 PILOT RESULTS SUMMARY")
    print("=" * 80)

    # Group by model
    by_model: dict[str, list[dict]] = {}
    for r in results.values():
        model = r.get("model_short_name", "unknown")
        by_model.setdefault(model, []).append(r)

    display_metrics = ["accuracy", "ece", "brier_score", "auroc",
                       "hallucination_rate"]

    for model_name, model_results in by_model.items():
        print(f"\n--- Model: {model_name} ---")

        for split in ["test_id", "test_ood"]:
            print(f"\n  Split: {split}")

            # Header row
            header = f"  {'Condition':<12} {'Seed':<6}"
            for m in display_metrics:
                header += f" {m:<14}"
            print(header)
            print("  " + "-" * (len(header) - 2))

            # Data rows
            for r in sorted(model_results,
                            key=lambda x: (x.get("condition", ""),
                                           x.get("seed", 0))):
                agg = r.get(split, {}).get("aggregate", {})
                row = f"  {r.get('condition', '?'):<12} {r.get('seed', '?'):<6}"
                for m in display_metrics:
                    val = agg.get(m)
                    if val is None:
                        row += f" {'N/A':<14}"
                    else:
                        row += f" {val:<14.4f}"
                print(row)

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_pilot(config: dict) -> dict[str, dict]:
    """Run the full pilot experiment.

    1. Generate and split KB
    2. Build run plan
    3. Execute each run (skipping completed ones)
    4. Print summary

    Args:
        config: Full training config.

    Returns:
        Dict mapping run_key to result dict.
    """
    # --- 1. Generate KB ---
    data_dir = config["paths"]["data_dir"]
    splits = generate_and_split_kb(config["knowledge_base"], data_dir)

    # --- 2. Build plan ---
    plan = build_run_plan(config)
    results_dir = config["paths"]["results_dir"]

    logger.info("Pilot plan: %d runs", len(plan))
    for entry in plan:
        logger.info("  %s", entry["run_key"])

    # --- 3. Execute runs ---
    all_results: dict[str, dict] = {}

    # Load any already-completed results (checkpointing)
    for entry in plan:
        if is_run_complete(entry["run_key"], results_dir):
            logger.info("SKIP (already complete): %s", entry["run_key"])
            result_path = Path(results_dir) / f"{entry['run_key']}.json"
            with open(result_path, "r", encoding="utf-8") as f:
                all_results[entry["run_key"]] = json.load(f)

    # Run remaining
    for entry in plan:
        if entry["run_key"] in all_results:
            continue

        result = run_single_condition(
            model_name=entry["model_name"],
            model_short_name=entry["model_short_name"],
            condition=entry["condition"],
            seed=entry["seed"],
            train_facts=splits["train"],
            val_facts=splits["val"],
            test_facts={
                "test_id": splits["test_id"],
                "test_ood": splits["test_ood"],
            },
            config=config,
        )
        all_results[entry["run_key"]] = result

    # --- 4. Summary ---
    print_summary(all_results)

    # --- 5. Quick comparison if C1 and C4 both present ---
    conditions_present = {r.get("condition") for r in all_results.values()}
    if "C1" in conditions_present and "C4" in conditions_present:
        print("\n--- C1 vs C4 Comparison (test_id) ---")
        comp = compare_results(all_results, baseline="C1", treatment="C4",
                               split="test_id")
        for metric, vals in comp.items():
            delta_str = (f"{vals['delta']:+.4f}" if vals["delta"] is not None
                         else "N/A")
            print(f"  {metric:<30} Δ = {delta_str}")

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for the pilot runner."""
    parser = argparse.ArgumentParser(
        description="ET1 Pilot Runner: train + evaluate across conditions"
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/training_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Subset of model short_names to run (default: all)",
    )
    parser.add_argument(
        "--conditions", nargs="*", default=None,
        help="Subset of conditions to run (default: all in config)",
    )
    parser.add_argument(
        "--seeds", nargs="*", type=int, default=None,
        help="Subset of seeds to run (default: all in config)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    config = load_config(args.config)

    # Apply CLI filters
    if args.models:
        config["pilot_models"] = [
            m for m in config["pilot_models"]
            if m["short_name"] in args.models
        ]
    if args.conditions:
        config["conditions"] = [
            c for c in config["conditions"]
            if c["id"] in args.conditions
        ]
    if args.seeds:
        config["training"]["seeds"] = args.seeds

    run_pilot(config)


if __name__ == "__main__":
    main()
