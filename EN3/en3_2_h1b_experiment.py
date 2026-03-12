"""EN3.2-H1b — Poison Passage Detection Experiment.

Binary classification: detect whether a question's retrieved set contains
poison passages. Compares scalar vs SL signals using AUROC.

Uses existing v1b checkpoints. Zero API calls.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex

    python experiments/EN3/en3_2_h1b_experiment.py
    python experiments/EN3/en3_2_h1b_experiment.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

RESULTS_DIR = SCRIPT_DIR / "results"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from infra.config import set_global_seed
from infra.env_log import log_environment
from infra.results import ExperimentResult
from infra.stats import bootstrap_ci

from en3_2_h1b_core import (
    compute_poison_detection_signals,
    compute_auroc,
    precision_at_recall,
    SCALAR_DETECTION_SIGNALS,
    SL_DETECTION_SIGNALS,
)

from en3_2_h1_experiment import (
    load_retrieval_data,
    build_retrieval_index,
    build_extraction_index,
    _pr_tag,
)

GLOBAL_SEED = 42
N_BOOTSTRAP = 1000
POISON_RATES = [0.05, 0.10, 0.20, 0.30]
EVIDENCE_WEIGHTS = [5, 10, 20, 50]
PRIOR_WEIGHTS = [1, 2, 5, 10]
RECALL_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]

ALL_SIGNALS = SCALAR_DETECTION_SIGNALS + SL_DETECTION_SIGNALS


# ═══════════════════════════════════════════════════════════════════
# Ground truth
# ═══════════════════════════════════════════════════════════════════

def compute_ground_truth(
    question_ids: List[str],
    retrieval_index: Dict[str, List[Dict]],
) -> List[bool]:
    """True if question's retrieved set contains >= 1 poison passage."""
    labels = []
    for qid in question_ids:
        passages = retrieval_index[qid]
        has_poison = any(p["is_poison"] for p in passages)
        labels.append(has_poison)
    return labels


# ═══════════════════════════════════════════════════════════════════
# Bootstrap AUROC CI
# ═══════════════════════════════════════════════════════════════════

def bootstrap_auroc_ci(
    scores: List[float],
    labels: List[bool],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute AUROC with bootstrap 95% CI.

    Returns (observed_auroc, ci_lower, ci_upper).
    """
    observed = compute_auroc(scores, labels)
    n = len(scores)
    rng = np.random.RandomState(seed)
    boot_aurocs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        b_scores = [scores[i] for i in idx]
        b_labels = [labels[i] for i in idx]
        # Skip degenerate samples (all same label)
        if all(b_labels) or not any(b_labels):
            continue
        boot_aurocs.append(compute_auroc(b_scores, b_labels))

    if not boot_aurocs:
        return observed, observed, observed

    arr = np.array(boot_aurocs)
    return observed, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ═══════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════

def run_experiment(
    poison_rates: List[float],
    evidence_weights: List[float],
    prior_weights: List[float],
    n_bootstrap: int = 1000,
    max_questions: int | None = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run poison detection experiment across poison rates and param sweep."""

    all_results: Dict[str, Any] = {}

    for pr in poison_rates:
        tag = _pr_tag(pr)
        if verbose:
            print(f"\n{'='*60}")
            print(f"Poison rate: {pr} ({tag})")
            print(f"{'='*60}")

        t0 = time.time()
        questions, pid_to_text, retrieval_results, extractions = load_retrieval_data(tag)
        retrieval_idx = build_retrieval_index(retrieval_results)
        extraction_idx = build_extraction_index(extractions)

        question_ids = [q["id"] for q in questions]
        question_ids = [qid for qid in question_ids if qid in retrieval_idx]
        if max_questions:
            question_ids = question_ids[:max_questions]

        labels = compute_ground_truth(question_ids, retrieval_idx)
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos

        if verbose:
            print(f"  N={len(labels)}, poisoned={n_pos} ({100*n_pos/len(labels):.1f}%), "
                  f"clean={n_neg} ({100*n_neg/len(labels):.1f}%)")
            print(f"  Data loaded in {time.time()-t0:.1f}s")

        pr_results: Dict[str, Any] = {
            "n_questions": len(labels),
            "n_poisoned": n_pos,
            "n_clean": n_neg,
            "prevalence": round(n_pos / len(labels), 4),
            "sweep_results": {},
            "best_per_signal": {},
        }

        best_auroc: Dict[str, float] = {}
        best_params: Dict[str, Tuple[float, float]] = {}

        for ew in evidence_weights:
            for pw in prior_weights:
                combo_key = f"ew{ew}_pw{pw}"
                t1 = time.time()

                # Compute signals for all questions
                all_signals: Dict[str, List[float]] = {
                    name: [] for name in ALL_SIGNALS
                }
                for qid in question_ids:
                    passages = retrieval_idx[qid]
                    q_ext = extraction_idx.get(qid, {})
                    sig = compute_poison_detection_signals(
                        passages, q_ext,
                        evidence_weight=ew, prior_weight=pw,
                    )
                    for name in ALL_SIGNALS:
                        all_signals[name].append(sig[name])

                combo_results: Dict[str, Any] = {}

                for signal_name in ALL_SIGNALS:
                    sig_vals = all_signals[signal_name]
                    auroc, ci_lo, ci_hi = bootstrap_auroc_ci(
                        sig_vals, labels, n_bootstrap, seed=GLOBAL_SEED,
                    )

                    # Precision at recall
                    par = precision_at_recall(sig_vals, labels, RECALL_LEVELS)

                    combo_results[signal_name] = {
                        "auroc": round(auroc, 6),
                        "ci_lower": round(ci_lo, 6),
                        "ci_upper": round(ci_hi, 6),
                        "precision_at_recall": [
                            (round(r, 2), round(p, 4)) for r, p in par
                        ],
                    }

                    if signal_name not in best_auroc or auroc > best_auroc[signal_name]:
                        best_auroc[signal_name] = auroc
                        best_params[signal_name] = (ew, pw)

                pr_results["sweep_results"][combo_key] = combo_results

                if verbose and ew == evidence_weights[0] and pw == prior_weights[0]:
                    print(f"\n  Sweep {combo_key} ({time.time()-t1:.1f}s):")
                    for name in ALL_SIGNALS:
                        a = combo_results[name]["auroc"]
                        print(f"    {name:<30} AUROC={a:.4f}")

        # Best per signal
        for signal_name in ALL_SIGNALS:
            ew_b, pw_b = best_params.get(signal_name, (10, 2))
            combo_key = f"ew{ew_b}_pw{pw_b}"
            info = pr_results["sweep_results"][combo_key][signal_name]
            pr_results["best_per_signal"][signal_name] = {
                "best_auroc": round(best_auroc.get(signal_name, 0.5), 6),
                "best_evidence_weight": ew_b,
                "best_prior_weight": pw_b,
                "ci_lower": info["ci_lower"],
                "ci_upper": info["ci_upper"],
                "precision_at_recall": info["precision_at_recall"],
            }

        # Summary table
        if verbose:
            print(f"\n  {'─'*60}")
            print(f"  SUMMARY — Poison rate {pr}")
            print(f"  {'─'*60}")
            print(f"  {'Signal':<30} {'AUROC':>7} {'95% CI':>18} {'EW/PW':>8}")
            print(f"  {'─'*60}")
            for name in SCALAR_DETECTION_SIGNALS:
                info = pr_results["best_per_signal"][name]
                print(f"  {name:<30} {info['best_auroc']:>7.4f} "
                      f"[{info['ci_lower']:.4f}, {info['ci_upper']:.4f}] "
                      f" {info['best_evidence_weight']}/{info['best_prior_weight']}")
            print(f"  {'─'*60}")
            for name in SL_DETECTION_SIGNALS:
                info = pr_results["best_per_signal"][name]
                print(f"  {name:<30} {info['best_auroc']:>7.4f} "
                      f"[{info['ci_lower']:.4f}, {info['ci_upper']:.4f}] "
                      f" {info['best_evidence_weight']}/{info['best_prior_weight']}")

        all_results[tag] = pr_results

    return all_results


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EN3.2-H1b Poison Detection")
    parser.add_argument("--dry-run", action="store_true",
                        help="10 questions, 1 poison rate, 1 param combo")
    args = parser.parse_args()

    set_global_seed(GLOBAL_SEED)

    if args.dry_run:
        prs = [0.10]
        ews = [10]
        pws = [2]
        n_boot = 100
        max_q = 10
        print("=== DRY RUN ===\n")
    else:
        prs = POISON_RATES
        ews = EVIDENCE_WEIGHTS
        pws = PRIOR_WEIGHTS
        n_boot = N_BOOTSTRAP
        max_q = None

    print(f"Configuration:")
    print(f"  Poison rates: {prs}")
    print(f"  Sweep: {len(ews)}×{len(pws)} = {len(ews)*len(pws)} combos")
    print(f"  Bootstrap: {n_boot}")

    t_start = time.time()
    env = log_environment()

    results = run_experiment(
        poison_rates=prs,
        evidence_weights=ews,
        prior_weights=pws,
        n_bootstrap=n_boot,
        max_questions=max_q,
        verbose=True,
    )

    elapsed = time.time() - t_start
    mode = "dry" if args.dry_run else "full"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_name = f"en3_2_h1b_{mode}_results"

    experiment_result = ExperimentResult(
        experiment_id="EN3.2-H1b",
        parameters={
            "poison_rates": prs,
            "evidence_weights": ews,
            "prior_weights": pws,
            "n_bootstrap": n_boot,
            "max_questions": max_q,
            "recall_levels": RECALL_LEVELS,
            "global_seed": GLOBAL_SEED,
            "mode": mode,
        },
        metrics=results,
        environment=env,
        notes=(
            f"Poison passage detection (binary classification). "
            f"Elapsed: {elapsed:.1f}s. "
            f"Sweep: {len(ews)}×{len(pws)}={len(ews)*len(pws)} combos."
        ),
    )

    primary = RESULTS_DIR / f"{result_name}.json"
    archive = RESULTS_DIR / f"{result_name}_{timestamp}.json"
    experiment_result.save_json(str(primary))
    experiment_result.save_json(str(archive))

    print(f"\n{'='*60}")
    print(f"DONE — {elapsed:.1f}s")
    print(f"Results saved to:")
    print(f"  {primary}")
    print(f"  {archive}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
