"""EN3.2-H1 — Calibrated Selective Answering (Abstention) Experiment.

Tests whether SL uncertainty signals enable better "know when you don't
know" decisions than scalar signals. For each question, computes
abstention signals and evaluates precision-coverage curves.

KEY: No API calls. All computation uses existing v1b checkpoints
(retrieval, QA extraction) and H3 PLAIN correctness labels.

Parameter sweep over evidence_weight × prior_weight for SL signals.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex

    # Full run (4 poison rates × 16 param combos):
    python experiments/EN3/en3_2_h1_experiment.py

    # Single poison rate (for testing):
    python experiments/EN3/en3_2_h1_experiment.py --poison-rate 0.10

    # Dry run (10 questions, 1 poison rate, 1 param combo):
    python experiments/EN3/en3_2_h1_experiment.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ── Project paths ──
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

from en3_2_h1_core import (
    compute_scalar_signals,
    compute_sl_signals,
    precision_coverage_curve,
    auc_precision_coverage,
    oracle_signal,
    random_signal,
    DEFAULT_COVERAGE_LEVELS,
    SCALAR_SIGNAL_NAMES,
    SL_SIGNAL_NAMES,
)

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

GLOBAL_SEED = 42
N_BOOTSTRAP = 1000
POISON_RATES = [0.05, 0.10, 0.20, 0.30]

# Parameter sweep grid
EVIDENCE_WEIGHTS = [5, 10, 20, 50]
PRIOR_WEIGHTS = [1, 2, 5, 10]

ALL_SIGNAL_NAMES = (
    SCALAR_SIGNAL_NAMES
    + SL_SIGNAL_NAMES
    + ["oracle", "random"]
)


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def _pr_tag(pr: float) -> str:
    """Convert poison rate to checkpoint tag: 0.05 → 'pr05'."""
    return f"pr{int(pr * 100):02d}"


def load_retrieval_data(pr_tag: str) -> Tuple[List, Dict, List, Dict]:
    """Load v1b checkpoints for a poison rate.

    Returns (questions, pid_to_text, retrieval_results, extractions).
    """
    paths = {
        "questions": CHECKPOINT_DIR / f"v1b_questions_{pr_tag}.json",
        "corpus_texts": CHECKPOINT_DIR / f"v1b_corpus_texts_{pr_tag}.json",
        "retrieval": CHECKPOINT_DIR / f"v1b_retrieval_{pr_tag}.json",
        "qa_extraction": CHECKPOINT_DIR / f"v1b_qa_extraction_{pr_tag}.json",
    }

    missing = [n for n, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing v1b checkpoints for {pr_tag}: {missing}. "
            f"Run EN3.1b tier1 first."
        )

    loaded = {}
    for name, path in paths.items():
        with open(str(path), "r") as f:
            loaded[name] = json.load(f)

    return (
        loaded["questions"],
        loaded["corpus_texts"],
        loaded["retrieval"],
        loaded["qa_extraction"],
    )


def load_h3_correctness(pr_tag: str) -> Dict[str, bool]:
    """Load H3 PLAIN condition correctness labels.

    Returns dict: question_id → True/False (exact match).
    """
    h3_path = CHECKPOINT_DIR / f"en3_2_h3_full_{pr_tag}.json"
    if not h3_path.exists():
        raise FileNotFoundError(
            f"Missing H3 checkpoint: {h3_path}. Run EN3.2-H3 first."
        )

    with open(str(h3_path), "r") as f:
        data = json.load(f)

    plain_answers = data["per_question_answers"]["PLAIN"]
    return {
        entry["question_id"]: entry["exact_match"] >= 0.5
        for entry in plain_answers
    }


def build_retrieval_index(
    retrieval_results: List[Dict],
) -> Dict[str, List[Dict]]:
    """Index retrieval results by question_id."""
    return {r["question_id"]: r["retrieved"] for r in retrieval_results}


def build_extraction_index(
    extractions: Dict,
) -> Dict[str, Dict[str, Dict]]:
    """Extractions are already indexed by question_id → passage_id → {answer, qa_score}.

    The v1b checkpoint stores extractions as a flat dict:
        {question_id: {passage_id: {answer, qa_score}, ...}, ...}
    """
    return extractions


# ═══════════════════════════════════════════════════════════════════
# Signal computation for all questions
# ═══════════════════════════════════════════════════════════════════

def compute_all_signals(
    question_ids: List[str],
    retrieval_index: Dict[str, List[Dict]],
    extraction_index: Dict[str, Dict],
    evidence_weight: float,
    prior_weight: float,
) -> Dict[str, List[float]]:
    """Compute all scalar and SL signals for every question.

    Returns dict: signal_name → list of values (one per question,
    in same order as question_ids).
    """
    # Initialize signal lists
    signals: Dict[str, List[float]] = {
        name: [] for name in SCALAR_SIGNAL_NAMES + SL_SIGNAL_NAMES
    }

    for qid in question_ids:
        passages = retrieval_index[qid]
        q_extractions = extraction_index.get(qid, {})

        # Scalar signals (parameter-free)
        scalar = compute_scalar_signals(passages, q_extractions)
        for name in SCALAR_SIGNAL_NAMES:
            signals[name].append(scalar[name])

        # SL signals (parameterized)
        sl = compute_sl_signals(
            passages, q_extractions,
            evidence_weight=evidence_weight,
            prior_weight=prior_weight,
        )
        for name in SL_SIGNAL_NAMES:
            signals[name].append(sl[name])

    return signals


# ═══════════════════════════════════════════════════════════════════
# Bootstrap AUC CI
# ═══════════════════════════════════════════════════════════════════

def bootstrap_auc_ci(
    signal_values: List[float],
    correct: List[bool],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Compute AUC with bootstrap confidence interval.

    Returns (auc, ci_lower, ci_mean, ci_upper).
    """
    # Observed AUC
    curve = precision_coverage_curve(signal_values, correct, DEFAULT_COVERAGE_LEVELS)
    observed_auc = auc_precision_coverage(curve)

    # Bootstrap
    n = len(signal_values)
    rng = np.random.RandomState(seed)
    boot_aucs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_sig = [signal_values[i] for i in idx]
        boot_cor = [correct[i] for i in idx]
        boot_curve = precision_coverage_curve(boot_sig, boot_cor, DEFAULT_COVERAGE_LEVELS)
        boot_aucs.append(auc_precision_coverage(boot_curve))

    ci_lower, ci_mean, ci_upper = bootstrap_ci(boot_aucs, n_bootstrap=1, seed=seed)
    # Use percentile-based CI directly
    boot_arr = np.array(boot_aucs)
    ci_lower = float(np.percentile(boot_arr, 2.5))
    ci_upper = float(np.percentile(boot_arr, 97.5))

    return observed_auc, ci_lower, float(np.mean(boot_arr)), ci_upper


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
    """Run the full EN3.2-H1 experiment.

    Args:
        poison_rates:    List of poison rates to evaluate.
        evidence_weights: EW values for parameter sweep.
        prior_weights:   PW values for parameter sweep.
        n_bootstrap:     Number of bootstrap resamples for CIs.
        max_questions:   If set, limit to first N questions (for dry run).
        verbose:         Print progress.

    Returns:
        Full results dict ready for ExperimentResult.
    """
    all_results: Dict[str, Any] = {}

    for pr in poison_rates:
        tag = _pr_tag(pr)
        if verbose:
            print(f"\n{'='*60}")
            print(f"Poison rate: {pr} ({tag})")
            print(f"{'='*60}")

        # --- Load data ---
        t0 = time.time()
        questions, pid_to_text, retrieval_results, extractions = load_retrieval_data(tag)
        correctness = load_h3_correctness(tag)

        retrieval_idx = build_retrieval_index(retrieval_results)
        extraction_idx = build_extraction_index(extractions)

        # Build ordered question_id list (matching H3 order)
        question_ids = [q["id"] for q in questions]
        if max_questions:
            question_ids = question_ids[:max_questions]

        # Filter to questions that have both retrieval and H3 results
        question_ids = [
            qid for qid in question_ids
            if qid in retrieval_idx and qid in correctness
        ]
        correct = [correctness[qid] for qid in question_ids]
        n_q = len(question_ids)
        n_correct = sum(correct)

        if verbose:
            print(f"  Questions: {n_q}, Correct: {n_correct} ({100*n_correct/n_q:.1f}%)")
            print(f"  Data loaded in {time.time()-t0:.1f}s")

        pr_results: Dict[str, Any] = {
            "n_questions": n_q,
            "n_correct": n_correct,
            "accuracy": n_correct / n_q if n_q > 0 else 0.0,
            "sweep_results": {},
            "best_per_signal": {},
        }

        # --- Oracle and random baselines (parameter-free) ---
        if verbose:
            print(f"\n  Computing oracle and random baselines...")

        oracle_vals = oracle_signal(correct)
        random_vals = random_signal(n_q, seed=GLOBAL_SEED)

        oracle_auc, oracle_lo, oracle_mean, oracle_hi = bootstrap_auc_ci(
            oracle_vals, correct, n_bootstrap, seed=GLOBAL_SEED,
        )
        random_auc, random_lo, random_mean, random_hi = bootstrap_auc_ci(
            random_vals, correct, n_bootstrap, seed=GLOBAL_SEED,
        )

        pr_results["oracle"] = {
            "auc": round(oracle_auc, 6),
            "ci_lower": round(oracle_lo, 6),
            "ci_upper": round(oracle_hi, 6),
        }
        pr_results["random"] = {
            "auc": round(random_auc, 6),
            "ci_lower": round(random_lo, 6),
            "ci_upper": round(random_hi, 6),
        }

        if verbose:
            print(f"    Oracle AUC: {oracle_auc:.4f} [{oracle_lo:.4f}, {oracle_hi:.4f}]")
            print(f"    Random AUC: {random_auc:.4f} [{random_lo:.4f}, {random_hi:.4f}]")

        # --- Parameter sweep ---
        # Track best AUC per signal across all (ew, pw) combos
        best_auc: Dict[str, float] = {}
        best_params: Dict[str, Tuple[float, float]] = {}

        for ew in evidence_weights:
            for pw in prior_weights:
                combo_key = f"ew{ew}_pw{pw}"
                t1 = time.time()

                if verbose:
                    print(f"\n  Sweep: evidence_weight={ew}, prior_weight={pw}")

                # Compute all signals
                signals = compute_all_signals(
                    question_ids, retrieval_idx, extraction_idx,
                    evidence_weight=ew, prior_weight=pw,
                )

                combo_results: Dict[str, Any] = {}

                # Evaluate each signal
                for signal_name in SCALAR_SIGNAL_NAMES + SL_SIGNAL_NAMES:
                    sig_vals = signals[signal_name]

                    # For scalar signals, invert those where lower = more confident
                    # sl_fused_uncertainty: lower u = more confident → invert
                    # sl_qa_fused_u: same
                    # sl_dual_fused_u: same
                    if signal_name in ("sl_fused_uncertainty", "sl_qa_fused_u", "sl_dual_fused_u"):
                        sig_vals = [1.0 - v for v in sig_vals]

                    auc, ci_lo, ci_mean, ci_hi = bootstrap_auc_ci(
                        sig_vals, correct, n_bootstrap, seed=GLOBAL_SEED,
                    )

                    # Also compute the full precision-coverage curve (for plotting)
                    curve = precision_coverage_curve(sig_vals, correct, DEFAULT_COVERAGE_LEVELS)

                    combo_results[signal_name] = {
                        "auc": round(auc, 6),
                        "ci_lower": round(ci_lo, 6),
                        "ci_upper": round(ci_hi, 6),
                        "curve": [(round(c, 4), round(p, 6)) for c, p in curve],
                    }

                    # Track best
                    if signal_name not in best_auc or auc > best_auc[signal_name]:
                        best_auc[signal_name] = auc
                        best_params[signal_name] = (ew, pw)

                    if verbose and signal_name in ("max_cosine", "sl_composite", "sl_dual_composite"):
                        print(f"    {signal_name}: AUC={auc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

                pr_results["sweep_results"][combo_key] = combo_results

                if verbose:
                    elapsed = time.time() - t1
                    print(f"    ({elapsed:.1f}s)")

        # --- Summarize best per signal ---
        for signal_name in SCALAR_SIGNAL_NAMES + SL_SIGNAL_NAMES:
            ew_best, pw_best = best_params.get(signal_name, (10, 2))
            combo_key = f"ew{ew_best}_pw{pw_best}"
            pr_results["best_per_signal"][signal_name] = {
                "best_auc": round(best_auc.get(signal_name, 0.0), 6),
                "best_evidence_weight": ew_best,
                "best_prior_weight": pw_best,
                "ci_lower": pr_results["sweep_results"][combo_key][signal_name]["ci_lower"],
                "ci_upper": pr_results["sweep_results"][combo_key][signal_name]["ci_upper"],
            }

        # --- Print summary table ---
        if verbose:
            print(f"\n  {'─'*55}")
            print(f"  SUMMARY — Poison rate {pr}")
            print(f"  {'─'*55}")
            print(f"  {'Signal':<25} {'AUC':>8} {'95% CI':>18} {'Best EW/PW':>12}")
            print(f"  {'─'*55}")

            # Oracle and random first
            print(f"  {'oracle':<25} {pr_results['oracle']['auc']:>8.4f} "
                  f"[{pr_results['oracle']['ci_lower']:.4f}, {pr_results['oracle']['ci_upper']:.4f}]")
            print(f"  {'random':<25} {pr_results['random']['auc']:>8.4f} "
                  f"[{pr_results['random']['ci_lower']:.4f}, {pr_results['random']['ci_upper']:.4f}]")
            print(f"  {'─'*55}")

            # Scalar
            for name in SCALAR_SIGNAL_NAMES:
                info = pr_results["best_per_signal"][name]
                print(f"  {name:<25} {info['best_auc']:>8.4f} "
                      f"[{info['ci_lower']:.4f}, {info['ci_upper']:.4f}] "
                      f"  ew={info['best_evidence_weight']}/pw={info['best_prior_weight']}")
            print(f"  {'─'*55}")

            # SL
            for name in SL_SIGNAL_NAMES:
                info = pr_results["best_per_signal"][name]
                print(f"  {name:<25} {info['best_auc']:>8.4f} "
                      f"[{info['ci_lower']:.4f}, {info['ci_upper']:.4f}] "
                      f"  ew={info['best_evidence_weight']}/pw={info['best_prior_weight']}")

        all_results[tag] = pr_results

    return all_results


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EN3.2-H1 Abstention Experiment")
    parser.add_argument("--dry-run", action="store_true",
                        help="10 questions, 1 poison rate, 1 param combo")
    parser.add_argument("--poison-rate", type=float, default=None,
                        help="Run single poison rate (e.g. 0.10)")
    parser.add_argument("--no-sweep", action="store_true",
                        help="Skip parameter sweep, use default ew=10/pw=2 only")
    args = parser.parse_args()

    set_global_seed(GLOBAL_SEED)

    # Configure run
    if args.dry_run:
        prs = [0.10]
        ews = [10]
        pws = [2]
        n_boot = 100
        max_q = 10
        print("=== DRY RUN: 10 questions, pr=0.10, ew=10/pw=2, 100 bootstrap ===\n")
    elif args.poison_rate is not None:
        prs = [args.poison_rate]
        ews = EVIDENCE_WEIGHTS if not args.no_sweep else [10]
        pws = PRIOR_WEIGHTS if not args.no_sweep else [2]
        n_boot = N_BOOTSTRAP
        max_q = None
    else:
        prs = POISON_RATES
        ews = EVIDENCE_WEIGHTS if not args.no_sweep else [10]
        pws = PRIOR_WEIGHTS if not args.no_sweep else [2]
        n_boot = N_BOOTSTRAP
        max_q = None

    print(f"Configuration:")
    print(f"  Poison rates: {prs}")
    print(f"  Evidence weights: {ews}")
    print(f"  Prior weights: {pws}")
    print(f"  Sweep combos: {len(ews) * len(pws)}")
    print(f"  Bootstrap: {n_boot}")
    print(f"  Max questions: {max_q or 'all'}")
    print(f"  Coverage levels: {len(DEFAULT_COVERAGE_LEVELS)} "
          f"({DEFAULT_COVERAGE_LEVELS[0]:.2f} to {DEFAULT_COVERAGE_LEVELS[-1]:.2f})")

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

    # --- Save results ---
    mode = "dry" if args.dry_run else "full"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_name = f"en3_2_h1_{mode}_results"

    experiment_result = ExperimentResult(
        experiment_id="EN3.2-H1",
        parameters={
            "poison_rates": prs,
            "evidence_weights": ews,
            "prior_weights": pws,
            "n_bootstrap": n_boot,
            "max_questions": max_q,
            "coverage_levels": DEFAULT_COVERAGE_LEVELS,
            "global_seed": GLOBAL_SEED,
            "mode": mode,
        },
        metrics=results,
        environment=env,
        notes=(
            f"Calibrated selective answering experiment. "
            f"Elapsed: {elapsed:.1f}s. "
            f"Sweep: {len(ews)}×{len(pws)}={len(ews)*len(pws)} combos. "
            f"Uses v1b retrieval checkpoints + H3 PLAIN correctness labels."
        ),
    )

    # Save primary + timestamped archive
    primary_path = RESULTS_DIR / f"{result_name}.json"
    archive_path = RESULTS_DIR / f"{result_name}_{timestamp}.json"

    experiment_result.save_json(str(primary_path))
    experiment_result.save_json(str(archive_path))

    print(f"\n{'='*60}")
    print(f"DONE — {elapsed:.1f}s total")
    print(f"Results saved to:")
    print(f"  {primary_path}")
    print(f"  {archive_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
