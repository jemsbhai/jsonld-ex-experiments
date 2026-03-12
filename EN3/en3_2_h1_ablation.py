"""EN3.2-H1 Ablation — Per-difficulty, signal combination, operating points, passage count.

All analyses use existing data from the full H1 run + v1b checkpoints.
No API calls. Pure computation.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN3/en3_2_h1_ablation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

RESULTS_DIR = SCRIPT_DIR / "results"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"

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

from en3_2_h1_experiment import (
    load_retrieval_data,
    load_h3_correctness,
    build_retrieval_index,
    build_extraction_index,
    compute_all_signals,
    bootstrap_auc_ci,
    _pr_tag,
)

GLOBAL_SEED = 42
N_BOOTSTRAP = 1000

# Best params from full run
BEST_EW = 5
BEST_PW = 1


# ═══════════════════════════════════════════════════════════════════
# Load difficulty labels from H3 checkpoints
# ═══════════════════════════════════════════════════════════════════

def load_h3_difficulty(pr_tag: str) -> Dict[str, str]:
    """Load difficulty labels from H3 PLAIN results."""
    h3_path = CHECKPOINT_DIR / f"en3_2_h3_full_{pr_tag}.json"
    with open(str(h3_path), "r") as f:
        data = json.load(f)
    return {
        entry["question_id"]: entry["difficulty"]
        for entry in data["per_question_answers"]["PLAIN"]
    }


# ═══════════════════════════════════════════════════════════════════
# Ablation 1: Per-difficulty AUC
# ═══════════════════════════════════════════════════════════════════

def ablation_per_difficulty(
    question_ids: List[str],
    signals: Dict[str, List[float]],
    correct: List[bool],
    difficulty: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Compute AUC broken down by difficulty class."""
    results = {}
    for diff_class in ["easy", "medium", "hard"]:
        mask = [difficulty.get(qid) == diff_class for qid in question_ids]
        n_in_class = sum(mask)
        if n_in_class < 10:
            results[diff_class] = {"n": n_in_class, "skip": True}
            continue

        idx = [i for i, m in enumerate(mask) if m]
        sub_correct = [correct[i] for i in idx]
        n_correct = sum(sub_correct)

        class_signals: Dict[str, Any] = {
            "n": n_in_class,
            "n_correct": n_correct,
            "accuracy": n_correct / n_in_class,
        }

        key_signals = ["max_qa_score", "max_cosine", "sl_composite",
                       "sl_dual_fused_belief", "sl_dual_composite", "sl_max_conflict"]

        for sig_name in key_signals:
            sig_vals_all = signals[sig_name]
            # Invert uncertainty signals
            if sig_name in ("sl_fused_uncertainty", "sl_qa_fused_u", "sl_dual_fused_u"):
                sig_vals_all = [1.0 - v for v in sig_vals_all]
            sub_sig = [sig_vals_all[i] for i in idx]
            auc, ci_lo, _, ci_hi = bootstrap_auc_ci(
                sub_sig, sub_correct, N_BOOTSTRAP, seed=GLOBAL_SEED
            )
            class_signals[sig_name] = {
                "auc": round(auc, 4),
                "ci": [round(ci_lo, 4), round(ci_hi, 4)],
            }

        results[diff_class] = class_signals

    return results


# ═══════════════════════════════════════════════════════════════════
# Ablation 2: Signal combination (linear)
# ═══════════════════════════════════════════════════════════════════

def ablation_signal_combination(
    signals: Dict[str, List[float]],
    correct: List[bool],
) -> Dict[str, Any]:
    """Test linear combinations of max_qa_score + SL signals.

    For each SL signal, sweep alpha in [0, 1]:
        combined = alpha * max_qa_score + (1 - alpha) * sl_signal
    Report best alpha and AUC for each SL signal.
    """
    results = {}
    max_qa = signals["max_qa_score"]

    sl_to_combine = [
        "sl_max_conflict", "sl_dual_fused_belief",
        "sl_dual_composite", "sl_fused_belief",
    ]

    alphas = [round(0.05 * i, 2) for i in range(21)]  # 0.00 to 1.00

    for sl_name in sl_to_combine:
        sl_vals = signals[sl_name]

        best_alpha = 1.0
        best_auc = 0.0
        alpha_curve = []

        for alpha in alphas:
            combined = [
                alpha * mq + (1.0 - alpha) * sv
                for mq, sv in zip(max_qa, sl_vals)
            ]
            curve = precision_coverage_curve(combined, correct, DEFAULT_COVERAGE_LEVELS)
            auc = auc_precision_coverage(curve)
            alpha_curve.append((alpha, round(auc, 6)))
            if auc > best_auc:
                best_auc = auc
                best_alpha = alpha

        # Bootstrap CI at best alpha
        combined_best = [
            best_alpha * mq + (1.0 - best_alpha) * sv
            for mq, sv in zip(max_qa, sl_vals)
        ]
        auc_obs, ci_lo, _, ci_hi = bootstrap_auc_ci(
            combined_best, correct, N_BOOTSTRAP, seed=GLOBAL_SEED
        )

        # Also get pure max_qa AUC for comparison
        qa_only_auc, qa_lo, _, qa_hi = bootstrap_auc_ci(
            max_qa, correct, N_BOOTSTRAP, seed=GLOBAL_SEED
        )

        improvement = auc_obs - qa_only_auc

        results[sl_name] = {
            "best_alpha": best_alpha,
            "best_combined_auc": round(best_auc, 6),
            "best_combined_ci": [round(ci_lo, 6), round(ci_hi, 6)],
            "qa_only_auc": round(qa_only_auc, 6),
            "qa_only_ci": [round(qa_lo, 6), round(qa_hi, 6)],
            "improvement": round(improvement, 6),
            "alpha_curve": alpha_curve,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# Ablation 3: Precision at specific operating points
# ═══════════════════════════════════════════════════════════════════

def ablation_operating_points(
    signals: Dict[str, List[float]],
    correct: List[bool],
) -> Dict[str, Any]:
    """Compare precision at key coverage levels: 70%, 80%, 90%, 95%."""
    key_coverages = [0.70, 0.80, 0.90, 0.95]
    key_signals = ["max_qa_score", "max_cosine",
                   "sl_dual_fused_belief", "sl_dual_composite", "sl_max_conflict"]

    results = {}
    for sig_name in key_signals:
        sig_vals = signals[sig_name]
        if sig_name in ("sl_fused_uncertainty", "sl_qa_fused_u", "sl_dual_fused_u"):
            sig_vals = [1.0 - v for v in sig_vals]
        curve = precision_coverage_curve(sig_vals, correct, key_coverages)
        results[sig_name] = {
            f"prec@{int(c*100)}": round(p, 4) for c, p in curve
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# Ablation 4: Passage count (top-k)
# ═══════════════════════════════════════════════════════════════════

def ablation_passage_count(
    question_ids: List[str],
    retrieval_index: Dict[str, List[Dict]],
    extraction_index: Dict[str, Dict],
    correct: List[bool],
) -> Dict[str, Any]:
    """Compute signals using top-k passages for k ∈ {3, 5, 7, 10}."""
    results = {}
    key_signals = ["max_qa_score", "sl_dual_fused_belief", "sl_dual_composite"]

    for k in [3, 5, 7, 10]:
        # Build truncated retrieval index
        trunc_retrieval = {}
        for qid in question_ids:
            trunc_retrieval[qid] = retrieval_index[qid][:k]

        sigs = compute_all_signals(
            question_ids, trunc_retrieval, extraction_index,
            evidence_weight=BEST_EW, prior_weight=BEST_PW,
        )

        k_results = {}
        for sig_name in key_signals:
            sig_vals = sigs[sig_name]
            auc, ci_lo, _, ci_hi = bootstrap_auc_ci(
                sig_vals, correct, N_BOOTSTRAP, seed=GLOBAL_SEED
            )
            k_results[sig_name] = {
                "auc": round(auc, 4),
                "ci": [round(ci_lo, 4), round(ci_hi, 4)],
            }

        results[f"top_{k}"] = k_results

    return results


# ═══════════════════════════════════════════════════════════════════
# Ablation 5: Correlation analysis
# ═══════════════════════════════════════════════════════════════════

def ablation_correlations(
    signals: Dict[str, List[float]],
    correct: List[bool],
) -> Dict[str, Any]:
    """Point-biserial correlation of each signal with correctness,
    plus inter-signal Pearson correlations."""
    from scipy.stats import pointbiserialr, pearsonr

    correct_int = [1 if c else 0 for c in correct]

    # Point-biserial with correctness
    pb_results = {}
    key_signals = ["max_cosine", "mean_cosine", "max_qa_score", "top1_qa_score",
                   "sl_fused_belief", "sl_max_conflict", "sl_composite",
                   "sl_dual_fused_belief", "sl_dual_composite"]

    for sig_name in key_signals:
        sig_vals = signals[sig_name]
        corr, pval = pointbiserialr(correct_int, sig_vals)
        pb_results[sig_name] = {
            "r_pb": round(corr, 4),
            "p_value": round(pval, 6),
        }

    # Inter-signal Pearson correlations (key pairs)
    inter_results = {}
    pairs = [
        ("max_qa_score", "sl_dual_fused_belief"),
        ("max_qa_score", "sl_max_conflict"),
        ("max_qa_score", "sl_dual_composite"),
        ("max_cosine", "sl_fused_belief"),
        ("sl_fused_belief", "sl_dual_fused_belief"),
    ]
    for a, b in pairs:
        r, p = pearsonr(signals[a], signals[b])
        inter_results[f"{a}_vs_{b}"] = {"pearson_r": round(r, 4), "p_value": round(p, 6)}

    return {"point_biserial": pb_results, "inter_signal": inter_results}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    set_global_seed(GLOBAL_SEED)
    t_start = time.time()
    env = log_environment()

    all_results: Dict[str, Any] = {}

    for pr in [0.05, 0.10, 0.20, 0.30]:
        tag = _pr_tag(pr)
        print(f"\n{'='*60}")
        print(f"Poison rate: {pr} ({tag})")
        print(f"{'='*60}")

        # Load data
        questions, pid_to_text, retrieval_results, extractions = load_retrieval_data(tag)
        correctness = load_h3_correctness(tag)
        difficulty = load_h3_difficulty(tag)

        retrieval_idx = build_retrieval_index(retrieval_results)
        extraction_idx = build_extraction_index(extractions)

        question_ids = [q["id"] for q in questions]
        question_ids = [qid for qid in question_ids
                        if qid in retrieval_idx and qid in correctness]
        correct = [correctness[qid] for qid in question_ids]

        # Compute signals at best params
        signals = compute_all_signals(
            question_ids, retrieval_idx, extraction_idx,
            evidence_weight=BEST_EW, prior_weight=BEST_PW,
        )

        pr_results: Dict[str, Any] = {}

        # Ablation 1: Per-difficulty
        print("  [1/5] Per-difficulty breakdown...")
        pr_results["per_difficulty"] = ablation_per_difficulty(
            question_ids, signals, correct, difficulty,
        )
        for dc in ["easy", "medium", "hard"]:
            info = pr_results["per_difficulty"][dc]
            if info.get("skip"):
                print(f"    {dc}: N={info['n']} (skipped)")
            else:
                print(f"    {dc}: N={info['n']}, acc={info['accuracy']:.3f}")
                qa = info.get("max_qa_score", {})
                sl = info.get("sl_dual_fused_belief", {})
                if qa and sl:
                    print(f"      max_qa_score AUC={qa['auc']:.4f}  sl_dual_fused_belief AUC={sl['auc']:.4f}")

        # Ablation 2: Signal combination
        print("  [2/5] Signal combination (max_qa + SL)...")
        pr_results["signal_combination"] = ablation_signal_combination(signals, correct)
        for sl_name, info in pr_results["signal_combination"].items():
            imp = info["improvement"]
            print(f"    max_qa + {sl_name}: best_alpha={info['best_alpha']:.2f} "
                  f"AUC={info['best_combined_auc']:.4f} (Δ={imp:+.4f})")

        # Ablation 3: Operating points
        print("  [3/5] Precision at operating points...")
        pr_results["operating_points"] = ablation_operating_points(signals, correct)
        for sig_name, precs in pr_results["operating_points"].items():
            print(f"    {sig_name:<25} " + "  ".join(f"{k}={v:.3f}" for k, v in precs.items()))

        # Ablation 4: Passage count
        print("  [4/5] Passage count ablation...")
        pr_results["passage_count"] = ablation_passage_count(
            question_ids, retrieval_idx, extraction_idx, correct,
        )
        for k_key, sigs in pr_results["passage_count"].items():
            qa = sigs["max_qa_score"]["auc"]
            sl = sigs["sl_dual_fused_belief"]["auc"]
            print(f"    {k_key}: max_qa={qa:.4f}  sl_dual_belief={sl:.4f}  Δ={sl-qa:+.4f}")

        # Ablation 5: Correlations
        print("  [5/5] Correlation analysis...")
        try:
            pr_results["correlations"] = ablation_correlations(signals, correct)
            pb = pr_results["correlations"]["point_biserial"]
            for sig_name, info in pb.items():
                print(f"    r_pb({sig_name}, correct) = {info['r_pb']:+.4f}  p={info['p_value']:.4e}")
        except ImportError:
            print("    scipy not available, skipping correlation analysis")
            pr_results["correlations"] = {"error": "scipy not installed"}

        all_results[tag] = pr_results

    elapsed = time.time() - t_start

    # Save
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = ExperimentResult(
        experiment_id="EN3.2-H1-ablation",
        parameters={
            "evidence_weight": BEST_EW,
            "prior_weight": BEST_PW,
            "n_bootstrap": N_BOOTSTRAP,
            "global_seed": GLOBAL_SEED,
        },
        metrics=all_results,
        environment=env,
        notes=f"H1 ablation: per-difficulty, signal combo, operating points, passage count, correlations. {elapsed:.1f}s.",
    )

    primary = RESULTS_DIR / "en3_2_h1_ablation_results.json"
    archive = RESULTS_DIR / f"en3_2_h1_ablation_results_{timestamp}.json"
    result.save_json(str(primary))
    result.save_json(str(archive))

    print(f"\n{'='*60}")
    print(f"DONE — {elapsed:.1f}s")
    print(f"Results saved to:")
    print(f"  {primary}")
    print(f"  {archive}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
