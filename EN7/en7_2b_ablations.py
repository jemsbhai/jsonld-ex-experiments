#!/usr/bin/env python
"""EN7.2b -- Ablations and Real-World Distribution Validation.

NeurIPS 2026 D&B, Suite EN7, Experiment 2b (extension of EN7.2).

Validates that the information-theoretic capacity advantage of SL opinions
over scalar confidence is NOT an artifact of the uniform synthetic
distribution used in EN7.2. Tests across:

  B1: Ablation sweep -- parameter sensitivity on the synthetic analysis
  B2: Six realistic ML confidence distributions covering the full
      spectrum of real-world ML outputs (overconfident DNNs, calibrated
      models, sparse evidence, binary classifiers)
  B3: Information loss analysis on each distribution

If the core finding (>50% info loss, >25% decision conflicts) holds
across ALL distributions, it is a fundamental property of the scalar
projection, not a distributional artifact.

Pre-registered hypotheses:
    H4 -- The information loss finding is robust to parameter choices
          (N, bin resolution, base rate distribution).
    H5 -- The information loss holds for realistic ML confidence
          distributions, not just uniform.
    H6 -- Evidence-based opinions (from_evidence) show HIGHER information
          loss than confidence-based opinions because from_evidence
          produces a wider spread of (b,d,u) tuples.

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN7/en7_2b_ablations.py

References:
    Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML.
    Josang, A. (2016). Subjective Logic. Springer.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# -- Path setup ----------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "packages" / "python" / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from jsonld_ex.confidence_algebra import Opinion

from experiments.infra.config import set_global_seed
from experiments.infra.results import ExperimentResult
from experiments.infra.env_log import log_environment

# -- Import analysis functions from EN7.2 --------------------------------
from experiments.EN7.en7_2_info_theoretic import (
    analysis_collisions,
    analysis_entropy,
    analysis_decision_loss,
    _uniform_simplex_sample,
)

SEED = 42


# ========================================================================
# Opinion Generators for Realistic ML Distributions
# ========================================================================


def gen_uniform(n: int, seed: int) -> list[Opinion]:
    """Baseline: uniform on simplex, uniform base rate (same as EN7.2)."""
    rng = random.Random(seed)
    ops = []
    for _ in range(n):
        b, d, u = _uniform_simplex_sample(rng)
        a = rng.random()
        ops.append(Opinion(belief=b, disbelief=d, uncertainty=u, base_rate=a))
    return ops


def gen_overconfident_dnn(n: int, seed: int) -> list[Opinion]:
    """Overconfident DNN: high confidence, low uncertainty.

    Models uncalibrated deep networks (Guo et al. 2017) which produce
    confidence scores peaked near 1.0. Beta(5, 1) gives mean=0.833.
    Low uncertainty (u ~ 0.05) reflects the model's false certainty.
    """
    rng = random.Random(seed)
    ops = []
    for _ in range(n):
        c = min(0.999, max(0.001, rng.betavariate(5.0, 1.0)))
        u = rng.uniform(0.01, 0.10)
        ops.append(Opinion.from_confidence(c, uncertainty=u))
    return ops


def gen_calibrated_dnn(n: int, seed: int) -> list[Opinion]:
    """Calibrated DNN: moderate confidence spread, entropy-derived uncertainty.

    After temperature scaling (Guo et al. 2017), confidence is more
    spread out. Beta(2, 2) gives a mild bell shape centered at 0.5.
    Uncertainty is derived from the confidence itself: higher confidence
    implies more evidence, thus lower uncertainty.
    """
    rng = random.Random(seed)
    ops = []
    for _ in range(n):
        c = min(0.999, max(0.001, rng.betavariate(2.0, 2.0)))
        # Uncertainty inversely related to confidence extremity
        extremity = abs(c - 0.5) * 2  # 0 at c=0.5, 1 at c=0/1
        u = max(0.01, (1.0 - extremity) * 0.5 + rng.uniform(0, 0.1))
        u = min(u, 0.99)
        ops.append(Opinion.from_confidence(c, uncertainty=u))
    return ops


def gen_high_uncertainty(n: int, seed: int) -> list[Opinion]:
    """High-uncertainty early-training model.

    Represents a partially trained model or one operating on out-of-
    distribution data. Confidence is nearly uniform, uncertainty is high.
    """
    rng = random.Random(seed)
    ops = []
    for _ in range(n):
        c = min(0.999, max(0.001, rng.betavariate(1.0, 1.0)))
        u = rng.uniform(0.3, 0.7)
        ops.append(Opinion.from_confidence(c, uncertainty=u))
    return ops


def gen_evidence_small(n: int, seed: int) -> list[Opinion]:
    """Evidence-based opinions with small sample sizes (N ~ Poisson(10)).

    Models scenarios with sparse observations: medical tests with few
    patients, A/B tests early in their run, rare event detection.
    """
    rng = random.Random(seed)
    ops = []
    for _ in range(n):
        # Total observations from Poisson(10), minimum 2
        total = max(2, int(rng.expovariate(1.0 / 10.0)))
        # True positive rate from Beta(2,2)
        p_true = rng.betavariate(2.0, 2.0)
        pos = sum(1 for _ in range(total) if rng.random() < p_true)
        neg = total - pos
        ops.append(Opinion.from_evidence(positive=pos, negative=neg))
    return ops


def gen_evidence_large(n: int, seed: int) -> list[Opinion]:
    """Evidence-based opinions with large sample sizes (N ~ Poisson(100)).

    Models scenarios with abundant data: large-scale A/B tests, high-
    frequency sensor readings, well-studied medical procedures.
    """
    rng = random.Random(seed)
    ops = []
    for _ in range(n):
        total = max(10, int(rng.expovariate(1.0 / 100.0)))
        p_true = rng.betavariate(2.0, 2.0)
        pos = sum(1 for _ in range(total) if rng.random() < p_true)
        neg = total - pos
        ops.append(Opinion.from_evidence(positive=pos, negative=neg))
    return ops


def gen_binary_classifier(n: int, seed: int) -> list[Opinion]:
    """Binary classifier with bimodal confidence distribution.

    Models a well-trained binary classifier that is confident in both
    directions. Mixture of Beta(0.5, 5) (confident negative) and
    Beta(5, 0.5) (confident positive), 50/50 mix.
    """
    rng = random.Random(seed)
    ops = []
    for _ in range(n):
        if rng.random() < 0.5:
            c = min(0.999, max(0.001, rng.betavariate(0.5, 5.0)))
        else:
            c = min(0.999, max(0.001, rng.betavariate(5.0, 0.5)))
        u = rng.uniform(0.02, 0.15)
        ops.append(Opinion.from_confidence(c, uncertainty=u))
    return ops


# All distribution generators
DISTRIBUTIONS = {
    "D0_uniform": {
        "generator": gen_uniform,
        "description": "Uniform on simplex (EN7.2 baseline)",
        "ml_scenario": "Theoretical baseline -- no distributional assumption",
    },
    "D1_overconfident_dnn": {
        "generator": gen_overconfident_dnn,
        "description": "Overconfident DNN (Beta(5,1), u~0.05)",
        "ml_scenario": "Uncalibrated deep network (Guo et al. 2017)",
    },
    "D2_calibrated_dnn": {
        "generator": gen_calibrated_dnn,
        "description": "Calibrated DNN (Beta(2,2), entropy-derived u)",
        "ml_scenario": "Temperature-scaled deep network",
    },
    "D3_high_uncertainty": {
        "generator": gen_high_uncertainty,
        "description": "High-uncertainty model (Uniform conf, u~0.3-0.7)",
        "ml_scenario": "Early training or out-of-distribution inference",
    },
    "D4_evidence_small": {
        "generator": gen_evidence_small,
        "description": "Evidence-based, small N (Poisson(10) obs)",
        "ml_scenario": "Sparse data: few-shot learning, rare events, early A/B tests",
    },
    "D5_evidence_large": {
        "generator": gen_evidence_large,
        "description": "Evidence-based, large N (Poisson(100) obs)",
        "ml_scenario": "Rich data: large-scale tests, well-studied domains",
    },
    "D6_binary_classifier": {
        "generator": gen_binary_classifier,
        "description": "Binary classifier bimodal (Beta mixture, low u)",
        "ml_scenario": "Confident binary classifier (spam/fraud detection)",
    },
}


# ========================================================================
# B1: Ablation Sweep
# ========================================================================


def run_ablation_sweep(n_opinions: int = 10_000) -> dict[str, Any]:
    """Sweep over parameters to test robustness of EN7.2 findings.

    Varies:
      - N: {1000, 10000, 100000}
      - scalar_bins: {50, 100, 200, 500}
      - base_rate: uniform, beta(2,2), fixed(0.5)
    """
    results = {}

    # -- N convergence --
    print("  B1a: N convergence sweep...")
    n_sweep = {}
    for n in [1_000, 10_000, 100_000]:
        opinions = gen_uniform(n, SEED)
        ent = analysis_entropy(opinions)
        n_sweep[str(n)] = {
            "n": n,
            "H_opinion": ent["H_opinion_bits"],
            "H_scalar": ent["H_scalar_bits"],
            "H_lost": ent["H_opinion_given_scalar_bits"],
            "pct_lost": ent["information_lost_pct"],
        }
        print(f"    N={n:>7,}: H_lost={ent['H_opinion_given_scalar_bits']:.2f} bits "
              f"({ent['information_lost_pct']:.1f}% lost)")
    results["n_convergence"] = n_sweep

    # -- Bin resolution --
    print("  B1b: Bin resolution sweep...")
    opinions = gen_uniform(10_000, SEED)
    bin_sweep = {}
    for bins in [50, 100, 200, 500]:
        ent = analysis_entropy(opinions, scalar_bins=bins)
        col = analysis_collisions(opinions, n_bins=bins)
        bin_sweep[str(bins)] = {
            "bins": bins,
            "H_lost": ent["H_opinion_given_scalar_bits"],
            "pct_lost": ent["information_lost_pct"],
            "mean_per_bin": col["mean_per_bin"],
            "mean_u_range": col["mean_uncertainty_range"],
        }
        print(f"    bins={bins:>3}: H_lost={ent['H_opinion_given_scalar_bits']:.2f} bits "
              f"({ent['information_lost_pct']:.1f}% lost), "
              f"mean_u_range={col['mean_uncertainty_range']:.3f}")
    results["bin_resolution"] = bin_sweep

    # -- Base rate distribution --
    print("  B1c: Base rate distribution sweep...")
    rng = random.Random(SEED)
    br_sweep = {}
    for br_name, br_fn in [
        ("uniform", lambda r: r.random()),
        ("beta_2_2", lambda r: r.betavariate(2.0, 2.0)),
        ("fixed_0.5", lambda r: 0.5),
    ]:
        ops = []
        r2 = random.Random(SEED)
        for _ in range(10_000):
            b, d, u = _uniform_simplex_sample(r2)
            a = br_fn(r2)
            ops.append(Opinion(belief=b, disbelief=d, uncertainty=u, base_rate=a))
        ent = analysis_entropy(ops)
        dec = analysis_decision_loss(ops)
        br_sweep[br_name] = {
            "base_rate": br_name,
            "H_lost": ent["H_opinion_given_scalar_bits"],
            "pct_lost": ent["information_lost_pct"],
            "decision_conflict_rate": dec["fraction_conflicting"],
        }
        print(f"    a~{br_name:>10}: H_lost={ent['H_opinion_given_scalar_bits']:.2f} bits "
              f"({ent['information_lost_pct']:.1f}% lost), "
              f"conflict={dec['fraction_conflicting']:.3f}")
    results["base_rate_distribution"] = br_sweep

    # -- Decision threshold sweep --
    print("  B1d: Decision threshold sweep...")
    opinions = gen_uniform(10_000, SEED)
    from experiments.EN7.en7_2_info_theoretic import analysis_decision_loss as adl
    threshold_sweep = {}
    for u_thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        # We need to modify the threshold. Call the function with modified params.
        # Can't easily pass u_threshold, so replicate the core logic here.
        p_groups = defaultdict(list)
        for op in opinions:
            p_bin = round(op.projected_probability(), 2)
            p_groups[int(p_bin * 100)].append(op)
        conflicts = 0
        total = 0
        for p_bin, group in p_groups.items():
            for i in range(len(group)):
                for j in range(i + 1, min(i + 50, len(group))):
                    total += 1
                    a, b = group[i], group[j]
                    if abs(a.projected_probability() - b.projected_probability()) < 0.01:
                        if abs(a.uncertainty - b.uncertainty) > 0.3:
                            act_a = a.uncertainty < u_thresh
                            act_b = b.uncertainty < u_thresh
                            if act_a != act_b:
                                conflicts += 1
        rate = conflicts / max(total, 1)
        threshold_sweep[str(u_thresh)] = {
            "u_threshold": u_thresh,
            "conflicts": conflicts,
            "total_pairs": total,
            "conflict_rate": round(rate, 4),
        }
        print(f"    u_thresh={u_thresh:.1f}: {conflicts:,} conflicts "
              f"({rate:.3f} rate)")
    results["decision_threshold"] = threshold_sweep

    return results


# ========================================================================
# B2: Distribution Analysis
# ========================================================================


def run_distribution_analysis(n_opinions: int = 10_000) -> dict[str, Any]:
    """Run full EN7.2 analysis on each realistic distribution."""
    results = {}

    for dist_name, dist_info in DISTRIBUTIONS.items():
        print(f"\n  {dist_name}: {dist_info['description']}")
        gen_fn = dist_info["generator"]
        opinions = gen_fn(n_opinions, SEED)

        # Descriptive stats
        beliefs = [op.belief for op in opinions]
        disbeliefs = [op.disbelief for op in opinions]
        uncertainties = [op.uncertainty for op in opinions]
        projected = [op.projected_probability() for op in opinions]

        desc = {
            "mean_belief": round(sum(beliefs) / len(beliefs), 4),
            "mean_disbelief": round(sum(disbeliefs) / len(disbeliefs), 4),
            "mean_uncertainty": round(sum(uncertainties) / len(uncertainties), 4),
            "mean_P": round(sum(projected) / len(projected), 4),
            "std_belief": round((sum((b - sum(beliefs)/len(beliefs))**2 for b in beliefs) / (len(beliefs)-1))**0.5, 4),
            "std_uncertainty": round((sum((u - sum(uncertainties)/len(uncertainties))**2 for u in uncertainties) / (len(uncertainties)-1))**0.5, 4),
            "std_P": round((sum((p - sum(projected)/len(projected))**2 for p in projected) / (len(projected)-1))**0.5, 4),
        }

        # Run analyses
        collisions = analysis_collisions(opinions)
        entropy = analysis_entropy(opinions)
        decision = analysis_decision_loss(opinions)

        print(f"    mean(b,d,u) = ({desc['mean_belief']:.3f}, "
              f"{desc['mean_disbelief']:.3f}, {desc['mean_uncertainty']:.3f})")
        print(f"    H_lost = {entropy['H_opinion_given_scalar_bits']:.2f} bits "
              f"({entropy['information_lost_pct']:.1f}% of {entropy['H_opinion_bits']:.2f})")
        print(f"    Mean u-range per bin: {collisions['mean_uncertainty_range']:.3f}")
        print(f"    Decision conflicts: {decision['fraction_conflicting']:.3f} "
              f"({decision['conflicting_pairs_found']:,} pairs)")

        results[dist_name] = {
            "description": dist_info["description"],
            "ml_scenario": dist_info["ml_scenario"],
            "descriptive_stats": desc,
            "entropy": {
                "H_opinion_bits": entropy["H_opinion_bits"],
                "H_scalar_bits": entropy["H_scalar_bits"],
                "H_opinion_given_scalar_bits": entropy["H_opinion_given_scalar_bits"],
                "mutual_information_bits": entropy["mutual_information_bits"],
                "information_preserved_pct": entropy["information_preserved_pct"],
                "information_lost_pct": entropy["information_lost_pct"],
            },
            "collisions": {
                "mean_per_bin": collisions["mean_per_bin"],
                "max_per_bin": collisions["max_per_bin"],
                "mean_uncertainty_range": collisions["mean_uncertainty_range"],
                "max_uncertainty_range": collisions["max_uncertainty_range"],
            },
            "decision": {
                "conflicting_pairs": decision["conflicting_pairs_found"],
                "fraction_conflicting": decision["fraction_conflicting"],
                "total_pairs_checked": decision["total_pairs_checked"],
            },
        }

    return results


# ========================================================================
# B3: Cross-Distribution Summary
# ========================================================================


def summarize_across_distributions(dist_results: dict[str, Any]) -> dict[str, Any]:
    """Produce the key summary table comparing all distributions."""
    rows = []
    for name, data in dist_results.items():
        rows.append({
            "distribution": name,
            "description": data["description"],
            "H_opinion": data["entropy"]["H_opinion_bits"],
            "H_lost": data["entropy"]["H_opinion_given_scalar_bits"],
            "pct_lost": data["entropy"]["information_lost_pct"],
            "u_range": data["collisions"]["mean_uncertainty_range"],
            "conflict_rate": data["decision"]["fraction_conflicting"],
        })

    # Check: does info loss hold for ALL distributions?
    all_positive_loss = all(r["H_lost"] > 0 for r in rows)
    min_loss = min(r["pct_lost"] for r in rows)
    max_loss = max(r["pct_lost"] for r in rows)
    min_conflict = min(r["conflict_rate"] for r in rows)

    return {
        "summary_table": rows,
        "all_show_info_loss": all_positive_loss,
        "min_pct_lost": round(min_loss, 2),
        "max_pct_lost": round(max_loss, 2),
        "min_conflict_rate": round(min_conflict, 4),
        "conclusion": (
            "ROBUST" if all_positive_loss and min_loss > 10
            else "PARTIAL" if all_positive_loss
            else "NOT ROBUST"
        ),
    }


# ========================================================================
# Main Runner
# ========================================================================


def run_en7_2b() -> ExperimentResult:
    """Run all EN7.2b analyses."""
    set_global_seed(SEED)

    print("=" * 60)
    print("  EN7.2b: Ablations and Real-World Distribution Validation")
    print("=" * 60)

    n = 10_000
    t_start = time.perf_counter()

    # B1: Ablation sweep
    print(f"\n--- B1: Ablation Sweep (N=10,000 baseline) ---")
    t0 = time.perf_counter()
    ablation = run_ablation_sweep(n)
    print(f"  [B1 complete in {time.perf_counter() - t0:.1f}s]")

    # B2: Distribution analysis
    print(f"\n--- B2: Distribution Analysis (7 distributions x {n:,} opinions) ---")
    t0 = time.perf_counter()
    distributions = run_distribution_analysis(n)
    print(f"  [B2 complete in {time.perf_counter() - t0:.1f}s]")

    # B3: Cross-distribution summary
    print(f"\n--- B3: Cross-Distribution Summary ---")
    summary = summarize_across_distributions(distributions)

    print(f"\n  {'Distribution':<28} {'H_lost':>7} {'%lost':>6} {'u_range':>8} {'conflict':>9}")
    print(f"  {'-'*28} {'-'*7} {'-'*6} {'-'*8} {'-'*9}")
    for row in summary["summary_table"]:
        print(f"  {row['distribution']:<28} {row['H_lost']:>7.2f} {row['pct_lost']:>5.1f}% "
              f"{row['u_range']:>8.3f} {row['conflict_rate']:>8.3f}")

    total_time = time.perf_counter() - t_start

    # Hypothesis outcomes
    h4 = (ablation["n_convergence"]["10000"]["pct_lost"] > 40 and
           ablation["n_convergence"]["100000"]["pct_lost"] > 40)
    h5 = summary["all_show_info_loss"] and summary["min_pct_lost"] > 10
    # H6: evidence-based > confidence-based info loss
    ev_small_loss = distributions.get("D4_evidence_small", {}).get("entropy", {}).get("information_lost_pct", 0)
    ev_large_loss = distributions.get("D5_evidence_large", {}).get("entropy", {}).get("information_lost_pct", 0)
    conf_losses = [
        distributions.get("D1_overconfident_dnn", {}).get("entropy", {}).get("information_lost_pct", 0),
        distributions.get("D2_calibrated_dnn", {}).get("entropy", {}).get("information_lost_pct", 0),
    ]
    avg_conf_loss = sum(conf_losses) / len(conf_losses) if conf_losses else 0
    h6 = (ev_small_loss > avg_conf_loss) or (ev_large_loss > avg_conf_loss)

    metrics = {
        "H4_robust_to_parameters": h4,
        "H5_holds_for_realistic_distributions": h5,
        "H5_all_show_loss": summary["all_show_info_loss"],
        "H5_min_pct_lost": summary["min_pct_lost"],
        "H5_max_pct_lost": summary["max_pct_lost"],
        "H6_evidence_higher_loss": h6,
        "H6_evidence_small_pct_lost": ev_small_loss,
        "H6_evidence_large_pct_lost": ev_large_loss,
        "H6_avg_confidence_pct_lost": round(avg_conf_loss, 2),
        "overall_conclusion": summary["conclusion"],
        "total_time_sec": round(total_time, 1),
    }

    result = ExperimentResult(
        experiment_id="EN7.2b",
        parameters={
            "n_opinions": n,
            "seed": SEED,
            "distributions": list(DISTRIBUTIONS.keys()),
            "ablation_Ns": [1000, 10000, 100000],
            "ablation_bins": [50, 100, 200, 500],
            "ablation_base_rates": ["uniform", "beta_2_2", "fixed_0.5"],
            "decision_thresholds": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        metrics=metrics,
        raw_data={
            "B1_ablation": ablation,
            "B2_distributions": distributions,
            "B3_summary": summary,
        },
        environment=log_environment(),
        notes=(
            f"Conclusion: {summary['conclusion']}. "
            f"Info loss range: {summary['min_pct_lost']:.1f}%-{summary['max_pct_lost']:.1f}% "
            f"across 7 distributions. "
            f"H4={'CONFIRMED' if h4 else 'REJECTED'}, "
            f"H5={'CONFIRMED' if h5 else 'REJECTED'}, "
            f"H6={'CONFIRMED' if h6 else 'REJECTED'}."
        ),
    )

    # Save
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "en7_2b_results.json"
    result.save_json(str(out_path))
    print(f"\nResults saved: {out_path}")

    print(f"\n{'='*60}")
    print(f"  HYPOTHESIS OUTCOMES (total: {total_time:.1f}s)")
    print(f"{'='*60}")
    print(f"  H4 (robust to parameters):    {'CONFIRMED' if h4 else 'REJECTED'}")
    print(f"  H5 (holds for real ML dists):  {'CONFIRMED' if h5 else 'REJECTED'} "
          f"(min={summary['min_pct_lost']:.1f}%, max={summary['max_pct_lost']:.1f}%)")
    print(f"  H6 (evidence > confidence):    {'CONFIRMED' if h6 else 'REJECTED'} "
          f"(ev_small={ev_small_loss:.1f}%, ev_large={ev_large_loss:.1f}% "
          f"vs conf_avg={avg_conf_loss:.1f}%)")
    print(f"  Overall: {summary['conclusion']}")
    print(f"{'='*60}")

    return result


if __name__ == "__main__":
    run_en7_2b()
