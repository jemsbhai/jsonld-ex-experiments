#!/usr/bin/env python
"""EA1.1 — Scalar Collapse Demonstration.

AAAI 2027, Suite EA1 (Algebra Superiority), Experiment 1.

Demonstrates that scalar confidence conflates distinct epistemic states
that Subjective Logic opinions preserve, leading to information loss in
downstream decision-making.

Protocol (from experiment roadmap):
    1. Construct 4 canonical opinion states, all with P(omega) = 0.5:
       A — Strong balanced evidence   (b=0.45, d=0.45, u=0.10, a=0.50)
       B — Total ignorance            (b=0.00, d=0.00, u=1.00, a=0.50)
       C — Moderate belief + uncertainty (b=0.30, d=0.10, u=0.60, a=1/3)
       D — Dogmatic coin flip          (b=0.50, d=0.50, u=0.00, a=0.50)
    2. Feed into 3 decision tasks:
       T1 — Accept/reject at threshold 0.5
       T2 — Request more data if uncertainty > 0.3
       T3 — Flag for human review if conflict > 0.2
    3. Quantify information loss via Shannon entropy of the (b, d, u) triple
       vs the single scalar.
    4. Extend to N_RANDOM random opinions — count how many distinct epistemic
       states collapse to the same scalar (within tolerance).

Note on canonical state C:
    To achieve P(omega) = 0.5 with b=0.3, d=0.1, u=0.6, we need
    P = b + a*u = 0.3 + a*0.6 = 0.5  =>  a = 1/3.
    This is a legitimate opinion with a non-default base rate,
    representing moderate belief with substantial uncertainty.

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EA1/ea1_1_scalar_collapse.py

Output:
    experiments/EA1/results/ea1_1_results.json                (latest)
    experiments/EA1/results/ea1_1_results_YYYYMMDD_HHMMSS.json (archived)

References:
    Josang, A. (2016). Subjective Logic: A Formalism for Reasoning Under
    Uncertainty. Springer.  ISBN 978-3-319-42337-1.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Path setup ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "packages" / "python" / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

# ── Imports: jsonld-ex ─────────────────────────────────────────────
from jsonld_ex.confidence_algebra import (
    Opinion,
    conflict_metric,
)

# ── Imports: experiment infrastructure ─────────────────────────────
from experiments.infra.config import set_global_seed, get_global_seed
from experiments.infra.results import ExperimentResult
from experiments.infra.env_log import log_environment

# ── Imports: third-party ───────────────────────────────────────────
import numpy as np

# ── Constants ──────────────────────────────────────────────────────
GLOBAL_SEED = 42
N_RANDOM = 1000
SCALAR_BIN_TOLERANCE = 1e-6  # Two scalars within this tolerance are "the same"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# =====================================================================
# Part 1: Canonical States and Decision Tasks
# =====================================================================

def build_canonical_states() -> Dict[str, Opinion]:
    """Construct the 4 canonical opinion states, all with P(omega) = 0.5.

    Returns:
        Dict mapping state label to Opinion.
    """
    states = {
        "A_strong_balanced": Opinion(
            belief=0.45, disbelief=0.45, uncertainty=0.10, base_rate=0.50
        ),
        "B_total_ignorance": Opinion(
            belief=0.00, disbelief=0.00, uncertainty=1.00, base_rate=0.50
        ),
        "C_moderate_belief": Opinion(
            belief=0.30, disbelief=0.10, uncertainty=0.60, base_rate=1.0 / 3.0
        ),
        "D_dogmatic_coinflip": Opinion(
            belief=0.50, disbelief=0.50, uncertainty=0.00, base_rate=0.50
        ),
    }

    # Verify all have P(omega) = 0.5
    for label, op in states.items():
        pp = op.projected_probability()
        assert abs(pp - 0.5) < 1e-12, (
            f"State {label} has P(omega) = {pp}, expected 0.5"
        )

    return states


def decision_accept_reject(opinion: Opinion, threshold: float = 0.5) -> str:
    """T1: Accept/reject based on scalar projected probability.

    A scalar system sees only P(omega) and must decide.
    """
    pp = opinion.projected_probability()
    if pp >= threshold:
        return "accept"
    else:
        return "reject"


def decision_request_data(opinion: Opinion, u_threshold: float = 0.3) -> str:
    """T2: Request more data if uncertainty exceeds threshold.

    Only SL can make this decision — scalar confidence has no uncertainty
    component to inspect.
    """
    if opinion.uncertainty > u_threshold:
        return "request_more_data"
    else:
        return "sufficient_data"


def decision_flag_review(opinion: Opinion, c_threshold: float = 0.2) -> str:
    """T3: Flag for human review if conflict exceeds threshold.

    Conflict measures the degree to which an opinion is internally
    contentious (high belief AND high disbelief).  Only SL provides this.
    """
    c = conflict_metric(opinion)
    if c > c_threshold:
        return "flag_for_review"
    else:
        return "auto_process"


def run_decision_tasks(
    states: Dict[str, Opinion],
) -> Dict[str, Dict[str, Any]]:
    """Run all 3 decision tasks on all canonical states.

    Returns:
        Dict[state_label, {task_name: {scalar_decision, sl_decision, ...}}]
    """
    results = {}
    for label, op in states.items():
        pp = op.projected_probability()
        conflict = conflict_metric(op)

        # T1: Scalar sees P=0.5 for ALL states => identical decision
        t1_scalar = "accept" if pp >= 0.5 else "reject"
        # SL could use uncertainty-aware threshold, but for fairness,
        # we show scalar makes the same decision for all 4 states.
        t1_sl = decision_accept_reject(op)

        # T2: Only SL can request more data
        t2_scalar = "no_uncertainty_signal"  # scalar has NO uncertainty info
        t2_sl = decision_request_data(op)

        # T3: Only SL can detect conflict
        t3_scalar = "no_conflict_signal"  # scalar has NO conflict info
        t3_sl = decision_flag_review(op)

        results[label] = {
            "projected_probability": pp,
            "uncertainty": op.uncertainty,
            "conflict": conflict,
            "opinion_bdu": [op.belief, op.disbelief, op.uncertainty],
            "base_rate": op.base_rate,
            "T1_scalar": t1_scalar,
            "T1_sl": t1_sl,
            "T2_scalar": t2_scalar,
            "T2_sl": t2_sl,
            "T3_scalar": t3_scalar,
            "T3_sl": t3_sl,
        }
    return results


# =====================================================================
# Part 2: Information-Theoretic Analysis
# =====================================================================

def shannon_entropy_bdu(opinion: Opinion) -> float:
    """Compute Shannon entropy of the (b, d, u) distribution.

    H(b, d, u) = -sum(p_i * log2(p_i)) for p_i in {b, d, u}

    This measures the "spread" of belief mass across the three components.
    A scalar collapses this to a single number and loses this information.
    """
    components = [opinion.belief, opinion.disbelief, opinion.uncertainty]
    h = 0.0
    for p in components:
        if p > 0:
            h -= p * math.log2(p)
    return h


def shannon_entropy_scalar(prob: float) -> float:
    """Compute Shannon entropy of a Bernoulli distribution with parameter prob.

    H(p) = -p*log2(p) - (1-p)*log2(1-p)

    This is the information content of the scalar projected probability.
    """
    if prob <= 0 or prob >= 1:
        return 0.0
    return -prob * math.log2(prob) - (1.0 - prob) * math.log2(1.0 - prob)


def information_loss(opinion: Opinion) -> float:
    """Quantify bits lost by collapsing opinion to scalar.

    The (b, d, u) triple lives in a 2-simplex (2 degrees of freedom).
    The scalar P(omega) is a 1D projection. Information loss is the
    difference in Shannon entropy of the BDU distribution vs the
    scalar Bernoulli distribution.

    Note: This is a descriptive measure of the *structural* information
    in the opinion, not a formal information-theoretic channel capacity.
    """
    h_bdu = shannon_entropy_bdu(opinion)
    h_scalar = shannon_entropy_scalar(opinion.projected_probability())
    return h_bdu - h_scalar


def run_information_analysis(
    states: Dict[str, Opinion],
) -> Dict[str, Dict[str, float]]:
    """Compute information-theoretic measures for all canonical states."""
    results = {}
    for label, op in states.items():
        results[label] = {
            "H_bdu": shannon_entropy_bdu(op),
            "H_scalar": shannon_entropy_scalar(op.projected_probability()),
            "information_loss_bits": information_loss(op),
            "projected_probability": op.projected_probability(),
        }
    return results


# =====================================================================
# Part 3: Random Opinion Collision Analysis
# =====================================================================

def generate_random_opinions(
    n: int,
    rng: np.random.RandomState,
) -> List[Opinion]:
    """Generate n random valid opinions with random base rates.

    Samples (b, d, u) uniformly from the 2-simplex using the
    Dirichlet(1, 1, 1) distribution, and base_rate uniformly from [0, 1].
    """
    opinions = []
    for _ in range(n):
        # Uniform on 2-simplex via Dirichlet(1,1,1)
        bdu = rng.dirichlet([1.0, 1.0, 1.0])
        a = rng.uniform(0.0, 1.0)
        opinions.append(
            Opinion(belief=bdu[0], disbelief=bdu[1], uncertainty=bdu[2], base_rate=a)
        )
    return opinions


def count_scalar_collisions(
    opinions: List[Opinion],
    tolerance: float = SCALAR_BIN_TOLERANCE,
) -> Dict[str, Any]:
    """Count how many distinct epistemic states collapse to the same scalar.

    Groups opinions by their projected probability (within tolerance).
    For each group, counts distinct (b, d, u, a) tuples.

    Returns:
        Dict with collision statistics.
    """
    # Bin by projected probability
    bins: Dict[int, List[Opinion]] = defaultdict(list)
    for op in opinions:
        pp = op.projected_probability()
        bin_key = round(pp / tolerance)
        bins[bin_key].append(op)

    # Analyze collisions
    n_bins = len(bins)
    n_opinions = len(opinions)
    max_collision_size = max(len(ops) for ops in bins.values())
    collision_sizes = [len(ops) for ops in bins.values() if len(ops) > 1]
    n_colliding_bins = len(collision_sizes)
    n_colliding_opinions = sum(collision_sizes)

    # For the largest collision bin, measure the diversity of opinions
    largest_bin_key = max(bins, key=lambda k: len(bins[k]))
    largest_bin = bins[largest_bin_key]
    largest_bin_pp = largest_bin[0].projected_probability()

    # Compute pairwise L2 distances in (b, d, u) space within largest bin
    if len(largest_bin) > 1:
        bdu_array = np.array([
            [op.belief, op.disbelief, op.uncertainty]
            for op in largest_bin
        ])
        # Pairwise L2 distances
        n_lb = len(largest_bin)
        dists = []
        for i in range(n_lb):
            for j in range(i + 1, n_lb):
                dists.append(float(np.linalg.norm(bdu_array[i] - bdu_array[j])))
        mean_l2 = float(np.mean(dists))
        max_l2 = float(np.max(dists))
        min_l2 = float(np.min(dists))
    else:
        mean_l2 = max_l2 = min_l2 = 0.0

    # Compute diversity: range of uncertainty values within largest bin
    u_values = [op.uncertainty for op in largest_bin]
    u_range = max(u_values) - min(u_values)

    # Entropy spread within collision bins
    entropy_spreads = []
    for ops in bins.values():
        if len(ops) > 1:
            entropies = [shannon_entropy_bdu(op) for op in ops]
            entropy_spreads.append(max(entropies) - min(entropies))

    mean_entropy_spread = float(np.mean(entropy_spreads)) if entropy_spreads else 0.0
    max_entropy_spread = float(np.max(entropy_spreads)) if entropy_spreads else 0.0

    # Decision divergence: how many opinions in collision bins would
    # get DIFFERENT SL decisions (T2: request_more_data)
    decision_divergence_count = 0
    decision_divergence_total = 0
    for ops in bins.values():
        if len(ops) > 1:
            decisions = set(decision_request_data(op) for op in ops)
            if len(decisions) > 1:
                decision_divergence_count += 1
            decision_divergence_total += 1

    # T3 divergence
    t3_divergence_count = 0
    t3_divergence_total = 0
    for ops in bins.values():
        if len(ops) > 1:
            decisions = set(decision_flag_review(op) for op in ops)
            if len(decisions) > 1:
                t3_divergence_count += 1
            t3_divergence_total += 1

    return {
        "n_opinions": n_opinions,
        "n_scalar_bins": n_bins,
        "n_colliding_bins": n_colliding_bins,
        "n_colliding_opinions": n_colliding_opinions,
        "max_collision_size": max_collision_size,
        "collision_rate": n_colliding_opinions / n_opinions if n_opinions > 0 else 0.0,
        "largest_bin": {
            "projected_probability": largest_bin_pp,
            "n_opinions": len(largest_bin),
            "uncertainty_range": u_range,
            "mean_l2_distance_bdu": mean_l2,
            "max_l2_distance_bdu": max_l2,
            "min_l2_distance_bdu": min_l2,
        },
        "entropy_spread_in_collision_bins": {
            "mean": mean_entropy_spread,
            "max": max_entropy_spread,
        },
        "T2_decision_divergence": {
            "bins_with_different_decisions": decision_divergence_count,
            "total_collision_bins": decision_divergence_total,
            "divergence_rate": (
                decision_divergence_count / decision_divergence_total
                if decision_divergence_total > 0 else 0.0
            ),
        },
        "T3_decision_divergence": {
            "bins_with_different_decisions": t3_divergence_count,
            "total_collision_bins": t3_divergence_total,
            "divergence_rate": (
                t3_divergence_count / t3_divergence_total
                if t3_divergence_total > 0 else 0.0
            ),
        },
    }


# =====================================================================
# Part 4: Systematic Scalar Value Sweep
# =====================================================================

def sweep_scalar_values(
    n_per_target: int,
    targets: List[float],
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    """For specific scalar targets, find diverse opinions that map to each.

    For each target P, generate random opinions and keep those within
    tolerance of P. Measures how many distinct states are possible.
    """
    results = {}
    for target in targets:
        collected = []
        attempts = 0
        max_attempts = n_per_target * 200  # generous budget

        while len(collected) < n_per_target and attempts < max_attempts:
            bdu = rng.dirichlet([1.0, 1.0, 1.0])
            a = rng.uniform(0.0, 1.0)
            op = Opinion(belief=bdu[0], disbelief=bdu[1], uncertainty=bdu[2], base_rate=a)
            if abs(op.projected_probability() - target) < 0.005:
                collected.append(op)
            attempts += 1

        if len(collected) < 2:
            results[f"P={target:.2f}"] = {
                "n_found": len(collected),
                "note": "insufficient samples found"
            }
            continue

        # Measure diversity
        u_values = [op.uncertainty for op in collected]
        conflict_values = [conflict_metric(op) for op in collected]
        entropy_values = [shannon_entropy_bdu(op) for op in collected]

        # Decision diversity
        t2_decisions = [decision_request_data(op) for op in collected]
        t3_decisions = [decision_flag_review(op) for op in collected]

        n_request_data = sum(1 for d in t2_decisions if d == "request_more_data")
        n_flag_review = sum(1 for d in t3_decisions if d == "flag_for_review")

        results[f"P={target:.2f}"] = {
            "n_found": len(collected),
            "attempts": attempts,
            "uncertainty": {
                "min": float(np.min(u_values)),
                "max": float(np.max(u_values)),
                "mean": float(np.mean(u_values)),
                "std": float(np.std(u_values)),
            },
            "conflict": {
                "min": float(np.min(conflict_values)),
                "max": float(np.max(conflict_values)),
                "mean": float(np.mean(conflict_values)),
                "std": float(np.std(conflict_values)),
            },
            "H_bdu": {
                "min": float(np.min(entropy_values)),
                "max": float(np.max(entropy_values)),
                "mean": float(np.mean(entropy_values)),
                "std": float(np.std(entropy_values)),
            },
            "T2_request_more_data_fraction": n_request_data / len(collected),
            "T3_flag_for_review_fraction": n_flag_review / len(collected),
        }

    return results


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    """Run EA1.1 — Scalar Collapse Demonstration."""

    print("=" * 70)
    print("EA1.1 — Scalar Collapse Demonstration")
    print("AAAI 2027, Suite EA1: Algebra Superiority")
    print("=" * 70)

    t_start = time.time()

    # ── Environment ──
    env = log_environment()
    set_global_seed(GLOBAL_SEED)
    rng = np.random.RandomState(GLOBAL_SEED)

    print(f"\n  Global seed: {GLOBAL_SEED}")
    print(f"  N random opinions: {N_RANDOM}")
    print(f"  Scalar bin tolerance: {SCALAR_BIN_TOLERANCE}")

    # ── Part 1: Canonical states ──
    print("\n--- Part 1: Canonical States & Decision Tasks ---")
    states = build_canonical_states()
    decision_results = run_decision_tasks(states)

    for label, res in decision_results.items():
        print(f"\n  {label}:")
        print(f"    P(omega) = {res['projected_probability']:.6f}")
        print(f"    (b, d, u) = ({res['opinion_bdu'][0]:.4f}, "
              f"{res['opinion_bdu'][1]:.4f}, {res['opinion_bdu'][2]:.4f})")
        print(f"    uncertainty = {res['uncertainty']:.4f}, "
              f"conflict = {res['conflict']:.4f}")
        print(f"    T1 scalar: {res['T1_scalar']:12s}  |  T1 SL: {res['T1_sl']}")
        print(f"    T2 scalar: {res['T2_scalar']:12s}  |  T2 SL: {res['T2_sl']}")
        print(f"    T3 scalar: {res['T3_scalar']:12s}  |  T3 SL: {res['T3_sl']}")

    # Count unique decisions
    scalar_t1 = set(r["T1_scalar"] for r in decision_results.values())
    sl_t2 = set(r["T2_sl"] for r in decision_results.values())
    sl_t3 = set(r["T3_sl"] for r in decision_results.values())

    print(f"\n  Scalar T1 decisions: {len(scalar_t1)} unique "
          f"(all identical: {len(scalar_t1) == 1})")
    print(f"  SL T2 decisions: {len(sl_t2)} unique "
          f"(distinguishes states: {len(sl_t2) > 1})")
    print(f"  SL T3 decisions: {len(sl_t3)} unique "
          f"(distinguishes states: {len(sl_t3) > 1})")

    # ── Part 2: Information-theoretic analysis ──
    print("\n--- Part 2: Information-Theoretic Analysis ---")
    info_results = run_information_analysis(states)

    for label, res in info_results.items():
        print(f"\n  {label}:")
        print(f"    H(b,d,u) = {res['H_bdu']:.4f} bits")
        print(f"    H(scalar) = {res['H_scalar']:.4f} bits")
        print(f"    Info loss = {res['information_loss_bits']:.4f} bits")

    # ── Part 3: Random collision analysis ──
    print(f"\n--- Part 3: Random Opinion Collision Analysis (N={N_RANDOM}) ---")
    random_opinions = generate_random_opinions(N_RANDOM, rng)
    collision_results = count_scalar_collisions(random_opinions)

    print(f"\n  Total opinions: {collision_results['n_opinions']}")
    print(f"  Unique scalar bins: {collision_results['n_scalar_bins']}")
    print(f"  Bins with collisions: {collision_results['n_colliding_bins']}")
    print(f"  Opinions in collision bins: {collision_results['n_colliding_opinions']}")
    print(f"  Max collision size: {collision_results['max_collision_size']}")
    print(f"  Collision rate: {collision_results['collision_rate']:.4f}")

    lb = collision_results["largest_bin"]
    print(f"\n  Largest collision bin:")
    print(f"    P(omega) ~ {lb['projected_probability']:.6f}")
    print(f"    N opinions: {lb['n_opinions']}")
    print(f"    Uncertainty range: {lb['uncertainty_range']:.4f}")
    print(f"    Mean L2 in (b,d,u): {lb['mean_l2_distance_bdu']:.4f}")
    print(f"    Max L2 in (b,d,u): {lb['max_l2_distance_bdu']:.4f}")

    t2d = collision_results["T2_decision_divergence"]
    t3d = collision_results["T3_decision_divergence"]
    print(f"\n  T2 decision divergence: {t2d['bins_with_different_decisions']}"
          f"/{t2d['total_collision_bins']} collision bins "
          f"({t2d['divergence_rate']:.2%})")
    print(f"  T3 decision divergence: {t3d['bins_with_different_decisions']}"
          f"/{t3d['total_collision_bins']} collision bins "
          f"({t3d['divergence_rate']:.2%})")

    # ── Part 4: Systematic scalar sweep ──
    print("\n--- Part 4: Scalar Value Sweep ---")
    sweep_targets = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_per_target = 100
    sweep_results = sweep_scalar_values(n_per_target, sweep_targets, rng)

    for target_label, res in sweep_results.items():
        if "note" in res:
            print(f"\n  {target_label}: {res['note']}")
            continue
        print(f"\n  {target_label} (n={res['n_found']}):")
        print(f"    Uncertainty range: [{res['uncertainty']['min']:.3f}, "
              f"{res['uncertainty']['max']:.3f}]")
        print(f"    Conflict range: [{res['conflict']['min']:.3f}, "
              f"{res['conflict']['max']:.3f}]")
        print(f"    H(b,d,u) range: [{res['H_bdu']['min']:.3f}, "
              f"{res['H_bdu']['max']:.3f}]")
        print(f"    T2 request-more-data: {res['T2_request_more_data_fraction']:.1%}")
        print(f"    T3 flag-for-review: {res['T3_flag_for_review_fraction']:.1%}")

    # ── Timing ──
    total_time = time.time() - t_start
    print(f"\n  Total wall time: {total_time:.1f}s")

    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "ea1_1_results.json"

    experiment_result = ExperimentResult(
        experiment_id="EA1.1",
        parameters={
            "global_seed": GLOBAL_SEED,
            "n_random_opinions": N_RANDOM,
            "scalar_bin_tolerance": SCALAR_BIN_TOLERANCE,
            "sweep_targets": sweep_targets,
            "n_per_sweep_target": n_per_target,
            "decision_thresholds": {
                "T1_accept_reject": 0.5,
                "T2_uncertainty": 0.3,
                "T3_conflict": 0.2,
            },
        },
        metrics={
            "canonical_states": {
                "n_states": 4,
                "all_same_projected_probability": True,
                "scalar_unique_T1_decisions": len(scalar_t1),
                "sl_unique_T2_decisions": len(sl_t2),
                "sl_unique_T3_decisions": len(sl_t3),
                "scalar_cannot_distinguish": len(scalar_t1) == 1,
                "sl_distinguishes_via_T2": len(sl_t2) > 1,
                "sl_distinguishes_via_T3": len(sl_t3) > 1,
            },
            "information_loss": {
                label: {
                    "H_bdu_bits": r["H_bdu"],
                    "H_scalar_bits": r["H_scalar"],
                    "loss_bits": r["information_loss_bits"],
                }
                for label, r in info_results.items()
            },
            "collision_analysis": {
                "n_opinions": collision_results["n_opinions"],
                "n_scalar_bins": collision_results["n_scalar_bins"],
                "collision_rate": collision_results["collision_rate"],
                "max_collision_size": collision_results["max_collision_size"],
                "T2_divergence_rate": t2d["divergence_rate"],
                "T3_divergence_rate": t3d["divergence_rate"],
            },
            "total_wall_time_seconds": round(total_time, 4),
        },
        raw_data={
            "canonical_decisions": decision_results,
            "information_analysis": info_results,
            "collision_analysis": collision_results,
            "sweep_analysis": sweep_results,
        },
        environment=env,
        notes=(
            "EA1.1: Scalar Collapse Demonstration. Shows that 4 epistemically "
            "distinct opinion states, all with P(omega)=0.5, produce identical "
            "scalar decisions but divergent SL decisions. Quantifies information "
            f"loss in bits and collision rate across {N_RANDOM} random opinions."
        ),
    )
    experiment_result.save_json(str(output_path))
    print(f"\n  Results saved to: {output_path}")

    # ── Timestamped archive ──
    ts = experiment_result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    archive_path = RESULTS_DIR / f"ea1_1_results_{ts}.json"
    experiment_result.save_json(str(archive_path))
    print(f"  Archived to:      {archive_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
