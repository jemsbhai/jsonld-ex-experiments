"""EN1.4 — Trust Discount Chain Experiment.

Comprehensive analysis of trust propagation through provenance chains.
All computation is pure mathematics — no GPU, no external data.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN1/en1_4_trust_chains.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from infra.config import set_global_seed
from infra.env_log import log_environment
from infra.results import ExperimentResult

from en1_4_core import (
    compute_sl_chain,
    compute_scalar_chain,
    compute_bayesian_chain,
    sl_chain_closed_form,
    sl_uncertainty_closed_form,
    compute_heterogeneous_sl_chain,
    compute_branching_provenance,
    compute_decision_divergence_point,
    compute_opinion_entropy,
    compute_information_loss_curve,
    run_trust_level_sweep,
    run_original_opinion_sweep,
    run_base_rate_sweep,
    run_chain_length_comparison,
)
from jsonld_ex.confidence_algebra import Opinion

GLOBAL_SEED = 42
MAX_CHAIN = 30

TRUST_LEVELS = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
BASE_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def run_experiment() -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    # ── 1. Core comparison: SL vs scalar vs Bayesian ──
    print("  Part 1: Core chain comparison")
    original = Opinion(0.85, 0.10, 0.05, base_rate=0.5)
    for tl in TRUST_LEVELS:
        trust = Opinion(tl, (1.0 - tl) * 0.5, (1.0 - tl) * 0.5)
        comparison = run_chain_length_comparison(
            trust, tl, original, max_length=MAX_CHAIN,
        )
        # Serialize opinions out of the result
        results[f"core_tl{int(tl*100)}"] = comparison

    # ── 2. Trust level sweep ──
    print("  Part 2: Trust level sweep")
    results["trust_sweep"] = run_trust_level_sweep(
        trust_levels=TRUST_LEVELS,
        original=Opinion(0.85, 0.10, 0.05, base_rate=0.5),
        max_length=MAX_CHAIN,
    )

    # ── 3. Base rate sweep (the key SL advantage) ──
    print("  Part 3: Base rate sweep")
    trust_br = Opinion(0.85, 0.05, 0.10)
    results["base_rate_sweep"] = {}
    for br in BASE_RATES:
        orig = Opinion(0.80, 0.10, 0.10, base_rate=br)
        sl_chain = compute_sl_chain(trust_br, orig, max_length=MAX_CHAIN)
        scalar_chain = compute_scalar_chain(
            0.85, orig.projected_probability(), max_length=MAX_CHAIN,
        )
        bayesian_chain = compute_bayesian_chain(
            0.85, orig.projected_probability(), max_length=MAX_CHAIN, prior=br,
        )
        results["base_rate_sweep"][br] = {
            "sl_probs": [op.projected_probability() for op in sl_chain],
            "sl_uncertainties": [op.uncertainty for op in sl_chain],
            "sl_beliefs": [op.belief for op in sl_chain],
            "scalar_probs": scalar_chain,
            "bayesian_probs": bayesian_chain,
            "sl_limit": br,
            "scalar_limit": 0.0,
            "bayesian_limit": br,
        }

    # ── 4. Decision divergence sweep ──
    print("  Part 4: Decision divergence")
    results["divergence"] = {}
    for tl in TRUST_LEVELS:
        trust = Opinion(tl, (1.0 - tl) * 0.5, (1.0 - tl) * 0.5)
        for br in [0.3, 0.5, 0.7]:
            orig = Opinion(0.85, 0.10, 0.05, base_rate=br)
            div = compute_decision_divergence_point(
                trust, tl, orig, threshold=0.5, max_length=50,
            )
            key = f"tl{int(tl*100)}_br{int(br*10)}"
            results["divergence"][key] = div

    # ── 5. Information content analysis ──
    print("  Part 5: Information content")
    trust_info = Opinion(0.85, 0.05, 0.10)
    orig_info = Opinion(0.80, 0.15, 0.05, base_rate=0.5)
    info_curve = compute_information_loss_curve(
        trust_info, 0.85, orig_info, max_length=MAX_CHAIN,
    )
    results["information_curve"] = info_curve

    # Verify non-monotonicity of BDU entropy (the finding)
    entropies = [entry["sl_entropy"] for entry in info_curve]
    has_increase = any(entropies[i] > entropies[i-1] + 1e-9
                       for i in range(1, len(entropies)))
    results["entropy_non_monotonic"] = has_increase
    results["entropy_peak_step"] = int(np.argmax(entropies))
    results["entropy_peak_value"] = float(max(entropies))

    # Evidence mass (the correct monotonic quantity)
    sl_chain_info = compute_sl_chain(trust_info, orig_info, max_length=MAX_CHAIN)
    evidence_mass = [op.belief + op.disbelief for op in sl_chain_info]
    results["evidence_mass_curve"] = [round(e, 9) for e in evidence_mass]
    results["evidence_mass_monotonic"] = all(
        evidence_mass[i] <= evidence_mass[i-1] + 1e-12
        for i in range(1, len(evidence_mass))
    )

    # ── 6. Heterogeneous chain analysis ──
    print("  Part 6: Heterogeneous chains")
    results["heterogeneous"] = {}

    # Uniform strong chain
    strong = Opinion(0.95, 0.02, 0.03)
    chain_strong = compute_heterogeneous_sl_chain([strong]*5,
        Opinion(0.85, 0.10, 0.05, base_rate=0.5))
    results["heterogeneous"]["uniform_strong"] = {
        "probs": [op.projected_probability() for op in chain_strong],
        "uncertainties": [op.uncertainty for op in chain_strong],
    }

    # One weak link
    weak = Opinion(0.30, 0.40, 0.30)
    chain_weak = compute_heterogeneous_sl_chain(
        [strong, strong, weak, strong, strong],
        Opinion(0.85, 0.10, 0.05, base_rate=0.5),
    )
    results["heterogeneous"]["one_weak_link"] = {
        "probs": [op.projected_probability() for op in chain_weak],
        "uncertainties": [op.uncertainty for op in chain_weak],
    }

    # Degrading trust (each link weaker)
    degrading = [Opinion(0.95-0.1*i, 0.02+0.05*i, 0.03+0.05*i) for i in range(5)]
    chain_degrade = compute_heterogeneous_sl_chain(
        degrading, Opinion(0.85, 0.10, 0.05, base_rate=0.5))
    results["heterogeneous"]["degrading_trust"] = {
        "probs": [op.projected_probability() for op in chain_degrade],
        "uncertainties": [op.uncertainty for op in chain_degrade],
        "trust_beliefs": [t.belief for t in degrading],
    }

    # ── 7. Branching provenance ──
    print("  Part 7: Branching provenance")
    results["branching"] = {}
    trust_branch = Opinion(0.80, 0.10, 0.10)
    orig_branch = Opinion(0.85, 0.10, 0.05, base_rate=0.5)

    for n_paths in [1, 2, 3, 4, 5]:
        for chain_len in [3, 5, 10]:
            trusts_pp = [[trust_branch] * chain_len] * n_paths
            result = compute_branching_provenance(orig_branch, trusts_pp)
            key = f"paths{n_paths}_len{chain_len}"
            results["branching"][key] = {
                "fused_prob": result["fused"].projected_probability(),
                "fused_uncertainty": result["fused"].uncertainty,
                "fused_belief": result["fused"].belief,
                "path_probs": [p.projected_probability() for p in result["paths"]],
                "path_uncertainties": [p.uncertainty for p in result["paths"]],
            }

    # ── 8. Closed-form verification (all steps, multiple configs) ──
    print("  Part 8: Closed-form verification")
    verification_errors = []
    for tl in TRUST_LEVELS:
        trust = Opinion(tl, (1.0 - tl) * 0.5, (1.0 - tl) * 0.5)
        for br in [0.2, 0.5, 0.8]:
            orig = Opinion(0.80, 0.10, 0.10, base_rate=br)
            chain = compute_sl_chain(trust, orig, max_length=MAX_CHAIN)
            for n in range(MAX_CHAIN + 1):
                computed_p = chain[n].projected_probability()
                formula_p = sl_chain_closed_form(tl, orig.projected_probability(), br, n)
                error = abs(computed_p - formula_p)
                if error > 1e-9:
                    verification_errors.append({
                        "tl": tl, "br": br, "n": n,
                        "computed": computed_p, "formula": formula_p, "error": error,
                    })

    results["closed_form_verification"] = {
        "total_checks": len(TRUST_LEVELS) * 3 * (MAX_CHAIN + 1),
        "errors": verification_errors,
        "max_error": max((e["error"] for e in verification_errors), default=0.0),
        "all_pass": len(verification_errors) == 0,
    }

    return results


def print_summary(results: Dict[str, Any]):
    print(f"\n{'='*70}")
    print("EN1.4 RESULTS SUMMARY")
    print(f"{'='*70}")

    # Core comparison at trust=0.85
    core = results.get("core_tl85", {})
    print(f"\n  Core (trust=0.85, original P=0.925, base_rate=0.5):")
    print(f"    SL limit:      {core.get('sl_limit', 'N/A')} (base rate)")
    print(f"    Scalar limit:  {core.get('scalar_limit', 'N/A')}")
    print(f"    Bayesian limit:{core.get('bayesian_limit', 'N/A')}")
    print(f"    SL half-life:  {core.get('sl_half_life', 'N/A')} steps")
    print(f"    Scalar half-life: {core.get('scalar_half_life', 'N/A')} steps")
    div = core.get("divergence_point")
    if div:
        print(f"    Divergence at step {div['divergence_length']}: "
              f"SL P={div['sl_prob']:.3f} ({'pos' if div['sl_decision'] else 'neg'}), "
              f"Scalar={div['scalar_prob']:.3f} ({'pos' if div['scalar_decision'] else 'neg'})")

    # Base rate sweep
    print(f"\n  Base rate sweep (trust=0.85, step 20):")
    print(f"    {'BR':>5} {'SL P(20)':>10} {'Scalar(20)':>12} {'Bayesian(20)':>14}")
    for br in BASE_RATES:
        data = results.get("base_rate_sweep", {}).get(br, {})
        sl_p = data.get("sl_probs", [0]*21)[20]
        sc_p = data.get("scalar_probs", [0]*21)[20]
        by_p = data.get("bayesian_probs", [0]*21)[20]
        print(f"    {br:>5.1f} {sl_p:>10.4f} {sc_p:>12.6f} {by_p:>14.4f}")

    # Divergence
    print(f"\n  Decision divergence (threshold=0.5):")
    for key, div in results.get("divergence", {}).items():
        if div is not None:
            print(f"    {key}: step {div['divergence_length']}, "
                  f"SL={div['sl_prob']:.3f}, scalar={div['scalar_prob']:.3f}")
        else:
            print(f"    {key}: no divergence within 50 steps")

    # Information
    print(f"\n  Information content:")
    print(f"    BDU entropy non-monotonic: {results.get('entropy_non_monotonic')}")
    print(f"    Entropy peak at step: {results.get('entropy_peak_step')}")
    print(f"    Evidence mass monotonic: {results.get('evidence_mass_monotonic')}")

    # Branching
    print(f"\n  Branching provenance (trust=0.80, chain_len=5):")
    for n_paths in [1, 2, 3, 5]:
        key = f"paths{n_paths}_len5"
        data = results.get("branching", {}).get(key, {})
        print(f"    {n_paths} paths: P={data.get('fused_prob', 0):.4f}, "
              f"u={data.get('fused_uncertainty', 0):.4f}")

    # Verification
    verif = results.get("closed_form_verification", {})
    print(f"\n  Closed-form verification:")
    print(f"    Total checks: {verif.get('total_checks', 0)}")
    print(f"    All pass: {verif.get('all_pass', False)}")
    print(f"    Max error: {verif.get('max_error', 0):.2e}")


def main():
    set_global_seed(GLOBAL_SEED)
    t_start = time.time()
    env = log_environment()

    print("EN1.4 — Trust Discount Chain Analysis")
    print("=" * 60)

    results = run_experiment()
    elapsed = time.time() - t_start

    print_summary(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_result = ExperimentResult(
        experiment_id="EN1.4",
        parameters={
            "trust_levels": TRUST_LEVELS,
            "base_rates": BASE_RATES,
            "max_chain_length": MAX_CHAIN,
            "global_seed": GLOBAL_SEED,
        },
        metrics=results,
        environment=env,
        notes=f"Trust discount chain analysis: {len(TRUST_LEVELS)} trust levels, "
              f"{len(BASE_RATES)} base rates, chains up to {MAX_CHAIN}. "
              f"Closed-form verified. {elapsed:.1f}s.",
    )

    primary = RESULTS_DIR / "en1_4_results.json"
    archive = RESULTS_DIR / f"en1_4_results_{timestamp}.json"
    experiment_result.save_json(str(primary))
    experiment_result.save_json(str(archive))

    print(f"\nDONE — {elapsed:.1f}s")
    print(f"Results: {primary}")


if __name__ == "__main__":
    main()
