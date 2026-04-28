"""Tests for EN5.6 -- End-to-End Attack Scenario Validation.

RED phase -- all tests should FAIL until en5_6_core.py is implemented.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest experiments/EN5/tests/test_en5_6.py -v

Hypotheses:
    H5.6a: Context injection detected by verify_integrity
    H5.6b: Internal URLs blocked by is_context_allowed
    H5.6c: Resource bombs defused by enforce_resource_limits
    H5.6d: Layered defense short-circuits at earliest check
    H5.6e: Security and SL metadata are orthogonal layers
"""
from __future__ import annotations

import json
import pytest

# -- Actual library imports --
from jsonld_ex.security import (
    compute_integrity,
    verify_integrity,
    is_context_allowed,
    enforce_resource_limits,
)

# -- Experiment core imports (SHOULD FAIL in RED phase) --
from EN5.en5_6_core import (
    # Scenario runners
    run_scenario_a_context_injection,
    run_scenario_b_ssrf_prevention,
    run_scenario_c_resource_exhaustion,
    run_scenario_d_layered_defense,
    run_scenario_e_sl_coexistence,
    # Result types
    ScenarioAResult,
    ScenarioBResult,
    ScenarioCResult,
    ScenarioDResult,
    ScenarioEResult,
    # Full experiment
    run_en5_6_full,
    EN56FullResult,
    # Helpers
    TAMPERING_STRATEGIES,
    SSRF_TEST_URLS,
)


# =====================================================================
# Scenario A: Context Injection (MITM Simulation)
# =====================================================================

class TestScenarioA:
    """Context injection attack — control group + treatment group."""

    def test_scenario_runs(self):
        result = run_scenario_a_context_injection()
        assert isinstance(result, ScenarioAResult)

    def test_control_silent_corruption(self):
        """WITHOUT @integrity, tampered context parses silently."""
        result = run_scenario_a_context_injection()
        assert result.control_tampered_parses is True, (
            "Control should show that tampered context parses without @integrity"
        )
        assert result.control_semantics_changed is True, (
            "Control should show semantic corruption from tampered context"
        )

    def test_treatment_tamper_detected(self):
        """WITH @integrity, all tampering strategies detected."""
        result = run_scenario_a_context_injection()
        assert result.total_strategies == len(TAMPERING_STRATEGIES)
        assert result.strategies_detected == result.total_strategies, (
            f"H5.6a: Only {result.strategies_detected}/{result.total_strategies} "
            f"strategies detected. Missed: {result.undetected_strategies}"
        )

    def test_valid_context_still_accepted(self):
        """Untampered context passes verification."""
        result = run_scenario_a_context_injection()
        assert result.valid_context_accepted is True

    def test_result_serializable(self):
        result = run_scenario_a_context_injection()
        serialized = json.dumps(result.to_dict())
        assert len(serialized) > 0


# =====================================================================
# Scenario B: SSRF Prevention
# =====================================================================

class TestScenarioB:
    """SSRF prevention via context allowlist."""

    def test_scenario_runs(self):
        result = run_scenario_b_ssrf_prevention()
        assert isinstance(result, ScenarioBResult)

    def test_all_internal_urls_blocked(self):
        result = run_scenario_b_ssrf_prevention()
        assert result.total_ssrf_urls >= 20
        assert result.ssrf_blocked == result.total_ssrf_urls, (
            f"H5.6b: {result.total_ssrf_urls - result.ssrf_blocked} SSRF URLs "
            f"not blocked: {result.unblocked_urls}"
        )

    def test_allowlisted_urls_accepted(self):
        result = run_scenario_b_ssrf_prevention()
        assert result.allowlisted_accepted >= 2  # At least 2 legit URLs

    def test_block_remote_rejects_all(self):
        result = run_scenario_b_ssrf_prevention()
        assert result.block_remote_rejects_all is True

    def test_result_serializable(self):
        result = run_scenario_b_ssrf_prevention()
        serialized = json.dumps(result.to_dict())
        assert len(serialized) > 0


# =====================================================================
# Scenario C: Resource Exhaustion Prevention
# =====================================================================

class TestScenarioC:
    """Resource bombs — measured controls + treatment."""

    def test_scenario_runs(self):
        result = run_scenario_c_resource_exhaustion()
        assert isinstance(result, ScenarioCResult)

    def test_control_measures_actual_cost(self):
        """Control group measures REAL resource consumption, not assumed."""
        result = run_scenario_c_resource_exhaustion()
        assert result.depth_bomb_unprotected_memory_bytes > 0, (
            "Control should measure actual memory for depth bomb"
        )
        assert result.size_bomb_unprotected_memory_bytes > 0
        assert result.width_bomb_unprotected_memory_bytes > 0

    def test_all_bombs_defused(self):
        """Treatment: all bombs rejected by enforce_resource_limits."""
        result = run_scenario_c_resource_exhaustion()
        assert result.depth_bomb_caught is True
        assert result.size_bomb_caught is True
        assert result.width_bomb_caught is True

    def test_rejection_fast(self):
        """Rejection should be fast (<100ms for each bomb)."""
        result = run_scenario_c_resource_exhaustion()
        assert result.depth_bomb_rejection_ms < 100, (
            f"Depth bomb rejection too slow: {result.depth_bomb_rejection_ms}ms"
        )
        assert result.size_bomb_rejection_ms < 100
        assert result.width_bomb_rejection_ms < 100

    def test_protection_factor_reported(self):
        """Protection factor = unprotected_time / rejection_time."""
        result = run_scenario_c_resource_exhaustion()
        assert result.depth_bomb_protection_factor > 0
        assert result.size_bomb_protection_factor > 0

    def test_no_crashes(self):
        result = run_scenario_c_resource_exhaustion()
        assert result.crashes == 0, f"Crashes: {result.crash_details}"

    def test_result_serializable(self):
        result = run_scenario_c_resource_exhaustion()
        serialized = json.dumps(result.to_dict())
        assert len(serialized) > 0


# =====================================================================
# Scenario D: Layered Defense Composition
# =====================================================================

class TestScenarioD:
    """Defense-in-depth — earliest check catches first."""

    def test_scenario_runs(self):
        result = run_scenario_d_layered_defense()
        assert isinstance(result, ScenarioDResult)

    def test_allowlist_catches_first(self):
        """When all checks fail, allowlist (cheapest) catches first."""
        result = run_scenario_d_layered_defense()
        assert result.first_check_catches == "allowlist", (
            f"Expected allowlist first, got: {result.first_check_catches}"
        )

    def test_fix_allowlist_then_integrity_catches(self):
        """After fixing allowlist, integrity catches next."""
        result = run_scenario_d_layered_defense()
        assert result.second_check_catches == "integrity"

    def test_fix_integrity_then_limits_catches(self):
        """After fixing integrity, resource limits catches next."""
        result = run_scenario_d_layered_defense()
        assert result.third_check_catches == "resource_limits"

    def test_fix_all_passes(self):
        """After fixing all issues, document passes full pipeline."""
        result = run_scenario_d_layered_defense()
        assert result.all_fixed_passes is True

    def test_result_serializable(self):
        result = run_scenario_d_layered_defense()
        serialized = json.dumps(result.to_dict())
        assert len(serialized) > 0


# =====================================================================
# Scenario E: Security + SL Metadata Coexistence
# =====================================================================

class TestScenarioE:
    """Security and SL operations are orthogonal."""

    def test_scenario_runs(self):
        result = run_scenario_e_sl_coexistence()
        assert isinstance(result, ScenarioEResult)

    def test_security_pipeline_passes_sl_doc(self):
        """Full security pipeline passes a richly-annotated SL document."""
        result = run_scenario_e_sl_coexistence()
        assert result.security_passes_sl_doc is True

    def test_sl_operations_work_after_security(self):
        """SL operations (annotate, validate, filter) work after security."""
        result = run_scenario_e_sl_coexistence()
        assert result.annotate_works is True
        assert result.validate_works is True
        assert result.filter_works is True

    def test_integrity_preserved_through_sl(self):
        """@integrity metadata survives SL operations."""
        result = run_scenario_e_sl_coexistence()
        assert result.integrity_preserved is True

    def test_tampered_context_caught_with_sl(self):
        """Security catches tampered context even with SL metadata present."""
        result = run_scenario_e_sl_coexistence()
        assert result.tamper_caught_with_sl is True

    def test_invalid_sl_doesnt_affect_security(self):
        """Invalid SL opinions don't cause security false positives."""
        result = run_scenario_e_sl_coexistence()
        assert result.invalid_sl_security_passes is True

    def test_security_doesnt_validate_sl(self):
        """Security layer does NOT validate SL content."""
        result = run_scenario_e_sl_coexistence()
        assert result.security_ignores_sl_content is True

    def test_result_serializable(self):
        result = run_scenario_e_sl_coexistence()
        serialized = json.dumps(result.to_dict())
        assert len(serialized) > 0


# =====================================================================
# Full EN5.6 integration
# =====================================================================

class TestFullEN56:
    """Full experiment runner."""

    def test_full_runs(self):
        result = run_en5_6_full()
        assert isinstance(result, EN56FullResult)
        assert result.scenario_a is not None
        assert result.scenario_b is not None
        assert result.scenario_c is not None
        assert result.scenario_d is not None
        assert result.scenario_e is not None

    def test_full_serializable(self):
        result = run_en5_6_full()
        serialized = json.dumps(result.to_dict())
        assert len(serialized) > 0
