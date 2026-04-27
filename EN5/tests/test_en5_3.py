"""Tests for EN5.3 -- Validation as Security Layer (Resource Limits).

RED phase -- all tests should FAIL until en5_3_core.py is implemented.
The jsonld_ex.security imports should succeed (testing the actual library).

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest experiments/EN5/tests/test_en5_3.py -v

Hypotheses:
    H5.3a: Size enforcement catches all oversized documents
    H5.3b: Depth enforcement catches all over-depth documents
    H5.3c: Default limits are safe (no OOM/stack overflow)
    H5.3d: Error messages are actionable (3 required fields)
    H5.3e: Timeout enforcement (investigation)
    H5.3f: Context depth vs graph depth (investigation)
"""
from __future__ import annotations

import json
import re
import pytest

# -- Actual library imports (these SHOULD succeed) --
from jsonld_ex.security import (
    enforce_resource_limits,
    DEFAULT_RESOURCE_LIMITS,
)

# -- Experiment core imports (these SHOULD FAIL in RED phase) --
from EN5.en5_3_core import (
    # Document generators
    generate_depth_bomb,
    generate_size_bomb,
    generate_width_bomb,
    generate_adversarial_documents,
    AdversarialDocType,
    ALL_ADVERSARIAL_TYPES,
    # Investigation runners
    run_timeout_investigation,
    run_context_vs_graph_depth_investigation,
    # Test suites
    run_size_enforcement_suite,
    run_depth_enforcement_suite,
    run_adversarial_detection_suite,
    run_error_actionability_suite,
    run_processing_order_suite,
    # Result types
    SizeEnforcementResult,
    DepthEnforcementResult,
    AdversarialDetectionResult,
    ErrorActionabilityResult,
    TimeoutInvestigationResult,
    DepthInvestigationResult,
    ProcessingOrderResult,
)


# =====================================================================
# 0. Smoke tests: verify enforce_resource_limits API
# =====================================================================

class TestResourceLimitsAPIAvailable:
    """Verify the actual library function exists and works."""

    def test_enforce_resource_limits_exists(self):
        assert callable(enforce_resource_limits)

    def test_default_limits_exist(self):
        assert isinstance(DEFAULT_RESOURCE_LIMITS, dict)
        assert "max_document_size" in DEFAULT_RESOURCE_LIMITS
        assert "max_graph_depth" in DEFAULT_RESOURCE_LIMITS

    def test_valid_document_passes(self):
        enforce_resource_limits({"key": "value"})  # Should not raise

    def test_valid_string_passes(self):
        enforce_resource_limits('{"key": "value"}')  # Should not raise

    def test_none_raises_typeerror(self):
        with pytest.raises(TypeError):
            enforce_resource_limits(None)

    def test_oversized_string_raises(self):
        with pytest.raises(ValueError, match="exceeds limit"):
            enforce_resource_limits('{"a": 1}', {"max_document_size": 2})

    def test_deep_document_raises(self):
        nested = {"a": {"b": {"c": {"d": "deep"}}}}  # depth=4
        with pytest.raises(ValueError, match="depth"):
            enforce_resource_limits(nested, {"max_graph_depth": 2})

    def test_list_document_passes(self):
        enforce_resource_limits([{"a": 1}, {"b": 2}])  # Should not raise

    def test_invalid_json_string_raises(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            enforce_resource_limits("{not json}")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="must be"):
            enforce_resource_limits(42)


# =====================================================================
# 1. Document generators
# =====================================================================

class TestDocumentGenerators:
    """Verify adversarial document generation."""

    def test_depth_bomb_creates_nested(self):
        doc = generate_depth_bomb(depth=50)
        assert isinstance(doc, dict)
        # Verify actual depth
        d = doc
        count = 0
        while isinstance(d, dict) and d:
            keys = list(d.keys())
            d = d[keys[0]] if keys else None
            count += 1
        assert count >= 50

    def test_depth_bomb_parametric(self):
        for target in [10, 50, 100, 200]:
            doc = generate_depth_bomb(depth=target)
            assert isinstance(doc, dict)

    def test_size_bomb_exceeds_default(self):
        doc = generate_size_bomb()
        serialized = json.dumps(doc)
        assert len(serialized) > DEFAULT_RESOURCE_LIMITS["max_document_size"]

    def test_size_bomb_parametric(self):
        doc = generate_size_bomb(target_bytes=1_000_000)
        serialized = json.dumps(doc)
        assert len(serialized) >= 1_000_000

    def test_width_bomb_creates_wide(self):
        doc = generate_width_bomb(n_keys=10_000)
        assert isinstance(doc, dict)
        assert len(doc) >= 10_000

    def test_adversarial_doc_types(self):
        """All adversarial doc types are defined."""
        assert len(ALL_ADVERSARIAL_TYPES) >= 8

    def test_generate_adversarial_returns_dict(self):
        for doc_type in ALL_ADVERSARIAL_TYPES:
            docs = generate_adversarial_documents(doc_type)
            assert isinstance(docs, list)
            assert len(docs) >= 1
            for doc in docs:
                assert isinstance(doc, (dict, list, str))


# =====================================================================
# 2. H5.3a — Size enforcement
# =====================================================================

class TestH53aSizeEnforcement:
    """enforce_resource_limits() catches all oversized documents."""

    def test_at_limit_passes(self):
        """Document exactly at the limit should pass."""
        # Create doc just under 1KB
        doc = {"data": "x" * 900}
        enforce_resource_limits(doc, {"max_document_size": 2000})

    def test_over_limit_raises(self):
        """Document over the limit should raise."""
        doc = {"data": "x" * 2000}
        with pytest.raises(ValueError, match="exceeds limit"):
            enforce_resource_limits(doc, {"max_document_size": 1000})

    def test_string_size_checked(self):
        """String input size is checked."""
        big_str = json.dumps({"data": "x" * 5000})
        with pytest.raises(ValueError, match="exceeds limit"):
            enforce_resource_limits(big_str, {"max_document_size": 1000})

    def test_size_bomb_caught_by_defaults(self):
        """Size bomb exceeding 10MB is caught by default limits."""
        doc = generate_size_bomb()
        with pytest.raises(ValueError, match="exceeds limit"):
            enforce_resource_limits(doc)

    def test_full_size_enforcement_suite(self):
        """Systematic size enforcement across multiple thresholds."""
        result = run_size_enforcement_suite()
        assert isinstance(result, SizeEnforcementResult)
        assert result.false_accepts == 0, (
            f"H5.3a: {result.false_accepts} oversized docs accepted"
        )
        assert result.false_rejects == 0, (
            f"H5.3a: {result.false_rejects} valid docs rejected"
        )


# =====================================================================
# 3. H5.3b — Depth enforcement
# =====================================================================

class TestH53bDepthEnforcement:
    """enforce_resource_limits() catches all over-depth documents."""

    def test_at_limit_passes(self):
        """Document at exactly max depth should pass."""
        doc = {"a": "leaf"}
        for _ in range(9):  # depth = 10
            doc = {"nested": doc}
        enforce_resource_limits(doc, {"max_graph_depth": 10})

    def test_over_limit_raises(self):
        """Document exceeding max depth should raise."""
        doc = {"a": "leaf"}
        for _ in range(15):  # depth = 16
            doc = {"nested": doc}
        with pytest.raises(ValueError, match="depth"):
            enforce_resource_limits(doc, {"max_graph_depth": 10})

    def test_depth_bomb_caught_by_defaults(self):
        """200-level depth bomb caught by default max_graph_depth=100."""
        doc = generate_depth_bomb(depth=200)
        with pytest.raises(ValueError, match="depth"):
            enforce_resource_limits(doc)

    def test_no_stack_overflow_at_500(self):
        """Depth=500 does not cause stack overflow (safety cap)."""
        doc = generate_depth_bomb(depth=500)
        # Should raise ValueError for depth, NOT RecursionError
        with pytest.raises(ValueError, match="depth"):
            enforce_resource_limits(doc)

    def test_full_depth_enforcement_suite(self):
        """Systematic depth enforcement across multiple thresholds."""
        result = run_depth_enforcement_suite()
        assert isinstance(result, DepthEnforcementResult)
        assert result.false_accepts == 0, (
            f"H5.3b: {result.false_accepts} over-depth docs accepted"
        )
        assert result.false_rejects == 0, (
            f"H5.3b: {result.false_rejects} valid-depth docs rejected"
        )
        assert result.stack_overflows == 0, (
            f"H5.3b: {result.stack_overflows} stack overflows (safety cap failed)"
        )


# =====================================================================
# 4. H5.3c — Default safety
# =====================================================================

class TestH53cDefaultSafety:
    """Default limits prevent OOM and stack overflow."""

    def test_depth_bomb_safe(self):
        """Depth bomb rejected without crash."""
        doc = generate_depth_bomb(depth=200)
        try:
            enforce_resource_limits(doc)
            pytest.fail("Should have raised ValueError")
        except ValueError:
            pass  # Expected
        except RecursionError:
            pytest.fail("Stack overflow — safety cap not working")

    def test_size_bomb_safe(self):
        """Size bomb rejected without OOM."""
        doc = generate_size_bomb()
        try:
            enforce_resource_limits(doc)
            pytest.fail("Should have raised ValueError")
        except ValueError:
            pass  # Expected
        except MemoryError:
            pytest.fail("OOM — size check should happen before processing")

    def test_width_bomb_safe(self):
        """Wide document (100K keys) handled without crash."""
        doc = generate_width_bomb(n_keys=100_000)
        # Width bomb may or may not exceed size limit; either way no crash
        try:
            enforce_resource_limits(doc)
        except ValueError:
            pass  # Size limit exceeded — that's fine


# =====================================================================
# 5. H5.3d — Error actionability
# =====================================================================

class TestH53dErrorActionability:
    """Error messages include: which limit, measured value, configured limit."""

    def test_size_error_has_measured_value(self):
        doc = {"data": "x" * 2000}
        try:
            enforce_resource_limits(doc, {"max_document_size": 100})
            pytest.fail("Should have raised")
        except ValueError as e:
            msg = str(e)
            # Should contain a number (the measured size)
            assert re.search(r"\d+", msg), f"No measured value in: {msg}"

    def test_size_error_has_limit_value(self):
        doc = {"data": "x" * 2000}
        try:
            enforce_resource_limits(doc, {"max_document_size": 100})
            pytest.fail("Should have raised")
        except ValueError as e:
            msg = str(e)
            assert "100" in msg, f"Limit value not in message: {msg}"

    def test_depth_error_has_measured_value(self):
        doc = generate_depth_bomb(depth=20)
        try:
            enforce_resource_limits(doc, {"max_graph_depth": 5})
            pytest.fail("Should have raised")
        except ValueError as e:
            msg = str(e)
            assert re.search(r"\d+", msg), f"No measured value in: {msg}"

    def test_depth_error_has_limit_value(self):
        doc = generate_depth_bomb(depth=20)
        try:
            enforce_resource_limits(doc, {"max_graph_depth": 5})
            pytest.fail("Should have raised")
        except ValueError as e:
            msg = str(e)
            assert "5" in msg, f"Limit value not in message: {msg}"

    def test_full_actionability_suite(self):
        """Systematic error message analysis."""
        result = run_error_actionability_suite()
        assert isinstance(result, ErrorActionabilityResult)
        assert result.messages_with_measured == result.total_errors, (
            f"Missing measured value in {result.total_errors - result.messages_with_measured} errors"
        )
        assert result.messages_with_limit == result.total_errors, (
            f"Missing limit value in {result.total_errors - result.messages_with_limit} errors"
        )


# =====================================================================
# 6. H5.3e — Timeout investigation
# =====================================================================

class TestH53eTimeoutInvestigation:
    """Investigate whether max_expansion_time is enforced."""

    def test_timeout_investigation_runs(self):
        result = run_timeout_investigation()
        assert isinstance(result, TimeoutInvestigationResult)

    def test_timeout_parameter_documented(self):
        """Investigation documents whether timeout is implemented."""
        result = run_timeout_investigation()
        assert result.parameter_exists is not None  # True or False
        assert isinstance(result.is_enforced, bool)
        assert isinstance(result.finding, str)
        assert len(result.finding) > 0

    def test_timeout_in_defaults(self):
        """max_expansion_time should at least exist in DEFAULT_RESOURCE_LIMITS."""
        assert "max_expansion_time" in DEFAULT_RESOURCE_LIMITS
        assert DEFAULT_RESOURCE_LIMITS["max_expansion_time"] == 30

    def test_timeout_result_serializable(self):
        result = run_timeout_investigation()
        d = result.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0


# =====================================================================
# 7. H5.3f — Context depth vs graph depth
# =====================================================================

class TestH53fContextVsGraphDepth:
    """Investigate context depth vs graph depth distinction."""

    def test_investigation_runs(self):
        result = run_context_vs_graph_depth_investigation()
        assert isinstance(result, DepthInvestigationResult)

    def test_investigation_documents_behavior(self):
        result = run_context_vs_graph_depth_investigation()
        assert isinstance(result.max_context_depth_exists, bool)
        assert isinstance(result.max_graph_depth_exists, bool)
        assert isinstance(result.are_separate_parameters, bool)
        assert isinstance(result.finding, str)
        assert len(result.finding) > 0

    def test_both_parameters_in_defaults(self):
        """Both parameters should at least exist in defaults."""
        assert "max_context_depth" in DEFAULT_RESOURCE_LIMITS
        assert "max_graph_depth" in DEFAULT_RESOURCE_LIMITS

    def test_investigation_result_serializable(self):
        result = run_context_vs_graph_depth_investigation()
        d = result.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0


# =====================================================================
# 8. Adversarial document detection
# =====================================================================

class TestAdversarialDetection:
    """Adversarial JSON-LD documents are caught."""

    def test_type_confusion_caught(self):
        """@type with wrong type (integer) is invalid."""
        docs = generate_adversarial_documents(AdversarialDocType.TYPE_CONFUSION)
        # At minimum, enforce_resource_limits shouldn't crash
        for doc in docs:
            try:
                enforce_resource_limits(doc)
            except (ValueError, TypeError):
                pass  # Expected for some

    def test_deep_nesting_caught(self):
        """200-level nesting exceeds default depth limit."""
        docs = generate_adversarial_documents(AdversarialDocType.DEEP_NESTING)
        for doc in docs:
            with pytest.raises(ValueError):
                enforce_resource_limits(doc)

    def test_full_adversarial_suite(self):
        """Run all adversarial document types."""
        result = run_adversarial_detection_suite()
        assert isinstance(result, AdversarialDetectionResult)
        assert result.total_types_tested >= 8
        assert result.crashes == 0, (
            f"Crashes on adversarial input: {result.crash_details}"
        )


# =====================================================================
# 9. Processing order
# =====================================================================

class TestProcessingOrder:
    """Verify cheap checks run before expensive checks."""

    def test_processing_order_suite_runs(self):
        result = run_processing_order_suite()
        assert isinstance(result, ProcessingOrderResult)

    def test_size_check_before_depth(self):
        """Size check (O(1)) should run before depth traversal (O(n))."""
        result = run_processing_order_suite()
        assert result.size_check_first is True, (
            "Size check should run before depth traversal"
        )

    def test_result_serializable(self):
        result = run_processing_order_suite()
        d = result.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0
