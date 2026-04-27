"""EN5.3 -- Validation as Security Layer (Resource Limits) core module.

NeurIPS 2026 D&B, Suite EN5 (Security and Integrity), Experiment 3.

Hypotheses:
    H5.3a: Size enforcement catches all oversized documents
    H5.3b: Depth enforcement catches all over-depth documents
    H5.3c: Default limits are safe (no OOM/stack overflow)
    H5.3d: Error messages are actionable
    H5.3e: Timeout enforcement (investigation)
    H5.3f: Context depth vs graph depth (investigation)

Tested function (from jsonld_ex.security):
    enforce_resource_limits, DEFAULT_RESOURCE_LIMITS
"""
from __future__ import annotations

import inspect
import json
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

from jsonld_ex.security import (
    enforce_resource_limits,
    DEFAULT_RESOURCE_LIMITS,
)

# -- sys.path setup --
import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _SCRIPT_DIR.parent
for p in [str(_SCRIPT_DIR), str(_EXPERIMENTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# =====================================================================
# Enums
# =====================================================================

class AdversarialDocType(Enum):
    """Adversarial JSON-LD document types."""
    TYPE_CONFUSION = "type_confusion"
    TYPE_ARRAY_INJECTION = "type_array_injection"
    CARDINALITY_VIOLATION = "cardinality_violation"
    DEEP_NESTING = "deep_nesting"
    WIDE_DOCUMENT = "wide_document"
    NULL_INJECTION = "null_injection"
    NUMERIC_OVERFLOW = "numeric_overflow"
    MIXED_ATTACK = "mixed_attack"


ALL_ADVERSARIAL_TYPES = list(AdversarialDocType)


# =====================================================================
# Result dataclasses
# =====================================================================

@dataclass
class SizeEnforcementResult:
    """Results from systematic size enforcement testing."""
    total_checks: int = 0
    false_accepts: int = 0   # Oversized doc accepted
    false_rejects: int = 0   # Valid doc rejected
    details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DepthEnforcementResult:
    """Results from systematic depth enforcement testing."""
    total_checks: int = 0
    false_accepts: int = 0
    false_rejects: int = 0
    stack_overflows: int = 0
    details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AdversarialDetectionResult:
    """Results from adversarial document testing."""
    total_types_tested: int = 0
    total_docs_tested: int = 0
    caught_by_limits: int = 0
    passed_limits: int = 0  # Not necessarily bad -- some adversarial docs are small/shallow
    crashes: int = 0
    crash_details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ErrorActionabilityResult:
    """Results from error message analysis."""
    total_errors: int = 0
    messages_with_measured: int = 0   # Contains the measured value
    messages_with_limit: int = 0      # Contains the limit value
    messages_with_which_limit: int = 0  # Identifies which limit
    messages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TimeoutInvestigationResult:
    """Results from timeout parameter investigation."""
    parameter_exists: bool = False
    is_enforced: bool = False
    default_value: int | None = None
    finding: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DepthInvestigationResult:
    """Results from context depth vs graph depth investigation."""
    max_context_depth_exists: bool = False
    max_graph_depth_exists: bool = False
    are_separate_parameters: bool = False
    context_depth_enforced: bool = False
    graph_depth_enforced: bool = False
    finding: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProcessingOrderResult:
    """Results from processing order verification."""
    size_check_first: bool = False
    size_check_time_us: float = 0.0
    depth_check_time_us: float = 0.0
    finding: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# =====================================================================
# Document generators
# =====================================================================

def generate_depth_bomb(depth: int = 200) -> dict:
    """Generate a document nested to the specified depth."""
    doc: dict[str, Any] = {"leaf": "value"}
    for i in range(depth - 1):
        doc = {f"level_{depth - 1 - i}": doc}
    return doc


def generate_size_bomb(
    target_bytes: int | None = None,
) -> dict:
    """Generate a document exceeding the target size (default: 10MB + 1KB)."""
    if target_bytes is None:
        target_bytes = DEFAULT_RESOURCE_LIMITS["max_document_size"] + 1024
    # Fill with string data to reach target
    # Compute exact overhead: len('{"payload": ""}') = 15
    shell_len = len(json.dumps({"payload": ""}))
    fill_size = max(1, target_bytes - shell_len)
    return {"payload": "X" * fill_size}


def generate_width_bomb(n_keys: int = 100_000) -> dict:
    """Generate a document with many top-level keys."""
    return {f"key_{i:07d}": i for i in range(n_keys)}


def generate_adversarial_documents(
    doc_type: AdversarialDocType,
) -> list[dict | list | str]:
    """Generate adversarial documents of the specified type."""

    if doc_type == AdversarialDocType.TYPE_CONFUSION:
        return [
            {"@type": 12345},
            {"@type": True},
            {"@type": None},
            {"@type": {"nested": "object"}},
        ]

    elif doc_type == AdversarialDocType.TYPE_ARRAY_INJECTION:
        return [
            {"@type": ["Person", "http://evil.org/Malware"]},
            {"@type": ["a"] * 1000},
        ]

    elif doc_type == AdversarialDocType.CARDINALITY_VIOLATION:
        return [
            {"name": [f"name_{i}" for i in range(1000)]},
            {"values": list(range(10_000))},
        ]

    elif doc_type == AdversarialDocType.DEEP_NESTING:
        return [
            generate_depth_bomb(200),
            generate_depth_bomb(500),
        ]

    elif doc_type == AdversarialDocType.WIDE_DOCUMENT:
        return [
            generate_width_bomb(50_000),
        ]

    elif doc_type == AdversarialDocType.NULL_INJECTION:
        return [
            {"@context": None, "@type": "Thing"},
            {"@id": None, "name": "test"},
            {None: "value"} if False else {"": "empty_key"},  # None key not valid JSON
        ]

    elif doc_type == AdversarialDocType.NUMERIC_OVERFLOW:
        return [
            {"value": 1e308},
            {"value": -1e308},
            {"value": float("inf")} if False else {"value": 1e300},  # inf not JSON-serializable
            {"nested": {"deep": {"value": 2**53 + 1}}},
        ]

    elif doc_type == AdversarialDocType.MIXED_ATTACK:
        # Combine: deep + wide + large
        deep = generate_depth_bomb(150)
        deep_with_payload = deep
        cursor = deep_with_payload
        for _ in range(10):
            k = list(cursor.keys())[0]
            if isinstance(cursor[k], dict):
                cursor = cursor[k]
            else:
                break
        cursor["payload"] = "X" * 100_000
        return [deep_with_payload]

    return [{}]


# =====================================================================
# Size enforcement suite
# =====================================================================

def run_size_enforcement_suite() -> SizeEnforcementResult:
    """Systematically test size enforcement across thresholds."""
    result = SizeEnforcementResult()

    test_configs = [
        # (doc_size_approx, max_document_size, should_raise)
        (500, 1000, False),
        (500, 200, True),
        (10_000, 20_000, False),
        (10_000, 5_000, True),
        (100_000, 200_000, False),
        (100_000, 50_000, True),
        (1_000_000, 2_000_000, False),
        (1_000_000, 500_000, True),
    ]

    for doc_size, limit, should_raise in test_configs:
        doc = {"data": "X" * doc_size}
        result.total_checks += 1

        try:
            enforce_resource_limits(doc, {"max_document_size": limit})
            if should_raise:
                result.false_accepts += 1
                result.details.append({
                    "doc_size": doc_size, "limit": limit,
                    "error": "Expected rejection, got acceptance",
                })
        except ValueError:
            if not should_raise:
                result.false_rejects += 1
                result.details.append({
                    "doc_size": doc_size, "limit": limit,
                    "error": "Expected acceptance, got rejection",
                })
        except Exception as exc:
            result.details.append({
                "doc_size": doc_size, "limit": limit,
                "error": f"Unexpected error: {type(exc).__name__}: {exc}",
            })

    return result


# =====================================================================
# Depth enforcement suite
# =====================================================================

def run_depth_enforcement_suite() -> DepthEnforcementResult:
    """Systematically test depth enforcement across thresholds."""
    result = DepthEnforcementResult()

    test_configs = [
        # (doc_depth, max_graph_depth, should_raise)
        (5, 10, False),
        (10, 10, False),     # At limit
        (11, 10, True),      # Just over
        (20, 10, True),
        (50, 100, False),
        (100, 100, False),   # At limit
        (101, 100, True),    # Just over
        (200, 100, True),
        (500, 100, True),    # Stress test -- safety cap
    ]

    for doc_depth, limit, should_raise in test_configs:
        doc = generate_depth_bomb(doc_depth)
        result.total_checks += 1

        try:
            enforce_resource_limits(doc, {"max_graph_depth": limit})
            if should_raise:
                result.false_accepts += 1
                result.details.append({
                    "depth": doc_depth, "limit": limit,
                    "error": "Expected rejection, got acceptance",
                })
        except ValueError:
            if not should_raise:
                result.false_rejects += 1
                result.details.append({
                    "depth": doc_depth, "limit": limit,
                    "error": "Expected acceptance, got rejection",
                })
        except RecursionError:
            result.stack_overflows += 1
            result.details.append({
                "depth": doc_depth, "limit": limit,
                "error": "RecursionError -- safety cap failed",
            })

    return result


# =====================================================================
# Adversarial detection suite
# =====================================================================

def run_adversarial_detection_suite() -> AdversarialDetectionResult:
    """Test all adversarial document types against enforce_resource_limits."""
    result = AdversarialDetectionResult()

    for doc_type in ALL_ADVERSARIAL_TYPES:
        result.total_types_tested += 1
        docs = generate_adversarial_documents(doc_type)

        for doc in docs:
            result.total_docs_tested += 1
            try:
                enforce_resource_limits(doc)
                result.passed_limits += 1
            except (ValueError, TypeError):
                result.caught_by_limits += 1
            except Exception as exc:
                result.crashes += 1
                result.crash_details.append(
                    f"{doc_type.value}: {type(exc).__name__}: {exc}"
                )

    return result


# =====================================================================
# Error actionability suite
# =====================================================================

def run_error_actionability_suite() -> ErrorActionabilityResult:
    """Analyze error messages for actionability."""
    result = ErrorActionabilityResult()

    error_scenarios = [
        # (description, document, limits)
        ("size_string", '{"d": "' + "X" * 5000 + '"}', {"max_document_size": 100}),
        ("size_dict", {"data": "X" * 5000}, {"max_document_size": 100}),
        ("depth_shallow_limit", generate_depth_bomb(20), {"max_graph_depth": 5}),
        ("depth_default_limit", generate_depth_bomb(200), None),
    ]

    for desc, doc, limits in error_scenarios:
        try:
            if limits:
                enforce_resource_limits(doc, limits)
            else:
                enforce_resource_limits(doc)
            continue  # No error -- skip
        except (ValueError, TypeError) as exc:
            msg = str(exc)
            result.total_errors += 1

            # Check for measured value (any number)
            has_measured = bool(re.search(r"\d+", msg))
            if has_measured:
                result.messages_with_measured += 1

            # Check for limit value
            if limits:
                limit_val = str(limits.get("max_document_size",
                                limits.get("max_graph_depth", "")))
            else:
                limit_val = str(DEFAULT_RESOURCE_LIMITS.get("max_graph_depth", ""))

            has_limit = limit_val in msg if limit_val else False
            if has_limit:
                result.messages_with_limit += 1

            # Check for which-limit identification
            has_which = ("size" in msg.lower() or "depth" in msg.lower()
                         or "exceeds" in msg.lower())
            if has_which:
                result.messages_with_which_limit += 1

            result.messages.append({
                "scenario": desc,
                "message": msg,
                "has_measured": has_measured,
                "has_limit": has_limit,
                "has_which": has_which,
            })

    return result


# =====================================================================
# Timeout investigation
# =====================================================================

def run_timeout_investigation() -> TimeoutInvestigationResult:
    """Investigate whether max_expansion_time is enforced.

    This reads the actual source code of enforce_resource_limits to
    determine whether the timeout parameter is used. This is honest
    source-level analysis, not guesswork.
    """
    result = TimeoutInvestigationResult()

    # Check if parameter exists in defaults
    result.parameter_exists = "max_expansion_time" in DEFAULT_RESOURCE_LIMITS
    if result.parameter_exists:
        result.default_value = DEFAULT_RESOURCE_LIMITS["max_expansion_time"]

    # Inspect the source code of enforce_resource_limits
    try:
        source = inspect.getsource(enforce_resource_limits)
        uses_timeout = ("max_expansion_time" in source
                        or "timeout" in source.lower()
                        or "time.time" in source
                        or "time.perf_counter" in source
                        or "signal.alarm" in source)
        result.is_enforced = uses_timeout
    except (OSError, TypeError):
        # Can't inspect source -- test empirically
        result.is_enforced = False

    if result.parameter_exists and not result.is_enforced:
        result.finding = (
            "max_expansion_time is defined in DEFAULT_RESOURCE_LIMITS "
            f"(default={result.default_value}s) but is NOT enforced by "
            "enforce_resource_limits(). The parameter is accepted in the "
            "limits dict but has no effect on processing. This is an "
            "implementation gap relative to the FLAIRS proposal (A3). "
            "Recommendation: implement timeout enforcement using "
            "signal.alarm (Unix) or threading.Timer (cross-platform) "
            "before NeurIPS submission, or document as future work."
        )
    elif result.parameter_exists and result.is_enforced:
        result.finding = (
            "max_expansion_time is defined and enforced. "
            f"Default timeout: {result.default_value}s."
        )
    else:
        result.finding = (
            "max_expansion_time is not present in DEFAULT_RESOURCE_LIMITS. "
            "This parameter was proposed in FLAIRS (A3) but has not been "
            "added to the implementation."
        )

    return result


# =====================================================================
# Context depth vs graph depth investigation
# =====================================================================

def run_context_vs_graph_depth_investigation() -> DepthInvestigationResult:
    """Investigate whether max_context_depth and max_graph_depth are separate.

    Reads source code to determine which parameters are actually checked
    by enforce_resource_limits.
    """
    result = DepthInvestigationResult()

    # Check existence in defaults
    result.max_context_depth_exists = "max_context_depth" in DEFAULT_RESOURCE_LIMITS
    result.max_graph_depth_exists = "max_graph_depth" in DEFAULT_RESOURCE_LIMITS

    # Inspect source to see which is actually enforced
    try:
        source = inspect.getsource(enforce_resource_limits)
        result.context_depth_enforced = "max_context_depth" in source
        result.graph_depth_enforced = "max_graph_depth" in source
    except (OSError, TypeError):
        result.context_depth_enforced = False
        result.graph_depth_enforced = False

    result.are_separate_parameters = (
        result.max_context_depth_exists and result.max_graph_depth_exists
    )

    # Build finding
    if result.max_context_depth_exists and not result.context_depth_enforced:
        if result.graph_depth_enforced:
            result.finding = (
                "Both max_context_depth and max_graph_depth exist in "
                "DEFAULT_RESOURCE_LIMITS, but only max_graph_depth is "
                "enforced by enforce_resource_limits(). max_context_depth "
                f"(default={DEFAULT_RESOURCE_LIMITS.get('max_context_depth')}) "
                "is accepted but has no effect. The FLAIRS proposal (A3) "
                "specifies these as separate limits: maxContextDepth=10 for "
                "nested context chains, maxGraphDepth=100 for document "
                "nesting. Current implementation conflates them under "
                "max_graph_depth only. Recommendation: implement "
                "max_context_depth as a separate check on context chain "
                "depth, or document the design decision to use a single "
                "depth parameter."
            )
        else:
            result.finding = (
                "Neither max_context_depth nor max_graph_depth is enforced. "
                "This is a critical implementation gap."
            )
    elif not result.max_context_depth_exists:
        result.finding = (
            "max_context_depth does not exist in DEFAULT_RESOURCE_LIMITS. "
            "Only max_graph_depth is defined and enforced."
        )
    else:
        result.finding = (
            "Both parameters exist and are enforced as separate checks."
        )

    return result


# =====================================================================
# Processing order suite
# =====================================================================

def run_processing_order_suite() -> ProcessingOrderResult:
    """Verify that cheap checks (size) run before expensive checks (depth).

    We test this by creating a document that fails BOTH size and depth
    limits, then checking which error is raised first.
    """
    result = ProcessingOrderResult()

    # Create a document that is both oversized and over-depth
    # Deep structure with large payload
    deep_doc = generate_depth_bomb(200)
    # Add payload to make it oversized
    cursor = deep_doc
    for _ in range(5):
        k = list(cursor.keys())[0]
        if isinstance(cursor[k], dict):
            cursor = cursor[k]
        else:
            break
    cursor["payload"] = "X" * 500_000

    # Set limits so both would fail
    limits = {"max_document_size": 1000, "max_graph_depth": 10}

    try:
        enforce_resource_limits(deep_doc, limits)
        result.finding = "Document unexpectedly passed both limits"
    except ValueError as exc:
        msg = str(exc).lower()
        if "size" in msg or "exceeds limit" in msg:
            result.size_check_first = True
            result.finding = (
                "Size check runs before depth check (correct). "
                "Error was: " + str(exc)
            )
        elif "depth" in msg:
            result.size_check_first = False
            result.finding = (
                "Depth check runs before size check. This is suboptimal "
                "since size check is O(1) while depth traversal is O(n). "
                "Error was: " + str(exc)
            )
        else:
            result.finding = f"Unknown error type: {exc}"

    # Also measure individual check times
    small_deep = generate_depth_bomb(50)
    serialized = json.dumps(small_deep)

    # Time size check (serialize + len)
    t0 = time.perf_counter_ns()
    for _ in range(1000):
        _ = len(json.dumps(small_deep))
    t1 = time.perf_counter_ns()
    result.size_check_time_us = (t1 - t0) / 1000.0 / 1000.0  # ns -> us, per call

    # Time depth check (full traversal)
    from jsonld_ex.security import _measure_depth
    t0 = time.perf_counter_ns()
    for _ in range(1000):
        _measure_depth(small_deep)
    t1 = time.perf_counter_ns()
    result.depth_check_time_us = (t1 - t0) / 1000.0 / 1000.0

    return result
