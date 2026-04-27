"""EN5.1 -- Context Integrity Verification core module.

NeurIPS 2026 D&B, Suite EN5 (Security and Integrity), Experiment 1.

Hypotheses:
    H5.1a: verify_integrity() detects ALL mutations (0 false negatives)
    H5.1b: verify_integrity() produces 0 false positives
    H5.1c: Round-trip property holds for all algorithms
    H5.1d: compute_integrity() is deterministic; dict key order invariant

Tested functions (from jsonld_ex.security):
    compute_integrity, verify_integrity, integrity_context
"""
from __future__ import annotations

import copy
import json
import random
import string
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from jsonld_ex.security import (
    compute_integrity,
    verify_integrity,
    SUPPORTED_ALGORITHMS,
)

# -- Add experiments root to sys.path for infra imports --
import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_ROOT))

from infra.config import set_global_seed
from infra.stats import bootstrap_ci


# =====================================================================
# Constants
# =====================================================================

PBT_EXAMPLES_PER_ALGORITHM = 10_000
LATENCY_TRIALS = 10_000
LATENCY_WARMUP = 100

CONTEXT_SIZE_TARGETS = [100, 1_000, 10_000, 100_000, 1_000_000]


# =====================================================================
# Enums
# =====================================================================

class MutationType(Enum):
    """Context mutation strategies for tamper detection testing."""
    BYTE_FLIP = "byte_flip"
    KEY_INSERT = "key_insert"
    KEY_DELETE = "key_delete"
    VALUE_SUBSTITUTE = "value_substitute"
    WHITESPACE_INJECT = "whitespace_inject"
    TRUNCATE = "truncate"
    KEY_RENAME = "key_rename"
    VALUE_RETYPE = "value_retype"


ALL_MUTATION_TYPES = list(MutationType)


# =====================================================================
# Result dataclasses
# =====================================================================

@dataclass
class IntegrityPBTResult:
    """Results from property-based integrity testing."""
    total_roundtrip_checks: int = 0
    total_mutation_checks: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    algorithms_tested: list[str] = field(default_factory=list)
    mutation_types_tested: list[str] = field(default_factory=list)
    false_negative_details: list[dict] = field(default_factory=list)
    false_positive_details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EdgeCaseEntry:
    """Single edge case result."""
    name: str
    passed: bool
    error: str | None = None


@dataclass
class IntegrityEdgeCaseResult:
    """Results from edge case testing."""
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    cases: list[EdgeCaseEntry] = field(default_factory=list)

    @property
    def failure_details(self) -> list[str]:
        return [f"{c.name}: {c.error}" for c in self.cases if not c.passed]

    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "cases": [asdict(c) for c in self.cases],
        }


@dataclass
class LatencyMeasurement:
    """Single latency measurement for one (size, algorithm, operation) combo."""
    size_bytes: int
    algorithm: str
    operation: str  # "compute" or "verify"
    mean_us: float = 0.0
    p50_us: float = 0.0
    p95_us: float = 0.0
    p99_us: float = 0.0
    ci_lower_us: float = 0.0
    ci_upper_us: float = 0.0
    n_trials: int = 0


@dataclass
class IntegrityLatencyResult:
    """Results from latency benchmarking."""
    measurements: list[LatencyMeasurement] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"measurements": [asdict(m) for m in self.measurements]}


# =====================================================================
# Context generators
# =====================================================================

def generate_context_by_size(target_bytes: int, seed: int = 42) -> str:
    """Generate a JSON context string of approximately target_bytes size.

    Creates a dict with enough keys to reach the target size,
    using schema.org-like vocabulary patterns.
    """
    rng = random.Random(seed)
    ctx: dict[str, str] = {}
    domains = [
        "http://schema.org/", "http://xmlns.com/foaf/0.1/",
        "http://purl.org/dc/terms/", "http://www.w3.org/2004/02/skos/core#",
    ]
    current = json.dumps(ctx, sort_keys=True)
    counter = 0
    while len(current.encode("utf-8")) < target_bytes:
        key = f"prop_{counter:06d}"
        domain = rng.choice(domains)
        value = f"{domain}{key}"
        ctx[key] = value
        current = json.dumps(ctx, sort_keys=True)
        counter += 1
    return current


def generate_realistic_jsonldex_context() -> dict[str, Any]:
    """Generate a context resembling real jsonld-ex usage."""
    return {
        "@vocab": "http://schema.org/",
        "confidence": "https://w3id.org/jsonld-ex/confidence",
        "source": "https://w3id.org/jsonld-ex/source",
        "extractedAt": "https://w3id.org/jsonld-ex/extractedAt",
        "validFrom": "https://w3id.org/jsonld-ex/validFrom",
        "validUntil": "https://w3id.org/jsonld-ex/validUntil",
        "vector": "https://w3id.org/jsonld-ex/vector",
        "integrity": "https://w3id.org/jsonld-ex/integrity",
        "name": "http://schema.org/name",
        "description": "http://schema.org/description",
        "dateCreated": "http://schema.org/dateCreated",
    }


def _random_context(rng: random.Random, depth: int = 0) -> dict[str, Any]:
    """Generate a random JSON-serializable context for PBT."""
    n_keys = rng.randint(1, 8)
    ctx: dict[str, Any] = {}
    for i in range(n_keys):
        key = f"k{''.join(rng.choices(string.ascii_lowercase, k=4))}_{i}"
        vtype = rng.choice(["str", "int", "float", "bool", "null", "nested"])
        if vtype == "str":
            ctx[key] = "".join(rng.choices(string.ascii_letters + string.digits, k=rng.randint(1, 30)))
        elif vtype == "int":
            ctx[key] = rng.randint(-10000, 10000)
        elif vtype == "float":
            ctx[key] = round(rng.uniform(-1000, 1000), 6)
        elif vtype == "bool":
            ctx[key] = rng.choice([True, False])
        elif vtype == "null":
            ctx[key] = None
        elif vtype == "nested" and depth < 3:
            ctx[key] = _random_context(rng, depth + 1)
        else:
            ctx[key] = rng.randint(0, 100)
    return ctx


# =====================================================================
# Mutation functions
# =====================================================================

def mutate_context_string(ctx: str, mutation: MutationType,
                          seed: int | None = None) -> str:
    """Apply a mutation to a JSON context string. Returns a different string."""
    rng = random.Random(seed)
    if not ctx:
        # Edge case: empty string, just append something
        return " "

    if mutation == MutationType.BYTE_FLIP:
        chars = list(ctx)
        idx = rng.randint(0, len(chars) - 1)
        # Flip to a different character
        old = chars[idx]
        while chars[idx] == old:
            chars[idx] = chr((ord(old) + rng.randint(1, 25)) % 128)
        return "".join(chars)

    elif mutation == MutationType.WHITESPACE_INJECT:
        idx = rng.randint(1, max(1, len(ctx) - 1))
        return ctx[:idx] + " " + ctx[idx:]

    elif mutation == MutationType.TRUNCATE:
        cut = max(1, len(ctx) - rng.randint(1, max(1, len(ctx) // 4)))
        return ctx[:cut]

    elif mutation == MutationType.KEY_RENAME:
        # Try to parse, rename a key, re-serialize
        try:
            d = json.loads(ctx)
            if isinstance(d, dict) and d:
                key = rng.choice(list(d.keys()))
                val = d.pop(key)
                d[key + "_mutated"] = val
                return json.dumps(d, sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: byte flip
        return mutate_context_string(ctx, MutationType.BYTE_FLIP, seed)

    else:
        # For dict-oriented mutations on a string, parse -> mutate -> re-serialize
        try:
            d = json.loads(ctx)
            if isinstance(d, dict):
                mutated = mutate_context_dict(d, mutation, seed)
                return json.dumps(mutated, sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback for non-parseable strings
        return mutate_context_string(ctx, MutationType.BYTE_FLIP, seed)


def mutate_context_dict(ctx: dict, mutation: MutationType,
                        seed: int | None = None) -> dict:
    """Apply a mutation to a context dict. Returns a different dict."""
    rng = random.Random(seed)
    result = copy.deepcopy(ctx)

    if not result:
        # Empty dict: can only insert
        result["_injected"] = "malicious"
        return result

    if mutation == MutationType.KEY_INSERT:
        new_key = f"_injected_{rng.randint(0, 99999)}"
        result[new_key] = "http://evil.org/injected"
        return result

    elif mutation == MutationType.KEY_DELETE:
        key = rng.choice(list(result.keys()))
        del result[key]
        return result

    elif mutation == MutationType.VALUE_SUBSTITUTE:
        key = rng.choice(list(result.keys()))
        old = result[key]
        if isinstance(old, str):
            result[key] = old + "_tampered"
        elif isinstance(old, (int, float)):
            result[key] = old + 1
        elif isinstance(old, bool):
            result[key] = not old
        elif old is None:
            result[key] = "not_null"
        elif isinstance(old, dict):
            result[key] = {"tampered": True}
        else:
            result[key] = "tampered"
        return result

    elif mutation == MutationType.KEY_RENAME:
        key = rng.choice(list(result.keys()))
        val = result.pop(key)
        result[key + "_renamed"] = val
        return result

    elif mutation == MutationType.VALUE_RETYPE:
        key = rng.choice(list(result.keys()))
        old = result[key]
        if isinstance(old, str):
            result[key] = len(old)  # str -> int
        elif isinstance(old, (int, float)):
            result[key] = str(old)  # number -> str
        elif isinstance(old, bool):
            result[key] = str(old)
        elif old is None:
            result[key] = 0
        else:
            result[key] = str(old)
        return result

    elif mutation in (MutationType.BYTE_FLIP, MutationType.WHITESPACE_INJECT,
                      MutationType.TRUNCATE):
        # These are string-level mutations; apply via serialize -> mutate -> parse
        serialized = json.dumps(result, sort_keys=True)
        mutated_str = mutate_context_string(serialized, mutation, seed)
        try:
            return json.loads(mutated_str)
        except json.JSONDecodeError:
            # Truncation or flip may break JSON; fall back to value substitution
            return mutate_context_dict(ctx, MutationType.VALUE_SUBSTITUTE, seed)

    # Fallback
    return mutate_context_dict(ctx, MutationType.VALUE_SUBSTITUTE, seed)


# =====================================================================
# PBT runner
# =====================================================================

def run_pbt_integrity_check(
    n_examples: int = PBT_EXAMPLES_PER_ALGORITHM,
    algorithms: list[str] | tuple[str, ...] = SUPPORTED_ALGORITHMS,
    mutation_types: list[MutationType] = ALL_MUTATION_TYPES,
    seed: int = 42,
) -> IntegrityPBTResult:
    """Run property-based integrity verification tests.

    For each random context:
      1. Compute hash -> verify round-trip (no false positives)
      2. For each mutation type: mutate -> verify fails (no false negatives)
    """
    rng = random.Random(seed)
    result = IntegrityPBTResult(
        algorithms_tested=list(algorithms),
        mutation_types_tested=[m.value for m in mutation_types],
    )

    for alg in algorithms:
        for i in range(n_examples):
            ctx = _random_context(rng)
            h = compute_integrity(ctx, algorithm=alg)

            # Round-trip check (H5.1b: no false positives)
            if not verify_integrity(ctx, h):
                result.false_positives += 1
                result.false_positive_details.append({
                    "algorithm": alg, "example": i,
                    "context_repr": repr(ctx)[:200],
                })
            result.total_roundtrip_checks += 1

            # Mutation checks (H5.1a: no false negatives)
            for mt in mutation_types:
                mutated = mutate_context_dict(copy.deepcopy(ctx), mt, seed=seed + i)
                if mutated == ctx:
                    # Mutation didn't change anything (rare edge case); skip
                    continue
                if verify_integrity(mutated, h):
                    result.false_negatives += 1
                    result.false_negative_details.append({
                        "algorithm": alg, "example": i,
                        "mutation": mt.value,
                        "original_repr": repr(ctx)[:200],
                        "mutated_repr": repr(mutated)[:200],
                    })
                result.total_mutation_checks += 1

    return result


# =====================================================================
# Edge case suite
# =====================================================================

def run_edge_case_suite() -> IntegrityEdgeCaseResult:
    """Run hand-crafted edge cases from EN5.1 Phase 2."""
    result = IntegrityEdgeCaseResult()
    cases: list[EdgeCaseEntry] = []

    def _run(name: str, fn):
        try:
            fn()
            cases.append(EdgeCaseEntry(name=name, passed=True))
        except Exception as exc:
            cases.append(EdgeCaseEntry(name=name, passed=False, error=str(exc)))

    # 1. Empty string round-trip
    def ec_empty_string():
        h = compute_integrity("")
        assert verify_integrity("", h), "Empty string round-trip failed"
    _run("empty_string_roundtrip", ec_empty_string)

    # 2. Empty dict round-trip
    def ec_empty_dict():
        h = compute_integrity({})
        assert verify_integrity({}, h), "Empty dict round-trip failed"
    _run("empty_dict_roundtrip", ec_empty_dict)

    # 3. Large context (1MB)
    def ec_large():
        ctx = generate_context_by_size(1_000_000)
        h = compute_integrity(ctx)
        assert verify_integrity(ctx, h), "1MB context round-trip failed"
    _run("large_context_1mb", ec_large)

    # 4. Unicode
    def ec_unicode():
        ctx = {"日本語": "値", "emoji": "data", "clef": "treble"}
        h = compute_integrity(ctx)
        assert verify_integrity(ctx, h), "Unicode round-trip failed"
    _run("unicode_roundtrip", ec_unicode)

    # 5. Nested 50 levels
    def ec_nested():
        ctx: dict[str, Any] = {"leaf": "value"}
        for _ in range(50):
            ctx = {"nested": ctx}
        h = compute_integrity(ctx)
        assert verify_integrity(ctx, h), "50-level nested round-trip failed"
    _run("nested_50_levels", ec_nested)

    # 6. Float determinism
    def ec_float():
        ctx = {"value": 0.1, "other": 0.2, "sum": 0.3}
        h1 = compute_integrity(ctx)
        h2 = compute_integrity(ctx)
        assert h1 == h2, f"Float non-deterministic: {h1} != {h2}"
    _run("float_determinism", ec_float)

    # 7. Key order invariance
    def ec_key_order():
        a = {"z": 1, "a": 2, "m": 3}
        b = {"a": 2, "m": 3, "z": 1}
        ha = compute_integrity(a)
        hb = compute_integrity(b)
        assert ha == hb, f"Key order not invariant: {ha} != {hb}"
    _run("key_order_invariance", ec_key_order)

    # 8. Single bit flip detected
    def ec_bit_flip():
        ctx = '{"key": "value_for_testing_integrity"}'
        h = compute_integrity(ctx)
        mutated = mutate_context_string(ctx, MutationType.BYTE_FLIP, seed=99)
        assert not verify_integrity(mutated, h), "Bit flip not detected"
    _run("single_bit_flip_detected", ec_bit_flip)

    # 9. Appended whitespace detected
    def ec_whitespace():
        ctx = '{"key": "value"}'
        h = compute_integrity(ctx)
        assert not verify_integrity(ctx + " ", h), "Appended whitespace not detected"
    _run("appended_whitespace_detected", ec_whitespace)

    # 10. None raises TypeError
    def ec_none():
        try:
            compute_integrity(None)
            raise AssertionError("None did not raise TypeError")
        except TypeError:
            pass  # Expected
    _run("none_raises_typeerror", ec_none)

    # 11. Non-serializable raises TypeError
    def ec_non_serializable():
        try:
            compute_integrity({"key": object()})
            raise AssertionError("Non-serializable did not raise TypeError")
        except TypeError:
            pass  # Expected
    _run("non_serializable_raises_typeerror", ec_non_serializable)

    # 12. Boolean and null values
    def ec_bool_null():
        ctx = {"active": True, "deprecated": False, "notes": None}
        h = compute_integrity(ctx)
        assert verify_integrity(ctx, h), "Bool/null round-trip failed"
    _run("boolean_null_roundtrip", ec_bool_null)

    # 13. List values
    def ec_list_values():
        ctx = {"items": [1, "two", 3.0, True, None], "count": 5}
        h = compute_integrity(ctx)
        assert verify_integrity(ctx, h), "List values round-trip failed"
    _run("list_values_roundtrip", ec_list_values)

    result.cases = cases
    result.total_cases = len(cases)
    result.passed = sum(1 for c in cases if c.passed)
    result.failed = sum(1 for c in cases if not c.passed)
    return result


# =====================================================================
# Latency benchmark
# =====================================================================

def run_latency_benchmark(
    sizes: list[int] | None = None,
    algorithms: list[str] | tuple[str, ...] | None = None,
    n_trials: int = LATENCY_TRIALS,
    n_warmup: int = LATENCY_WARMUP,
) -> IntegrityLatencyResult:
    """Benchmark compute_integrity and verify_integrity latency.

    Returns measurements with mean, p50, p95, p99, and bootstrap 95% CI.
    """
    if sizes is None:
        sizes = CONTEXT_SIZE_TARGETS
    if algorithms is None:
        algorithms = list(SUPPORTED_ALGORITHMS)

    result = IntegrityLatencyResult()

    for size in sizes:
        ctx_str = generate_context_by_size(size)
        actual_size = len(ctx_str.encode("utf-8"))

        for alg in algorithms:
            # Pre-compute hash for verify benchmarks
            h = compute_integrity(ctx_str, algorithm=alg)

            # --- Benchmark compute_integrity ---
            # Warm up
            for _ in range(n_warmup):
                compute_integrity(ctx_str, algorithm=alg)

            compute_times = []
            for _ in range(n_trials):
                t0 = time.perf_counter_ns()
                compute_integrity(ctx_str, algorithm=alg)
                t1 = time.perf_counter_ns()
                compute_times.append((t1 - t0) / 1000.0)  # ns -> us

            arr = np.array(compute_times)
            ci_lo, _, ci_hi = bootstrap_ci(arr)
            result.measurements.append(LatencyMeasurement(
                size_bytes=actual_size,
                algorithm=alg,
                operation="compute",
                mean_us=float(np.mean(arr)),
                p50_us=float(np.percentile(arr, 50)),
                p95_us=float(np.percentile(arr, 95)),
                p99_us=float(np.percentile(arr, 99)),
                ci_lower_us=ci_lo,
                ci_upper_us=ci_hi,
                n_trials=n_trials,
            ))

            # --- Benchmark verify_integrity ---
            for _ in range(n_warmup):
                verify_integrity(ctx_str, h)

            verify_times = []
            for _ in range(n_trials):
                t0 = time.perf_counter_ns()
                verify_integrity(ctx_str, h)
                t1 = time.perf_counter_ns()
                verify_times.append((t1 - t0) / 1000.0)

            arr = np.array(verify_times)
            ci_lo, _, ci_hi = bootstrap_ci(arr)
            result.measurements.append(LatencyMeasurement(
                size_bytes=actual_size,
                algorithm=alg,
                operation="verify",
                mean_us=float(np.mean(arr)),
                p50_us=float(np.percentile(arr, 50)),
                p95_us=float(np.percentile(arr, 95)),
                p99_us=float(np.percentile(arr, 99)),
                ci_lower_us=ci_lo,
                ci_upper_us=ci_hi,
                n_trials=n_trials,
            ))

    return result
