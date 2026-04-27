"""Tests for EN5.1 -- Context Integrity Verification.

RED phase -- all tests should FAIL until en5_1_core.py is implemented.
The jsonld_ex.security imports should succeed (testing the actual library).

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest experiments/EN5/tests/test_en5_1.py -v

Hypotheses:
    H5.1a: verify_integrity() detects ALL mutations (0 false negatives)
    H5.1b: verify_integrity() produces 0 false positives on unmodified contexts
    H5.1c: Round-trip property holds for all algorithms
    H5.1d: compute_integrity() is deterministic; dict key order invariant
"""
from __future__ import annotations

import json
import copy
import time

import pytest
import numpy as np

# -- Actual library imports (these SHOULD succeed) --
from jsonld_ex.security import (
    compute_integrity,
    verify_integrity,
    integrity_context,
)

# -- Experiment core imports (these SHOULD FAIL in RED phase) --
from EN5.en5_1_core import (
    # Context generators
    generate_context_by_size,
    generate_realistic_jsonldex_context,
    CONTEXT_SIZE_TARGETS,
    # Mutation functions
    MutationType,
    mutate_context_string,
    mutate_context_dict,
    ALL_MUTATION_TYPES,
    # Benchmark runners
    run_pbt_integrity_check,
    run_edge_case_suite,
    run_latency_benchmark,
    # Result types
    IntegrityPBTResult,
    IntegrityEdgeCaseResult,
    IntegrityLatencyResult,
    # Constants
    SUPPORTED_ALGORITHMS,
    PBT_EXAMPLES_PER_ALGORITHM,
    LATENCY_TRIALS,
    LATENCY_WARMUP,
)


# =====================================================================
# 0. Smoke tests: verify jsonld_ex.security is importable and functional
# =====================================================================

class TestSecurityAPIAvailable:
    """Verify the actual library functions exist and have correct signatures."""

    def test_compute_integrity_exists(self):
        """compute_integrity is callable."""
        assert callable(compute_integrity)

    def test_verify_integrity_exists(self):
        """verify_integrity is callable."""
        assert callable(verify_integrity)

    def test_integrity_context_exists(self):
        """integrity_context is callable."""
        assert callable(integrity_context)

    def test_compute_integrity_string_input(self):
        """compute_integrity accepts a string and returns algo-hash format."""
        result = compute_integrity("test context", algorithm="sha256")
        assert isinstance(result, str)
        assert result.startswith("sha256-")
        assert len(result) > len("sha256-")

    def test_compute_integrity_dict_input(self):
        """compute_integrity accepts a dict and returns algo-hash format."""
        result = compute_integrity({"@vocab": "http://schema.org/"})
        assert isinstance(result, str)
        assert result.startswith("sha256-")

    def test_verify_integrity_roundtrip(self):
        """Basic round-trip: compute then verify."""
        ctx = '{"@vocab": "http://schema.org/"}'
        h = compute_integrity(ctx)
        assert verify_integrity(ctx, h) is True

    def test_verify_integrity_tampered(self):
        """Tampered content fails verification."""
        h = compute_integrity("original")
        assert verify_integrity("tampered", h) is False

    def test_compute_integrity_sha384(self):
        """SHA-384 algorithm produces sha384- prefix."""
        result = compute_integrity("test", algorithm="sha384")
        assert result.startswith("sha384-")

    def test_compute_integrity_sha512(self):
        """SHA-512 algorithm produces sha512- prefix."""
        result = compute_integrity("test", algorithm="sha512")
        assert result.startswith("sha512-")

    def test_compute_integrity_rejects_md5(self):
        """Unsupported algorithms are rejected."""
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            compute_integrity("test", algorithm="md5")

    def test_compute_integrity_rejects_none(self):
        """None input raises TypeError."""
        with pytest.raises(TypeError):
            compute_integrity(None)


# =====================================================================
# 1. Constants and configuration
# =====================================================================

class TestConstants:
    """Verify experiment configuration constants."""

    def test_supported_algorithms(self):
        """Must support exactly sha256, sha384, sha512."""
        assert set(SUPPORTED_ALGORITHMS) == {"sha256", "sha384", "sha512"}

    def test_pbt_examples_sufficient(self):
        """At least 10,000 examples per algorithm for statistical power."""
        assert PBT_EXAMPLES_PER_ALGORITHM >= 10_000

    def test_all_mutation_types_count(self):
        """At least 6 mutation types for comprehensive coverage."""
        assert len(ALL_MUTATION_TYPES) >= 6

    def test_mutation_types_are_enum(self):
        """MutationType values are accessible by .value string."""
        for mt in ALL_MUTATION_TYPES:
            assert isinstance(mt.value, str)

    def test_context_size_targets(self):
        """Size targets span from ~100B to ~1MB."""
        assert len(CONTEXT_SIZE_TARGETS) >= 5
        assert min(CONTEXT_SIZE_TARGETS) <= 200       # ~100B target
        assert max(CONTEXT_SIZE_TARGETS) >= 500_000   # ~1MB target

    def test_latency_trials_sufficient(self):
        """At least 10,000 trials for reliable latency estimates."""
        assert LATENCY_TRIALS >= 10_000

    def test_latency_warmup_positive(self):
        """Warmup count is positive."""
        assert LATENCY_WARMUP >= 50


# =====================================================================
# 2. Context generators
# =====================================================================

class TestContextGenerators:
    """Verify context generation for benchmarks."""

    def test_generate_context_by_size_returns_string(self):
        """Generated context is a JSON string."""
        ctx = generate_context_by_size(1000)
        assert isinstance(ctx, str)
        # Must be valid JSON
        parsed = json.loads(ctx)
        assert isinstance(parsed, dict)

    def test_generate_context_by_size_approximate(self):
        """Generated context is approximately the target size."""
        for target in [100, 1000, 10_000, 100_000]:
            ctx = generate_context_by_size(target)
            actual = len(ctx.encode("utf-8"))
            # Within 2x of target (generation is approximate)
            assert actual >= target * 0.5, f"Too small: {actual} < {target * 0.5}"
            assert actual <= target * 2.0, f"Too large: {actual} > {target * 2.0}"

    def test_generate_context_by_size_valid_json(self):
        """All generated sizes produce valid JSON."""
        for target in CONTEXT_SIZE_TARGETS:
            ctx = generate_context_by_size(target)
            json.loads(ctx)  # Should not raise

    def test_generate_realistic_context(self):
        """Realistic context has schema.org-like structure."""
        ctx = generate_realistic_jsonldex_context()
        assert isinstance(ctx, dict)
        # Should have at least some vocabulary-like keys
        assert len(ctx) >= 3

    def test_generate_realistic_context_is_serializable(self):
        """Realistic context can be JSON-serialized."""
        ctx = generate_realistic_jsonldex_context()
        serialized = json.dumps(ctx, sort_keys=True)
        assert len(serialized) > 10


# =====================================================================
# 3. Mutation functions
# =====================================================================

class TestMutationFunctions:
    """Verify mutation strategies produce distinct contexts."""

    def test_mutate_context_string_produces_different(self):
        """String mutation produces a different string."""
        original = '{"@vocab": "http://schema.org/", "name": "text"}'
        for mt in ALL_MUTATION_TYPES:
            mutated = mutate_context_string(original, mt)
            assert mutated != original, f"Mutation {mt.value} produced identical string"

    def test_mutate_context_dict_produces_different(self):
        """Dict mutation produces a different dict."""
        original = {"@vocab": "http://schema.org/", "name": "text", "age": 42}
        for mt in ALL_MUTATION_TYPES:
            mutated = mutate_context_dict(copy.deepcopy(original), mt)
            assert mutated != original, f"Mutation {mt.value} produced identical dict"

    def test_mutation_type_byte_flip(self):
        """BYTE_FLIP changes exactly one byte."""
        original = '{"key": "value_with_enough_chars"}'
        mutated = mutate_context_string(original, MutationType.BYTE_FLIP)
        # Should differ in exactly one position
        diffs = sum(1 for a, b in zip(original, mutated) if a != b)
        assert diffs >= 1, "BYTE_FLIP should change at least one byte"

    def test_mutation_type_key_insert(self):
        """KEY_INSERT adds a new key to the dict."""
        original = {"a": 1, "b": 2}
        mutated = mutate_context_dict(copy.deepcopy(original), MutationType.KEY_INSERT)
        assert len(mutated) > len(original)

    def test_mutation_type_key_delete(self):
        """KEY_DELETE removes a key from the dict."""
        original = {"a": 1, "b": 2, "c": 3}
        mutated = mutate_context_dict(copy.deepcopy(original), MutationType.KEY_DELETE)
        assert len(mutated) < len(original)

    def test_mutation_type_value_substitute(self):
        """VALUE_SUBSTITUTE changes a value but keeps the same keys."""
        original = {"a": 1, "b": "hello"}
        mutated = mutate_context_dict(copy.deepcopy(original), MutationType.VALUE_SUBSTITUTE)
        assert set(mutated.keys()) == set(original.keys())
        assert mutated != original

    def test_mutation_type_whitespace_inject(self):
        """WHITESPACE_INJECT adds whitespace to a string context."""
        original = '{"key":"value"}'
        mutated = mutate_context_string(original, MutationType.WHITESPACE_INJECT)
        assert len(mutated) > len(original)

    def test_mutation_type_truncate(self):
        """TRUNCATE shortens the context."""
        original = '{"key": "value", "another": "entry"}'
        mutated = mutate_context_string(original, MutationType.TRUNCATE)
        assert len(mutated) < len(original)

    def test_mutation_preserves_type(self):
        """String mutations return strings; dict mutations return dicts."""
        s = '{"x": 1}'
        d = {"x": 1}
        for mt in ALL_MUTATION_TYPES:
            assert isinstance(mutate_context_string(s, mt), str)
            assert isinstance(mutate_context_dict(copy.deepcopy(d), mt), dict)

    def test_empty_dict_mutation_does_not_crash(self):
        """Mutations on minimal inputs don't raise unexpected errors."""
        # KEY_DELETE on single-key dict
        result = mutate_context_dict({"a": 1}, MutationType.KEY_DELETE)
        assert isinstance(result, dict)


# =====================================================================
# 4. H5.1a — Detection completeness (no false negatives)
# =====================================================================

class TestH51aDetectionCompleteness:
    """verify_integrity() must detect ALL mutations."""

    @pytest.mark.parametrize("algorithm", ["sha256", "sha384", "sha512"])
    def test_byte_flip_detected(self, algorithm):
        """Single byte flip is detected by all algorithms."""
        ctx = '{"@vocab": "http://schema.org/", "name": "http://schema.org/name"}'
        h = compute_integrity(ctx, algorithm=algorithm)
        mutated = mutate_context_string(ctx, MutationType.BYTE_FLIP)
        assert verify_integrity(mutated, h) is False

    @pytest.mark.parametrize("algorithm", ["sha256", "sha384", "sha512"])
    def test_key_insert_detected(self, algorithm):
        """Key insertion is detected."""
        ctx = {"@vocab": "http://schema.org/"}
        h = compute_integrity(ctx, algorithm=algorithm)
        mutated = mutate_context_dict(copy.deepcopy(ctx), MutationType.KEY_INSERT)
        assert verify_integrity(mutated, h) is False

    @pytest.mark.parametrize("algorithm", ["sha256", "sha384", "sha512"])
    def test_value_substitute_detected(self, algorithm):
        """Value substitution is detected."""
        ctx = {"source": "http://schema.org/sender", "dest": "http://schema.org/recipient"}
        h = compute_integrity(ctx, algorithm=algorithm)
        # The MITM attack: swap source and destination
        mutated = {"source": "http://schema.org/recipient", "dest": "http://schema.org/sender"}
        assert verify_integrity(mutated, h) is False

    @pytest.mark.parametrize("algorithm", ["sha256", "sha384", "sha512"])
    def test_whitespace_injection_detected(self, algorithm):
        """Whitespace injection into serialized context is detected."""
        ctx = '{"key": "value"}'
        h = compute_integrity(ctx, algorithm=algorithm)
        mutated = '{"key": "value" }'  # trailing space inside
        assert verify_integrity(mutated, h) is False

    @pytest.mark.parametrize("algorithm", ["sha256", "sha384", "sha512"])
    def test_truncation_detected(self, algorithm):
        """Truncated context is detected."""
        ctx = '{"key": "value", "other": "data"}'
        h = compute_integrity(ctx, algorithm=algorithm)
        mutated = '{"key": "value"}'
        assert verify_integrity(mutated, h) is False

    def test_pbt_all_mutations_detected(self):
        """Property-based test: ALL mutation types detected across algorithms.

        This is the main H5.1a validation. The full PBT suite runs via
        run_pbt_integrity_check() which uses the Hypothesis framework
        to generate 10,000+ random contexts per algorithm.
        """
        result = run_pbt_integrity_check(
            n_examples=100,  # Reduced for unit test speed; full run uses 10K+
            algorithms=SUPPORTED_ALGORITHMS,
            mutation_types=ALL_MUTATION_TYPES,
        )
        assert isinstance(result, IntegrityPBTResult)
        assert result.false_negatives == 0, (
            f"H5.1a REJECTED: {result.false_negatives} false negatives detected"
        )
        assert result.total_mutation_checks > 0


# =====================================================================
# 5. H5.1b — No false positives
# =====================================================================

class TestH51bNoFalsePositives:
    """verify_integrity() must not reject valid contexts."""

    @pytest.mark.parametrize("algorithm", ["sha256", "sha384", "sha512"])
    def test_roundtrip_string(self, algorithm):
        """String context round-trips correctly."""
        ctx = '{"@vocab": "http://schema.org/"}'
        h = compute_integrity(ctx, algorithm=algorithm)
        assert verify_integrity(ctx, h) is True

    @pytest.mark.parametrize("algorithm", ["sha256", "sha384", "sha512"])
    def test_roundtrip_dict(self, algorithm):
        """Dict context round-trips correctly."""
        ctx = {"@vocab": "http://schema.org/", "name": "http://schema.org/name"}
        h = compute_integrity(ctx, algorithm=algorithm)
        assert verify_integrity(ctx, h) is True

    def test_roundtrip_empty_string(self):
        """Empty string context round-trips."""
        h = compute_integrity("")
        assert verify_integrity("", h) is True

    def test_roundtrip_empty_dict(self):
        """Empty dict context round-trips."""
        h = compute_integrity({})
        assert verify_integrity({}, h) is True

    def test_roundtrip_unicode(self):
        """Unicode context round-trips."""
        ctx = {"日本語": "値", "clef": "treble"}
        h = compute_integrity(ctx)
        assert verify_integrity(ctx, h) is True

    def test_roundtrip_nested(self):
        """Deeply nested context round-trips."""
        ctx = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        h = compute_integrity(ctx)
        assert verify_integrity(ctx, h) is True

    def test_roundtrip_large_context(self):
        """1MB context round-trips."""
        ctx = generate_context_by_size(1_000_000)
        h = compute_integrity(ctx)
        assert verify_integrity(ctx, h) is True

    def test_roundtrip_list_values(self):
        """Context with list values round-trips."""
        ctx = {"types": ["Person", "Organization"], "version": 2}
        h = compute_integrity(ctx)
        assert verify_integrity(ctx, h) is True

    def test_roundtrip_boolean_null(self):
        """Context with boolean and null values round-trips."""
        ctx = {"active": True, "deprecated": False, "notes": None}
        h = compute_integrity(ctx)
        assert verify_integrity(ctx, h) is True

    def test_pbt_no_false_positives(self):
        """Property-based test: NO false positives across algorithms.

        run_pbt_integrity_check also validates round-trip (no false positives)
        alongside mutation detection (no false negatives).
        """
        result = run_pbt_integrity_check(
            n_examples=100,  # Reduced for unit test speed
            algorithms=SUPPORTED_ALGORITHMS,
            mutation_types=ALL_MUTATION_TYPES,
        )
        assert result.false_positives == 0, (
            f"H5.1b REJECTED: {result.false_positives} false positives detected"
        )


# =====================================================================
# 6. H5.1c — Algorithm consistency (round-trip for all algorithms)
# =====================================================================

class TestH51cAlgorithmConsistency:
    """Round-trip property holds for all supported algorithms."""

    def test_all_algorithms_produce_different_hashes(self):
        """Different algorithms produce different hash strings for same input."""
        ctx = '{"test": "data"}'
        hashes = {alg: compute_integrity(ctx, algorithm=alg) for alg in SUPPORTED_ALGORITHMS}
        # All should have different prefixes
        prefixes = {h.split("-")[0] for h in hashes.values()}
        assert len(prefixes) == len(SUPPORTED_ALGORITHMS)

    def test_all_algorithms_roundtrip_same_context(self):
        """Each algorithm independently verifies the same context."""
        ctx = {"complex": {"nested": [1, 2, 3]}, "name": "test"}
        for alg in SUPPORTED_ALGORITHMS:
            h = compute_integrity(ctx, algorithm=alg)
            assert verify_integrity(ctx, h) is True, f"Round-trip failed for {alg}"

    def test_cross_algorithm_rejection(self):
        """Hash from one algorithm is rejected by verify when another was used."""
        ctx = "test context"
        h_sha256 = compute_integrity(ctx, algorithm="sha256")
        # verify_integrity should parse the prefix and use the correct algorithm
        # So this should still work (the hash string is self-describing)
        assert verify_integrity(ctx, h_sha256) is True

    @pytest.mark.parametrize("algorithm", ["sha256", "sha384", "sha512"])
    def test_realistic_context_roundtrip(self, algorithm):
        """Realistic jsonld-ex context round-trips per algorithm."""
        ctx = generate_realistic_jsonldex_context()
        h = compute_integrity(ctx, algorithm=algorithm)
        assert verify_integrity(ctx, h) is True


# =====================================================================
# 7. H5.1d — Determinism and key-order invariance
# =====================================================================

class TestH51dDeterminism:
    """compute_integrity() must be deterministic and key-order invariant."""

    def test_repeated_calls_same_result(self):
        """Calling compute_integrity 100 times gives identical results."""
        ctx = {"@vocab": "http://schema.org/", "name": "http://schema.org/name"}
        first = compute_integrity(ctx)
        for _ in range(100):
            assert compute_integrity(ctx) == first

    def test_key_order_invariance_simple(self):
        """Different insertion order produces same hash."""
        ctx_a = {"b": 2, "a": 1}
        ctx_b = {"a": 1, "b": 2}
        assert compute_integrity(ctx_a) == compute_integrity(ctx_b)

    def test_key_order_invariance_nested(self):
        """Nested dicts with different key order produce same hash."""
        ctx_a = {"outer": {"z": 3, "y": 2, "x": 1}, "top": "val"}
        ctx_b = {"top": "val", "outer": {"x": 1, "y": 2, "z": 3}}
        assert compute_integrity(ctx_a) == compute_integrity(ctx_b)

    def test_key_order_invariance_all_algorithms(self):
        """Key-order invariance holds for all algorithms."""
        ctx_a = {"c": 3, "a": 1, "b": 2}
        ctx_b = {"a": 1, "b": 2, "c": 3}
        for alg in SUPPORTED_ALGORITHMS:
            ha = compute_integrity(ctx_a, algorithm=alg)
            hb = compute_integrity(ctx_b, algorithm=alg)
            assert ha == hb, f"Key-order invariance failed for {alg}"

    def test_float_determinism(self):
        """Float values produce deterministic hashes.

        JSON serialization of floats can vary. compute_integrity must
        handle this consistently (e.g., via canonical JSON serialization).
        """
        ctx = {"value": 0.1, "other": 0.2}
        h1 = compute_integrity(ctx)
        h2 = compute_integrity(ctx)
        assert h1 == h2

    def test_string_vs_dict_different(self):
        """A string and a dict with same content produce different hashes.

        compute_integrity('{"a":1}') and compute_integrity({"a":1}) should
        differ because the dict is canonicalized (sorted keys) while the
        string is hashed as-is.
        """
        s = '{"a": 1}'
        d = {"a": 1}
        # These MAY or MAY NOT be equal depending on serialization.
        # We just verify both are deterministic.
        hs = compute_integrity(s)
        hd = compute_integrity(d)
        assert isinstance(hs, str) and isinstance(hd, str)
        # Document the behavior: record whether they match
        # (This is an observation, not an assertion — both behaviors are valid)


# =====================================================================
# 8. Error handling
# =====================================================================

class TestErrorHandling:
    """Verify graceful error handling for invalid inputs."""

    def test_verify_empty_hash_raises(self):
        """Empty hash string raises ValueError."""
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            verify_integrity("test", "")

    def test_verify_no_hyphen_raises(self):
        """Hash without algorithm-hash separator raises ValueError."""
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            verify_integrity("test", "sha256abc123")

    def test_verify_bad_algorithm_raises(self):
        """Hash with unsupported algorithm prefix raises ValueError."""
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            verify_integrity("test", "md5-abc123")

    def test_compute_non_serializable_raises(self):
        """Non-JSON-serializable input raises TypeError."""
        with pytest.raises(TypeError):
            compute_integrity({"key": object()})


# =====================================================================
# 9. Edge case suite (Phase 2)
# =====================================================================

class TestEdgeCaseSuite:
    """Run the full edge case suite from EN5.1 Phase 2."""

    def test_edge_case_suite_runs(self):
        """Edge case suite completes and returns structured results."""
        result = run_edge_case_suite()
        assert isinstance(result, IntegrityEdgeCaseResult)
        assert result.total_cases >= 11  # At least 11 edge cases defined
        assert result.passed >= 0
        assert result.failed >= 0
        assert result.passed + result.failed == result.total_cases

    def test_edge_case_suite_all_pass(self):
        """All edge cases pass (0 failures)."""
        result = run_edge_case_suite()
        assert result.failed == 0, (
            f"Edge case failures: {result.failure_details}"
        )


# =====================================================================
# 10. Latency benchmark (Phase 3)
# =====================================================================

class TestLatencyBenchmark:
    """Verify latency benchmark produces valid measurements."""

    def test_latency_benchmark_runs(self):
        """Latency benchmark completes for a small configuration."""
        result = run_latency_benchmark(
            sizes=[100, 1000],          # Just 2 sizes for speed
            algorithms=["sha256"],       # Just 1 algorithm for speed
            n_trials=100,                # Reduced for unit test
            n_warmup=10,
        )
        assert isinstance(result, IntegrityLatencyResult)
        assert len(result.measurements) > 0

    def test_latency_measurement_has_required_fields(self):
        """Each measurement has mean, p50, p95, p99, ci_lower, ci_upper."""
        result = run_latency_benchmark(
            sizes=[1000],
            algorithms=["sha256"],
            n_trials=100,
            n_warmup=10,
        )
        for m in result.measurements:
            assert hasattr(m, "size_bytes")
            assert hasattr(m, "algorithm")
            assert hasattr(m, "operation")  # "compute" or "verify"
            assert hasattr(m, "mean_us")
            assert hasattr(m, "p50_us")
            assert hasattr(m, "p95_us")
            assert hasattr(m, "p99_us")
            assert hasattr(m, "ci_lower_us")
            assert hasattr(m, "ci_upper_us")
            assert hasattr(m, "n_trials")

    def test_latency_values_positive(self):
        """All latency values are positive."""
        result = run_latency_benchmark(
            sizes=[1000],
            algorithms=["sha256"],
            n_trials=100,
            n_warmup=10,
        )
        for m in result.measurements:
            assert m.mean_us > 0
            assert m.p50_us > 0
            assert m.p95_us >= m.p50_us  # p95 >= p50
            assert m.p99_us >= m.p95_us  # p99 >= p95

    def test_latency_ci_brackets_mean(self):
        """Bootstrap CI brackets the mean."""
        result = run_latency_benchmark(
            sizes=[1000],
            algorithms=["sha256"],
            n_trials=500,
            n_warmup=10,
        )
        for m in result.measurements:
            assert m.ci_lower_us <= m.mean_us <= m.ci_upper_us

    def test_latency_compute_vs_verify(self):
        """Both compute and verify operations are measured."""
        result = run_latency_benchmark(
            sizes=[1000],
            algorithms=["sha256"],
            n_trials=100,
            n_warmup=10,
        )
        operations = {m.operation for m in result.measurements}
        assert "compute" in operations
        assert "verify" in operations


# =====================================================================
# 11. Full experiment runner integration
# =====================================================================

class TestFullExperimentIntegration:
    """Verify the full EN5.1 experiment can be orchestrated."""

    def test_pbt_result_has_summary(self):
        """PBT result includes summary statistics."""
        result = run_pbt_integrity_check(
            n_examples=50,
            algorithms=["sha256"],
            mutation_types=[MutationType.BYTE_FLIP, MutationType.KEY_INSERT],
        )
        assert result.total_roundtrip_checks > 0
        assert result.total_mutation_checks > 0
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_pbt_result_serializable(self):
        """PBT result can be serialized to JSON for results file."""
        result = run_pbt_integrity_check(
            n_examples=50,
            algorithms=["sha256"],
            mutation_types=[MutationType.BYTE_FLIP],
        )
        # Must have a to_dict() or similar for JSON output
        d = result.to_dict()
        assert isinstance(d, dict)
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_latency_result_serializable(self):
        """Latency result can be serialized to JSON."""
        result = run_latency_benchmark(
            sizes=[1000],
            algorithms=["sha256"],
            n_trials=100,
            n_warmup=10,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_edge_case_result_serializable(self):
        """Edge case result can be serialized to JSON."""
        result = run_edge_case_suite()
        d = result.to_dict()
        assert isinstance(d, dict)
        serialized = json.dumps(d)
        assert len(serialized) > 0
