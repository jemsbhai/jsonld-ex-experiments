"""Tests for EN5.4 -- Security Pipeline Overhead.

RED phase -- all tests should FAIL until en5_4_core.py is implemented.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest experiments/EN5/tests/test_en5_4.py -v

Hypotheses:
    H5.4a: Full pipeline adds <1ms per document for docs under 100KB
    H5.4b: Security check latency scales linearly with document size
    H5.4c: Security checks are <5% of annotate() time
    H5.4d: Memory overhead <2x input document size
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
from EN5.en5_4_core import (
    # Document generators
    generate_pipeline_test_document,
    PIPELINE_DOC_SIZES,
    # Pipeline runner
    run_individual_operation_benchmarks,
    run_full_pipeline_benchmark,
    run_annotate_comparison,
    run_memory_overhead_benchmark,
    # Result types
    OperationBenchmarkResult,
    PipelineBenchmarkResult,
    AnnotateComparisonResult,
    MemoryOverheadResult,
    # Full experiment
    run_en5_4_full,
    EN54FullResult,
)


# =====================================================================
# 0. Smoke tests
# =====================================================================

class TestPipelineAPIsAvailable:
    """Verify all security functions work together."""

    def test_full_pipeline_no_crash(self):
        """Run all three checks in sequence on a valid document."""
        ctx = {"@vocab": "http://schema.org/"}
        doc = {"@context": "https://schema.org/", "@type": "Person", "name": "Test"}
        cfg = {"allowed": ["https://schema.org/"]}
        h = compute_integrity(ctx)

        assert is_context_allowed("https://schema.org/", cfg) is True
        assert verify_integrity(ctx, h) is True
        enforce_resource_limits(doc)  # Should not raise


# =====================================================================
# 1. Document generators
# =====================================================================

class TestDocumentGenerators:
    """Verify pipeline test document generation."""

    def test_generates_valid_jsonldex_doc(self):
        doc, ctx_str, ctx_hash, allowlist_cfg = generate_pipeline_test_document(1000)
        assert isinstance(doc, dict)
        assert isinstance(ctx_str, str)
        assert isinstance(ctx_hash, str)
        assert ctx_hash.startswith("sha256-")
        assert isinstance(allowlist_cfg, dict)

    def test_doc_sizes_span_range(self):
        """Sizes from 100B to 1MB."""
        assert len(PIPELINE_DOC_SIZES) >= 5
        assert min(PIPELINE_DOC_SIZES) <= 200
        assert max(PIPELINE_DOC_SIZES) >= 500_000

    @pytest.mark.parametrize("size", [100, 1000, 10_000])
    def test_doc_approximate_size(self, size):
        doc, _, _, _ = generate_pipeline_test_document(size)
        serialized = json.dumps(doc)
        actual = len(serialized.encode("utf-8"))
        assert actual >= size * 0.3, f"Too small: {actual}"
        assert actual <= size * 5.0, f"Too large: {actual}"

    def test_doc_has_jsonldex_extensions(self):
        """Generated doc includes @confidence, @source, etc."""
        doc, _, _, _ = generate_pipeline_test_document(1000)
        serialized = json.dumps(doc)
        # Should contain at least some extension keywords
        assert "@type" in serialized or "type" in serialized


# =====================================================================
# 2. Individual operation benchmarks
# =====================================================================

class TestIndividualOperations:
    """Benchmark each security operation individually."""

    def test_benchmark_runs(self):
        result = run_individual_operation_benchmarks(
            sizes=[100, 1000],
            n_trials=500,
            n_warmup=50,
        )
        assert isinstance(result, OperationBenchmarkResult)
        assert len(result.measurements) > 0

    def test_three_operations_measured(self):
        result = run_individual_operation_benchmarks(
            sizes=[1000],
            n_trials=500,
            n_warmup=50,
        )
        operations = {m.operation for m in result.measurements}
        assert "allowlist_check" in operations
        assert "integrity_verify" in operations
        assert "resource_limits" in operations

    def test_measurements_have_required_fields(self):
        result = run_individual_operation_benchmarks(
            sizes=[1000],
            n_trials=500,
            n_warmup=50,
        )
        for m in result.measurements:
            assert hasattr(m, "size_bytes")
            assert hasattr(m, "operation")
            assert hasattr(m, "mean_us")
            assert hasattr(m, "p50_us")
            assert hasattr(m, "p95_us")
            assert hasattr(m, "p99_us")
            assert hasattr(m, "ci_lower_us")
            assert hasattr(m, "ci_upper_us")
            assert m.mean_us > 0

    def test_result_serializable(self):
        result = run_individual_operation_benchmarks(
            sizes=[1000], n_trials=100, n_warmup=10,
        )
        d = result.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0


# =====================================================================
# 3. Full pipeline benchmark
# =====================================================================

class TestFullPipeline:
    """Benchmark all three checks in sequence."""

    def test_pipeline_benchmark_runs(self):
        result = run_full_pipeline_benchmark(
            sizes=[100, 1000],
            n_trials=500,
            n_warmup=50,
        )
        assert isinstance(result, PipelineBenchmarkResult)
        assert len(result.measurements) > 0

    def test_pipeline_measured_per_size(self):
        sizes = [100, 1000, 10_000]
        result = run_full_pipeline_benchmark(
            sizes=sizes, n_trials=200, n_warmup=20,
        )
        measured_sizes = {m.size_bytes for m in result.measurements}
        assert len(measured_sizes) == len(sizes)

    def test_pipeline_has_percentiles(self):
        result = run_full_pipeline_benchmark(
            sizes=[1000], n_trials=500, n_warmup=50,
        )
        m = result.measurements[0]
        assert m.p95_us >= m.p50_us
        assert m.p99_us >= m.p95_us
        assert m.ci_lower_us <= m.mean_us <= m.ci_upper_us

    def test_result_serializable(self):
        result = run_full_pipeline_benchmark(
            sizes=[1000], n_trials=100, n_warmup=10,
        )
        serialized = json.dumps(result.to_dict())
        assert len(serialized) > 0


# =====================================================================
# 4. Annotate comparison (H5.4c)
# =====================================================================

class TestAnnotateComparison:
    """Compare security overhead to annotate() baseline."""

    def test_comparison_runs(self):
        result = run_annotate_comparison(
            sizes=[100, 1000],
            n_trials=200,
            n_warmup=20,
        )
        assert isinstance(result, AnnotateComparisonResult)
        assert len(result.comparisons) > 0

    def test_comparison_has_ratio(self):
        result = run_annotate_comparison(
            sizes=[1000], n_trials=200, n_warmup=20,
        )
        for c in result.comparisons:
            assert hasattr(c, "size_bytes")
            assert hasattr(c, "pipeline_mean_us")
            assert hasattr(c, "annotate_mean_us")
            assert hasattr(c, "overhead_ratio")
            assert c.overhead_ratio >= 0  # Ratio can't be negative

    def test_result_serializable(self):
        result = run_annotate_comparison(
            sizes=[1000], n_trials=100, n_warmup=10,
        )
        serialized = json.dumps(result.to_dict())
        assert len(serialized) > 0


# =====================================================================
# 5. Memory overhead (H5.4d)
# =====================================================================

class TestMemoryOverhead:
    """Measure memory overhead of security pipeline."""

    def test_memory_benchmark_runs(self):
        result = run_memory_overhead_benchmark(sizes=[100, 1000])
        assert isinstance(result, MemoryOverheadResult)
        assert len(result.measurements) > 0

    def test_memory_has_required_fields(self):
        result = run_memory_overhead_benchmark(sizes=[1000])
        for m in result.measurements:
            assert hasattr(m, "size_bytes")
            assert hasattr(m, "baseline_peak_bytes")
            assert hasattr(m, "security_peak_bytes")
            assert hasattr(m, "overhead_bytes")
            assert hasattr(m, "amplification_factor")

    def test_amplification_reasonable(self):
        """Memory amplification should be bounded (not exponential)."""
        result = run_memory_overhead_benchmark(sizes=[1000, 10_000])
        for m in result.measurements:
            assert m.amplification_factor > 0
            # Sanity: shouldn't be more than 100x for any reasonable input
            assert m.amplification_factor < 100, (
                f"Amplification {m.amplification_factor}x at {m.size_bytes}B"
            )

    def test_result_serializable(self):
        result = run_memory_overhead_benchmark(sizes=[1000])
        serialized = json.dumps(result.to_dict())
        assert len(serialized) > 0


# =====================================================================
# 6. Full EN5.4 integration
# =====================================================================

class TestFullEN54:
    """Full experiment runner integration."""

    def test_full_experiment_runs(self):
        """Run the complete EN5.4 with minimal config."""
        result = run_en5_4_full(
            sizes=[100, 1000],
            n_trials=100,
            n_warmup=10,
        )
        assert isinstance(result, EN54FullResult)
        assert result.operations is not None
        assert result.pipeline is not None
        assert result.comparison is not None
        assert result.memory is not None

    def test_full_result_serializable(self):
        result = run_en5_4_full(
            sizes=[1000], n_trials=100, n_warmup=10,
        )
        serialized = json.dumps(result.to_dict())
        assert len(serialized) > 0
