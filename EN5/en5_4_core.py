"""EN5.4 -- Security Pipeline Overhead core module.

NeurIPS 2026 D&B, Suite EN5 (Security and Integrity), Experiment 4.

Hypotheses:
    H5.4a: Full pipeline adds <1ms per doc for docs under 100KB
    H5.4b: Security check latency scales linearly with doc size
    H5.4c: Security checks are <5% of annotate() time
    H5.4d: Memory overhead <2x input document size

Measures honest overhead of security pipeline relative to EN4.1 baselines.
No inflated metrics. No adversarial-vs-local comparisons.
"""
from __future__ import annotations

import gc
import json
import random
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from jsonld_ex.security import (
    compute_integrity,
    verify_integrity,
    is_context_allowed,
    enforce_resource_limits,
)
from jsonld_ex.ai_ml import annotate

# -- sys.path setup --
import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _SCRIPT_DIR.parent
for p in [str(_SCRIPT_DIR), str(_EXPERIMENTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from infra.stats import bootstrap_ci


# =====================================================================
# Constants
# =====================================================================

PIPELINE_DOC_SIZES = [100, 1_000, 10_000, 100_000, 1_000_000]


# =====================================================================
# Result dataclasses
# =====================================================================

@dataclass
class OperationMeasurement:
    """Single latency measurement."""
    size_bytes: int
    operation: str
    mean_us: float = 0.0
    p50_us: float = 0.0
    p95_us: float = 0.0
    p99_us: float = 0.0
    ci_lower_us: float = 0.0
    ci_upper_us: float = 0.0
    n_trials: int = 0


@dataclass
class OperationBenchmarkResult:
    """Individual operation benchmark results."""
    measurements: list[OperationMeasurement] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"measurements": [asdict(m) for m in self.measurements]}


@dataclass
class PipelineMeasurement:
    """Full pipeline measurement for one doc size."""
    size_bytes: int
    mean_us: float = 0.0
    p50_us: float = 0.0
    p95_us: float = 0.0
    p99_us: float = 0.0
    ci_lower_us: float = 0.0
    ci_upper_us: float = 0.0
    n_trials: int = 0


@dataclass
class PipelineBenchmarkResult:
    """Full pipeline benchmark results."""
    measurements: list[PipelineMeasurement] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"measurements": [asdict(m) for m in self.measurements]}


@dataclass
class ComparisonEntry:
    """Pipeline vs annotate comparison for one doc size."""
    size_bytes: int
    pipeline_mean_us: float = 0.0
    annotate_mean_us: float = 0.0
    overhead_ratio: float = 0.0  # pipeline / annotate


@dataclass
class AnnotateComparisonResult:
    """Pipeline vs annotate comparison results."""
    comparisons: list[ComparisonEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"comparisons": [asdict(c) for c in self.comparisons]}


@dataclass
class MemoryMeasurement:
    """Memory measurement for one doc size."""
    size_bytes: int
    baseline_peak_bytes: int = 0
    security_peak_bytes: int = 0
    overhead_bytes: int = 0
    amplification_factor: float = 0.0


@dataclass
class MemoryOverheadResult:
    """Memory overhead results."""
    measurements: list[MemoryMeasurement] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"measurements": [asdict(m) for m in self.measurements]}


@dataclass
class EN54FullResult:
    """Complete EN5.4 results."""
    operations: OperationBenchmarkResult | None = None
    pipeline: PipelineBenchmarkResult | None = None
    comparison: AnnotateComparisonResult | None = None
    memory: MemoryOverheadResult | None = None

    def to_dict(self) -> dict:
        return {
            "operations": self.operations.to_dict() if self.operations else None,
            "pipeline": self.pipeline.to_dict() if self.pipeline else None,
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "memory": self.memory.to_dict() if self.memory else None,
        }


# =====================================================================
# Document generator
# =====================================================================

def generate_pipeline_test_document(
    target_bytes: int, seed: int = 42,
) -> tuple[dict, str, str, dict]:
    """Generate a realistic jsonld-ex document with security metadata.

    Returns:
        (document, context_string, context_hash, allowlist_config)
    """
    rng = random.Random(seed)

    ctx = {
        "@vocab": "http://schema.org/",
        "confidence": "https://w3id.org/jsonld-ex/confidence",
        "source": "https://w3id.org/jsonld-ex/source",
    }
    ctx_str = json.dumps(ctx, sort_keys=True)
    ctx_hash = compute_integrity(ctx_str)
    allowlist_cfg = {"allowed": ["https://schema.org/"]}

    # Build document with enough data to approach target size
    doc: dict[str, Any] = {
        "@context": "https://schema.org/",
        "@type": "Dataset",
        "name": "Pipeline Test Dataset",
        "description": "Generated for EN5.4 overhead benchmarking",
    }

    # Fill with annotated items to reach target size
    items = []
    current_size = len(json.dumps(doc))
    counter = 0
    while current_size < target_bytes:
        item = {
            f"field_{counter}": annotate(
                f"value_{counter}_{rng.randint(0, 99999)}",
                confidence=round(rng.random(), 4),
                source=f"model:test-v{rng.randint(1, 5)}",
            )
        }
        items.append(item)
        current_size += len(json.dumps(item))
        counter += 1

    if items:
        doc["items"] = items

    return doc, ctx_str, ctx_hash, allowlist_cfg


# =====================================================================
# Timing helper
# =====================================================================

def _timed_trials(fn, n_trials: int, n_warmup: int) -> OperationMeasurement:
    """Run fn() n_trials times, return timing stats in microseconds."""
    for _ in range(n_warmup):
        fn()

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)

    arr = np.array(times)
    ci_lo, _, ci_hi = bootstrap_ci(arr)

    return OperationMeasurement(
        size_bytes=0,  # Caller fills this
        operation="",  # Caller fills this
        mean_us=float(np.mean(arr)),
        p50_us=float(np.percentile(arr, 50)),
        p95_us=float(np.percentile(arr, 95)),
        p99_us=float(np.percentile(arr, 99)),
        ci_lower_us=ci_lo,
        ci_upper_us=ci_hi,
        n_trials=n_trials,
    )


# =====================================================================
# Individual operation benchmarks
# =====================================================================

def run_individual_operation_benchmarks(
    sizes: list[int] | None = None,
    n_trials: int = 10_000,
    n_warmup: int = 100,
) -> OperationBenchmarkResult:
    """Benchmark each security operation individually."""
    if sizes is None:
        sizes = PIPELINE_DOC_SIZES

    result = OperationBenchmarkResult()

    for size in sizes:
        doc, ctx_str, ctx_hash, cfg = generate_pipeline_test_document(size)
        doc_str = json.dumps(doc)
        actual_size = len(doc_str.encode("utf-8"))

        # 1. Allowlist check
        m = _timed_trials(
            lambda: is_context_allowed("https://schema.org/", cfg),
            n_trials, n_warmup,
        )
        m.size_bytes = actual_size
        m.operation = "allowlist_check"
        result.measurements.append(m)

        # 2. Integrity verification
        m = _timed_trials(
            lambda: verify_integrity(ctx_str, ctx_hash),
            n_trials, n_warmup,
        )
        m.size_bytes = actual_size
        m.operation = "integrity_verify"
        result.measurements.append(m)

        # 3. Resource limits
        m = _timed_trials(
            lambda: enforce_resource_limits(doc),
            n_trials, n_warmup,
        )
        m.size_bytes = actual_size
        m.operation = "resource_limits"
        result.measurements.append(m)

    return result


# =====================================================================
# Full pipeline benchmark
# =====================================================================

def run_full_pipeline_benchmark(
    sizes: list[int] | None = None,
    n_trials: int = 10_000,
    n_warmup: int = 100,
) -> PipelineBenchmarkResult:
    """Benchmark all three security checks in sequence."""
    if sizes is None:
        sizes = PIPELINE_DOC_SIZES

    result = PipelineBenchmarkResult()

    for size in sizes:
        doc, ctx_str, ctx_hash, cfg = generate_pipeline_test_document(size)
        actual_size = len(json.dumps(doc).encode("utf-8"))

        def full_pipeline():
            is_context_allowed("https://schema.org/", cfg)
            verify_integrity(ctx_str, ctx_hash)
            enforce_resource_limits(doc)

        m = _timed_trials(full_pipeline, n_trials, n_warmup)

        result.measurements.append(PipelineMeasurement(
            size_bytes=actual_size,
            mean_us=m.mean_us,
            p50_us=m.p50_us,
            p95_us=m.p95_us,
            p99_us=m.p99_us,
            ci_lower_us=m.ci_lower_us,
            ci_upper_us=m.ci_upper_us,
            n_trials=n_trials,
        ))

    return result


# =====================================================================
# Annotate comparison
# =====================================================================

def run_annotate_comparison(
    sizes: list[int] | None = None,
    n_trials: int = 10_000,
    n_warmup: int = 100,
) -> AnnotateComparisonResult:
    """Compare security pipeline overhead to annotate() time."""
    if sizes is None:
        sizes = PIPELINE_DOC_SIZES

    result = AnnotateComparisonResult()

    for size in sizes:
        doc, ctx_str, ctx_hash, cfg = generate_pipeline_test_document(size)
        actual_size = len(json.dumps(doc).encode("utf-8"))

        # Pipeline timing
        def full_pipeline():
            is_context_allowed("https://schema.org/", cfg)
            verify_integrity(ctx_str, ctx_hash)
            enforce_resource_limits(doc)

        pm = _timed_trials(full_pipeline, n_trials, n_warmup)

        # Annotate timing (annotate a single value — the atomic operation)
        am = _timed_trials(
            lambda: annotate(
                "test_value",
                confidence=0.85,
                source="model:test-v1",
                extracted_at="2026-01-01T00:00:00Z",
            ),
            n_trials, n_warmup,
        )

        ratio = pm.mean_us / am.mean_us if am.mean_us > 0 else float("inf")

        result.comparisons.append(ComparisonEntry(
            size_bytes=actual_size,
            pipeline_mean_us=pm.mean_us,
            annotate_mean_us=am.mean_us,
            overhead_ratio=round(ratio, 4),
        ))

    return result


# =====================================================================
# Memory overhead benchmark
# =====================================================================

def run_memory_overhead_benchmark(
    sizes: list[int] | None = None,
    n_iterations: int = 100,
) -> MemoryOverheadResult:
    """Measure memory overhead of security pipeline via tracemalloc."""
    if sizes is None:
        sizes = PIPELINE_DOC_SIZES

    result = MemoryOverheadResult()

    for size in sizes:
        doc, ctx_str, ctx_hash, cfg = generate_pipeline_test_document(size)
        input_size = len(json.dumps(doc).encode("utf-8"))

        # Baseline: json.loads only
        gc.collect()
        doc_str = json.dumps(doc)
        tracemalloc.start()
        for _ in range(n_iterations):
            json.loads(doc_str)
        _, baseline_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Security pipeline
        gc.collect()
        tracemalloc.start()
        for _ in range(n_iterations):
            is_context_allowed("https://schema.org/", cfg)
            verify_integrity(ctx_str, ctx_hash)
            enforce_resource_limits(doc)
        _, security_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        overhead = security_peak - baseline_peak
        amp = security_peak / max(1, input_size)

        result.measurements.append(MemoryMeasurement(
            size_bytes=input_size,
            baseline_peak_bytes=baseline_peak,
            security_peak_bytes=security_peak,
            overhead_bytes=overhead,
            amplification_factor=round(amp, 4),
        ))

    return result


# =====================================================================
# Full experiment
# =====================================================================

def run_en5_4_full(
    sizes: list[int] | None = None,
    n_trials: int = 10_000,
    n_warmup: int = 100,
) -> EN54FullResult:
    """Run the complete EN5.4 experiment."""
    if sizes is None:
        sizes = PIPELINE_DOC_SIZES

    return EN54FullResult(
        operations=run_individual_operation_benchmarks(sizes, n_trials, n_warmup),
        pipeline=run_full_pipeline_benchmark(sizes, n_trials, n_warmup),
        comparison=run_annotate_comparison(sizes, n_trials, n_warmup),
        memory=run_memory_overhead_benchmark(sizes),
    )
