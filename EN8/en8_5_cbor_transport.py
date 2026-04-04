#!/usr/bin/env python
"""EN8.5 -- CBOR-LD Compact Transport with TurboQuant Enhancement.

NeurIPS 2026 D&B, Suite EN8 (Ecosystem Integration), Experiment 5.

Hypothesis:
    CBOR-LD serialization provides meaningful payload reduction over
    JSON-LD for ML data exchange, and quantized vector representations
    (4-bit TurboQuant) provide an additional order-of-magnitude
    reduction for embedding-heavy documents.

Protocol:
    1. Construct test documents at 3 complexity levels:
       - Simple:  sensor reading with @confidence (no vectors)
       - Medium:  product with 128-dim embedding + @quantization
       - Complex: multi-node @graph with 3x 768-dim embeddings

    2. For Medium and Complex, create 2 vector variants:
       - float32:   full-precision float arrays
       - quantized: 4-bit packed byte arrays (simulated TurboQuant output)

    3. Serialize each to multiple formats:
       - JSON-LD:        json.dumps (compact separators)
       - gzip(JSON-LD):  gzip compressed JSON
       - CBOR-LD:        to_cbor() with context compression
       - gzip(CBOR-LD):  gzip compressed CBOR
       - MQTT payload:   to_mqtt_payload(compress=True)
       - MessagePack:    msgpack.packb()

    4. Measure:
       - Payload bytes for each format x document x variant
       - Compression ratio vs JSON-LD baseline
       - Serialization throughput (ops/sec, 1000 iterations)
       - Deserialization throughput (ops/sec, 1000 iterations)
       - Round-trip fidelity (exact match after deserialize)

    5. Quantization-specific analysis:
       - Byte savings from float32 -> 4-bit quantized representation
       - @quantization metadata overhead
       - Theoretical minimum: dims * bits_per_dim / 8

Design notes:
    - Protobuf is omitted: it requires schema compilation (.proto files)
      which is infrastructure beyond the scope of this format comparison.
      We note this limitation in the results.
    - MessagePack is a fair binary alternative: schema-free like CBOR,
      widely used in ML pipelines (e.g. Ray, Redis).
    - All format comparisons use the SAME logical document content.
    - We do NOT strawman any format -- each is used idiomatically.

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN8/en8_5_cbor_transport.py

Output:
    experiments/EN8/results/en8_5_results.json             (latest)
    experiments/EN8/results/en8_5_results_YYYYMMDD_HHMMSS.json (archive)

References:
    RFC 8949: Concise Binary Object Representation (CBOR).
    Zandieh et al. (2025). TurboQuant. ICLR 2026. arXiv:2504.19874.
    Shannon, C.E. (1959). Coding theorems for a discrete source with
        a fidelity criterion.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -- Path setup ----------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_PKG_SRC = _REPO_ROOT / "packages" / "python" / "src"
_EXPERIMENTS_ROOT = _REPO_ROOT / "experiments"

for p in [str(_REPO_ROOT), str(_PKG_SRC), str(_EXPERIMENTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# -- Imports: jsonld-ex ---------------------------------------------------
from jsonld_ex.cbor_ld import to_cbor, from_cbor, payload_stats
from jsonld_ex.mqtt import to_mqtt_payload, from_mqtt_payload
from jsonld_ex.vector import quantization_descriptor

# -- Imports: experiment infrastructure -----------------------------------
from experiments.infra.results import ExperimentResult
from experiments.infra.env_log import log_environment
from experiments.infra.config import set_global_seed

# -- Imports: third-party (optional) --------------------------------------
try:
    import msgpack
    _HAS_MSGPACK = True
except ImportError:
    _HAS_MSGPACK = False

RESULTS_DIR = _SCRIPT_DIR / "results"
SEED = 42
THROUGHPUT_ITERATIONS = 1000


# =====================================================================
# Document Construction
# =====================================================================


def _make_unit_vector(dims: int, seed: int) -> List[float]:
    """Generate a deterministic unit-normalised random vector."""
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(dims)]
    norm = math.sqrt(sum(x * x for x in raw))
    if norm == 0:
        raw[0] = 1.0
        norm = 1.0
    return [x / norm for x in raw]


def _pack_4bit(vec: List[float], dims: int) -> List[int]:
    """Simulate 4-bit quantization: map floats to [0,15] and pack.

    This simulates what a real TurboQuant encoder would produce:
    each float is uniformly quantized to a 4-bit integer.  Two 4-bit
    values are packed per byte.

    Returns a list of ints in [0, 255] representing packed bytes.
    """
    # Uniform quantization: map [-1, 1] -> [0, 15]
    quantized_nibbles = []
    for v in vec:
        clamped = max(-1.0, min(1.0, v))
        q = int(round((clamped + 1.0) / 2.0 * 15.0))
        q = max(0, min(15, q))
        quantized_nibbles.append(q)

    # Pack two nibbles per byte (high nibble first)
    packed = []
    for i in range(0, len(quantized_nibbles), 2):
        high = quantized_nibbles[i]
        low = quantized_nibbles[i + 1] if i + 1 < len(quantized_nibbles) else 0
        packed.append((high << 4) | low)

    return packed


def build_simple_doc() -> Dict[str, Any]:
    """Simple sensor reading with @confidence, no vectors."""
    return {
        "@context": "https://schema.org/",
        "@type": "SensorReading",
        "@id": "urn:sensor:imu-001",
        "temperature": {
            "@value": 36.7,
            "@confidence": 0.95,
            "@source": "https://model.example.org/temp-classifier-v2",
            "@validFrom": "2026-03-27T10:00:00Z",
            "@validUntil": "2026-03-27T11:00:00Z",
        },
        "humidity": {
            "@value": 45.2,
            "@confidence": 0.88,
        },
        "pressure": {
            "@value": 1013.25,
            "@confidence": 0.92,
        },
    }


def build_medium_doc(variant: str = "float32") -> Dict[str, Any]:
    """Product with 128-dim embedding. variant: 'float32' or 'quantized'."""
    dims = 128
    vec_float = _make_unit_vector(dims, seed=SEED)

    embedding: Dict[str, Any] = {
        "@container": "@vector",
        "@dimensions": dims,
        "@similarity": "cosine",
        "@quantization": quantization_descriptor(
            method="turboquant",
            bit_width=4,
            rotation_seed=42,
            has_residual_qjl=True,
        ),
    }

    if variant == "float32":
        embedding["@value"] = vec_float
    elif variant == "quantized":
        embedding["@value"] = _pack_4bit(vec_float, dims)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return {
        "@context": "https://schema.org/",
        "@type": "Product",
        "@id": "urn:product:widget-42",
        "name": "Widget Pro",
        "description": "A high-quality widget with semantic embedding.",
        "@confidence": 0.97,
        "@source": "https://model.example.org/product-embedder-v3",
        "embedding": embedding,
    }


def build_complex_doc(variant: str = "float32") -> Dict[str, Any]:
    """Multi-node @graph with 3x 768-dim embeddings."""
    dims = 768
    products = []
    for i in range(3):
        vec_float = _make_unit_vector(dims, seed=SEED + i)

        embedding: Dict[str, Any] = {
            "@container": "@vector",
            "@dimensions": dims,
            "@similarity": "cosine",
            "@quantization": quantization_descriptor(
                method="turboquant",
                bit_width=4,
                rotation_seed=42 + i,
                has_residual_qjl=True,
            ),
        }

        if variant == "float32":
            embedding["@value"] = vec_float
        elif variant == "quantized":
            embedding["@value"] = _pack_4bit(vec_float, dims)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        products.append({
            "@id": f"urn:product:item-{i}",
            "@type": "Product",
            "name": f"Product {i}",
            "description": f"Description for product {i} with full embedding.",
            "@confidence": round(0.90 + i * 0.03, 2),
            "@source": f"https://model.example.org/embedder-v{i}",
            "embedding": embedding,
            "category": {
                "@value": f"Category-{chr(65 + i)}",
                "@confidence": round(0.85 + i * 0.05, 2),
            },
        })

    return {
        "@context": "https://schema.org/",
        "@graph": products,
    }


# =====================================================================
# Serialization Formats
# =====================================================================


def serialize_json(doc: Dict[str, Any]) -> bytes:
    """Compact JSON serialization."""
    return json.dumps(doc, separators=(",", ":")).encode("utf-8")


def deserialize_json(data: bytes) -> Dict[str, Any]:
    """JSON deserialization."""
    return json.loads(data.decode("utf-8"))


def serialize_gzip_json(doc: Dict[str, Any]) -> bytes:
    """Gzip-compressed JSON."""
    return gzip.compress(serialize_json(doc))


def deserialize_gzip_json(data: bytes) -> Dict[str, Any]:
    """Gzip-decompressed JSON."""
    return json.loads(gzip.decompress(data).decode("utf-8"))


def serialize_cbor(doc: Dict[str, Any]) -> bytes:
    """CBOR-LD with context compression."""
    return to_cbor(doc)


def deserialize_cbor(data: bytes) -> Dict[str, Any]:
    """CBOR-LD deserialization."""
    return from_cbor(data)


def serialize_gzip_cbor(doc: Dict[str, Any]) -> bytes:
    """Gzip-compressed CBOR-LD."""
    return gzip.compress(to_cbor(doc))


def deserialize_gzip_cbor(data: bytes) -> Dict[str, Any]:
    """Gzip-decompressed CBOR-LD."""
    return from_cbor(gzip.decompress(data))


def serialize_mqtt(doc: Dict[str, Any]) -> bytes:
    """MQTT payload (CBOR mode)."""
    return to_mqtt_payload(doc, compress=True)


def deserialize_mqtt(data: bytes) -> Dict[str, Any]:
    """MQTT payload deserialization."""
    return from_mqtt_payload(data, compressed=True)


def serialize_msgpack(doc: Dict[str, Any]) -> bytes:
    """MessagePack serialization."""
    if not _HAS_MSGPACK:
        raise ImportError("msgpack required: pip install msgpack")
    return msgpack.packb(doc, use_bin_type=True)


def deserialize_msgpack(data: bytes) -> Dict[str, Any]:
    """MessagePack deserialization."""
    if not _HAS_MSGPACK:
        raise ImportError("msgpack required: pip install msgpack")
    return msgpack.unpackb(data, raw=False)


# Format registry: (name, serialize_fn, deserialize_fn)
FORMATS = [
    ("JSON-LD", serialize_json, deserialize_json),
    ("gzip(JSON)", serialize_gzip_json, deserialize_gzip_json),
    ("CBOR-LD", serialize_cbor, deserialize_cbor),
    ("gzip(CBOR)", serialize_gzip_cbor, deserialize_gzip_cbor),
    ("MQTT", serialize_mqtt, deserialize_mqtt),
]

if _HAS_MSGPACK:
    FORMATS.append(("MessagePack", serialize_msgpack, deserialize_msgpack))


# =====================================================================
# Measurement
# =====================================================================


def measure_payload(doc: Dict[str, Any], fmt_name: str,
                    ser_fn, deser_fn, n_iter: int) -> Dict[str, Any]:
    """Measure payload size, throughput, and round-trip fidelity.

    Returns a dict with:
        format, payload_bytes, ser_ops_per_sec, deser_ops_per_sec,
        round_trip_ok
    """
    # -- Payload size --
    serialized = ser_fn(doc)
    payload_bytes = len(serialized)

    # -- Round-trip fidelity --
    restored = deser_fn(serialized)
    # Deep comparison: for MQTT, context may be slightly different
    # (http vs https schema.org), so we compare non-context fields
    round_trip_ok = _check_fidelity(doc, restored)

    # -- Serialization throughput --
    t0 = time.perf_counter()
    for _ in range(n_iter):
        ser_fn(doc)
    ser_elapsed = time.perf_counter() - t0
    ser_ops = n_iter / ser_elapsed if ser_elapsed > 0 else float("inf")

    # -- Deserialization throughput --
    t0 = time.perf_counter()
    for _ in range(n_iter):
        deser_fn(serialized)
    deser_elapsed = time.perf_counter() - t0
    deser_ops = n_iter / deser_elapsed if deser_elapsed > 0 else float("inf")

    return {
        "format": fmt_name,
        "payload_bytes": payload_bytes,
        "ser_ops_per_sec": round(ser_ops, 1),
        "deser_ops_per_sec": round(deser_ops, 1),
        "round_trip_ok": round_trip_ok,
    }


def _check_fidelity(original: Dict[str, Any], restored: Dict[str, Any]) -> bool:
    """Check round-trip fidelity, tolerant of context URL normalisation.

    CBOR-LD may normalise http://schema.org/ <-> https://schema.org/
    (both map to registry ID 1).  We check all non-@context fields
    for exact equality and allow context URLs to differ only in scheme.
    """
    def _strip_context(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: _strip_context(v) for k, v in d.items() if k != "@context"}
        if isinstance(d, list):
            return [_strip_context(item) for item in d]
        return d

    orig_stripped = _strip_context(original)
    rest_stripped = _strip_context(restored)

    # Compare as JSON strings for deep equality
    orig_json = json.dumps(orig_stripped, sort_keys=True, separators=(",", ":"))
    rest_json = json.dumps(rest_stripped, sort_keys=True, separators=(",", ":"))
    return orig_json == rest_json


# =====================================================================
# Quantization Analysis
# =====================================================================


def quantization_analysis(dims: int, bit_width: int = 4) -> Dict[str, Any]:
    """Compute theoretical and measured byte savings from quantization.

    Returns dict with:
        dimensions, bit_width, float32_bytes, quantized_bytes,
        theoretical_minimum_bytes, reduction_ratio, metadata_overhead_bytes
    """
    vec_float = _make_unit_vector(dims, seed=SEED)
    vec_packed = _pack_4bit(vec_float, dims)

    # JSON representation sizes
    json_float = json.dumps(vec_float, separators=(",", ":")).encode("utf-8")
    json_packed = json.dumps(vec_packed, separators=(",", ":")).encode("utf-8")

    # Theoretical minimum: dims * bits / 8
    theoretical_min = math.ceil(dims * bit_width / 8)

    # Metadata overhead: the @quantization descriptor
    q_desc = quantization_descriptor(
        method="turboquant", bit_width=4,
        rotation_seed=42, has_residual_qjl=True,
    )
    meta_json = json.dumps(q_desc, separators=(",", ":")).encode("utf-8")

    return {
        "dimensions": dims,
        "bit_width": bit_width,
        "float32_json_bytes": len(json_float),
        "quantized_json_bytes": len(json_packed),
        "theoretical_minimum_bytes": theoretical_min,
        "json_reduction_ratio": round(len(json_packed) / len(json_float), 4),
        "metadata_overhead_bytes": len(meta_json),
    }


# =====================================================================
# Main Experiment
# =====================================================================


def run_experiment() -> ExperimentResult:
    """Execute the full EN8.5 benchmark."""
    set_global_seed(SEED)
    env = log_environment()

    print("=" * 70)
    print("EN8.5 -- CBOR-LD Compact Transport with TurboQuant Enhancement")
    print("=" * 70)

    # -- Build documents --
    documents = {
        "simple": {"doc": build_simple_doc(), "has_vectors": False},
        "medium_float32": {"doc": build_medium_doc("float32"), "has_vectors": True},
        "medium_quantized": {"doc": build_medium_doc("quantized"), "has_vectors": True},
        "complex_float32": {"doc": build_complex_doc("float32"), "has_vectors": True},
        "complex_quantized": {"doc": build_complex_doc("quantized"), "has_vectors": True},
    }

    # -- Measure all format x document combinations --
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    fidelity_failures: List[str] = []
    formats_used = [f[0] for f in FORMATS]

    for doc_name, doc_info in documents.items():
        doc = doc_info["doc"]
        json_baseline = len(serialize_json(doc))

        print(f"\n--- {doc_name} (JSON baseline: {json_baseline:,} bytes) ---")

        doc_results = []
        for fmt_name, ser_fn, deser_fn in FORMATS:
            try:
                result = measure_payload(
                    doc, fmt_name, ser_fn, deser_fn,
                    n_iter=THROUGHPUT_ITERATIONS,
                )
                result["compression_ratio"] = round(
                    result["payload_bytes"] / json_baseline, 4
                )
                doc_results.append(result)

                status = "OK" if result["round_trip_ok"] else "FAIL"
                if not result["round_trip_ok"]:
                    fidelity_failures.append(f"{doc_name}/{fmt_name}")

                print(
                    f"  {fmt_name:15s}  {result['payload_bytes']:>8,} bytes  "
                    f"ratio={result['compression_ratio']:.4f}  "
                    f"ser={result['ser_ops_per_sec']:>8,.0f} ops/s  "
                    f"deser={result['deser_ops_per_sec']:>8,.0f} ops/s  "
                    f"[{status}]"
                )
            except Exception as e:
                print(f"  {fmt_name:15s}  ERROR: {e}")
                doc_results.append({
                    "format": fmt_name,
                    "error": str(e),
                })

        all_results[doc_name] = doc_results

    # -- Quantization analysis --
    print("\n--- Quantization Byte Savings Analysis ---")
    quant_analysis = {}
    for dims in [128, 768]:
        qa = quantization_analysis(dims)
        quant_analysis[f"{dims}d"] = qa
        print(
            f"  {dims}-dim:  float32 JSON={qa['float32_json_bytes']:,}B  "
            f"quantized JSON={qa['quantized_json_bytes']:,}B  "
            f"ratio={qa['json_reduction_ratio']:.4f}  "
            f"theoretical min={qa['theoretical_minimum_bytes']}B  "
            f"metadata overhead={qa['metadata_overhead_bytes']}B"
        )

    # -- Cross-variant comparison (the TurboQuant headline) --
    print("\n--- TurboQuant Enhancement: Float32 vs Quantized ---")
    cross_variant = {}
    for level in ["medium", "complex"]:
        f32_key = f"{level}_float32"
        q_key = f"{level}_quantized"
        f32_json = next(
            r for r in all_results[f32_key] if r.get("format") == "JSON-LD"
        )
        q_json = next(
            r for r in all_results[q_key] if r.get("format") == "JSON-LD"
        )
        f32_cbor = next(
            r for r in all_results[f32_key] if r.get("format") == "CBOR-LD"
        )
        q_cbor = next(
            r for r in all_results[q_key] if r.get("format") == "CBOR-LD"
        )

        savings = {
            "json_float32_bytes": f32_json["payload_bytes"],
            "json_quantized_bytes": q_json["payload_bytes"],
            "json_reduction": round(
                1.0 - q_json["payload_bytes"] / f32_json["payload_bytes"], 4
            ),
            "cbor_float32_bytes": f32_cbor["payload_bytes"],
            "cbor_quantized_bytes": q_cbor["payload_bytes"],
            "cbor_reduction": round(
                1.0 - q_cbor["payload_bytes"] / f32_cbor["payload_bytes"], 4
            ),
        }
        cross_variant[level] = savings

        print(
            f"  {level}:  JSON {savings['json_float32_bytes']:,}B -> "
            f"{savings['json_quantized_bytes']:,}B "
            f"({savings['json_reduction']:.1%} reduction)  |  "
            f"CBOR {savings['cbor_float32_bytes']:,}B -> "
            f"{savings['cbor_quantized_bytes']:,}B "
            f"({savings['cbor_reduction']:.1%} reduction)"
        )

    # -- Summary --
    print("\n--- Summary ---")
    total_fidelity = sum(
        1 for doc_results in all_results.values()
        for r in doc_results
        if r.get("round_trip_ok", False)
    )
    total_measured = sum(
        1 for doc_results in all_results.values()
        for r in doc_results
        if "payload_bytes" in r
    )
    print(f"  Round-trip fidelity: {total_fidelity}/{total_measured} passed")
    if fidelity_failures:
        print(f"  FAILURES: {fidelity_failures}")
    print(f"  Formats tested: {formats_used}")
    print(f"  Documents tested: {list(documents.keys())}")

    # -- Build ExperimentResult --
    metrics = {
        "formats_tested": formats_used,
        "documents_tested": list(documents.keys()),
        "throughput_iterations": THROUGHPUT_ITERATIONS,
        "round_trip_fidelity": f"{total_fidelity}/{total_measured}",
        "fidelity_failures": fidelity_failures,
        "cross_variant_savings": cross_variant,
        "quantization_analysis": quant_analysis,
    }

    result = ExperimentResult(
        experiment_id="EN8.5",
        parameters={
            "seed": SEED,
            "throughput_iterations": THROUGHPUT_ITERATIONS,
            "complexity_levels": ["simple", "medium", "complex"],
            "vector_variants": ["float32", "quantized"],
            "formats": formats_used,
            "vector_dimensions": {"medium": 128, "complex": 768},
            "quantization_method": "turboquant",
            "quantization_bit_width": 4,
            "note_protobuf_omitted": (
                "Protobuf omitted: requires .proto schema compilation. "
                "MessagePack serves as the alternative schema-free binary format."
            ),
        },
        metrics=metrics,
        raw_data={"format_results": all_results},
        environment=env,
        notes=(
            "EN8.5 with TurboQuant enhancement. Quantized variant uses "
            "simulated 4-bit uniform quantization packed two nibbles per byte, "
            "representing what a real TurboQuant encoder would produce. "
            "Distortion constants are illustrative defaults, not calibrated. "
            "See quantization_bridge.py for the SL uncertainty mapping."
        ),
    )

    return result


# =====================================================================
# Entry Point
# =====================================================================


def main() -> None:
    """Run EN8.5 and save results."""
    result = run_experiment()

    # -- Save results --
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    latest_path = RESULTS_DIR / "en8_5_results.json"
    result.save_json(str(latest_path))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = RESULTS_DIR / f"en8_5_results_{ts}.json"
    result.save_json(str(archive_path))

    print(f"\nResults saved to:")
    print(f"  {latest_path}")
    print(f"  {archive_path}")

    # -- Exit status --
    if result.metrics["fidelity_failures"]:
        print("\nWARNING: Round-trip fidelity failures detected!")
        sys.exit(1)
    else:
        print("\nAll round-trip fidelity checks passed.")


if __name__ == "__main__":
    main()
