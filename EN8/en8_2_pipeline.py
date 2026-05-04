"""EN8.2 — IoT Sensor Pipeline: Annotation, SSN/SOSA, Transport, Quality.

Provides the full jsonld-ex IoT pipeline:
  annotate() → to_ssn() → to_cbor() / to_mqtt() / to_coap()
  plus data quality detection using annotation metadata.
"""
from __future__ import annotations

import inspect
import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import sys
_PKG_SRC = Path(__file__).resolve().parent.parent.parent / "packages" / "python" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import jsonld_ex as jx


# =====================================================================
# 1. Annotation Pipeline
# =====================================================================

def annotate_sensor_reading(
    value: float,
    sensor_id: str,
    unit: str = "celsius",
    timestamp: Optional[str] = None,
    sigma: Optional[float] = None,
    calibrated_at: Optional[str] = None,
    calibration_method: Optional[str] = None,
    calibration_authority: Optional[str] = None,
) -> dict:
    """Annotate a single sensor reading with full metadata.

    Wraps ``jsonld_ex.annotate()`` with IoT-specific defaults.

    Returns:
        Annotated JSON-LD value dict.
    """
    confidence = None
    if sigma is not None and sigma > 0:
        # Map sigma to confidence: higher noise → lower confidence
        # Using a simple sigmoid: conf = 1 / (1 + sigma)
        confidence = 1.0 / (1.0 + sigma)

    return jx.annotate(
        value=value,
        confidence=confidence,
        source=sensor_id,
        extracted_at=timestamp,
        method="direct-measurement",
        measurement_uncertainty=sigma,
        unit=unit,
        calibrated_at=calibrated_at,
        calibration_method=calibration_method,
        calibration_authority=calibration_authority,
    )


def annotate_sensor_batch(
    readings: List[Dict[str, Any]],
) -> List[dict]:
    """Annotate a batch of sensor readings."""
    return [annotate_sensor_reading(**r) for r in readings]


# =====================================================================
# 2. SSN/SOSA Interoperability
# =====================================================================

def readings_to_ssn(annotated_doc: dict) -> Tuple[dict, Any]:
    """Convert an annotated sensor reading to SSN/SOSA.

    Wraps the annotated value in a minimal JSON-LD document, then
    calls ``jx.to_ssn()``.

    Returns:
        (ssn_document, conversion_report)
    """
    # Build a document wrapping the annotated value
    source = annotated_doc.get("@source", "unknown-sensor")
    doc = {
        "@context": {"@vocab": "https://schema.org/"},
        "@type": "Observation",
        "@id": f"obs-{id(annotated_doc) % 100000}",
        "temperature": annotated_doc,
    }
    return jx.to_ssn(doc)


def ssn_round_trip(annotated_doc: dict) -> Tuple[dict, dict, Any]:
    """Full round-trip: annotated → SSN/SOSA → annotated.

    Returns:
        (original_annotation, recovered_annotation, conversion_report)
    """
    ssn_doc, report_fwd = readings_to_ssn(annotated_doc)
    recovered_doc, report_back = jx.from_ssn(ssn_doc)

    # Extract the recovered annotation from the document
    recovered_annotation = recovered_doc.get("temperature", {})
    if isinstance(recovered_annotation, dict):
        return annotated_doc, recovered_annotation, report_back
    else:
        return annotated_doc, {"@value": recovered_annotation}, report_back


def categorize_ssn_triples(ssn_doc: dict) -> Dict[str, int]:
    """Categorize SSN/SOSA output triples by namespace.

    Returns:
        Dict with counts: native (sosa:/ssn:), allied (qudt:/xsd:),
        extension (jsonld-ex:).
    """
    counts = {"native": 0, "allied": 0, "extension": 0}

    ssn_json = json.dumps(ssn_doc)

    # Count namespace occurrences in property URIs
    native_patterns = [
        "sosa/", "sosa:", "ssn/", "ssn:",
        "www.w3.org/ns/sosa", "www.w3.org/ns/ssn",
    ]
    allied_patterns = [
        "qudt", "XMLSchema", "xsd:", "rdfs:",
    ]
    extension_patterns = [
        "jsonld-ex",
    ]

    # Parse the graph structure to count actual property uses
    graph = ssn_doc.get("@graph", [ssn_doc])
    if not isinstance(graph, list):
        graph = [graph]

    for node in graph:
        if not isinstance(node, dict):
            continue
        for key, val in node.items():
            if key.startswith("@"):
                continue  # skip JSON-LD keywords
            is_native = any(p in key for p in native_patterns)
            is_allied = any(p in key for p in allied_patterns)
            is_extension = any(p in key for p in extension_patterns)

            if is_native:
                counts["native"] += 1
            elif is_allied:
                counts["allied"] += 1
            elif is_extension:
                counts["extension"] += 1
            # else: uncategorized (schema.org etc)

    return counts


def validate_ssn_structure(ssn_doc: dict) -> Dict[str, bool]:
    """Validate structural compliance of generated SSN/SOSA.

    Checks for required SSN/SOSA patterns:
    - At least one sosa:Observation node
    - At least one sosa:Result node
    - At least one sosa:Sensor node
    """
    ssn_json = json.dumps(ssn_doc)

    return {
        "has_observation": "sosa/Observation" in ssn_json or "sosa:Observation" in ssn_json,
        "has_result": "sosa/Result" in ssn_json or "sosa:Result" in ssn_json,
        "has_sensor": "sosa/Sensor" in ssn_json or "sosa:Sensor" in ssn_json,
        "has_observable_property": (
            "sosa/ObservableProperty" in ssn_json or
            "sosa:ObservableProperty" in ssn_json
        ),
    }


# =====================================================================
# 3. Transport Efficiency
# =====================================================================

def encode_cbor_batch(docs: List[dict]) -> List[int]:
    """Encode each document as CBOR-LD, return list of byte sizes."""
    sizes = []
    for doc in docs:
        # Wrap in document for to_cbor
        wrapper = {
            "@context": {"@vocab": "https://schema.org/"},
            "@type": "Observation",
            "value": doc,
        }
        cbor_bytes = jx.to_cbor(wrapper)
        sizes.append(len(cbor_bytes))
    return sizes


def encode_json_batch(docs: List[dict]) -> List[int]:
    """Encode each document as JSON, return list of byte sizes."""
    sizes = []
    for doc in docs:
        wrapper = {
            "@context": {"@vocab": "https://schema.org/"},
            "@type": "Observation",
            "value": doc,
        }
        json_bytes = json.dumps(wrapper).encode("utf-8")
        sizes.append(len(json_bytes))
    return sizes


def mqtt_derive_batch(docs: List[dict]) -> List[Dict[str, Any]]:
    """Derive MQTT metadata for each document."""
    results = []
    for doc in docs:
        wrapper = {
            "@context": {"@vocab": "https://schema.org/"},
            "@type": "Observation",
            "@id": doc.get("@source", "obs"),
            "value": doc,
        }
        topic = jx.derive_mqtt_topic(wrapper)
        qos = jx.derive_mqtt_qos(wrapper)
        payload = jx.to_mqtt_payload(wrapper)
        results.append({
            "topic": topic,
            "qos": qos,
            "payload_size": len(payload),
        })
    return results


def coap_derive_batch(docs: List[dict]) -> List[Dict[str, Any]]:
    """Derive CoAP options for each document."""
    results = []
    for doc in docs:
        wrapper = {
            "@context": {"@vocab": "https://schema.org/"},
            "@type": "Observation",
            "@id": doc.get("@source", "obs"),
            "value": doc,
        }
        options = jx.derive_coap_options(wrapper)
        results.append(options)
    return results


# =====================================================================
# 4. Data Quality Detection
# =====================================================================

def compute_disagreement_signal(
    readings: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute per-sensor mean disagreement from multi-sensor readings.

    For each sensor, compute its mean absolute deviation from the
    consensus of all OTHER sensors at each time step.

    Args:
        readings: Dict mapping sensor_id -> array of values.

    Returns:
        Dict mapping sensor_id -> mean disagreement score.
    """
    sensor_ids = list(readings.keys())
    n = len(next(iter(readings.values())))
    result = {}

    for sid in sensor_ids:
        others = [s for s in sensor_ids if s != sid]
        abs_devs = []
        for t in range(n):
            if np.isnan(readings[sid][t]):
                continue
            other_vals = [readings[s][t] for s in others
                          if not np.isnan(readings[s][t])]
            if len(other_vals) >= 1:
                consensus = np.mean(other_vals)
                abs_devs.append(abs(readings[sid][t] - consensus))
        result[sid] = float(np.mean(abs_devs)) if abs_devs else 0.0

    return result


def compute_calibration_age_signal(
    timestamps: List[str],
    calibration_date: str,
) -> List[float]:
    """Compute calibration age (days) at each timestamp.

    Args:
        timestamps: List of ISO 8601 timestamp strings.
        calibration_date: ISO 8601 string of last calibration.

    Returns:
        List of ages in days.
    """
    cal_dt = datetime.fromisoformat(calibration_date.replace("Z", "+00:00"))
    ages = []
    for ts in timestamps:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        delta = (dt - cal_dt).total_seconds() / 86400.0
        ages.append(max(0.0, delta))
    return ages


def detect_drift(
    readings: Dict[str, np.ndarray],
    n_chunks: int = 10,
) -> Dict[str, float]:
    """Detect time-varying bias (drift) via chunked deviation analysis.

    Splits each sensor's time series into n_chunks, computes median
    deviation from consensus in each chunk, and reports the range
    of chunk-level biases as the drift score.

    Args:
        readings: Dict mapping sensor_id -> array of values.
        n_chunks: Number of temporal chunks.

    Returns:
        Dict mapping sensor_id -> drift score (range of chunk biases).
    """
    sensor_ids = list(readings.keys())
    n = len(next(iter(readings.values())))
    chunk_size = max(1, n // n_chunks)
    result = {}

    for sid in sensor_ids:
        others = [s for s in sensor_ids if s != sid]
        chunk_biases = []

        for c in range(n_chunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, n)
            devs = []
            for t in range(start, end):
                if np.isnan(readings[sid][t]):
                    continue
                other_vals = [readings[s][t] for s in others
                              if not np.isnan(readings[s][t])]
                if len(other_vals) >= 1:
                    devs.append(readings[sid][t] - np.mean(other_vals))
            if devs:
                chunk_biases.append(float(np.median(devs)))

        if len(chunk_biases) >= 2:
            result[sid] = float(max(chunk_biases) - min(chunk_biases))
        else:
            result[sid] = 0.0

    return result


def detect_outliers_by_annotation(
    values: np.ndarray,
    sigma: float,
    threshold_sigmas: float = 5.0,
    window: int = 20,
    neighbor_values: dict = None,
) -> np.ndarray:
    """Detect outliers using annotation-derived uncertainty.

    Two modes:
    1. Single-sensor (neighbor_values=None): flag readings deviating
       from a local rolling median. Works for isolated spikes only.
    2. Cross-sensor (neighbor_values provided): flag readings deviating
       from the neighbor consensus. Works for sustained failures.

    Cross-sensor mode is strongly preferred for real IoT data where
    sensor failures produce long consecutive runs, not isolated spikes.

    Args:
        values: Array of sensor readings.
        sigma: Known measurement noise (from annotation metadata).
        threshold_sigmas: Multiplier for outlier threshold.
        window: Rolling window size for single-sensor mode.
        neighbor_values: Dict mapping neighbor_id -> array of values.
            If provided, uses cross-sensor consensus instead of
            rolling median.

    Returns:
        Boolean array, True = flagged as outlier.
    """
    n = len(values)
    flagged = np.zeros(n, dtype=bool)

    if neighbor_values is not None:
        # Cross-sensor mode: compare against neighbor consensus
        neighbor_arrays = list(neighbor_values.values())
        for i in range(n):
            if np.isnan(values[i]):
                continue
            neighbor_vals = [arr[i] for arr in neighbor_arrays
                            if i < len(arr) and not np.isnan(arr[i])]
            if len(neighbor_vals) < 1:
                continue
            consensus = np.median(neighbor_vals)
            if abs(values[i] - consensus) > threshold_sigmas * sigma:
                flagged[i] = True
    else:
        # Single-sensor mode: rolling median (isolated spikes only)
        for i in range(n):
            if np.isnan(values[i]):
                continue
            start = max(0, i - window // 2)
            end = min(n, i + window // 2 + 1)
            local = values[start:end]
            local_clean = local[~np.isnan(local)]
            if len(local_clean) < 3:
                continue
            local_median = np.median(local_clean)
            if abs(values[i] - local_median) > threshold_sigmas * sigma:
                flagged[i] = True

    return flagged


# =====================================================================
# 5. Code Complexity Comparison
# =====================================================================

def pipeline_jsonldex(
    sensors: List[Dict[str, Any]],
    readings: List[Dict[str, Any]],
) -> List[dict]:
    """Full IoT pipeline using jsonld-ex.

    Annotate readings → convert to SSN/SOSA → encode as CBOR-LD.
    Developer never references SSN/SOSA URIs directly.
    """
    results = []
    for r in readings:
        # Step 1: Annotate
        annotated = jx.annotate(
            value=r["value"],
            source=r["sensor_id"],
            unit=r.get("unit", "celsius"),
            extracted_at=r.get("timestamp"),
            measurement_uncertainty=r.get("sigma"),
            method="direct-measurement",
        )

        # Step 2: Build document
        doc = {
            "@context": {"@vocab": "https://schema.org/"},
            "@type": "Observation",
            "@id": f"obs-{r['sensor_id']}-{r.get('timestamp', '')}",
            "temperature": annotated,
        }

        # Step 3: Convert to SSN/SOSA
        ssn_doc, _ = jx.to_ssn(doc)

        # Step 4: Encode as CBOR-LD
        cbor_bytes = jx.to_cbor(doc)

        results.append(ssn_doc)
    return results


def pipeline_rdflib(
    sensors: List[Dict[str, Any]],
    readings: List[Dict[str, Any]],
) -> List[Any]:
    """Equivalent IoT pipeline using manual rdflib SSN/SOSA construction.

    Developer must know and reference SSN/SOSA ontology URIs directly.
    This is the idiomatic way a competent RDF developer would do it.
    """
    try:
        from rdflib import Graph, Namespace, Literal, BNode, URIRef
        from rdflib.namespace import RDF, XSD, RDFS
    except ImportError:
        return []

    SOSA = Namespace("http://www.w3.org/ns/sosa/")
    SSN = Namespace("http://www.w3.org/ns/ssn/")
    SSN_SYS = Namespace("http://www.w3.org/ns/ssn/systems/")
    QUDT = Namespace("http://qudt.org/schema/qudt/")
    SCHEMA = Namespace("https://schema.org/")

    results = []
    for r in readings:
        g = Graph()
        g.bind("sosa", SOSA)
        g.bind("ssn", SSN)
        g.bind("ssn-system", SSN_SYS)
        g.bind("qudt", QUDT)
        g.bind("schema", SCHEMA)

        obs_id = URIRef(f"obs-{r['sensor_id']}-{r.get('timestamp', '')}")
        sensor_id = URIRef(r["sensor_id"])
        result_node = BNode()
        prop_node = URIRef("temperature")

        # Observation
        g.add((obs_id, RDF.type, SOSA.Observation))
        g.add((obs_id, SOSA.madeBySensor, sensor_id))
        g.add((obs_id, SOSA.observedProperty, prop_node))
        g.add((obs_id, SOSA.hasResult, result_node))

        # Result
        g.add((result_node, RDF.type, SOSA.Result))
        g.add((result_node, QUDT.numericValue,
               Literal(r["value"], datatype=XSD.double)))
        g.add((result_node, QUDT.unit,
               Literal(r.get("unit", "celsius"))))

        # Sensor
        g.add((sensor_id, RDF.type, SOSA.Sensor))

        # System capability (if sigma provided)
        if r.get("sigma") is not None:
            cap_node = BNode()
            acc_node = BNode()
            g.add((sensor_id, SSN_SYS.hasSystemCapability, cap_node))
            g.add((cap_node, RDF.type, SSN_SYS.SystemCapability))
            g.add((cap_node, SSN_SYS.hasSystemProperty, acc_node))
            g.add((acc_node, RDF.type, SSN_SYS.Accuracy))
            g.add((acc_node, QUDT.numericValue,
                   Literal(r["sigma"], datatype=XSD.double)))

        # Observable property
        g.add((prop_node, RDF.type, SOSA.ObservableProperty))

        results.append(g)
    return results


def count_loc(func) -> int:
    """Count non-blank, non-comment lines in a function's source."""
    source = inspect.getsource(func)
    lines = source.split("\n")
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        count += 1
    return count


def count_ssn_uris(func) -> int:
    """Count distinct SSN/SOSA URI references in a function's source.

    Looks for references to sosa:, ssn:, SOSA., SSN., SSN_SYS.
    """
    source = inspect.getsource(func)
    patterns = [
        r"SOSA\.\w+",
        r"SSN\.\w+",
        r"SSN_SYS\.\w+",
        r"sosa:",
        r"ssn:",
        r"ssn-system:",
        r"www\.w3\.org/ns/sosa",
        r"www\.w3\.org/ns/ssn",
    ]
    uris = set()
    for pattern in patterns:
        matches = re.findall(pattern, source)
        uris.update(matches)
    return len(uris)
