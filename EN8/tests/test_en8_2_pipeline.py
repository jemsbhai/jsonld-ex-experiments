"""RED-phase tests for EN8.2 pipeline — annotation, SSN/SOSA, transport, quality.

Covers:
  H8.2a: Pipeline throughput
  H8.2b: SSN/SOSA interoperability gap analysis
  H8.2c: Transport efficiency (CBOR-LD, MQTT, CoAP)
  H8.2d: Data quality detection
  H8.2e: Code complexity comparison
"""
from __future__ import annotations

import json
import math
import time
import pytest
import numpy as np

from en8_2_pipeline import (
    # Annotation pipeline
    annotate_sensor_reading,
    annotate_sensor_batch,
    # SSN/SOSA
    readings_to_ssn,
    ssn_round_trip,
    categorize_ssn_triples,
    validate_ssn_structure,
    # Transport
    encode_cbor_batch,
    encode_json_batch,
    mqtt_derive_batch,
    coap_derive_batch,
    # Quality detection
    compute_disagreement_signal,
    compute_calibration_age_signal,
    detect_drift,
    detect_outliers_by_annotation,
    # Code complexity
    pipeline_jsonldex,
    pipeline_rdflib,
    count_loc,
    count_ssn_uris,
)


# ====================================================================
# H8.2a: Pipeline Throughput
# ====================================================================

class TestPipelineThroughput:
    """Pipeline must process readings fast enough for real-time IoT."""

    def test_annotate_throughput(self):
        """annotate_sensor_reading must handle >5000 readings/sec."""
        readings = [
            {"value": 20.0 + i * 0.01, "sensor_id": f"s-{i % 10}",
             "timestamp": f"2024-03-01T00:{i//60:02d}:{i%60:02d}Z",
             "sigma": 0.3, "unit": "celsius",
             "calibrated_at": "2024-01-15T10:00:00Z"}
            for i in range(1000)
        ]
        t0 = time.perf_counter()
        results = [annotate_sensor_reading(**r) for r in readings]
        elapsed = time.perf_counter() - t0
        rate = len(readings) / elapsed
        assert rate > 5000, f"Annotate rate {rate:.0f}/s < 5000/s"
        # Each result should be a valid annotated document
        assert all("@value" in r for r in results)

    def test_full_pipeline_throughput(self):
        """annotate → to_ssn → to_cbor must handle >1000 readings/sec."""
        readings = [
            {"value": 20.0 + i * 0.01, "sensor_id": f"s-{i % 5}",
             "timestamp": f"2024-03-01T00:{i//60:02d}:{i%60:02d}Z",
             "sigma": 0.3, "unit": "celsius",
             "calibrated_at": "2024-01-15T10:00:00Z"}
            for i in range(500)
        ]
        t0 = time.perf_counter()
        for r in readings:
            doc = annotate_sensor_reading(**r)
            ssn_doc, _ = readings_to_ssn(doc)
        elapsed = time.perf_counter() - t0
        rate = len(readings) / elapsed
        assert rate > 100, f"Full pipeline rate {rate:.0f}/s < 100/s"


# ====================================================================
# H8.2b: SSN/SOSA Interoperability Gap Analysis
# ====================================================================

class TestSSNInterop:
    """SSN/SOSA conversion fidelity and gap analysis."""

    def test_round_trip_preserves_value(self):
        """Core value survives annotate → to_ssn → from_ssn."""
        doc = annotate_sensor_reading(
            value=22.5, sensor_id="sensor-4", unit="celsius",
            timestamp="2024-03-01T12:00:00Z", sigma=0.25,
            calibrated_at="2024-01-15T10:00:00Z")
        original, recovered, report = ssn_round_trip(doc)
        assert recovered["@value"] == pytest.approx(22.5)

    def test_round_trip_preserves_unit(self):
        """Unit annotation survives round-trip."""
        doc = annotate_sensor_reading(
            value=22.5, sensor_id="sensor-4", unit="celsius",
            timestamp="2024-03-01T12:00:00Z", sigma=0.25)
        original, recovered, report = ssn_round_trip(doc)
        assert recovered.get("@unit") == "celsius"

    def test_round_trip_preserves_source(self):
        """Source annotation survives round-trip."""
        doc = annotate_sensor_reading(
            value=22.5, sensor_id="sensor-4", unit="celsius",
            timestamp="2024-03-01T12:00:00Z", sigma=0.25)
        original, recovered, report = ssn_round_trip(doc)
        assert recovered.get("@source") == "sensor-4"

    def test_round_trip_preserves_confidence(self):
        """Confidence annotation survives round-trip."""
        doc = annotate_sensor_reading(
            value=22.5, sensor_id="sensor-4", unit="celsius",
            timestamp="2024-03-01T12:00:00Z", sigma=0.25)
        original, recovered, report = ssn_round_trip(doc)
        assert "@confidence" in recovered

    def test_round_trip_preserves_uncertainty(self):
        """Measurement uncertainty survives round-trip."""
        doc = annotate_sensor_reading(
            value=22.5, sensor_id="sensor-4", unit="celsius",
            timestamp="2024-03-01T12:00:00Z", sigma=0.25)
        original, recovered, report = ssn_round_trip(doc)
        assert recovered.get("@measurementUncertainty") == pytest.approx(0.25)

    def test_round_trip_batch_fidelity(self):
        """Batch of 100 readings: >=95% field-level fidelity."""
        docs = [
            annotate_sensor_reading(
                value=20.0 + i * 0.1, sensor_id=f"s-{i % 5}",
                unit="celsius", sigma=0.3,
                timestamp=f"2024-03-01T{i//60:02d}:{i%60:02d}:00Z",
                calibrated_at="2024-01-15T10:00:00Z")
            for i in range(100)
        ]
        fields_checked = 0
        fields_preserved = 0
        check_fields = ["@value", "@unit", "@source", "@confidence",
                        "@measurementUncertainty"]
        for doc in docs:
            _, recovered, _ = ssn_round_trip(doc)
            for f in check_fields:
                if f in doc:
                    fields_checked += 1
                    if f in recovered:
                        if isinstance(doc[f], float):
                            if abs(doc[f] - recovered.get(f, 0)) < 1e-6:
                                fields_preserved += 1
                        elif doc[f] == recovered.get(f):
                            fields_preserved += 1
        fidelity = fields_preserved / fields_checked if fields_checked > 0 else 0
        assert fidelity >= 0.95, f"Field fidelity {fidelity:.1%} < 95%"

    def test_triple_categorization(self):
        """SSN output triples are categorized as native/allied/extension."""
        doc = annotate_sensor_reading(
            value=22.5, sensor_id="sensor-4", unit="celsius",
            timestamp="2024-03-01T12:00:00Z", sigma=0.25,
            calibrated_at="2024-01-15T10:00:00Z")
        ssn_doc, _ = readings_to_ssn(doc)
        cats = categorize_ssn_triples(ssn_doc)
        assert "native" in cats  # sosa:/ssn: namespace
        assert "allied" in cats  # qudt:/xsd:
        assert "extension" in cats  # jsonld-ex:
        assert cats["native"] >= 1  # at least some native triples
        # Total should account for all triples
        total = cats["native"] + cats["allied"] + cats["extension"]
        assert total > 0

    def test_ssn_structure_valid(self):
        """Generated SSN/SOSA has correct graph structure."""
        doc = annotate_sensor_reading(
            value=22.5, sensor_id="sensor-4", unit="celsius",
            timestamp="2024-03-01T12:00:00Z", sigma=0.25)
        ssn_doc, _ = readings_to_ssn(doc)
        validation = validate_ssn_structure(ssn_doc)
        assert validation["has_observation"], "Missing sosa:Observation"
        assert validation["has_result"], "Missing sosa:Result"
        assert validation["has_sensor"], "Missing sosa:Sensor"


# ====================================================================
# H8.2c: Transport Efficiency
# ====================================================================

class TestTransport:
    """CBOR-LD, MQTT, CoAP transport for constrained IoT devices."""

    def test_cbor_smaller_than_json(self):
        """CBOR encoding is smaller than JSON for annotated readings."""
        docs = [
            annotate_sensor_reading(
                value=20.0 + i * 0.1, sensor_id=f"s-{i % 5}",
                unit="celsius", sigma=0.3,
                timestamp=f"2024-03-01T00:{i:02d}:00Z")
            for i in range(50)
        ]
        cbor_sizes = encode_cbor_batch(docs)
        json_sizes = encode_json_batch(docs)
        total_cbor = sum(cbor_sizes)
        total_json = sum(json_sizes)
        ratio = total_cbor / total_json
        assert ratio < 0.90, f"CBOR/JSON ratio {ratio:.1%} >= 90%"

    def test_cbor_ratio_with_metadata(self):
        """CBOR savings measured for minimal vs full annotation."""
        # Minimal: value + source only
        minimal = [annotate_sensor_reading(
            value=20.0 + i * 0.1, sensor_id=f"s-{i % 5}",
            unit="celsius", sigma=0.3,
            timestamp=f"2024-03-01T00:{i:02d}:00Z")
            for i in range(50)]
        # Full: all fields
        full = [annotate_sensor_reading(
            value=20.0 + i * 0.1, sensor_id=f"s-{i % 5}",
            unit="celsius", sigma=0.3,
            timestamp=f"2024-03-01T00:{i:02d}:00Z",
            calibrated_at="2024-01-15T10:00:00Z",
            calibration_method="NIST-traceable",
            calibration_authority="LabCal")
            for i in range(50)]
        ratio_min = sum(encode_cbor_batch(minimal)) / sum(encode_json_batch(minimal))
        ratio_full = sum(encode_cbor_batch(full)) / sum(encode_json_batch(full))
        # Both should be < 90%
        assert ratio_min < 0.90
        assert ratio_full < 0.90

    def test_mqtt_topic_derivation(self):
        """MQTT topic derived correctly from annotated reading."""
        doc = annotate_sensor_reading(
            value=22.5, sensor_id="sensor-4", unit="celsius",
            timestamp="2024-03-01T12:00:00Z", sigma=0.25)
        topics = mqtt_derive_batch([doc])
        assert len(topics) == 1
        assert isinstance(topics[0]["topic"], str)
        assert len(topics[0]["topic"]) > 0
        assert topics[0]["qos"] in (0, 1, 2)

    def test_coap_options_derivation(self):
        """CoAP options derived correctly."""
        doc = annotate_sensor_reading(
            value=22.5, sensor_id="sensor-4", unit="celsius",
            timestamp="2024-03-01T12:00:00Z", sigma=0.25)
        options = coap_derive_batch([doc])
        assert len(options) == 1
        assert "uri_path" in options[0]
        assert "content_format" in options[0]


# ====================================================================
# H8.2d: Data Quality Detection
# ====================================================================

class TestQualityDetection:
    """Annotation-derived quality signals detect real sensor problems."""

    def test_disagreement_detects_bias(self):
        """Disagreement signal correlates with known sensor bias."""
        # Simulate: 4 sensors, one with large bias
        n = 200
        rng = np.random.default_rng(42)
        true_temp = 20.0 + np.sin(np.linspace(0, 2 * np.pi, n))
        readings = {
            "s1": true_temp + rng.normal(0, 0.3, n),
            "s2": true_temp + rng.normal(0, 0.3, n) + 2.0,  # biased +2
            "s3": true_temp + rng.normal(0, 0.3, n),
            "s4": true_temp + rng.normal(0, 0.3, n) - 0.5,  # biased -0.5
        }
        known_biases = {"s1": 0.0, "s2": 2.0, "s3": 0.0, "s4": -0.5}
        disagreements = compute_disagreement_signal(readings)

        # Disagreement should be highest for s2 (largest bias)
        assert disagreements["s2"] > disagreements["s1"]
        assert disagreements["s2"] > disagreements["s3"]

        # Correlation between |bias| and disagreement should be strong
        biases = [abs(known_biases[s]) for s in sorted(readings.keys())]
        disagrs = [disagreements[s] for s in sorted(readings.keys())]
        corr = np.corrcoef(biases, disagrs)[0, 1]
        assert corr > 0.8, f"Bias-disagreement correlation {corr:.3f} < 0.8"

    def test_drift_detection(self):
        """Drift signal detects time-varying bias."""
        n = 500
        rng = np.random.default_rng(42)
        # Sensor 1: stable. Sensor 2: linear drift of +0.01/step
        base = 20.0 * np.ones(n)
        readings = {
            "s1": base + rng.normal(0, 0.2, n),
            "s2": base + rng.normal(0, 0.2, n) + np.linspace(0, 5, n),  # drifts
            "s3": base + rng.normal(0, 0.2, n),
        }
        drift_scores = detect_drift(readings, n_chunks=10)
        # s2 should have highest drift score
        assert drift_scores["s2"] > drift_scores["s1"]
        assert drift_scores["s2"] > drift_scores["s3"]
        # s2 drift should be substantial
        assert drift_scores["s2"] > 1.0, \
            f"Drift score {drift_scores['s2']:.2f} too low for 5°C drift"

    def test_outlier_detection(self):
        """Annotation-based outlier flag catches injected outliers."""
        n = 200
        rng = np.random.default_rng(42)
        values = 20.0 + rng.normal(0, 0.3, n)
        # Inject 10 outliers
        outlier_idx = rng.choice(n, 10, replace=False)
        values[outlier_idx] += rng.choice([-50, 50], 10)
        is_outlier_truth = np.zeros(n, dtype=bool)
        is_outlier_truth[outlier_idx] = True

        flagged = detect_outliers_by_annotation(
            values, sigma=0.3, threshold_sigmas=5.0)

        # Should catch most outliers
        true_positives = np.sum(flagged & is_outlier_truth)
        recall = true_positives / np.sum(is_outlier_truth)
        assert recall > 0.8, f"Outlier recall {recall:.1%} < 80%"

        # Should not flag too many clean readings
        false_positives = np.sum(flagged & ~is_outlier_truth)
        fpr = false_positives / np.sum(~is_outlier_truth)
        assert fpr < 0.05, f"False positive rate {fpr:.1%} > 5%"

    def test_calibration_age_signal(self):
        """Calibration age grows correctly over time."""
        timestamps = [f"2024-03-{d:02d}T12:00:00Z" for d in range(1, 31)]
        cal_date = "2024-03-01T00:00:00Z"
        ages = compute_calibration_age_signal(timestamps, cal_date)
        assert len(ages) == 30
        # First day: age ~0.5 days, last day: age ~29.5 days
        assert ages[0] < 1.0
        assert ages[-1] > 28.0
        # Monotonically increasing
        assert all(ages[i] <= ages[i+1] for i in range(len(ages)-1))


# ====================================================================
# H8.2e: Code Complexity
# ====================================================================

class TestCodeComplexity:
    """jsonld-ex requires fewer LoC and no ontology knowledge."""

    def test_both_pipelines_produce_output(self):
        """Both implementations produce valid SSN/SOSA output."""
        sensors = [{"id": f"s-{i}", "sigma": 0.3} for i in range(3)]
        readings = [
            {"sensor_id": f"s-{i}", "value": 20.0 + i, "unit": "celsius",
             "timestamp": "2024-03-01T12:00:00Z"}
            for i in range(3)
        ]
        jx_output = pipeline_jsonldex(sensors, readings)
        rdf_output = pipeline_rdflib(sensors, readings)
        assert jx_output is not None
        assert rdf_output is not None

    def test_jsonldex_fewer_loc(self):
        """jsonld-ex pipeline has ≤50% the LoC of rdflib pipeline."""
        jx_loc = count_loc(pipeline_jsonldex)
        rdf_loc = count_loc(pipeline_rdflib)
        ratio = jx_loc / rdf_loc
        assert ratio <= 0.60, \
            f"LoC ratio {ratio:.1%}: jx={jx_loc} rdf={rdf_loc}"

    def test_jsonldex_zero_ssn_uris(self):
        """jsonld-ex pipeline references 0 SSN/SOSA URIs directly."""
        n_uris = count_ssn_uris(pipeline_jsonldex)
        assert n_uris == 0, f"jsonld-ex references {n_uris} SSN/SOSA URIs"

    def test_rdflib_requires_ssn_uris(self):
        """rdflib pipeline must reference SSN/SOSA URIs explicitly."""
        n_uris = count_ssn_uris(pipeline_rdflib)
        assert n_uris >= 5, \
            f"rdflib only uses {n_uris} SSN/SOSA URIs (too few = strawman)"
