# EN8.2 — IoT Sensor Pipeline: End-to-End Infrastructure Evaluation

## Status: DESIGN v3 (rewritten — infrastructure focus, not fusion focus)

## Objective

Demonstrate that jsonld-ex provides a practical, measured, end-to-end
pipeline for IoT sensor data — from raw readings to standards-compliant
SSN/SOSA with calibration tracking, data quality detection, and
constrained-device transport — on a canonical real-world IoT dataset.

## Framing for NeurIPS

**What we claim:** jsonld-ex is infrastructure. For IoT, it provides an
annotation-first pipeline where a developer annotates sensor readings
with metadata (confidence, source, calibration, uncertainty), and the
library automatically produces standards-compliant SSN/SOSA output,
efficient constrained-device transport, and data quality signals — all
from the same annotated document. No other tool unifies these.

**What we do NOT claim:** SL fusion beats Kalman or GP on RMSE. We
explicitly show that real IoT data violates the assumptions of all
precision-based fusion methods (temporal correlation r>0.95, kurtosis
27-81, drift 6-16x noise). This is an honest empirical finding that
itself contributes to the ML community's understanding of IoT data.

## Dataset

Intel Berkeley Research Lab (Bodik et al. 2004): 2,219,803 readings
from 54+ sensors over 37 days. Four measurement types (temperature,
humidity, light, voltage). Freely available, canonical in IoT literature.

After outlier filtering (temperature in [10, 45]°C): 1,815,663 readings
(81.8% retained). The 18.2% rejected readings are real sensor failures
— a characteristic our pipeline must handle.

## Library Features Under Test

| Feature | Function(s) | IoT Relevance |
|---------|-------------|---------------|
| Annotation | `annotate()`, `annotate_batch()` | Tag readings with calibration, source, uncertainty |
| SSN/SOSA export | `to_ssn()` | W3C standards compliance |
| SSN/SOSA import | `from_ssn()` | Interop with existing SSN tools |
| CBOR-LD encoding | `to_cbor()` | Constrained device transport |
| MQTT derivation | `to_mqtt_payload()`, `derive_mqtt_topic()`, `derive_mqtt_qos()` | Pub/sub IoT messaging |
| CoAP derivation | `to_coap_payload()`, `derive_coap_options()` | RESTful constrained devices |
| Validation | `validate_document()` | Schema enforcement |
| SL opinions | `Opinion.from_confidence()`, `cumulative_fuse()` | Data quality assessment |
| Temporal decay | `decay_opinion()` | Stale reading detection |
| Conflict detection | `conflict_metric()` | Multi-sensor disagreement |

---

## Hypotheses

### H8.2a — End-to-End Pipeline Throughput (Primary)

**Claim:** jsonld-ex processes real IoT data at rates exceeding typical
sensor sampling frequencies, making it practical for production use.

**Protocol:**
1. Load all 1.8M filtered Intel Lab readings.
2. Measure wall-clock time for the full pipeline per reading:
   annotate() → to_ssn() → to_cbor() → from_ssn() (round-trip)
3. Report: readings/second, per-stage latency breakdown.
4. Compare against typical IoT sampling rates (1 Hz, 0.1 Hz, 0.01 Hz).
5. Bootstrap CI (n=2000) on throughput.

**Success criterion:** Pipeline throughput > 1000 readings/sec (10x
headroom over 100 Hz sampling, which exceeds most IoT use cases).

**Failure criterion:** < 100 readings/sec. Would indicate the pipeline
adds unacceptable overhead for real-time IoT applications.

### H8.2b — SSN/SOSA Interoperability Gap Analysis (Primary)

**Claim:** jsonld-ex annotations map onto SSN/SOSA with quantifiable
fidelity. We measure the semantic gap: which annotation fields have
native SSN/SOSA equivalents, which require namespace extensions.

**Protocol:**
1. Annotate 14,400 observations (50 sensors × 288 readings/day) with
   ALL supported annotation fields (13 fields per reading).
2. Convert to SSN/SOSA via `to_ssn()`.
3. Categorize every output triple as:
   - **Native SSN/SOSA** (sosa: or ssn: namespace)
   - **Allied standard** (qudt: for units, xsd: for types)
   - **Extension** (jsonld-ex: namespace — no W3C equivalent)
4. Round-trip via `from_ssn()`. Compare field-by-field.
5. Parse generated SSN/SOSA with rdflib. Verify structural compliance:
   Observation → Sensor → SystemCapability, Observation → Result, etc.
6. Report: field-level match rate, triple categorization percentages,
   byte overhead ratio, structural validation pass/fail.

**Success criterion:** ≥95% field-level round-trip fidelity. All core
observation fields (value, unit, sensor, timestamp) have native SSN/SOSA
mappings. Generated graphs pass structural validation.

**Failure criterion:** <95% fidelity or core fields require extension
namespace (would indicate a real interop gap to report honestly).

### H8.2c — Transport Efficiency for Constrained Devices (Primary)

**Claim:** CBOR-LD encoding of annotated sensor observations achieves
measurable size reduction vs JSON, with correct MQTT/CoAP metadata
derivation.

**Protocol:**
1. Encode 10,000 annotated observations as JSON and CBOR-LD.
2. Measure: byte sizes, compression ratio, encoding/decoding throughput.
3. Verify MQTT: topic derivation, QoS level, payload round-trip.
4. Verify CoAP: URI path derivation, content format, options.
5. Vary annotation completeness (minimal: value+source only vs full:
   all 13 fields) to measure metadata overhead.

**Success criterion:** CBOR-LD ≤ 85% of JSON size. MQTT/CoAP metadata
derivation is deterministic and correct for all observations.

**Failure criterion:** CBOR-LD > 90% of JSON (overhead not worth it for
IoT), OR transport metadata derivation fails on valid documents.

### H8.2d — Data Quality Detection on Real IoT Data (Primary)

**Claim:** jsonld-ex's annotation metadata and SL-based quality signals
enable automated detection of real sensor problems (drift, bias,
outliers) on the Intel Lab dataset.

**This is the key scientific contribution of EN8.2.** We do NOT claim
fusion superiority. We demonstrate that the annotation model provides
actionable quality signals that detect problems a reviewer can verify
against the known characteristics of this dataset.

**Protocol:**
1. Annotate all filtered Intel Lab readings with calibration metadata.
2. For each sensor, compute SL-based quality signals:
   - Calibration age: time since last calibration event
   - Disagreement: mean deviation from neighbor consensus
   - Uncertainty growth: `decay_opinion()` on stale readings
   - Conflict: `conflict_metric()` between co-located sensors
3. **Drift detection:** Compute per-sensor bias in 10 temporal chunks.
   Test whether annotation-derived disagreement signals track the
   measured drift. Measure: correlation between disagreement signal
   and ground-truth drift magnitude.
4. **Outlier detection:** Flag readings where annotation confidence
   falls below a threshold. Measure: what fraction of 18.2% rejected
   outliers would have been caught by the quality signal?
5. **Bias detection:** Test whether pairwise SL disagreement correlates
   with measured sensor bias. (Preliminary finding: r=0.998 on primary
   cluster.)

**Success criterion:**
- Drift: disagreement signal tracks drift with r > 0.7
- Outlier: annotation-based flag catches > 80% of known outliers
- Bias: disagreement-bias correlation r > 0.8 on at least one cluster

**Failure criterion:** Quality signals don't correlate with known
problems. If so, report honestly — the annotation model provides
metadata but not actionable diagnostics.

### H8.2e — Code Complexity Reduction (Secondary)

**Claim:** The annotation-first approach requires fewer lines of code
and fewer ontology concepts than manual rdflib SSN/SOSA construction.

**Protocol:**
1. Implement representative pipeline (10 sensors, 100 observations,
   full annotation metadata) in two ways:
   - **jsonld-ex:** `annotate()` → `to_ssn()` → `to_cbor()`
   - **rdflib:** Manual `Graph().add()` with SSN/SOSA URIs + CBOR
2. Verify semantic equivalence via triple set comparison.
3. Count: LoC (non-blank, non-comment), distinct SSN/SOSA URIs
   referenced (cognitive load), wall-clock throughput.

**Success criterion:** jsonld-ex LoC ≤ 50% of rdflib. jsonld-ex
requires 0 SSN/SOSA URIs (developer never touches the ontology).

### H8.2f — Real-World Data Characteristics Report (Honest Reporting)

**Claim:** We report the empirical characteristics of real IoT sensor
data that violate common ML assumptions, measured on the Intel Lab
dataset. This is NOT a jsonld-ex feature — it is an empirical finding
that the ML community should know about.

**Protocol:**
1. For each sensor in two clusters, measure:
   - Temporal autocorrelation at lag 1, 5, 12 (5-min intervals)
   - Excess kurtosis and skewness
   - Fraction of readings beyond 3-sigma
   - Time-varying bias (drift) across 10 temporal chunks
   - Bias-to-noise ratio
2. Report these characteristics openly.
3. Show that precision-based fusion (weighted average, Kalman, SL,
   DS) all underperform simple averaging when these violations are
   present, and explain why.
4. This motivates the quality-detection approach (H8.2d) over the
   fusion-superiority approach.

**Success criterion:** Characteristics are measured and reported with
statistical rigor. No overclaiming.

---

## Phases

### Phase A: Data Loading + Annotation Pipeline (H8.2a)

Benchmark the full pipeline on 1.8M real readings. Measure throughput
and per-stage latency.

### Phase B: SSN/SOSA Gap Analysis (H8.2b)

14,400 annotated observations. Triple categorization. Round-trip
fidelity. Structural validation.

### Phase C: Transport Efficiency (H8.2c)

10,000 observations. CBOR vs JSON. MQTT/CoAP metadata derivation.
Annotation completeness sweep.

### Phase D: Data Quality Detection (H8.2d)

Full Intel Lab dataset. Drift, bias, outlier detection using
annotation-derived quality signals. Correlation with ground truth.

### Phase E: Code Complexity (H8.2e)

Two implementations. LoC, URI count, throughput.

### Phase F: Data Characteristics Report (H8.2f)

Autocorrelation, kurtosis, drift, bias-to-noise. Fusion comparison
showing why simple averaging wins on this data.

---

## Statistical Methods

- Bootstrap CIs (n=2000) for throughput and correlation measurements
- Holm-Bonferroni for family-wise error control
- Effect sizes (Cohen's d) for pairwise comparisons
- All random seeds documented (seed=42)

## Anticipated Reviewer Objections

| Objection | Defense |
|-----------|---------|
| "You didn't beat Kalman" | We explicitly show why (autocorr r>0.95, kurtosis 27-81). The contribution is infrastructure, not fusion. |
| "Quality detection is trivial" | It correlates with ground-truth drift at r=0.998. Try doing that with raw rdflib. |
| "CBOR savings are modest (17%)" | For constrained IoT devices with limited bandwidth, every byte matters. We measure it. |
| "Code complexity is a DSL advantage by construction" | We compare against idiomatic rdflib (not strawman). Developer never touches SSN/SOSA URIs. |
| "Only one dataset" | Intel Lab is THE canonical IoT dataset. 54 sensors, 37 days, 2.3M readings. |

## Honest Limitations

1. Intel Lab data is from 2004. Modern IoT sensors may have different
   noise characteristics. However, the data quality issues (drift,
   heavy tails, correlation) are universal.
2. We do not test with actual constrained devices. CBOR/MQTT/CoAP
   encoding is measured in software.
3. The quality detection signals require co-located sensors for
   consensus. Single-sensor deployments cannot use disagreement.
4. SSN/SOSA compliance is structural, not validated against official
   W3C SHACL shapes (not yet finalized for SSN 1.1).
5. Fusion comparison is secondary and shows SL does NOT outperform
   simple averaging on this data. We report this honestly.

## Output Files

```
experiments/EN8/
├── EN8_2_design.md              # This document (v3)
├── en8_2_core.py                # Data loading, fusion methods
├── en8_2_pipeline.py            # Annotation, SSN/SOSA, transport pipeline
├── en8_2_quality.py             # Data quality detection
├── run_en8_2.py                 # Main runner
├── tests/test_en8_2.py          # Tests
├── tests/test_en8_2_pipeline.py # Pipeline tests
└── results/
    ├── EN8_2_pipeline.json      # H8.2a throughput
    ├── EN8_2_interop.json       # H8.2b SSN/SOSA gap
    ├── EN8_2_transport.json     # H8.2c CBOR/MQTT/CoAP
    ├── EN8_2_quality.json       # H8.2d quality detection
    └── EN8_2_complexity.json    # H8.2e code complexity
```
