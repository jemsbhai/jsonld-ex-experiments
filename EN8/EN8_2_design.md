# EN8.2 — IoT Sensor Pipeline: SSN/SOSA Integration

## Status: DESIGN v2 (hardened for NeurIPS reviewer scrutiny)

## Objective

Quantify the interoperability gap between ML/AI annotation metadata and the
W3C SSN/SOSA ontology, demonstrate that jsonld-ex bridges that gap with
measured fidelity, and evaluate whether Subjective Logic fusion provides
principled advantages over standard baselines for heterogeneous sensor
state estimation — including under model misspecification.

## Framing for NeurIPS

This experiment contributes to the paper's **interoperability** and **ML pipeline**
narratives. The scientific questions are:

1. **What information is lost** when ML annotation metadata crosses the
   SSN/SOSA ontology boundary, and what is the cost of preserving it?
2. **Does the annotation-first approach** (annotate → export) reduce cognitive
   complexity compared to the ontology-first approach (construct SSN directly)?
3. **Can SL fusion with temporal decay** match or exceed standard baselines
   (including Kalman filtering) for sensor state estimation — and when does
   it fail?

This is NOT a claim that jsonld-ex is a better sensor fusion engine than Kalman
filters. The claim is that jsonld-ex provides **interoperable ML infrastructure**
where uncertainty-aware fusion, standards-compliant export, and calibration
tracking emerge as natural side-effects of the annotation model.

## Library Features Under Test

- `annotate()` — attach confidence, source, temporal, calibration metadata
- `to_ssn()` / `from_ssn()` — bidirectional SSN/SOSA conversion (owl_interop.py)
- `Opinion.from_evidence()` — evidence-based SL opinion construction
- `cumulative_fuse()` — multi-sensor fusion
- `decay_opinion()` — temporal decay for stale readings
- All calibration annotations: `@measurementUncertainty`, `@calibratedAt`,
  `@calibrationMethod`, `@calibrationAuthority`, `@unit`

---

## Hypotheses

### H8.2a — SSN/SOSA Interoperability Gap Analysis (Primary)

**Claim:** jsonld-ex annotations map onto SSN/SOSA with quantifiable fidelity:
some fields have native SSN/SOSA equivalents, others require namespace
extensions. The generated SSN/SOSA graphs are standards-compliant.

**Why this matters (anticipated objection: "just a unit test"):** The question
is NOT "does to_ssn/from_ssn work?" — that's a unit test. The question is:
"What is the semantic gap between ML annotation metadata and the W3C sensor
ontology, and what does bridging it cost?" This is an empirical interoperability
measurement.

**Protocol:**
1. Generate 14,400 annotated observations across 50 sensors. Every observation
   populates ALL supported annotation fields (7 core + 6 calibration/aggregation).
2. Convert each parent document to SSN/SOSA via `to_ssn()`.
3. Categorize every output triple as:
   - **Native SSN/SOSA** (uses sosa: or ssn: namespace) — zero extension needed
   - **Extension** (uses jsonld-ex: namespace) — SSN/SOSA has no equivalent
   - **Bridge** (uses qudt: or other standard namespace) — covered by allied standards
4. Convert back via `from_ssn()`. Compare all fields.
5. **Standards compliance check:** Parse the generated SSN/SOSA JSON-LD with
   rdflib, verify all sosa:/ssn: terms resolve to the W3C ontology, verify
   the graph structure follows SSN/SOSA patterns (Observation→Sensor→Capability,
   Observation→FeatureOfInterest, Observation→ObservableProperty).
6. Report:
   - Field-level match rate per annotation type (7 core + 6 extended)
   - Triple categorization: % native, % extension, % bridge
   - Byte overhead ratio: SSN representation / jsonld-ex representation
   - Information density: unique annotation fields per kilobyte
   - Any ConversionReport warnings or errors

**Success criterion:** ≥99% field-level fidelity for round-trip. All 7 core
annotation fields have a standards-compliant SSN/SOSA mapping (native or bridge).
Generated graphs pass structural validation.

**Failure criterion:** <99% fidelity, OR silent data corruption, OR core fields
require jsonld-ex namespace extension (indicating a real interop gap that we
must honestly report).

**If the experiment fails:** Report the gap honestly. Document which ML
annotation concepts have NO W3C standard equivalent — this is itself a useful
finding for the standards community.

---

### H8.2b — Cognitive Complexity Reduction (Primary)

**Claim:** The annotation-first approach (jsonld-ex) requires the developer
to know fewer ontology concepts and write fewer lines of code than the
ontology-first approach (rdflib), while producing equivalent SSN/SOSA output.

**Anticipated objection: "Comparing abstraction levels is trivially favorable."**

**Counter-design:** We do NOT compare "high-level API vs raw triples" like a
strawman. We compare two *idiomatic, best-practice implementations* of the
same task, measuring multiple complexity dimensions:

**Protocol:**
1. Define a representative IoT pipeline task:
   - 10 sensors, 100 observations
   - Each observation: value, unit, source sensor, timestamp, confidence,
     measurement uncertainty, calibration metadata (date, method, authority)
   - Output: a valid SSN/SOSA observation graph
2. Implement in three ways:
   - **jsonld-ex path:** `annotate()` → `to_ssn()` (the framework approach)
   - **rdflib idiomatic:** Using rdflib Namespace objects, Graph helpers,
     BNode factories — the way a competent RDF developer would write it.
     NO deliberately verbose code.
   - **rdflib + helper functions:** A thin wrapper that a developer might
     write to reduce boilerplate (e.g., `add_observation(g, sensor, value, ...)`)
     — the "what if you rolled your own?" baseline.
3. Verify semantic equivalence: parse both outputs with rdflib, compare
   triple sets (modulo blank node renaming).
4. Measure:
   - **LoC** (non-blank, non-comment, non-import)
   - **Distinct SSN/SOSA URIs referenced** — how many ontology terms must the
     developer know? (Cognitive load metric.)
   - **Cyclomatic complexity** via radon (if applicable to comparison functions)
   - **Wall-clock throughput** for 1,000 iterations (bootstrap CI, n=1000)
5. Equivalence verification: both approaches produce the same triples.

**Success criterion:** jsonld-ex LoC ≤ 50% of idiomatic rdflib; jsonld-ex requires
0 SSN/SOSA URIs (developer never touches the ontology); throughput within 20%
of rdflib (no unacceptable overhead).

**Failure criterion:** LoC ratio < 2x against idiomatic rdflib, OR jsonld-ex
requires developer to reference SSN/SOSA URIs directly, OR >50% slower.

**Fairness controls:**
- The rdflib implementation is written to be idiomatic (Namespace objects, not
  raw URI strings; helper patterns where natural).
- Both implementations are committed to the repo for reviewer inspection.
- LoC is machine-counted (no cherry-picking).
- The rdflib+helper baseline tests "what if the developer already wrapped rdflib?"

---

### H8.2c — Multi-Sensor Fusion with Baselines (Primary)

**Claim:** SL fusion of co-located sensors with heterogeneous reliability
profiles matches or exceeds standard baselines on state estimation accuracy.

**Anticipated objection: "You built the noise model and the SL constructor from
the same model — of course SL wins."**

**Counter-design:** We include (a) a proper Kalman filter baseline, (b) a
misspecified-model condition where the assumed noise profiles are wrong, and
(c) pre-registered failure criteria. If SL loses to Kalman, we report it.

**Protocol:**
1. **Ground truth:** Temperature signal with slow sinusoidal variation
   (period=24h, amplitude=5°C, mean=20°C) plus a step change at t=12h (+2°C)
   to test transient response.
2. **Three co-located sensor types:**
   - Type A: High-precision (σ=0.1°C), well-calibrated, continuous
   - Type B: Medium-precision (σ=0.5°C), less-frequent calibration, continuous
   - Type C: High-precision (σ=0.15°C), intermittent (realistic bursty dropout
     modeled as a hidden Markov process: P(gap|active)=0.05, P(active|gap)=0.3,
     NOT uniform random)
3. **1,000 time points**, 5 independent trials (different random seeds).
4. **Six conditions:**
   - C1: Scalar average (equal weight) — naive baseline
   - C2: Scalar weighted average (inverse-variance, σ from spec sheet)
   - C3: Kalman filter (standard sensor fusion baseline, scipy.linalg)
   - C4: SL fusion (`Opinion.from_evidence()` + `cumulative_fuse()`)
   - C5: SL fusion + decay (`decay_opinion()` for stale Type C readings)
   - C6: SL fusion + decay, **misspecified model** (assumed σ_A=0.3, σ_B=0.2,
     σ_C=0.5 — i.e., developer got the noise profiles backwards for A and B)
5. **Metrics per condition:**
   - RMSE (primary)
   - MAE
   - 95% prediction interval coverage (SL and Kalman only — scalars don't have intervals)
   - Interval width (narrower is better, given equal coverage)
6. **Statistical analysis:**
   - Bootstrap CI (n=2000) on RMSE differences: SL vs C1, SL vs C2, SL vs C3
   - Cohen's d effect size for each pairwise comparison
   - Holm-Bonferroni for family-wise error control (5 pairwise comparisons)
7. **Ablation:** Compare C4 vs C5 to isolate the contribution of temporal decay.
8. **Robustness:** Compare C5 vs C6 to measure degradation under misspecification.

**Success criteria:**
- C5 (SL+decay) RMSE ≤ C3 (Kalman) RMSE, OR within 10% (demonstrating
  competitiveness, not necessarily superiority).
- C5 beats C1 and C2 with bootstrap CI entirely below zero.
- C5 beats C4 during Type C gap periods (decay contribution is measurable).
- C6 (misspecified) degrades gracefully — RMSE increase ≤ 50% vs C5.

**Failure criteria and honest reporting plan:**
- If Kalman beats SL: Report honestly. The contribution is interoperability
  infrastructure, not SOTA fusion. Note that Kalman requires explicit state-space
  model specification while SL opinions arise naturally from the annotation model.
- If SL doesn't beat scalar weighted average: Investigate why. Likely cause:
  constant-u degeneracy (see EN3.4 finding). Report the condition under which
  SL fusion adds value (heterogeneous reliability + missing data).
- If misspecified model collapses SL: Report the fragility. Compare to Kalman's
  robustness under the same misspecification.

---

## Removed: H8.2c (Temporal Decay Coherence)

**Rationale for removal:** Temporal decay invariants (b+d+u=1, monotonic
uncertainty) are already exhaustively verified in EN7.1 (10,000 random opinions
via Hypothesis PBT). Repeating them here would be padding. Instead, decay is
tested as an ablation arm within H8.2c (C4 vs C5).

---

## Phases

### Phase A: Sensor Simulation & Data Generation

Generate the full 50-sensor, 24-hour dataset with realistic characteristics:

**Three sensor classes (50 total):**
- **Environmental** (20 sensors): temperature, humidity, pressure.
  σ ∈ {0.1, 0.3, 0.5}°C/% RH/hPa, calibrated monthly, linear drift between
  calibrations (0.01°C/day for chemical, none for environmental).
- **Motion** (15 sensors): acceleration (3-axis), angular velocity.
  σ ∈ {0.05, 0.2} m/s², calibrated quarterly.
- **Chemical** (15 sensors): pH, dissolved oxygen, conductivity.
  σ ∈ {0.02, 0.1}, calibrated weekly, **non-linear drift** modeled as a
  random walk (σ_drift=0.005/day) to test calibration tracking.

**Observation schedule:**
- Most sensors: 1 reading every 5 minutes (288/day × 50 sensors ~ 14,400)
- Type C (intermittent): HMM-based dropout pattern
- Calibration events logged with timestamp + method + authority

Each observation is annotated with ALL supported fields.
Random seed: 42 (documented, reproducible).

### Phase B: Interoperability Gap Analysis (H8.2a)

1. Convert all 14,400 observations to SSN/SOSA
2. Triple categorization (native / extension / bridge)
3. Round-trip and field-level comparison
4. Structural validation via rdflib parsing
5. Byte overhead analysis

### Phase C: Cognitive Complexity Comparison (H8.2b)

1. Write three implementations (jsonld-ex, rdflib idiomatic, rdflib+helpers)
2. Verify semantic equivalence
3. Count LoC, distinct URIs, throughput
4. All three committed for reviewer transparency

### Phase D: Multi-Sensor Fusion with Baselines (H8.2c)

1. Co-located sensor simulation (3 types, 1,000 time points, 5 trials)
2. Six conditions including Kalman and misspecified model
3. RMSE/MAE/coverage/width analysis with bootstrap CIs
4. Ablation (decay contribution) and robustness (misspecification)

## Statistical Methods

- Bootstrap CIs (n=2000) for all RMSE and timing comparisons
- Cohen's d effect size for all pairwise comparisons
- Holm-Bonferroni for family-wise error control
- 5 independent trials with different seeds for fusion (report mean ± std)
- All random seeds documented

## Output Files

```
experiments/EN8/
├── EN8_2_design.md              # This document
├── en8_2_core.py                # Simulation, annotation, round-trip, gap analysis
├── en8_2_complexity.py          # 3 implementations + LoC counting
├── en8_2_fusion.py              # 6-condition fusion comparison
├── run_en8_2.py                 # Main runner (phases: a, b, c, d)
├── tests/
│   └── test_en8_2.py            # RED-phase tests
└── results/
    ├── EN8_2_interop.json       # H8.2a results
    ├── EN8_2_complexity.json    # H8.2b results
    └── EN8_2_fusion.json        # H8.2c results
```

## Dependencies

- jsonld-ex >= 0.7.2 (installed)
- rdflib (for baseline comparison + SSN validation)
- numpy, scipy (statistics, Kalman filter)
- No GPU required
- No external APIs

## Anticipated Reviewer Objections & Defenses

| Objection | Defense |
|-----------|---------|
| "Round-trip is a unit test" | We measure interoperability gap (native vs extension triples), not just "does it work." The gap analysis is the finding. |
| "LoC comparison is trivially favorable" | Three-way comparison including idiomatic rdflib and rdflib+helpers. Cognitive complexity metric (distinct URIs). All code committed for inspection. |
| "Synthetic data, no real IoT" | Acknowledged as limitation. We include misspecified-model condition. Contribution is infrastructure, not SOTA fusion. |
| "Why not just use Kalman?" | We include Kalman as a baseline. If Kalman wins on RMSE, we report honestly. SL's value is that fusion + interop + calibration tracking are unified. |
| "Decay was already tested in EN7.1" | H8.2c (old) removed. Decay tested as ablation arm (C4 vs C5), not re-verified. |
| "Constant-u degeneracy (EN3.4)" | Per-sensor evidence counts vary naturally (Type A=continuous high-N, Type C=intermittent low-N), breaking degeneracy. Verified. |

## Honest Limitations (pre-registered)

1. All data is synthetic. Real-world IoT validation is future work.
2. Kalman filter may outperform SL fusion on RMSE — the SL contribution is
   unified metadata, not SOTA state estimation.
3. SSN/SOSA compliance is structural, not validated against official W3C
   SHACL shapes (which are not yet finalized for SSN 1.1).
4. The LoC comparison measures code volume, not development time or error rate.
5. Chemical sensor drift model is a random walk approximation; real sensor
   drift is typically more complex (temperature-dependent, hysteresis).
