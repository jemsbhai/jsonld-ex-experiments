# EN8.10 — Multi-Format Interop Pipeline (Revised)

## Motivation

jsonld-ex claims to serve as an interoperability hub — data can flow through
multiple W3C standards (PROV-O, RDF-Star, SHACL, SSN/SOSA, Croissant) without
information loss for fields within each standard's semantic scope. This
experiment measures that claim empirically with chained round-trip conversions
on both synthetic and real-world data.

## Key Insight: Chained Round-Trips

The converters are NOT composable as a linear pipeline (each expects jsonld-ex
input). The actual pipeline is **chained round-trips**:

```
doc₀ →[to_prov_o]→ PROV-O →[from_prov_o]→ doc₁
doc₁ →[to_rdf_star]→ N-Triples →[from_rdf_star]→ doc₂
doc₂ →[shape_to_shacl]→ SHACL →[shacl_to_shape]→ doc₃  (shape portion only)
doc₃ →[to_ssn]→ SSN/SOSA →[from_ssn]→ doc₄
doc₄ →[to_croissant]→ Croissant →[from_croissant]→ doc₅
```

Cumulative fidelity = how much of doc₀ survives in doc₅.

## Distinction from EN2.3

EN2.3 (NOT YET DONE) tests per-format round-trip fidelity independently.
EN8.10 tests the CUMULATIVE pipeline — chaining multiple round-trips and
measuring error propagation. EN8.10 also adds the rdflib baseline comparison
(both fidelity and LoC) and real-world validation.

---

## Two-Phase Design

### Phase 1: Systematic/Synthetic (~100 docs)
Controlled coverage of every annotation field, every edge case, every
converter. Ensures we are not missing any field mappings. This is the
"unit test" of the pipeline.

### Phase 2: Real-World (~200+ docs)
Documents drawn from completed experiments — actual NER annotations,
actual sensor readings, actual dataset cards. Proves ecological validity.

**Real data sources:**
- **EN1.1** (CoNLL-2003 NER): Per-token predictions with confidence scores
  from multiple models → annotated NER documents
- **EN8.2** (Intel Lab IoT): Real sensor readings from 54 Mica2Dot motes,
  already annotated with jsonld-ex (annotate_sensor_reading()) → IoT docs
- **EN2.4** (Croissant cards): 10 real MLCommons Croissant cards in
  experiments/EN2/croissant_cards/ (COCO, Fashion-MNIST, Titanic, etc.)
- **EN3.4** (FHIR clinical): Synthea patient resources with dual-NER
  confidence annotations → clinical ML docs
- **EN8.6** (KG merge): DBpedia×Wikidata merge results with conflict
  resolution and provenance → multi-source KG docs

---

## Document Categories

### Category A: General ML Annotations
Fields: @confidence, @source, @method, @extractedAt, @humanVerified,
@derivedFrom, @delegatedBy, @invalidatedAt, @invalidationReason, @value

Applicable pipeline: PROV-O → RDF-Star (2 stages)

Phase 1 (20 docs): Controlled field combinations
- 5× NER annotation docs (token-level predictions with confidence)
- 5× Multi-source fusion docs (conflicting assertions with provenance)
- 5× Delegation chain docs (agent → delegator chains)
- 5× Invalidation docs (retracted claims with reasons)

Phase 2 (~80 docs):
- ~30× from EN1.1 (CoNLL-2003 sentence-level NER annotations)
- ~30× from EN8.6 (KG merge conflict resolutions with provenance)
- ~20× from EN3.4 (clinical entity extractions with SL confidence)

### Category B: IoT/Sensor Annotations
Fields: All of Category A PLUS @measurementUncertainty, @unit,
@calibratedAt, @calibrationMethod, @calibrationAuthority,
@aggregationMethod, @aggregationWindow, @aggregationCount

Applicable pipeline: PROV-O → RDF-Star → SSN/SOSA (3 stages)

Phase 1 (15 docs): Controlled IoT field combinations
- 5× Temperature/humidity sensor readings with calibration
- 5× Aggregated time-series observations
- 5× Multi-sensor readings with measurement uncertainty

Phase 2 (~60 docs):
- ~60× from EN8.2 (Intel Lab mote readings, already annotated)

### Category C: Dataset Metadata
Fields: Dataset card fields (name, description, license, creator,
distribution, recordSet, field definitions)

Applicable pipeline: Croissant (1 stage)
If docs also contain annotated values → additionally PROV-O + RDF-Star.

Phase 1 (10 docs): Synthetic dataset cards covering varied domains
- 2× NLP, 2× Vision, 2× Tabular, 2× Audio, 2× Multimodal

Phase 2 (10 docs):
- 10× from EN2/croissant_cards/ (real MLCommons Croissant cards)

### Category D: Validation Shapes
Fields: @shape definitions with @required, @type, @minimum, @maximum,
@minLength, @maxLength, @pattern, @in, @or/@and/@not, @extends,
@lessThan, @class, @uniqueLang, @qualifiedShape, @if/@then/@else

Applicable pipeline: SHACL (1 stage, also OWL as secondary)

Phase 1 (10 docs):
- 2× Simple shapes (required + type only)
- 2× Shapes with numeric/string constraints
- 2× Shapes with logical combinators (@or/@and/@not)
- 2× Shapes with @extends inheritance
- 2× Complex shapes with conditional + cross-property + qualified

Phase 2 (5 docs):
- 5× from EN8.1 (SHACL shapes used in real validation experiments)

### Category E: Kitchen Sink
Fields: ALL annotation types combined — annotated values WITH shapes
WITH dataset metadata WITH sensor fields.

Applicable pipeline: ALL 5 stages (PROV-O → RDF-Star → SHACL → SSN → Croissant)

Phase 1 (5 docs):
- IoT dataset cards with validation shapes and annotated sensor readings
- Designed to stress-test the full pipeline

Phase 2 (5 docs):
- Composite documents merging real data from EN8.2 + EN2.4 + EN8.1

**Total: Phase 1 = 60 docs, Phase 2 = 160+ docs, Combined = 220+ docs**

---

## Hypotheses

### H8.10a — In-Scope Field Fidelity (Per-Format)
Each format converter preserves ≥ 95% of fields within its semantic scope
during a single round-trip.

**Scope definitions:**
- PROV-O scope: @confidence, @source, @extractedAt, @method, @humanVerified,
  @derivedFrom, @delegatedBy, @invalidatedAt, @invalidationReason, @value
- RDF-Star scope: ALL annotation fields (complete _ANNOTATION_FIELDS list)
- SHACL scope: All @shape constraint keywords
- SSN/SOSA scope: @value, @source, @extractedAt, @method, @confidence,
  @measurementUncertainty, @unit, @aggregation*, @calibration*
- Croissant scope: All dataset-level fields

Fidelity is measured as SEMANTIC equivalence (values must match, not just
field existence). Type coercion (int↔float) is tolerated with explicit
tracking.

**Failure criterion:** < 90% for any format within scope → BUG, investigate.

### H8.10b — Out-of-Scope Field Characterization
Fields outside a converter's scope are CHARACTERIZED (not pass/fail):
  (a) Preserved unchanged (passthrough)
  (b) Dropped (information loss)

**Expected losses (from source code analysis):**
- PROV-O drops: @unit, @measurementUncertainty, @calibration*, @aggregation*,
  @mediaType, @contentUrl, @contentHash, @translatedFrom, @translationModel
- RDF-Star: preserves ALL annotation fields (widest scope)
- SHACL: Only processes shapes; annotations pass through untouched
- SSN/SOSA drops: @derivedFrom, @delegatedBy, @invalidatedAt
- Croissant: Context swap only; extension fields may survive as unknown keys

This characterization IS the contribution — an empirical map of what each
W3C standard can and cannot express.

### H8.10c — Cumulative Pipeline Fidelity
Chaining all applicable round-trips:
  - Category A (2 stages): ≥ 85% cumulative fidelity
  - Category B (3 stages): ≥ 70% cumulative fidelity
  - Category E (5 stages): ≥ 50% cumulative fidelity

**Failure criterion:** < 40% for any category → pipeline is not viable.

### H8.10d — Error Propagation Is Additive
Information loss is ADDITIVE, not multiplicative:
  - Fields successfully round-tripped by stage N are NOT corrupted by stage N+1
  - Cumulative loss ≈ union of per-stage out-of-scope losses

Measured as: for each field that survived stage N, probability of surviving
stage N+1 (should be ≈ 1.0 if the field is in stage N+1's scope).

**Failure criterion:** A stage corrupts fields preserved by all prior stages.

### H8.10e — Code Complexity (LoC)
Implementing PROV-O + RDF-Star pipeline with rdflib + manual glue requires
≥ 3× more lines of code than jsonld-ex.

LoC counted: executable lines only (no comments, blanks, imports).
Both implementations achieve equivalent fidelity on the same test set.

**Failure criterion:** < 2× LoC ratio → integration value claim weakens.
Report honestly regardless.

### H8.10f — Baseline Fidelity Parity
rdflib baseline achieves comparable per-stage fidelity (within 5pp) for
PROV-O and RDF-Star stages.

**Rationale:** If rdflib achieves similar fidelity with more code, the value
is integration convenience, not algorithmic advantage. (Same honest finding
as EN8.6's B5 baseline.)

### H8.10g — Phase 1 → Phase 2 Transfer
Fidelity patterns observed in Phase 1 (synthetic) transfer to Phase 2
(real-world): same fields lost, same fields preserved, within ±5pp.

**Failure criterion:** > 10pp gap between synthetic and real-world fidelity
for any pipeline stage → synthetic docs are not representative.

---

## Fidelity Measurement

### Field Extraction
```python
def extract_fields(doc: dict) -> dict[str, Any]:
    """Extract all fields as flattened path→value pairs.

    Example: {"name": {"@value": "Alice", "@confidence": 0.9}}
    → {"name.@value": "Alice", "name.@confidence": 0.9}

    Excludes @context (format-specific, expected to change).
    Normalizes @id with UUIDs to stable identifiers.
    """
```

### Semantic Comparison
Not just structural matching — VALUES must be semantically equivalent:
- Numeric: |a - b| < ε (ε = 1e-10 for float comparisons)
- String: exact match after whitespace normalization
- Boolean: exact match
- List: order-independent set comparison for multi-value fields
  (@derivedFrom, @delegatedBy)
- Type coercion: int(22) == float(22.0) is TRANSFORMED, not LOST

### Per-Stage Metrics
For each round-trip stage:
- **preserved**: field present in both, semantically identical value
- **transformed**: field present in both, different representation
  (tracked WITH the transformation type: int→float, etc.)
- **lost**: field in input, absent in output
- **gained**: field in output, absent in input (conversion artifacts)
- **corrupted**: field in both, semantically DIFFERENT value (most serious)

### Aggregate Metrics
- **fidelity** = preserved / total_input_fields × 100%
- **semantic_fidelity** = (preserved + transformed) / total_input_fields × 100%
  (transformed fields carry information, just in different form)
- **loss_rate** = lost / total_input_fields × 100%
- **corruption_rate** = corrupted / total_input_fields × 100% (should be 0)

### Per-Field Survival Matrix
A matrix: rows = annotation fields, columns = pipeline stages.
Cell = fraction of documents where that field survived that stage.
This is a key visualization for the paper — shows at a glance which
W3C standards cover which jsonld-ex features.

### Statistical Rigor
- Bootstrap 95% CIs (n=2000) for all aggregate metrics
- Per-category × per-stage breakdown with CIs
- Effect sizes: absolute difference in fidelity between stages
- Phase 1 vs Phase 2 comparison with paired analysis

---

## Baseline: rdflib + Manual Glue

### What the Baseline Does
For PROV-O and RDF-Star stages, implement equivalent conversions using:
- rdflib for RDF graph construction and serialization
- Manual Python code for mapping jsonld-ex annotations ↔ RDF triples
- No jsonld-ex library calls

### What We Measure
1. **LoC**: Total executable lines (excluding comments, blanks, imports)
2. **Fidelity**: Same field-level metrics as jsonld-ex pipeline
3. **Time**: Wall-clock time for the full pipeline (informational)

### Scope
Baseline covers PROV-O + RDF-Star only. These are the formats where rdflib
provides native support. Extending to SSN/SHACL/Croissant would require
additional manual ontology mapping — which is exactly the complexity jsonld-ex
eliminates. The LoC comparison ONLY counts PROV-O + RDF-Star for fairness.

---

## Directory Structure

```
experiments/EN8/
├── EN8_10_design.md           # This document
├── en8_10_core.py             # Doc generators, fidelity measurement, pipeline
├── en8_10_real_world.py       # Phase 2: load real data from prior experiments
├── en8_10_baseline.py         # rdflib baseline (PROV-O + RDF-Star)
├── run_en8_10.py              # Full-scale runner (Phase 1 + Phase 2)
├── tests/
│   └── test_en8_10.py         # RED-phase tests
└── results/
    ├── en8_10_results.json    # Primary results (both phases)
    └── en8_10_YYYYMMDD_HHMMSS.json  # Timestamped archive
```

---

## Execution Plan

1. Write RED-phase tests (test_en8_10.py)
2. Implement en8_10_core.py:
   a. Field extraction and semantic comparison
   b. Synthetic document generators (5 categories)
   c. Per-stage round-trip runner
   d. Cumulative pipeline runner
   e. Per-field survival matrix computation
3. Implement en8_10_real_world.py:
   a. Load EN1.1 NER data → annotated docs
   b. Load EN8.2 IoT data → annotated docs
   c. Load EN2/croissant_cards/ → dataset docs
   d. Load EN3.4 clinical data → annotated docs
   e. Load EN8.6 merge data → KG docs
   f. Compose kitchen sink docs from real data
4. Implement en8_10_baseline.py (rdflib PROV-O + RDF-Star)
5. Write run_en8_10.py
6. Run Phase 1, diagnose results
7. Run Phase 2, compare with Phase 1
8. Save results + append to FINDINGS.md

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| PROV-O drops IoT fields | Lower cumulative fidelity | Expected — characterize |
| RDF-Star loses nested structure | doc₂ degraded | By design; measure |
| SHACL @if/@then/@else lost | Shape fidelity < 95% | Check conditional markers |
| Kitchen sink < 50% cumulative | Weak full-pipeline claim | Honest reporting; ID bottleneck |
| rdflib baseline is concise | LoC ratio < 2× | Report honestly |
| Type coercion false negatives | Inflated loss count | Tolerant comparison |
| Phase 1→2 gap > 10pp | Synthetic not representative | Revise synthetic docs |
| Real data loading fails | Phase 2 blocked | Fall back to subset |
| EN1.1 raw data not available | Fewer Cat A docs | Use EN8.6 + EN3.4 instead |

## Anticipated NeurIPS Reviewer Questions

**Q: "Why not test with other tools like Apache Jena or JSON-LD Playground?"**
A: Our claim is about jsonld-ex as an integration hub, not about superiority
over RDF frameworks. rdflib is the most widely-used Python RDF toolkit and
a fair baseline. Apache Jena is Java — different ecosystem.

**Q: "The fidelity thresholds seem arbitrary."**
A: Thresholds are PRE-REGISTERED (set before running). We report all raw
numbers regardless of threshold. The thresholds provide falsifiability.

**Q: "How do you handle documents where not all stages apply?"**
A: Each category has a defined pipeline. We don't force documents through
inapplicable stages. Kitchen sink (Category E) is specifically designed to
test ALL stages on a single document.

**Q: "Is cumulative fidelity meaningful? Wouldn't real users only convert
to one target format?"**
A: The cumulative pipeline tests COMPOSABILITY — can jsonld-ex serve as
a common interchange format across a heterogeneous ecosystem? Even if
real users convert to one format, the ability to chain conversions without
degradation proves architectural soundness.

**Q: "The rdflib baseline only covers 2 of 5 formats."**
A: By design. rdflib natively supports RDF operations (PROV-O, RDF-Star).
The remaining formats (SHACL shapes, SSN/SOSA observations, Croissant
datasets) would require domain-specific ontology mapping code — exactly
the integration work jsonld-ex eliminates. We are transparent about
baseline scope.
