# EN3.4 — FHIR R4 Clinical Data Exchange with Confidence-Aware NER Fusion

## Experiment Design Document (v2 — hardened for NeurIPS reviewer rigor)

**NeurIPS 2026 D&B, Suite EN3 (ML Pipeline Integration), Experiment 4**
**Status:** Design phase
**Priority:** HIGH / Primary
**GPU required:** Yes (GLiNER inference). Local RTX 4090.
**Venv:** `experiments\.venv-gliner`

---

## 1. Research Question

Can jsonld-ex's Subjective Logic (SL) framework enable confidence-aware
clinical data exchange workflows that standard FHIR R4 cannot express,
and does SL fusion of a generalist + specialist NER model produce better
clinical entity extraction than either model alone?

## 2. Experimental Structure

Three phases with distinct claims:

- **Phase A0 — Calibration Analysis (methodological prerequisite):**
  Measures Expected Calibration Error (ECE) for both models on dev set.
  Produces reliability diagrams. Derives principled per-model uncertainty
  parameters. This is NOT optional — it is required to make Phase A1
  scientifically defensible.

- **Phase A1 — NER Fusion Evaluation (the ML claim):**
  Tests whether SL fusion of GLiNER2 (generalist) + GLiNER-BioMed
  (biomedical specialist) improves entity-level F1 on established
  biomedical NER benchmarks, against FIVE baselines including standard
  ensemble methods.

- **Phase B — FHIR Clinical Pipeline (the systems/interop claim):**
  Demonstrates that jsonld-ex enables clinical data exchange workflows
  impossible with plain FHIR, using Synthea synthetic patient data.

---

## 3. Datasets

### Phase A: Benchmarks with Ground Truth

**Dataset 1: BC5CDR** (BioCreative V Chemical-Disease Relation)
- Source: 1,500 PubMed abstracts
- Annotations: 4,409 chemicals + 5,818 diseases (expert-annotated)
- Access: Public via HuggingFace `tner/bc5cdr`, no DUA
- Entity types: Chemical (B-Chemical, I-Chemical), Disease (B-Disease, I-Disease)
- Split: train (5,228 sentences), dev (5,330), test (5,865)
- Why: (a) GLiNER-BioMed was evaluated on this — published baselines exist;
       (b) Chemical ≈ medication, Disease ≈ diagnosis — maps to FHIR entities;
       (c) Widely recognized by NeurIPS reviewers.
- Citation: Wei et al., 2016. "Assessing the state of the art in biomedical
  relation extraction." Database.

**Dataset 2: MedMentions (ST21pv subset)**
- Source: 4,392 PubMed abstracts, 350K+ mentions linked to UMLS concepts
- Access: Public, CC0 license, GitHub `chanzuckerberg/MedMentions`,
  HuggingFace `bigbio/medmentions`
- Entity types: 21 UMLS semantic types including Chemicals & Drugs,
  Disorders, Procedures, Anatomy, Devices, Living Beings
- Why: (a) Breadth — 21 entity types tests generalization;
       (b) Hard benchmark (SOTA F1 ≈ 0.63) — room for honest findings;
       (c) Tests whether fusion helps across many categories or only specific ones.
- Citation: Mohan & Li, 2019. "MedMentions: A Large Biomedical Corpus
  Annotated with UMLS Concepts." AKBC.

### Phase B: Clinical Pipeline

**Dataset 3: Synthea FHIR R4**
- Source: Pre-generated 1,180 patient bundles (already downloaded)
- Location: `E:\data\code\claudecode\jsonld\jsonld-ex\data\synthea\fhir_r4\`
- Access: Public, Apache 2.0 + CC0 data
- Citation: Walonoski et al., 2018. "Synthea: An Approach, Method, and
  Software Mechanism for Generating Synthetic Patients." JAMIA.

---

## 4. Models

### GLiNER2 (Generalist)
- Model: `fastino/gliner2-base-v1`
- Architecture: Span-matching with dot-product + sigmoid scoring
- Zero-shot: Uses natural language entity descriptions
- Package: `gliner2`
- Confidence mechanism: Sigmoid span score (NOT softmax)

### GLiNER-BioMed (Biomedical Specialist)
- Model: `Ihor/gliner-biomed-base-v1.0`
- Architecture: Domain-adapted GLiNER with biomedical pre-training
- Zero-shot: Uses natural language entity descriptions
- Package: `gliner`
- Confidence mechanism: Same as GLiNER but domain-calibrated
- Published F1 on BC5CDR: Available in Ihor et al., 2025

### Why This Pairing Matters
The generalist-vs-specialist contrast creates a natural test for SL fusion:
- GLiNER-BioMed SHOULD outperform GLiNER2 on biomedical entities
  (if not, we report honestly — H3.4d addresses this)
- The interesting question is whether fusion ADDS VALUE beyond the
  better individual model
- Different calibration profiles (general vs domain-adapted) test
  whether SL handles confidence asymmetry correctly
- If GLiNER-BioMed alone is sufficient, SL fusion adds overhead
  without benefit — and we report that honestly

---

## 5. Methodological Hardening (Anti-Reviewer-Attack Design)

This section documents five specific defenses against anticipated
NeurIPS reviewer attacks. Each defense is integrated into the protocol.

### Defense 1: Principled Uncertainty from Dev-Set Calibration

**Attack:** "Your uncertainty assignment u = 1 - score is a mechanical
reparameterization. The SL opinion carries no information beyond the
scalar."

**Defense:** We estimate per-model uncertainty from Expected Calibration
Error (ECE) on the dev set:

```
Phase A0 protocol:
  1. Run each model on dev set, collect (predicted_score, is_correct) pairs
  2. Bin predictions into 10 equal-width bins by confidence
  3. For each bin: ECE_bin = |accuracy_bin - confidence_bin| × (n_bin / n_total)
  4. ECE_model = sum(ECE_bin)
  5. Assign: u_model = clamp(ECE_model + ε, min=0.02, max=0.50)
     where ε = 0.02 is a floor preventing overconfident uncertainty
  6. Then: opinion = Opinion.from_confidence(score, uncertainty=u_model)
```

**Rationale:** A model with high calibration error genuinely has more
epistemic uncertainty. GLiNER-BioMed (domain-adapted) should have lower
ECE than GLiNER2 (generalist) on biomedical text. SL fusion then
correctly weights the specialist more heavily — because it earns lower
uncertainty through better calibration, not through mechanical assignment.

**Ablation:** We also run with u = 1 - score (mechanical) to show that
calibration-derived uncertainty produces different (hopefully better)
fusion outcomes. This ablation is reported in the results regardless
of direction.

### Defense 2: Calibration Analysis as Mandatory Sub-Experiment

**Attack:** "Did you calibrate? GLiNER scores are not probabilities."

**Defense:** Phase A0 produces:
- Reliability diagrams for both models on dev set (10-bin)
- ECE values (scalar) for both models
- Overconfidence/underconfidence analysis per bin
- Temperature scaling: fit T on dev set, report ECE before and after

This becomes a finding in its own right: "GLiNER-BioMed is better
calibrated on biomedical text than GLiNER2" (or "both are poorly
calibrated, and here is how we handle that").

### Defense 3: Five Baselines Including Standard Ensemble Methods

**Attack:** "How is this different from standard ensemble methods?
Show me something majority voting or stacking can't do."

**Defense:** Six total conditions (1 SL + 5 baselines):

| Condition | Description | What it tests |
|-----------|-------------|---------------|
| B1: GLiNER2 alone | Single model, optimized threshold | Individual generalist |
| B2: GLiNER-BioMed alone | Single model, optimized threshold | Individual specialist |
| B3: Union | Accept if EITHER model predicts (OR logic) | Max-recall ensemble |
| B4: Intersection | Accept only if BOTH predict (AND logic) | Max-precision ensemble |
| B5: Scalar average | Average scores, accept above threshold | Standard confidence ensemble |
| SL: SL fusion | cumulative_fuse + conflict-based abstention | Our method |

The SL condition must beat ALL five baselines to claim H3.4a and H3.4b.
If SL beats B1-B2 but ties B5 (scalar average), H3.4b is REJECTED —
SL adds complexity without benefit over simple averaging.

If SL ties B3 or B4, but provides CONFLICT DETECTION that B3/B4 cannot,
that is still a contribution (H3.4c, H3.4f) — but a weaker one.

### Defense 4: Primary Hypothesis Designation + Multiple Testing Correction

**Attack:** "Six hypotheses without correction — you're fishing."

**Defense:**
- **PRIMARY hypothesis:** H3.4a (fusion improves F1 on BC5CDR).
  This is tested at α = 0.05 without correction.
- **SECONDARY hypotheses:** H3.4b through H3.4f.
  These are tested with Holm-Bonferroni correction (k=5, family-wise α = 0.05).
- Both corrected and uncorrected p-values are reported in the results.
- Pre-registration: hypotheses are written before any data is examined.
  The design doc timestamp serves as the pre-registration record.

### Defense 5: Effect Sizes, CI Widths, and Statistical Power

**Attack:** "Your F1 improvement is 0.3 percentage points. Is that noise?"

**Defense:**
- All F1 comparisons report bootstrap 95% CI (n=1000, seed=42).
- Report CI width for each metric. If CI for (F1_SL - F1_best_baseline)
  includes zero, the result is INCONCLUSIVE, not "accepted."
- Report Cohen's h (arcsine-transformed proportion difference) for each
  F1 comparison as standardized effect size.
- If the CI width on BC5CDR exceeds 2 percentage points F1, we note
  insufficient statistical power and suggest larger datasets for
  future work.

---

## 6. Phase A Hypotheses (Falsifiable, with Explicit Failure Criteria)

**Pre-registration note:** These hypotheses are fixed before examining
any data. The design doc creation timestamp serves as evidence.

### H3.4a — Fusion Improves F1 (BC5CDR) [PRIMARY]
**Claim:** SL cumulative fusion of GLiNER2 + GLiNER-BioMed achieves
higher entity-level F1 than ALL five baselines (B1-B5) on BC5CDR test set.
**Metric:** Entity-level F1 (strict span match), micro-averaged.
**Statistical test:** Bootstrap CI on F1 difference (SL - best baseline).
If lower bound of 95% CI > 0, ACCEPTED. Otherwise REJECTED or INCONCLUSIVE.
**Significance level:** α = 0.05 (primary, uncorrected).
**Effect size:** Cohen's h reported.
**Failure criterion:** Lower bound of bootstrap 95% CI on
(F1_SL - F1_best_baseline) ≤ 0.
**If fails:** Report REJECTED. Analyze per-entity-type (Chemical vs Disease).
Analyze whether calibration-derived uncertainty helped vs hurt.

### H3.4b — SL Fusion Outperforms Scalar Average [SECONDARY]
**Claim:** SL fusion outperforms naive scalar confidence averaging (B5).
**Metric:** Entity-level F1 (strict span match).
**Statistical test:** Bootstrap CI on F1 difference (SL - B5).
**Significance level:** α = 0.05, Holm-Bonferroni corrected (k=5).
**Failure criterion:** Corrected lower bound of 95% CI ≤ 0.
**If fails:** Report REJECTED. SL's three-component representation adds
complexity without measurable benefit over simple averaging.

### H3.4c — Conflict Detection Correlates with Error [SECONDARY]
**Claim:** High SL conflict_metric correlates with incorrect entity
extractions.
**Metric:** Spearman ρ between conflict score and entity-level error.
**Statistical test:** Spearman correlation with two-sided p-value.
**Significance level:** α = 0.05, Holm-Bonferroni corrected (k=5).
**Failure criterion:** Corrected p > 0.05 OR ρ ≤ 0.3.
**If fails:** Report REJECTED. Conflict detection does not reliably
signal errors for this task.

### H3.4d — Specialist Outperforms Generalist [SECONDARY]
**Claim:** GLiNER-BioMed (B2) achieves higher F1 than GLiNER2 (B1) on
BC5CDR.
**Statistical test:** Bootstrap CI on F1 difference (B2 - B1).
**Significance level:** α = 0.05, Holm-Bonferroni corrected (k=5).
**Failure criterion:** Corrected lower bound of 95% CI ≤ 0.
**If fails:** The "specialist vs generalist" framing is invalid. Report
honestly. The fusion analysis still has value — it becomes a "two
comparable models" story instead.

### H3.4e — Fusion Generalizes to Rich Taxonomy (MedMentions) [SECONDARY]
**Claim:** SL fusion improves entity-level F1 on MedMentions ST21pv
(21 entity types) compared to the best individual model.
**Statistical test:** Bootstrap CI on F1 difference.
**Significance level:** α = 0.05, Holm-Bonferroni corrected (k=5).
**Failure criterion:** Corrected lower bound of 95% CI ≤ 0.
**If fails:** Report REJECTED. Fusion may only help for narrow taxonomies.

### H3.4f — Abstention Improves Precision [SECONDARY]
**Claim:** Abstaining on high-conflict entities (conflict_metric > τ)
improves precision without catastrophic recall loss (recall drop < 10
percentage points absolute).
**Protocol:** Sweep τ ∈ {0.3, 0.4, 0.5, 0.6, 0.7}. Optimize τ on dev
set. Evaluate on test set.
**Statistical test:** Bootstrap CI on precision difference at best τ.
**Significance level:** α = 0.05, Holm-Bonferroni corrected (k=5).
**Failure criterion:** No threshold achieves precision improvement
with corrected p < 0.05 AND recall drop < 10pp.
**If fails:** Report REJECTED. Conflict-based abstention is too
aggressive or not selective enough.

---

## 7. Phase B Hypotheses (Falsifiable, with Explicit Failure Criteria)

### H3.4g — FHIR Round-Trip Fidelity
**Claim:** jsonld-ex FHIR round-trip preserves all structural fields
for the 10 resource types used in Phase B.
**Protocol:** For each resource: from_fhir() → to_fhir(). Compare
original vs round-tripped resource at field level.
**Metric:** Field-level preservation rate.
**Failure criterion:** Any structural field loss.
**Note:** This is a LIGHT test. EN8.11 does exhaustive round-trip
across all 32 resource types.

### H3.4h — Annotation Overhead
**Claim:** jsonld-ex annotation overhead (SL extension on FHIR resources)
is < 15% of original resource size in bytes.
**Failure criterion:** Overhead ≥ 15%.

### H3.4i — Query Expressiveness
**Claim:** jsonld-ex enables ≥ 5 query types that plain FHIR cannot
express.
**Protocol:** Enumerate and demonstrate queries:
  1. Filter observations by AI confidence level
  2. Track provenance of AI-suggested diagnoses
  3. Fuse predictions from multiple NLP models with conflict detection
  4. Apply temporal decay to old observations
  5. Abstain on high-conflict entity extractions
  6. Threshold-filter resources by uncertainty component
**Failure criterion:** < 5 query types demonstrated with working code.

---

## 8. Protocol: Phase A0 (Calibration Analysis — Prerequisite)

### Step A0.1: Run Both Models on Dev Set
```
For each model in {GLiNER2, GLiNER-BioMed}:
  For each dev set sentence:
    Extract entities with threshold=0.01 (very low — capture everything)
    Record: (span, type, raw_score)
  
  Align predictions to ground truth (strict span match)
  Build calibration table: [(predicted_score, is_correct_entity), ...]
```

### Step A0.2: Compute ECE
```
For each model:
  Bin predictions into 10 equal-width bins [0.0-0.1), [0.1-0.2), ..., [0.9-1.0]
  For each bin:
    accuracy = n_correct / n_total
    confidence = mean(predicted_scores)
    weight = n_total / N
    ece_contribution = weight × |accuracy - confidence|
  ECE = sum(ece_contributions)
```

### Step A0.3: Reliability Diagram
```
Plot: x = mean predicted confidence per bin, y = actual accuracy per bin
Diagonal = perfect calibration
Record: overconfidence (points below diagonal) vs underconfidence (above)
```

### Step A0.4: Temperature Scaling (Optional Calibration)
```
Fit temperature T on dev set by minimizing negative log-likelihood:
  calibrated_score = sigmoid(logit(raw_score) / T)
Report: ECE_before, ECE_after, T_fitted for each model
If temperature scaling substantially improves ECE:
  Run Phase A1 with BOTH raw and calibrated scores (ablation)
```

### Step A0.5: Derive Per-Model Uncertainty
```
u_gliner2 = clamp(ECE_gliner2 + 0.02, min=0.02, max=0.50)
u_biomed  = clamp(ECE_biomed + 0.02,  min=0.02, max=0.50)
Record these values — they are used in Phase A1 opinion construction
```

### Step A0.6: Report Calibration Findings
```
Mandatory outputs:
  - ECE for each model (scalar)
  - Reliability diagram (plot or data table)
  - Temperature T for each model
  - Per-model uncertainty values u_gliner2, u_biomed
  - Finding: "GLiNER-BioMed is {better/worse/similarly} calibrated
    compared to GLiNER2 on biomedical text (ECE = X vs Y)"
```

---

## 9. Protocol: Phase A1 (NER Fusion Evaluation)

### Step A1.1: Data Loading
```
Load BC5CDR via HuggingFace `datasets` (tner/bc5cdr)
Load MedMentions via HuggingFace `datasets` (bigbio/medmentions)
Verify: token counts, entity counts, split sizes match published stats
```

### Step A1.2: Entity Label Mapping for GLiNER
GLiNER models use natural language entity descriptions, not BIO tags.
Define label mappings:

**BC5CDR:**
- "chemical compound or drug" → maps to B-Chemical/I-Chemical
- "disease or medical condition" → maps to B-Disease/I-Disease

**MedMentions ST21pv (21 types):**
- Map each UMLS semantic type to a natural language description
- Example: T047 "Disease or Syndrome" → "disease or syndrome"
- Example: T121 "Pharmacologic Substance" → "pharmacological substance or drug"
- Full mapping defined in code, documented in design

### Step A1.3: Model Inference
```
For each dataset split (dev, test):
  For each sentence/abstract:
    Run GLiNER2 with threshold=0.3 (low, to capture borderline entities)
    Run GLiNER-BioMed with threshold=0.3
    Record: span text, span start/end, entity type, confidence score
```

### Step A1.4: Span Alignment
Both models may produce overlapping or partially matching spans for the
same text region. Alignment procedure:

```
For each text region:
  1. Collect all predicted spans from both models
  2. Group spans by overlap (IoU > 0.5 at character level)
  3. For each group:
     a. If both models predict the same type → fuse
     b. If models predict different types → conflict case (type disagreement)
     c. If only one model predicts → single-source opinion

Ablation: also run with IoU > 0.0 (any overlap) and exact match only.
Report sensitivity to alignment threshold.
```

### Step A1.5: Opinion Construction (Calibration-Derived Uncertainty)
```
PRIMARY METHOD (calibration-derived):
  For each GLiNER2 prediction:
    opinion = Opinion.from_confidence(score, uncertainty=u_gliner2)
  For each GLiNER-BioMed prediction:
    opinion = Opinion.from_confidence(score, uncertainty=u_biomed)
  Where u_gliner2, u_biomed are from Phase A0 Step A0.5.

ABLATION (mechanical):
  opinion = Opinion.from_confidence(score, uncertainty=max(0.01, 1 - score))
  This is the naive reparameterization — reported for comparison.
```

### Step A1.6: Fusion + Decision
```
For aligned span pairs:
  fused = cumulative_fuse(opinion_model_a, opinion_model_b)
  conflict = conflict_metric(fused)
  
  # Decision rule:
  if conflict > abstention_threshold:
      action = ABSTAIN
  elif fused.projected_probability() > acceptance_threshold:
      action = ACCEPT (entity type from higher-confidence model)
  else:
      action = REJECT

Threshold optimization:
  Sweep acceptance_threshold ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
  Sweep abstention_threshold ∈ {0.3, 0.4, 0.5, 0.6, 0.7}
  Optimize on dev set (maximize F1), evaluate on test set
```

### Step A1.7: All Six Conditions
```
B1: GLiNER2 alone
  - Accept entities above optimized threshold (dev set)
  
B2: GLiNER-BioMed alone
  - Accept entities above optimized threshold (dev set)

B3: Union ensemble
  - Accept entity if EITHER model predicts it above its own
    optimized threshold (OR logic)
  - Entity type from higher-confidence model if types match;
    if types disagree, take from higher-confidence model

B4: Intersection ensemble
  - Accept entity only if BOTH models predict it above their
    respective thresholds (AND logic)
  - Entity type from higher-confidence model

B5: Scalar average ensemble
  - For aligned spans: average the two confidence scores
  - Accept above optimized threshold (dev set)
  - For single-model spans: use that model's score directly

SL: SL fusion with conflict-based abstention
  - Full SL pipeline per Step A1.6
  - Reports both F1 and abstention rate
```

### Step A1.8: Evaluation
```
Entity-level evaluation (strict span match):
  - Predicted span boundaries must EXACTLY match ground truth
  - Predicted entity type must match ground truth type
  - Metrics per condition:
    - Precision, Recall, F1 (micro-averaged)
    - Bootstrap 95% CI (n=1000, seed=42) for each metric
    - Bootstrap 95% CI on pairwise F1 DIFFERENCES (SL - B_i)
    - Cohen's h for each F1 comparison
    - p-value from permutation test (10,000 permutations)
  - Holm-Bonferroni correction applied to secondary hypotheses

Per-entity-type breakdown (BC5CDR):
  - Separate P/R/F1 for Chemical and Disease
  - Report whether fusion helps more for one type than the other
```

### Step A1.9: Ablations (BC5CDR only)
```
1. Extraction threshold sensitivity: {0.1, 0.2, 0.3, 0.4, 0.5}
2. Abstention threshold sensitivity: {0.3, 0.4, 0.5, 0.6, 0.7}
3. Uncertainty assignment: calibration-derived vs mechanical (u=1-score)
4. Span alignment IoU: exact match vs IoU>0.5 vs any overlap
5. Temperature scaling: raw scores vs calibrated scores
6. Per-entity-type breakdown: Chemical vs Disease
```

### Step A1.10: Conflict-Error Correlation (H3.4c)
```
For each entity span where both models produce predictions:
  1. Compute pairwise conflict: con(a,b) = b_a*d_b + d_a*b_b
  2. Determine correctness against ground truth (binary)
  3. Compute Spearman rank correlation (conflict, is_error)
  4. Report: ρ, p-value, 95% CI on ρ via bootstrap
  5. Visualization: scatter plot of conflict vs error rate (binned)
```

---

## 10. Protocol: Phase B (FHIR Clinical Pipeline)

### Step B.1: Load Synthea Bundles
```
Load 100 patient bundles from data/synthea/fhir_r4/
Extract 10 resource types:
  Patient, Observation, Condition, MedicationRequest,
  DiagnosticReport, Procedure, Immunization, Encounter,
  AllergyIntolerance, CarePlan
Count total resources per type
```

### Step B.2: Extract Narrative Text
```
For resource types with narrative text fields:
  Observation.code.text, Condition.code.text,
  DiagnosticReport.conclusion, MedicationRequest.medicationCodeableConcept.text
  
  Collect all text snippets for NER processing
```

### Step B.3: Run NER + SL Fusion on Clinical Text
```
Use same pipeline as Phase A1 (GLiNER2 + GLiNER-BioMed + SL fusion)
Use calibration-derived uncertainty values from Phase A0
Annotate each FHIR resource with extracted entities and fused opinions
Use jsonld-ex annotate() / from_fhir() / to_fhir()
```

### Step B.4: Demonstrate Impossible-with-FHIR Workflows
Implement and measure 6+ query types (see H3.4i):
- Confidence filtering, provenance tracking, temporal decay,
  conflict-based abstention, uncertainty-component queries,
  multi-model fusion with attribution

### Step B.5: Round-Trip Fidelity (Light)
```
For each resource:
  original = load_fhir_resource()
  converted = from_fhir(original)
  annotated = annotate(converted, ...)
  stripped = strip_annotations(annotated)
  roundtripped = to_fhir(stripped)
  compare(original, roundtripped)  # field-level diff
```

### Step B.6: Overhead Measurement
```
For each resource:
  base_size = len(json.dumps(original))
  annotated_size = len(json.dumps(annotated))
  overhead_pct = (annotated_size - base_size) / base_size * 100
Report: mean, median, p95, p99 overhead across all resources
Bootstrap CI (n=1000) on mean overhead
```

---

## 11. Implementation Structure

```
experiments/EN3/
├── EN3_4_design.md            # This document
├── en3_4_calibration.py       # Phase A0: calibration analysis
├── en3_4_core.py              # Phase A1: NER fusion evaluation
├── en3_4_phase_b.py           # Phase B: FHIR pipeline
├── run_en3_4.py               # Full experiment runner (all phases)
├── tests/
│   ├── test_en3_4_calibration.py  # Phase A0 tests
│   ├── test_en3_4.py              # Phase A1 tests
│   └── test_en3_4_phase_b.py      # Phase B tests
└── results/
    ├── EN3_4_calibration.json     # Phase A0 results
    ├── EN3_4_phase_a.json         # Phase A1 results
    ├── EN3_4_phase_b.json         # Phase B results
    └── (timestamped archives)
```

---

## 12. Dependencies

**Venv:** `experiments\.venv-gliner` (already created)
**Packages:** gliner2, gliner, datasets, seqeval, torch (CUDA),
             transformers, numpy, scipy
**jsonld-ex:** Installed in editable mode in the venv

**Packages already in requirements-gliner.txt:**
- `datasets` (HuggingFace, for BC5CDR + MedMentions loading)
- `seqeval` (for entity-level F1 evaluation)
- `scipy` (for Spearman correlation in H3.4c)

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| GLiNER-BioMed doesn't outperform GLiNER2 | H3.4d explicitly tests this; report honestly |
| Fusion hurts F1 (noise from generalist) | Report REJECTED for H3.4a; analyze per-entity-type |
| Span alignment is noisy | IoU ablation: exact, IoU>0.5, any overlap |
| GLiNER confidence scores are uncalibrated | Phase A0 measures ECE; temp scaling offered |
| Calibration-derived u ≈ mechanical u | Ablation compares both; report if no difference |
| MedMentions too hard for zero-shot GLiNER | Report actual F1; document power limitation |
| SL ≈ scalar average (no unique contribution) | H3.4b tests this directly; H3.4c/f test conflict detection as differentiator |
| Multiple testing inflates false positives | Holm-Bonferroni on secondary hypotheses |
| Effect too small to detect | Report CI widths; flag if inconclusive |
| Synthea narrative text is formulaic | Acknowledge limitation; Phase B is systems demo |
| n2c2 2018 would be a stronger dataset | Acknowledge in limitations; BC5CDR + MedMentions as accessible alternatives |

---

## 14. Ethical Considerations

- All datasets are publicly available and de-identified
- Synthea generates fully synthetic patients — no real patient data
- No IRB required
- We do NOT claim clinical deployment readiness
- We DO claim: the framework enables confidence-aware exchange,
  which is a prerequisite for safer clinical AI workflows

---

## 15. What Constitutes Success vs Failure

**Strong success:** H3.4a ACCEPTED (primary) + majority of secondary
hypotheses ACCEPTED after Holm-Bonferroni correction. Effect sizes
meaningful (Cohen's h > 0.2).

**Moderate success:** H3.4a ACCEPTED but H3.4b REJECTED (SL ≈ scalar
average). The fusion helps, but SL specifically doesn't add value over
simple averaging. Contribution shifts to conflict detection (H3.4c, H3.4f).

**Partial success:** H3.4a REJECTED but H3.4c + H3.4f ACCEPTED. Fusion
doesn't improve raw F1, but conflict detection enables precision-recall
tradeoffs impossible with scalar methods. Nuanced finding.

**Honest failure:** H3.4a + H3.4b + H3.4c all REJECTED. SL fusion does
not improve entity extraction and conflict detection doesn't correlate
with errors. We report this honestly with full analysis of WHY.
Phase B still has value as a systems demonstration.

**Regardless of Phase A outcome:** Phase B demonstrates jsonld-ex's
FHIR interop capabilities. The query expressiveness (H3.4i) and
round-trip fidelity (H3.4g) claims are independent of NER fusion results.

---

## 16. Anticipated Reviewer Questions (Pre-Emptive Answers)

**Q: "Why not use n2c2 2018 or i2b2 2010?"**
A: n2c2 datasets are temporarily unavailable (portal down as of April
2026). BC5CDR and MedMentions are established benchmarks with published
GLiNER-BioMed baselines for direct comparison. We acknowledge n2c2 as
a stronger clinical-text benchmark in limitations.

**Q: "Why only two models? Real ensembles use 5+."**
A: Our contribution is NOT ensemble NER. It is a confidence-aware data
exchange framework. Two models are sufficient to demonstrate the SL
fusion mechanism. EN1.1 already demonstrates 5-model NER fusion.

**Q: "The improvement is small. Is SL worth the complexity?"**
A: We report effect sizes and CI widths. If the improvement is not
statistically significant, we say so. The value of SL is not just
accuracy — it is CONFLICT DETECTION (H3.4c) and PRINCIPLED ABSTENTION
(H3.4f), which scalar methods cannot provide. Even if F1 is equal,
knowing WHEN to not trust the prediction has clinical value.

**Q: "This is PubMed text, not clinical notes."**
A: Acknowledged in limitations. BC5CDR and MedMentions are the standard
accessible benchmarks for biomedical NER. Phase B uses Synthea clinical
data for the systems demonstration. n2c2 2018 (clinical discharge
summaries) would strengthen this if access becomes available.

**Q: "How does this relate to the FLAIRS paper?"**
A: FLAIRS proposed the framework. This paper VALIDATES it empirically
with rigorous baselines and statistical testing. Zero content duplication.
