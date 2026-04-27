# ET1: Semantic Annotation in Training Data — Experimental Protocol

**Experiment Suite:** ET1 (Training-Time Effects)
**Working Title:** "Does Semantic Annotation Structure in Training Data Affect Model Calibration and Uncertainty Awareness?"
**Authors:** Muntaser Syed, Marius Silaghi, Sheikh Abujar, Rwaida Alssadi
**Target Venue:** NeurIPS 2026 (Main Track or Datasets & Benchmarks)
**Protocol Version:** 0.1 (Draft — Pre-pilot)
**Date:** April 27, 2026

---

## 0. Document Purpose

This document specifies a complete experimental protocol *before any code is written or any data is collected*. It serves as a de facto pre-registration: every hypothesis, metric, control condition, statistical test, and failure criterion is declared here. Deviations from this protocol during execution will be documented as protocol amendments with justification. Results will be reported regardless of whether they support our hypotheses.

---

## 1. Research Questions

**Primary question:** Does the representational format of training data — specifically, the presence of Subjective Logic (SL) confidence annotations, provenance metadata, and semantic type structure provided by jsonld-ex — causally affect a fine-tuned language model's downstream behavior with respect to calibration, uncertainty expression, and structured output fidelity?

**Decomposed into three falsifiable hypotheses:**

**H1 — Format Grounding:** A model fine-tuned on jsonld-ex-formatted training data produces structurally valid jsonld-ex outputs at a significantly higher rate than models fine-tuned on plain text, plain JSON, or un-annotated JSON-LD, when prompted to express facts in jsonld-ex format.

- *Null:* Output validity rates do not differ significantly across training conditions.
- *Expected effect size:* Large (Cohen's h > 0.8). If the model cannot learn the output format from supervised examples, this is a model-capacity failure, not a hypothesis failure.

**H2 — Calibration Transfer:** A model fine-tuned on data where facts are paired with semantically meaningful SL confidence annotations produces better-calibrated confidence estimates on *held-out novel facts* than models fine-tuned on the same facts without confidence annotations, or with randomized confidence annotations.

- *Null:* Expected Calibration Error (ECE) does not differ significantly between conditions.
- *Critical distinction:* We must demonstrate calibration on **novel facts not seen during training**, to distinguish genuine calibration transfer from memorized confidence parroting. We must also show that **randomized confidences (condition C6) do not produce the same effect**, to distinguish learning from annotation signal versus learning from annotation structure.
- *Expected effect size:* Small-to-medium (Cohen's d ≈ 0.3–0.5). We have no prior work to base this on; if the effect is smaller than d=0.2, we will report a null result.

**H3 — Uncertainty Awareness:** A model fine-tuned on jsonld-ex data — where some facts carry high SL belief, some carry high uncertainty, and some are explicitly invalidated — exhibits more appropriate epistemic behavior on ambiguous or unanswerable test queries: lower hallucination rate, higher selective-prediction accuracy, and greater willingness to abstain.

- *Null:* Hallucination rate and abstention appropriateness do not differ significantly across conditions.
- *Expected effect size:* Small-to-medium (Cohen's d ≈ 0.3–0.5).
- *Hardest to measure:* "Appropriate uncertainty" requires ground-truth annotations on test items designating whether a correct response should be confident, hedged, or an abstention. We construct this ground truth as part of the knowledge base (Section 4).

---

## 2. Theoretical Motivation

A harsh reviewer will ask: **"Why on Earth would you expect the format of training data to matter? The factual content is the same."** This section provides the mechanistic argument.

### 2.1 The Representation Hypothesis

Language models learn not only factual associations but also the *structure of how information is presented*. This is well-established: models trained on code learn syntax; models trained on markdown learn formatting conventions; models trained on chain-of-thought prompts learn to reason step-by-step (Wei et al., 2022). We extend this observation: if training data consistently pairs facts with structured uncertainty metadata, the model may learn the *co-occurrence pattern* between fact-types and uncertainty-levels, effectively acquiring a rudimentary calibration prior.

### 2.2 Information-Theoretic Argument

An SL opinion ω = (b, d, u, a) encodes strictly more information than a scalar confidence value c = b + a·u (demonstrated in experiment EN7.2 of the existing roadmap). During training, gradient updates on jsonld-ex-annotated data encode this richer signal into the model's representations. The question is whether this additional information is *recoverable* at inference time — whether the model can generalize the pattern "facts of type X tend to have high uncertainty" to novel facts of type X.

### 2.3 Calibration-by-Example

Human calibration improves with exposure to calibrated feedback (Lichtenstein et al., 1982). We hypothesize an analogous effect for models: a model repeatedly exposed to training examples where well-sourced facts are marked b=0.95, u=0.03 and poorly-sourced facts are marked b=0.40, u=0.50 may learn to internalize this mapping. The mechanism is ordinary supervised learning — the model learns to predict the annotation fields, and this prediction generalizes.

### 2.4 What This is NOT

We are **not** claiming that:
- jsonld-ex magically makes models smarter
- Annotation structure is more important than data quality or quantity
- Small-model pilot results will transfer to frontier models without further validation

We **are** claiming that representational format is an underexplored axis of training data curation, and that structured uncertainty metadata is a specific format intervention worth measuring.

---

## 3. Experimental Conditions

Seven training conditions, each using **identical factual content** in different representational formats. This is a between-subjects design: each condition produces a separate fine-tuned model.

### 3.1 Condition Table

| ID | Name | Format | Purpose |
|----|------|--------|---------|
| C1 | Plain Text | Natural language sentences | **Primary control.** The default training format. |
| C2 | Plain JSON | Flat key-value JSON | **Structural control.** Tests whether JSON structure alone matters. |
| C3 | JSON-LD | JSON-LD with @context, @type, @id | **Semantic control.** Tests whether Linked Data semantics (without uncertainty) matter. |
| C4 | jsonld-ex (Full) | JSON-LD + SL opinions, provenance, temporal validity | **Primary treatment.** The full jsonld-ex annotation. |
| C5 | Verbose Text | Plain text padded with metadata prose to match C4 token count | **Token-length control.** Isolates the "more tokens" confound. |
| C6 | jsonld-ex (Random) | Same structure as C4 but with **randomized** SL opinion values | **Signal control.** Tests whether the model learns from annotation *structure* or annotation *signal*. |
| C7 | Scalar Confidence | JSON with a single numeric `confidence` field, no SL opinion | **Representation control.** Tests whether the full SL tuple matters versus a simple scalar. |

### 3.2 Confound Analysis

| Confound | Risk | Controlled By |
|----------|------|---------------|
| Token count | C4 has more tokens; model sees more gradient updates per fact | C5 (matched token count) |
| JSON structure | Any JSON structure might help, not just jsonld-ex | C2 (plain JSON), C3 (JSON-LD) |
| Presence of numbers | Confidence values are numeric; their mere presence might help | C6 (random numbers in same positions) |
| Annotation signal vs structure | Is it the SL algebra or just having any confidence? | C6 (structure without signal) + C7 (signal without SL structure) |
| Base model contamination | Models may already know some facts | Mitigated by synthetic/fictional knowledge base (Section 4) |
| LoRA capacity | Adapter may lack parameters for meta-learning | Mitigated by trying multiple ranks; full fine-tune on smallest model |
| Model scale | Effects may be scale-dependent | 3 model sizes in pilot; flag as limitation |

### 3.3 Reviewer-Anticipated Objections

**"C5 is a weak control — padding with irrelevant prose is not the same as informative metadata."**
Response: This is intentional. C5 isolates the pure token-count effect. If C4 outperforms C5, the advantage is from *informative content*, not token count. If C5 matches C4, the advantage is merely from more training tokens (a damaging finding for our hypothesis, which we would report honestly).

**"C6 is your most important control. If C6 matches C4, your entire thesis collapses."**
Response: Agreed. This is exactly why C6 exists. If the model learns equal calibration from random confidences as from meaningful confidences, then the hypothesis that *SL signal* matters is falsified. We would report this as a null result on H2 and pivot to analyzing what *structure-only* benefits remain (H1 would likely still hold).

**"Why not include PROV-O or schema.org annotations as additional conditions?"**
Response: Scope. Seven conditions × 3 models × 3 seeds = 63 fine-tuning runs. Adding more conditions is possible in a full study but would exceed pilot scope. We note this as future work.

---

## 4. Knowledge Base Construction

### 4.1 Design Principles

The knowledge base (KB) must satisfy:

1. **Zero contamination:** All entities must be fictional to avoid base-model memorization confounds. The model should have zero prior knowledge of any fact in the KB.
2. **Structural realism:** Entity types, relation types, and fact patterns must mirror real-world knowledge graphs (companies, people, locations, events, scientific findings).
3. **Confidence diversity:** Ground-truth confidence levels must span the full [0, 1] range with principled assignment criteria.
4. **Provenance diversity:** Facts must vary in source count, source reliability, and source agreement.
5. **Temporal diversity:** Some facts must have temporal validity windows; some must be time-invariant.
6. **Sufficient scale:** Enough facts for meaningful calibration measurement (minimum ~1000 test items across 10 calibration bins = 100 per bin).
7. **Internal consistency:** The fictional world must be self-consistent — no contradictions within high-confidence facts.

### 4.2 Fictional World: "Meridian"

We construct a fictional world called **Meridian** — an alternate-geography setting with its own companies, research institutions, cities, and scientific discoveries. This ensures zero overlap with any real-world pre-training data.

**Entity types and approximate counts:**

| Entity Type | Count | Relations |
|-------------|-------|-----------|
| Organizations (companies, labs, NGOs) | 40 | headquartered_in, founded_by, CEO, industry, revenue, subsidiary_of |
| People (executives, researchers, public figures) | 60 | works_at, born_in, studied_at, authored, collaborated_with |
| Locations (cities, regions, countries) | 25 | located_in, capital_of, population, climate |
| Research Findings (scientific claims) | 50 | discovered_by, published_in, replicated_by, contradicted_by |
| Products / Technologies | 30 | developed_by, released_date, based_on, superseded_by |
| Events (conferences, mergers, incidents) | 20 | occurred_at, involved, date, outcome |
| Total entities | ~225 | |

**Target fact count:** 3,500 facts total.

### 4.3 Ground-Truth Confidence Assignment

Each fact is assigned a ground-truth confidence tier based on principled criteria. The model never sees these tier labels — they are for evaluation only.

| Tier | Confidence Range | SL Opinion Profile | Criteria | Approx. % of KB |
|------|-----------------|-------------------|----------|-----------------|
| T1: Established | b ∈ [0.90, 0.99] | High belief, near-zero uncertainty | Multiple corroborating reliable sources, long temporal validity, no contradictions | 30% |
| T2: Probable | b ∈ [0.70, 0.89] | Moderate-high belief, low uncertainty | 2–3 sources with minor disagreements, recent | 25% |
| T3: Uncertain | b ∈ [0.40, 0.69] | Moderate belief, substantial uncertainty | Single source, or sources with moderate disagreement | 20% |
| T4: Speculative | b ∈ [0.15, 0.39] | Low belief, high uncertainty | Unreliable source, or sparse evidence, or temporal decay applied | 15% |
| T5: Contested | b ∈ [0.30, 0.60], d ∈ [0.20, 0.50] | Significant disbelief AND belief, low uncertainty | Sources actively contradict each other — NOT ignorance but conflict | 10% |

**Critical design note on T5:** Contested facts are the SL acid test. Scalar confidence conflates T3 (uncertain due to ignorance) with T5 (uncertain due to conflict). SL distinguishes them: T3 has high u; T5 has low u but both b and d are substantial. If training on SL-annotated data teaches models to distinguish these states, that is the strongest possible evidence for H2.

### 4.4 Data Splits

| Split | Fact Count | Purpose | Overlap with Training |
|-------|-----------|---------|----------------------|
| Train | 2,000 | Fine-tuning | — |
| Validation | 500 | Hyperparameter tuning, early stopping | Zero entity overlap with train (different Meridian entities) |
| Test-ID | 500 | In-distribution test — same entity types, new facts about seen entities | Same entities as train, but held-out facts |
| Test-OOD | 500 | Out-of-distribution — new entity types not in train (e.g., fictional sports teams, geological features) | Zero entity-type overlap with train |

**Why two test sets:** Test-ID measures interpolation (can the model generalize to new facts about known entities?). Test-OOD measures extrapolation (does calibration transfer to entirely novel domains?). Reviewers will ask about both.

### 4.5 Knowledge Base Generation Process

1. **Schema design:** Define entity types, relation types, and cardinality constraints manually.
2. **Entity generation:** Use a seeded random generator to create fictional names (verified against Wikipedia to confirm non-existence). Names must be phonetically plausible but verifiably fictional.
3. **Fact generation:** For each entity, generate facts according to the schema. Assign confidence tiers using the criteria in §4.3.
4. **SL opinion generation:** For each fact, generate a specific SL opinion (b, d, u, a) consistent with its tier, with controlled variance within tiers (not all T1 facts have identical opinions).
5. **Provenance generation:** For each fact, generate 1–5 fictional source identifiers with reliability ratings.
6. **Temporal metadata:** For ~40% of facts, assign valid_from/valid_until timestamps.
7. **Consistency check:** Automated verification that no two high-confidence facts contradict each other; contradictions exist only in T5 (contested) facts, by design.
8. **Contamination audit:** Spot-check 100 entity names against Google/Wikipedia to verify fictionality.

---

## 5. Training Data Format Specification

For each fact in the KB, we generate a training example in instruction-tuning format: a prompt (question) and a response (answer in the target format). All conditions use **identical prompts**; only the response format differs.

### 5.1 Prompt Design

Each fact maps to 1–3 question-answer pairs. Questions are natural language, designed to elicit the fact.

**Example fact:** *NovaTech Industries is headquartered in Cedarpoint, Valoria. (Tier T1, b=0.94, d=0.02, u=0.04, source: Valorian Corporate Registry + Meridian Business Index)*

### 5.2 Format Templates

**C1 — Plain Text:**
```
User: Where is NovaTech Industries headquartered?
Assistant: NovaTech Industries is headquartered in Cedarpoint, Valoria.
```

**C2 — Plain JSON:**
```
User: Where is NovaTech Industries headquartered?
Assistant: {"entity": "NovaTech Industries", "relation": "headquartered_in", "value": "Cedarpoint, Valoria"}
```

**C3 — JSON-LD:**
```
User: Where is NovaTech Industries headquartered?
Assistant: {
  "@context": "https://schema.org/",
  "@type": "Organization",
  "@id": "urn:meridian:novatech",
  "name": "NovaTech Industries",
  "location": {
    "@type": "Place",
    "name": "Cedarpoint, Valoria"
  }
}
```

**C4 — jsonld-ex (Full):**
```
User: Where is NovaTech Industries headquartered?
Assistant: {
  "@context": ["https://schema.org/", "https://jsonld-ex.org/context/v1"],
  "@type": "Organization",
  "@id": "urn:meridian:novatech",
  "name": "NovaTech Industries",
  "location": {
    "@type": "Place",
    "name": "Cedarpoint, Valoria",
    "@opinion": {"belief": 0.94, "disbelief": 0.02, "uncertainty": 0.04, "base_rate": 0.5},
    "@source": ["urn:meridian:vcr-2024", "urn:meridian:mbi-2024"],
    "@method": "cross_reference_verification",
    "@valid_from": "2019-06-01"
  }
}
```

**C5 — Verbose Text (token-matched to C4):**
```
User: Where is NovaTech Industries headquartered?
Assistant: NovaTech Industries is headquartered in Cedarpoint, Valoria. This information was verified through cross-referencing the Valorian Corporate Registry (2024 edition) and the Meridian Business Index (2024 edition). The registration has been active since June 2019. The verification process involves checking multiple business registries for consistency. Both sources independently confirm the Cedarpoint, Valoria location.
```

**C6 — jsonld-ex (Randomized):**
```
User: Where is NovaTech Industries headquartered?
Assistant: {
  "@context": ["https://schema.org/", "https://jsonld-ex.org/context/v1"],
  "@type": "Organization",
  "@id": "urn:meridian:novatech",
  "name": "NovaTech Industries",
  "location": {
    "@type": "Place",
    "name": "Cedarpoint, Valoria",
    "@opinion": {"belief": 0.31, "disbelief": 0.47, "uncertainty": 0.22, "base_rate": 0.5},
    "@source": ["urn:meridian:vcr-2024", "urn:meridian:mbi-2024"],
    "@method": "cross_reference_verification",
    "@valid_from": "2019-06-01"
  }
}
```

*(Note: C6 opinion values are uniformly random subject to b+d+u=1. They bear no relationship to the fact's actual confidence tier. The opinion for a T1 fact might show high disbelief; a T4 fact might show high belief. This is by design.)*

**C7 — Scalar Confidence:**
```
User: Where is NovaTech Industries headquartered?
Assistant: {"entity": "NovaTech Industries", "relation": "headquartered_in", "value": "Cedarpoint, Valoria", "confidence": 0.96}
```

*(Note: C7 confidence = projected probability P(ω) = b + a·u from the ground-truth SL opinion. This gives the model a scalar confidence signal without the full SL tuple.)*

### 5.3 Token Count Analysis

Before running experiments, we will compute and report the mean token count per training example for each condition. C5 will be calibrated to match C4's mean token count (±5%).

---

## 6. Model Selection

### 6.1 Pilot Models

| Model | Parameters | Why |
|-------|-----------|-----|
| SmolLM2-135M (HuggingFaceTB/SmolLM2-135M-Instruct) | 135M | Fastest iteration. Pipeline validation. If 135M shows zero signal, it's a capacity issue, not a hypothesis issue. |
| SmolLM2-360M (HuggingFaceTB/SmolLM2-360M-Instruct) | 360M | Slightly more representational capacity. First check for scale effects. |
| Qwen2.5-0.5B (Qwen/Qwen2.5-0.5B-Instruct) | 500M | Stronger instruction-following baseline. Better tokenizer for JSON. |

### 6.2 Scale-Up Models (Post-Pilot, if pilot shows signal)

| Model | Parameters | Fine-tuning Method |
|-------|-----------|-------------------|
| Qwen2.5-1.5B-Instruct | 1.5B | QLoRA (4-bit) — fits on 4090 |
| Llama-3.2-3B-Instruct | 3B | QLoRA (4-bit) — fits on 4090 |
| Phi-3.5-mini-instruct | 3.8B | QLoRA (4-bit) — fits on 4090 |

### 6.3 Why Small Models Are Sufficient for a Pilot

**Potential reviewer objection:** "Small models can't learn complex meta-patterns."

**Response:** The pilot has two possible outcomes, both informative:
1. **Signal detected at small scale:** Strong evidence for the hypothesis, warranting scale-up.
2. **No signal at small scale:** We *cannot* conclude the hypothesis is false — only that small models lack capacity. We would report this honestly and either (a) scale up to 3B+ models, or (b) reframe as a negative result at small scale with a clear "future work: larger models" direction.

We will **never** claim a null pilot result falsifies the hypothesis. We will also **never** torture small-model results to find significance. The pilot's purpose is pipeline validation and preliminary signal detection.

---

## 7. Training Configuration

### 7.1 Fine-tuning Method

**LoRA** (Low-Rank Adaptation) for all pilot models. Rationale: parameter-efficient, fast iteration, well-understood.

**For SmolLM2-135M only:** We also run **full fine-tuning** as a comparison, since the model is small enough. If LoRA shows no signal but full fine-tuning does, the bottleneck is adapter capacity, not the hypothesis.

### 7.2 Hyperparameters (Fixed Across Conditions)

All hyperparameters are **identical** across conditions to ensure fair comparison. The only independent variable is training data format.

| Parameter | Value | Justification |
|-----------|-------|---------------|
| LoRA rank (r) | 16 | Standard for small models; sufficient for format learning |
| LoRA alpha | 32 | α = 2r is standard |
| LoRA target modules | All linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj) | Maximizes adapter expressiveness |
| LoRA dropout | 0.05 | Mild regularization |
| Learning rate | 2e-4 | Standard for LoRA |
| LR scheduler | Cosine with warmup | |
| Warmup steps | 10% of total steps | |
| Batch size (effective) | 16 | Gradient accumulation if needed |
| Max sequence length | 1024 tokens | Sufficient for all conditions including C4 |
| Epochs | 5 | With early stopping on validation loss (patience=2) |
| Optimizer | AdamW | |
| Weight decay | 0.01 | |
| BF16 / FP16 | BF16 if supported, else FP16 | |
| Gradient checkpointing | Enabled | Memory savings |
| Random seeds | {42, 137, 2024} | 3 seeds per condition for variance estimation |

### 7.3 Hyperparameter Sensitivity

**Potential reviewer objection:** "Your results may be hyperparameter-sensitive. Maybe C4 needs different hyperparameters than C1."

**Response:** We intentionally use identical hyperparameters to test whether format matters *given standard training*. If C4 requires special hyperparameters to show an effect, that weakens the practical value of the finding. We report this as a negative if applicable. However, as a supplementary analysis, we run a small grid search for C1 and C4 only (LR ∈ {1e-4, 2e-4, 5e-4}, rank ∈ {8, 16, 32}) and report whether the best hyperparameters differ.

---

## 8. Evaluation Protocol

### 8.1 Evaluation Prompts

At evaluation time, **all models receive the same prompts regardless of training condition.** We use two prompt families:

**Family A — Neutral Prompts (Primary):**
```
User: Where is NovaTech Industries headquartered? Provide your answer and rate your confidence from 0 to 1.
```
No format is specified. We observe what format each model naturally produces and parse accordingly.

**Family B — Format-Specific Prompts (Supplementary):**
```
User: Where is NovaTech Industries headquartered? Express your answer as a jsonld-ex annotated JSON-LD document with your confidence opinion.
```
All models receive this prompt. C4 models should excel; the question is whether they produce *meaningful* confidence values.

**Family C — Abstention Prompts (H3 only):**
```
User: What was the outcome of the Valoria-Kasmir trade dispute in 2023? If you are not confident in your answer, say "I don't have enough information to answer confidently."
```
These target T4 (speculative) and T5 (contested) facts. The correct behavior is to express uncertainty, not to hallucinate a confident answer.

### 8.2 Hypothesis-Specific Evaluation

#### H1 — Format Grounding

**Procedure:** Present all models with Family B prompts for 500 test facts. Parse outputs.

**Metrics:**
- **JSON validity rate:** % of outputs that parse as valid JSON
- **JSON-LD validity rate:** % of valid JSON outputs that are valid JSON-LD (have @context, @type)
- **jsonld-ex compliance rate:** % of JSON-LD outputs that contain valid @opinion fields (b+d+u=1, all ∈ [0,1], base_rate ∈ [0,1])
- **Schema compliance rate:** % of outputs with correct @type for the entity

**Statistical test:** χ² test of independence (condition × validity), followed by pairwise Fisher's exact tests with Holm-Šídák correction.

#### H2 — Calibration Transfer

**Procedure:** Present all models with Family A prompts for 1000 test facts (500 Test-ID + 500 Test-OOD). Extract stated confidence from the response (parsed from JSON if structured, or from natural language if text). Map against ground-truth correctness.

**Confidence extraction:** For each model output, we extract a confidence value:
- If the output contains a numeric confidence field (C4, C7 trained models may produce this): use it directly.
- If the output contains an SL opinion: compute projected probability P(ω) = b + a·u.
- If the output is text with verbal confidence expressions (e.g., "I am fairly certain"): map using a pre-defined verbal-to-numeric scale (1.0 = certain, 0.8 = fairly certain, 0.5 = uncertain, 0.2 = doubtful, 0.0 = no idea). This mapping is fixed before evaluation and reported in the paper.

**Metrics:**
- **Expected Calibration Error (ECE):** 15-bin equal-width. Lower is better. $$ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|$$
- **Maximum Calibration Error (MCE):** Worst-case bin. $$MCE = \max_m |acc(B_m) - conf(B_m)|$$
- **Brier Score:** Mean squared error between stated confidence and binary correctness indicator. $$BS = \frac{1}{n} \sum_i (c_i - o_i)^2$$
- **Adaptive ECE (AECE):** Equal-mass binning (15 bins) to handle uneven confidence distributions.
- **Reliability diagram:** Plotted for each condition. Visual comparison in paper.

**Statistical test:** Paired bootstrap test (n=10,000 resamples) for ECE differences between conditions. Report 95% bootstrap CIs.

**Critical integrity check:** If C4 achieves low ECE primarily by outputting constant confidence (e.g., always 0.5), that is *not* calibration — it's trivial hedging. We compute **confidence histogram entropy** and require that C4's confidence distribution has comparable spread to ground truth. A well-calibrated model should produce varied confidences, not a point mass.

#### H3 — Uncertainty Awareness

**Procedure:** Present all models with Family C prompts for the full test set. Classify model responses into:
- **Confident correct:** Model gives a correct answer with confidence > 0.7
- **Confident incorrect (hallucination):** Model gives an incorrect answer with confidence > 0.7
- **Hedged correct:** Model gives a correct answer with confidence 0.3–0.7 or verbal hedging
- **Hedged incorrect:** Model gives an incorrect answer with hedging
- **Abstained:** Model explicitly declines to answer or says it doesn't know

**Metrics:**
- **Hallucination rate:** % of responses that are Confident Incorrect
- **Selective Prediction Accuracy (SPA):** Accuracy computed only on non-abstained responses. Higher is better.
- **Abstention Appropriateness:** Among abstentions, what fraction involved T4/T5 (genuinely uncertain) facts? Higher is better.
- **AUROC of Uncertainty:** Treat the model's stated confidence as a classifier score for correctness. Compute area under ROC curve. A well-calibrated model has AUROC near 1.0.
- **Tier Discrimination:** For each confidence tier (T1–T5), report the model's mean stated confidence. A good model shows monotonic decrease from T1 to T4. T5 (contested) is the acid test — the model should express moderate confidence with conflict indicators, not simply low confidence.

**Statistical test:** McNemar's test for hallucination rate (pairwise between conditions). Bootstrap CIs for AUROC differences.

### 8.3 Response Parsing

Model outputs will vary in format. We implement a robust parser with the following fallback chain:

1. Attempt JSON parse → extract fields
2. If JSON fails, attempt regex extraction of key-value patterns
3. If regex fails, use a rule-based NLP parser for verbal confidence and factual claims
4. If all parsing fails, mark as "unparseable" and report the rate per condition

**The parser is implemented and unit-tested before any model is trained.** Parser code is identical across conditions.

---

## 9. Statistical Analysis Plan

### 9.1 Primary Comparisons

We pre-register the following pairwise comparisons as primary (hypothesis-driven), to be corrected for multiple testing:

| Comparison | Tests Hypothesis | Expected Direction |
|------------|-----------------|-------------------|
| C4 vs C1 | All three | C4 better on H1, H2, H3 |
| C4 vs C6 | H2, H3 | C4 better (signal vs random signal) |
| C4 vs C7 | H2, H3 | C4 better (SL tuple vs scalar) |
| C4 vs C3 | H2, H3 | C4 better (annotation value, beyond LD structure) |
| C4 vs C5 | H2, H3 | C4 better (semantic content vs token padding) |
| C3 vs C2 | H1 | C3 better (LD semantics help format learning) |
| C7 vs C1 | H2 | C7 better (even scalar confidence helps) |

**Total primary comparisons:** 7 per metric family. Holm-Šídák correction applied.

### 9.2 Multiple Comparisons Correction

With 7 comparisons × 3 hypothesis families = 21 primary tests, we use Holm-Šídák (step-down) correction rather than Bonferroni (too conservative).

### 9.3 Effect Sizes

We report Cohen's d (continuous metrics) or Cohen's h (proportions) for all comparisons, not just p-values. A significant p-value with negligible effect size is not a meaningful result.

### 9.4 Variance Estimation

Each condition × model is run with 3 random seeds. We report mean ± standard deviation across seeds. If between-seed variance exceeds between-condition variance, we increase to 5 seeds.

### 9.5 Bayesian Supplementary Analysis

As a supplementary analysis (not primary), we compute Bayes factors for the key comparisons (C4 vs C1, C4 vs C6) using a JZS prior. This provides evidence for H0 if results are null — p-values cannot distinguish "no effect" from "insufficient power."

---

## 10. Power Analysis

### 10.1 Sample Size Justification

For ECE with 15 bins and 1000 test examples: ~67 examples per bin. Following Kumar et al. (2019), ECE estimation requires ≥50 examples per bin for stable estimates. 1000 is adequate.

For hallucination rate with expected rate ~20% (control) and ~12% (treatment): McNemar's test with n=1000 paired observations, α=0.05 (after correction), power=0.80 requires effect size of approximately 8 percentage points — consistent with our expected effect.

For AUROC difference of 0.05 (small but meaningful): DeLong test with n=1000 per group has >0.80 power at α=0.05.

### 10.2 Pilot-Specific Power Limitations

With 3 seeds per condition and small models, the pilot is **underpowered for definitive claims.** We acknowledge this explicitly. The pilot's purpose is:
1. Validate the experimental pipeline (data generation, training, evaluation)
2. Estimate effect sizes for power analysis of the full study
3. Detect gross effects (Cohen's d > 0.8) if present
4. Identify failure modes early

---

## 11. Pre-Registered Failure Modes

We enumerate specific outcomes that would falsify or weaken each hypothesis, and commit to reporting them honestly.

### 11.1 Hypothesis-Killing Results

| Result | Kills | Interpretation |
|--------|-------|----------------|
| C4 ≈ C6 on ECE and hallucination rate | H2 | Model learns from annotation *structure*, not *signal*. The specific SL values don't matter. |
| C4 ≈ C1 on all metrics | All | Format has no effect. Null result. |
| C4 ≈ C7 on ECE | H2 (partially) | Scalar confidence is sufficient; full SL tuple adds nothing for calibration. |
| C5 ≈ C4 on all metrics | All | The effect is simply from more training tokens, not semantic content. |
| All models output near-constant confidence | H2 | Models lack capacity or training signal for confidence diversity. Reconsider approach. |
| Unparseable output rate > 50% for any condition | H1 | Model cannot learn the format at this scale. Increase model size or training data. |

### 11.2 Weakening Results

| Result | Weakens | Interpretation |
|--------|---------|----------------|
| Effect appears on Test-ID but not Test-OOD | H2, H3 | Calibration doesn't generalize to new domains. Effect is domain-specific. |
| Effect appears only at 500M parameters, not 135M | Generality | Scale-dependent effect. Still valid but limited. |
| C7 ≈ C4 but both > C1 | H2 (SL-specific) | Confidence annotation helps, but SL algebra adds no value over scalar. Important finding either way. |
| High between-seed variance | All | Unstable results. Need more seeds or larger models. |

### 11.3 Commitment

**If the pilot produces null results, we will:**
1. Report the null honestly with effect size estimates and confidence intervals
2. Analyze whether model capacity is the bottleneck (compare 135M vs 500M trends)
3. Decide whether to scale up (if trend suggests larger models might show effects) or pivot (if no trend exists)
4. Never p-hack, never test additional metrics post-hoc without declaring them exploratory, never drop conditions that gave unfavorable results

---

## 12. Implementation Plan

### 12.1 Software Dependencies

```
# Core
torch >= 2.1
transformers >= 4.40
peft >= 0.10  (LoRA)
datasets >= 2.19
accelerate >= 0.30
bitsandbytes >= 0.43  (QLoRA quantization)

# Evaluation
scikit-learn  (calibration metrics)
scipy  (statistical tests)
numpy
matplotlib  (reliability diagrams)

# jsonld-ex
jsonld-ex >= 0.7.1  (for training data generation)

# Experiment tracking
json  (lightweight result logging; no MLflow dependency for pilot)
```

### 12.2 Project Structure

```
experiments/ET1/
├── PROTOCOL.md                 # This document
├── README.md                   # Quick-start instructions
├── requirements.txt
├── configs/
│   └── training_config.yaml    # All hyperparameters (single source of truth)
├── src/
│   ├── __init__.py
│   ├── knowledge_base.py       # KB generation and management
│   ├── data_formatter.py       # Converts KB facts to 7 training conditions
│   ├── train.py                # Fine-tuning script
│   ├── evaluate.py             # Evaluation suite
│   ├── parse_response.py       # Output parsing (JSON, text, hybrid)
│   ├── metrics.py              # ECE, Brier, AUROC, hallucination rate
│   └── statistical_tests.py   # Bootstrap CIs, McNemar, Holm-Šídák
├── tests/
│   ├── __init__.py
│   ├── test_knowledge_base.py
│   ├── test_data_formatter.py
│   ├── test_parse_response.py
│   ├── test_metrics.py
│   └── test_statistical_tests.py
├── data/
│   ├── meridian_kb.json        # Generated knowledge base
│   ├── train/                  # Formatted training data per condition
│   ├── val/
│   └── test/
├── results/
│   ├── pilot/                  # Raw results per model × condition × seed
│   └── analysis/               # Aggregated analysis, plots
└── notebooks/
    └── analysis.ipynb          # Result visualization and reporting
```

### 12.3 Development Order (TDD)

1. **Knowledge base generator** — tests first, then implementation
2. **Data formatter** — tests for each of 7 conditions, verify token counts, verify SL validity
3. **Response parser** — tests for all expected output formats, edge cases
4. **Metrics module** — tests against known-calibration synthetic data
5. **Statistical tests module** — tests against analytically known distributions
6. **Training script** — integration test with tiny model + tiny dataset
7. **Evaluation script** — integration test with saved checkpoint
8. **End-to-end pipeline** — smoke test: generate KB → format → train 1 epoch → evaluate

---

## 13. Ethical Considerations

### 13.1 No Human Subjects

This study uses synthetic data and pretrained models. No IRB approval is required.

### 13.2 Compute and Environmental Cost

Pilot estimate: 63 fine-tuning runs × ~15 minutes each (small models) = ~16 GPU-hours on a single RTX 4090. Modest energy footprint. Full study (larger models) would be ~100 GPU-hours. We report total compute in the paper.

### 13.3 Dual Use

The finding that training data format affects model calibration could be used to *improve* calibration (beneficial) or to *manipulate* calibration in adversarial ways (e.g., training models to be overconfident by design). We discuss this in the paper's broader impacts section.

### 13.4 Reproducibility

- All code, data, and configs will be released publicly on GitHub.
- All random seeds are fixed and reported.
- All models are publicly available on HuggingFace.
- The knowledge base generator is deterministic given its seed.

---

## 14. Relationship to Existing Roadmap

This experiment suite (ET1) is **orthogonal** to the existing NeurIPS experiment roadmap (EN1–EN8). The existing roadmap studies jsonld-ex as an **inference-time tool** (fusion, filtering, interop). ET1 studies jsonld-ex as a **training-time intervention**. There is zero content overlap.

However, positive results from ET1 would significantly strengthen the NeurIPS paper's overall narrative: jsonld-ex is useful not only at inference time for structured data exchange, but also at training time as a data curation format that improves model behavior. This elevates jsonld-ex from "a useful library" to "a paradigm for ML data representation."

If ET1 results are strong enough, they could constitute a **separate paper** (ICML, ICLR, or EMNLP) on the relationship between training data representation and model calibration, with jsonld-ex as the specific instantiation.

---

## 15. Timeline

| Week | Milestone |
|------|-----------|
| W1 | Protocol review, knowledge base schema design, TDD: KB generator tests + implementation |
| W2 | TDD: Data formatter tests + implementation, response parser tests + implementation |
| W3 | TDD: Metrics + statistical tests. Integration test of full pipeline with dummy data. |
| W4 | Pilot runs: SmolLM2-135M × 7 conditions × 3 seeds = 21 runs |
| W5 | Pilot runs: SmolLM2-360M + Qwen2.5-0.5B = 42 more runs. Result analysis. |
| W6 | Decision point: signal/no-signal. If signal → scale-up plan. If null → honest write-up. |

---

## 16. Open Questions (To Resolve Before Coding)

1. **Prompt template for confidence elicitation:** Should we use structured output prompts (e.g., "respond in JSON with 'answer' and 'confidence' fields") or free-form ("answer and state your confidence")? Structured is easier to parse but may bias toward JSON-trained models.

2. **SmolLM2 tokenizer:** Does the SmolLM2 tokenizer handle JSON well, or does it fragment JSON syntax tokens? Poor tokenization would disadvantage C2–C7. We should check tokenizer behavior before committing.

3. **Full fine-tuning vs LoRA for 135M:** At 135M parameters, full fine-tuning is feasible (~540MB model). Should we run both and compare? This adds 21 more runs but controls for LoRA capacity.

4. **Verbal confidence mapping:** The mapping from verbal expressions to numeric values (§8.2) is somewhat arbitrary. Should we use a secondary LLM to extract numeric confidence from verbose responses? This adds complexity but reduces mapping bias.

5. **What constitutes a "correct" answer for Meridian facts?** We need a deterministic grading function. Exact match is too strict (different phrasings of the same fact); fuzzy match risks false positives. Proposed: normalized Levenshtein distance < 0.3 OR entity name match + relation match.

---

*End of Protocol v0.1*

*This document will be updated with protocol amendments as decisions are made. All amendments will be timestamped and justified.*
