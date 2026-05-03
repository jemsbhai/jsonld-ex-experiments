# EN4.2 — Baseline Comparisons: Dempster-Shafer vs Subjective Logic

## Status: DESIGN v1 (hardened for NeurIPS)

## Why This Experiment Exists

The single most dangerous reviewer question for this paper is:

> "Subjective Logic IS Dempster-Shafer theory. What does Jøsang's
> formulation add over the classical DS combination rule? Show me
> empirically."

We currently have NO empirical answer anywhere in the paper. This
experiment provides one, using the real NER datasets and cached model
predictions we already have.

## Background: DS vs SL — The Actual Mathematical Differences

For a binary frame Θ = {entity, ¬entity}:

**Mass function representation (shared):**
- m(entity) = b, m(¬entity) = d, m(Θ) = u, where b + d + u = 1

**Classical Dempster's Rule (Dempster 1967, Shafer 1976):**
```
K       = b₁d₂ + d₁b₂                          (conflict mass)
m(ent)  = (b₁b₂ + b₁u₂ + u₁b₂) / (1 - K)      (normalized)
m(¬ent) = (d₁d₂ + d₁u₂ + u₁d₂) / (1 - K)
m(Θ)    = (u₁u₂) / (1 - K)
```
Decision: max(m(ent), m(¬ent)) — no prior adjustment.

**Yager's Rule (Yager 1987):**
```
m(ent)  = b₁b₂ + b₁u₂ + u₁b₂                   (unnormalized)
m(¬ent) = d₁d₂ + d₁u₂ + u₁d₂
m(Θ)    = u₁u₂ + K                              (conflict → uncertainty)
```
Decision: max(m(ent), m(¬ent)) — conflict becomes ignorance.

**SL Cumulative Fusion (Jøsang 2016, §12.3):**
```
κ       = u₁ + u₂ - u₁u₂                       (uncertainty denominator)
b_fused = (b₁u₂ + b₂u₁) / κ                    (NO b₁b₂ term)
d_fused = (d₁u₂ + d₂u₁) / κ
u_fused = (u₁u₂) / κ
```
Decision: P = b + a·u > threshold — **prior-adjusted** via base rate a.

**Key empirical differences to test:**

| Aspect | Classical DS | Yager | SL Cumulative |
|--------|-------------|-------|---------------|
| Conflict handling | Normalize away (1-K) | → uncertainty | Separate metric |
| Belief-belief interaction | b₁b₂ in numerator | b₁b₂ in numerator | NO b₁b₂ term |
| Prior/base rate | None | None | Explicit a parameter |
| Decision function | max(m) | max(m) | P = b + a·u |
| Pathological K→1 | Blow-up (Zadeh) | Pushes to uncertainty | Stable |

---

## Hypotheses

### H4.2a — SL vs DS on NER Fusion: Overall F1 (Primary)

**Claim:** SL cumulative fusion achieves equal or better entity-level F1
than classical Dempster's rule and Yager's rule on real NER benchmarks.

**Protocol:**
1. Load cached predictions for three datasets:
   - CoNLL-2003 test set (5 models: spaCy, Flair, Stanza, HuggingFace, GLiNER2)
   - BC5CDR test set (2 models: GLiNER2, GLiNER-BioMed)
   - MedMentions-corrected test set (2 models: GLiNER2, GLiNER-BioMed)
2. For each entity/token position, construct mass functions from model scores.
   Mass construction: same as EN1.1/EN3.4 — `from_confidence(score, uncertainty=u)`
   yields (b, d, u). Use the same per-model or per-bin uncertainty values
   already calibrated in those experiments.
3. Apply four combination rules:
   - **DS-classical:** Dempster's normalized rule
   - **DS-Yager:** Yager's conflict-to-uncertainty rule
   - **SL-cumulative:** Jøsang's cumulative fusion (our library)
   - **SL-averaging:** Jøsang's averaging fusion (our library)
4. For DS methods, predict entity if m(entity) > m(¬entity).
   For SL methods, predict entity if P = b + a·u > threshold (optimized on dev).
5. Evaluate: entity-level P, R, F1 (exact span match).
6. Bootstrap CI (n=2000) on F1 differences: SL-cumulative vs DS-classical.
7. Report per-dataset and pooled.

**Success criterion:** SL F1 ≥ DS F1 on at least 2 of 3 datasets, with
bootstrap CI including or above zero.

**Failure criterion:** DS consistently outperforms SL. If so, investigate
whether the base rate a is miscalibrated and report honestly.

### H4.2b — Conflict Robustness: Zadeh's Paradox (Primary)

**Claim:** Classical DS produces counterintuitive results under high
inter-source conflict, while SL degrades gracefully.

**Protocol:**
1. **Synthetic Zadeh scenario:** Two sources with high but conflicting evidence.
   - Source 1: m(A)=0.9, m(B)=0.1, m(Θ)=0.0 — strongly believes A
   - Source 2: m(A)=0.0, m(B)=0.9, m(Θ)=0.1 — strongly believes B
   - DS result: m(A)=0.0, m(B)=0.90/0.91≈0.989 — ALL evidence for B,
     despite one source strongly supporting A. This is Zadeh's paradox.
   - SL result: should produce high uncertainty and low belief for either.
2. **Systematic sweep:** Generate 10,000 opinion pairs with controlled conflict
   levels (K ∈ [0, 0.2, 0.4, 0.6, 0.8, 0.95]).
   For each conflict level: compare DS vs SL fused uncertainty u_fused.
3. **Real-data conflict partition:** From each NER dataset, partition entity
   positions by conflict level (quartiles). Compare DS vs SL F1 within
   each conflict quartile.

**Success criterion:** SL maintains meaningful uncertainty under high conflict
(u_fused > 0.3 when K > 0.6). DS should show pathological certainty.
In real data, SL should outperform DS specifically in high-conflict partitions.

**Failure criterion:** SL and DS behave identically even at K > 0.8.

### H4.2c — Base Rate Effect (Ablation)

**Claim:** SL's base rate parameter a provides a principled advantage over
DS under class imbalance — the projected probability P = b + a·u reverts
to the prior when evidence is sparse, while DS has no such mechanism.

**Protocol:**
1. On each NER dataset, vary the SL base rate:
   a ∈ {0.1, 0.2, 0.3, 0.5 (default), 0.7, empirical_class_prior}
2. DS methods are invariant to base rate (they don't have one).
3. Measure F1 at each base rate setting.
4. Report: optimal base rate per dataset, F1 sensitivity to base rate,
   whether empirical class prior is near-optimal.
5. Specifically analyze the MedMentions dataset (sparse entities, high
   class imbalance) where base rate should matter most.

**Success criterion:** Empirical class prior as base rate outperforms
a=0.5 by ≥0.5pp F1 on at least one dataset. Base rate sensitivity
curve is smooth (no pathological sensitivity).

**Failure criterion:** Base rate has zero effect (all F1s identical).
This would mean the advantage over DS is purely in the combination rule,
not the base rate — still reportable but a different finding.

### H4.2d — Abstention Under Conflict (SL Advantage)

**Claim:** SL's conflict_metric provides an abstention signal that DS
methods cannot natively produce, yielding higher precision on
non-abstained predictions.

**Protocol:**
1. On each dataset, apply conflict-based abstention (SL only) at
   thresholds: 0.05, 0.10, 0.15, 0.20, 0.30, 0.50.
2. For DS, construct comparable abstention signals:
   - DS-conflict: K = b₁d₂ + d₁b₂ (the normalization factor)
   - DS-uncertainty: m(Θ) after combination
3. Compare abstention quality: AUROC as error detector.
4. At matched abstention rates: compare precision on accepted predictions.

**Success criterion:** SL conflict_metric AUROC ≥ DS-K AUROC. SL achieves
higher precision at matched abstention rates.

**Failure criterion:** DS-K provides equivalent abstention quality to SL
conflict_metric. If so, the conflict detection advantage is an
implementation convenience, not a theoretical one — still report honestly.

---

## Datasets and Data Sources

| Dataset | Source | Models | Format | Cached? |
|---------|--------|--------|--------|---------|
| CoNLL-2003 | EN1.1 checkpoints | 5 (spaCy, Flair, Stanza, HF, GLiNER2) | token-level {tag, confidence} | ✅ |
| BC5CDR | EN3.4 checkpoints | 2 (GLiNER2, GLiNER-BioMed) | span-level {start, end, type, score} | ✅ |
| MedMentions (corrected) | EN3.4 checkpoints | 2 (GLiNER2, GLiNER-BioMed) | span-level | ✅ |
| Synthetic (Zadeh) | Generated | N/A | Controlled mass functions | N/A |

**No new model runs required.** All predictions are cached from prior experiments.

---

## Statistical Methods

- Bootstrap CIs (n=2000) for all F1 and AUROC differences
- Cohen's d effect sizes for all pairwise comparisons
- Holm-Bonferroni for family-wise error control (4 hypotheses)
- Paired analysis: same entity positions, different combination rules
- All random seeds documented (seed=42 for bootstrap)

---

## Implementation Plan

```
experiments/EN4/
├── EN4_2_design.md              # This document
├── en4_2_ds_comparison.py       # Classical DS, Yager, SL comparison
├── run_en4_2.py                 # Main runner
├── tests/
│   ├── conftest.py              # Path setup
│   └── test_en4_2.py            # RED-phase tests
└── results/
    └── EN4_2_ds_comparison.json # Primary results
```

---

## Dependencies

- jsonld-ex >= 0.7.2 (SL fusion operators)
- numpy (DS combination rule implementation)
- scipy (bootstrap)
- No external packages beyond what's already installed
- No GPU required

---

## What We Expect to Find (Honest Pre-Registration)

1. **Low-conflict regime (most NER data):** DS and SL should produce SIMILAR
   results. The NER models we use are reasonably calibrated and rarely produce
   extreme disagreement. If SL massively outperforms DS here, we should be
   suspicious of our implementation.

2. **High-conflict regime:** SL should outperform DS because DS normalizes
   away conflict while SL preserves it. This is the theoretical advantage.

3. **Base rate effect:** Likely small on CoNLL-2003 (balanced entity classes)
   but meaningful on MedMentions (sparse entities, high class imbalance).

4. **Abstention:** SL's conflict_metric should provide a cleaner error signal
   than DS's K factor because SL separates combination from conflict assessment.

5. **Overall F1 difference:** Likely SMALL (< 1pp). The contribution is not
   "SL beats DS by 10%" — it's "SL provides the same fusion quality PLUS
   base rate handling, conflict detection, and abstention, within a unified
   framework that also does SSN/SOSA, FHIR, PROV-O, etc."

---

## Anticipated Reviewer Objections

| Objection | Defense |
|-----------|---------|
| "The differences are tiny" | Acknowledged in pre-registration. SL's advantage is the unified framework, not raw fusion RMSE. Show that DS + manual base rate + manual conflict ≈ SL's built-in functionality. |
| "Why not TBM (Smets)?" | TBM uses unnormalized DS + pignistic probability. Yager's rule approximates this. TBM discussed in Related Work. |
| "Your DS implementation might be wrong" | Unit-test against published textbook examples (Shafer 1976, Zadeh 1979). All code committed for inspection. |
| "Only 3 datasets" | These are canonical benchmarks with cached multi-model predictions. Adding more datasets requires new model runs — out of scope. |
| "3-frame DS (PER/ORG/LOC/MISC) not binary" | Valid — EN1.1 is multi-class. We implement both binary (entity-vs-not) and multi-frame DS. Binary is the primary comparison; multi-frame is an ablation. |

---

## Preliminary Finding: Structural Divergence (from RED-phase diagnostics)

**Discovery during test development (pre-experiment):**

DS and SL are NOT "approximately the same under low conflict." They are
structurally different combination rules that diverge at ALL conflict levels.

**Root cause:** DS includes a b₁b₂ mutual reinforcement term in the belief
numerator that SL explicitly excludes. This term is always non-negative,
so DS systematically produces higher fused belief than SL from the same
inputs.

**Measured divergence (500 random pairs, seed=42):**

| Conflict Level | Mean Belief Divergence (DS − SL) | n |
|---------------|----------------------------------|---|
| K < 0.10      | ~0.10                            | ~11 |
| K ∈ [0.10, 0.30) | ~0.12                        | ~32 |
| K ≥ 0.30      | ~0.12                            | varies |

Correlation between divergence and b₁b₂/(1-K): r = 0.24 (weak).
The divergence is a structural property of the formula, not driven
primarily by any single input variable.

**Implication:** DS is systematically more aggressive — it gives higher
confidence when both sources agree. Whether this helps or hurts on real
NER data is an empirical question that the full experiment will answer.
If the sources are well-calibrated, DS's mutual reinforcement may be
overconfident. If sources are conservative, it may help.

**This finding motivates an additional ablation (H4.2e).**

---

### H4.2e — Mutual Reinforcement Ablation (NEW)

**Question:** Is DS's b₁b₂ mutual reinforcement term beneficial or
harmful for NER fusion? Can we isolate the contribution of each
structural difference between DS and SL?

**Protocol:**

Five conditions on all three NER datasets, all using the same mass
functions from cached predictions:

| ID | Method | Combination Rule | Decision Rule | Isolates |
|----|--------|-----------------|---------------|----------|
| C1 | DS-classical | Dempster (with b₁b₂) | max(m) | Baseline DS |
| C2 | DS + base rate | Dempster (with b₁b₂) | P = b + a·u | Base rate effect on DS |
| C3 | Yager | Conflict → Θ | max(m) | Conflict handling |
| C4 | SL-cumulative | Jøsang (no b₁b₂) | P = b + a·u | Full SL |
| C5 | SL-averaging | Jøsang averaging | P = b + a·u | Conservative SL |

Key comparisons:
- C1 vs C2: does adding a base rate to DS help? (Isolates base rate.)
- C1 vs C4: full DS vs full SL (combined effect of formula + base rate.)
- C2 vs C4: same decision rule, different combination (isolates b₁b₂ term.)
- C3 vs C4: different conflict handling, both without normalization.

**Success criterion:** At least one comparison shows a statistically
significant F1 difference (bootstrap CI excludes zero).

**If C1 ≈ C4:** The structural differences don't matter in practice
for well-calibrated NER. Report this honestly — the value of SL is
then purely in the unified framework, not the combination rule.

**If C2 ≈ C4:** The base rate is the key contribution, not the
combination rule. The paper should emphasize the base rate mechanism.

**If C2 ≠ C4:** The b₁b₂ term itself affects outcomes. Report which
direction (DS or SL) performs better and analyze why.

1. The comparison uses post-hoc opinion construction (from model scores),
   not end-to-end evidential models. This favors SL's design assumptions.
2. Three NER datasets from two domains (news, biomedical). We do not claim
   generalizability to all ML fusion tasks.
3. DS and SL differences are expected to be small on well-calibrated models.
   The paper's contribution is the unified infrastructure, not fusion superiority.
4. Multi-frame DS (for multi-class NER) requires implementation choices
   (how to handle multi-class conflict) that could affect results. We document
   our choices and test alternatives.
