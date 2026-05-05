# EN8.6 — Graph Merge and Diff Operations: Full Experiment Design

**Suite:** EN8 (NeurIPS 2026, Primary)
**Priority:** HIGH
**Placement:** Primary — Section 5 (Feature Showcase: Graph Operations)
**Est. Total Time:** 4–6 hours
**Date:** 2026-05-04
**Author:** Muntaser Syed

---

## Motivation

Knowledge graph merge is a fundamental operation in ML pipelines: multi-model
extraction, federated learning aggregation, and multi-source data integration all
produce overlapping KG fragments that must be reconciled. Standard RDF tools
(rdflib `Graph.__iadd__`) provide set-union of triples — when two sources assert
different values for the same property, both triples coexist with no principled
resolution. JSON Schema merges are typically ad-hoc (deepmerge, jsonmerge) with
no confidence semantics.

jsonld-ex's `merge_graphs` provides confidence-aware merge: when sources agree,
confidence is boosted (noisy-OR/average/max); when sources conflict, the winner
is selected by confidence-based strategy (highest, weighted_vote, recency) with
a full audit trail (`MergeReport`). `diff_graphs` provides semantic diff between
JSON-LD graphs, ignoring annotation metadata.

The core claim is NOT that confidence-aware merge is a novel algorithm (it is
essentially argmax-by-confidence or noisy-OR boosting). The claim is that
jsonld-ex provides this as an integrated, auditable pipeline operation on JSON-LD
graphs — something no existing JSON-LD/RDF tool offers out of the box.

---

## Anticipated NeurIPS Reviewer Objections

| Objection | How We Address |
|-----------|---------------|
| "Synthetic data designed to make your method win" | Phase 1 sweeps confidence-correctness correlation from r=-0.3 (adversarial) to r=0.8 (ideal). Phase 2 uses real-world DBpedia×Wikidata data. |
| "100 nodes is trivial" | Scale from 100 to 5000 nodes with explicit scaling analysis. |
| "Why not compare against real KG merge tools?" | We compare 5 baselines: rdflib union, rdflib+random, rdflib+majority, rdflib+most-recent, and custom rdflib+confidence (reimplemented). |
| "Confidence correlated with correctness is circular" | Correlation is a controlled parameter we sweep, not an assumption. We explicitly measure the threshold where advantage disappears. |
| "The diff function is trivial" | Acknowledged as utility; tested for correctness, not novelty. |
| "What about multi-way conflicts?" | Explicitly tested: 2-way and 3-way conflicts with multi-valued properties. |
| "What about miscalibrated confidence?" | Robustness analysis with 4 calibration regimes. |
| "Does this work on real data?" | Phase 2: DBpedia × Wikidata overlapping entities with curated ground truth. |

---

## Phase 1: Synthetic Data (Controlled Ground Truth)

### Data Generation

**Domain:** Academic researcher knowledge graph.

**Ground-truth KG:** 500 entities (researchers), each with 10 typed properties:
- `name` (string, unique, invariant)
- `affiliation` (string, may vary between sources)
- `field` (string)
- `h_index` (integer)
- `email` (string)
- `publications_count` (integer)
- `country` (string)
- `active` (boolean)
- `homepage` (URL)
- `orcid` (string, unique identifier)

**Source generation (3 overlapping KGs):**
- KG_A: entities 1–300 (60% of total)
- KG_B: entities 101–400 (60% of total)
- KG_C: entities 201–500 (60% of total)
- Pairwise overlap: ~40% (200 entities)
- Three-way overlap: ~20% (100 entities)
- Unique to each source: ~20% (100 entities)

For each entity in each source:
- Each property has `p_correct` probability of carrying the ground-truth value
- Incorrect properties receive a plausible but wrong value from same domain
- Confidence is assigned based on `calibration_regime` (see below)

**Controlled Parameters (full sweep):**

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `corruption_rate` | {0.05, 0.10, 0.20, 0.40} | % of properties per source that are incorrect |
| `calibration_regime` | {ideal, noisy, uncalibrated, adversarial} | See below |
| `n_sources` | {2, 3, 5} | Number of overlapping sources |
| `graph_size` | {100, 500, 1000, 5000} | Nodes per source |
| `overlap_rate` | {0.2, 0.4, 0.6} | Fraction of entities shared between source pairs |

**Calibration Regimes:**
1. **Ideal** (r ≈ 0.7–0.8): Correct values get conf ~ Beta(8,2) (mean 0.8); incorrect get conf ~ Beta(3,5) (mean 0.375). Strong positive correlation between confidence and correctness.
2. **Noisy** (r ≈ 0.3–0.4): Correct values get conf ~ Beta(5,3) (mean 0.625); incorrect get conf ~ Beta(3,4) (mean 0.43). Weak but positive correlation.
3. **Uncalibrated** (r ≈ 0): All values get conf ~ Beta(4,4) (mean 0.5) regardless of correctness. Confidence is uninformative.
4. **Adversarial** (r ≈ -0.3): Incorrect values get HIGHER confidence than correct ones. Correct get conf ~ Beta(3,5) (mean 0.375); incorrect get conf ~ Beta(8,2) (mean 0.8).

**Special conflict construction (for H8.6b):**
- 20% of conflicts are "majority-correct": the correct value is asserted by 2+
  sources with moderate confidence (0.65–0.80 each), while one source asserts
  the wrong value with high confidence (0.82–0.90). This tests whether
  `weighted_vote` correctly aggregates multiple corroborating sources against
  a single high-confidence outlier.

### Hypotheses

**H8.6a (Conflict Resolution Accuracy Under Ideal Calibration):**
When confidence is positively correlated with correctness (ideal regime),
`merge_graphs(strategy="highest")` selects the ground-truth-correct value for
≥80% of conflicting properties. All baselines that ignore confidence achieve
≤ majority-vote accuracy (~67% for 3 sources, ~50% for 2 sources).

*Failure criterion:* jsonld-ex highest-confidence accuracy < 75%, OR rdflib
naive union matches jsonld-ex accuracy.

**H8.6b (Weighted Vote vs Highest — Strategic Divergence):**
On the "majority-correct" conflict subset (correct value supported by multiple
lower-confidence sources vs. one high-confidence incorrect source),
`weighted_vote` outperforms `highest` by ≥ 10pp accuracy. On all other
conflicts, `highest` ≥ `weighted_vote` - 5pp (i.e., no worse within tolerance).

*Failure criterion:* No measurable divergence between strategies, OR
`weighted_vote` is uniformly worse.

**H8.6c (Calibration Sensitivity — Honest Boundary):**
The accuracy advantage of confidence-aware merge over majority-vote is
monotonically related to calibration quality:
- Ideal: Δ accuracy > 15pp
- Noisy: Δ accuracy > 5pp
- Uncalibrated: Δ accuracy < 0 (majority vote retains counting advantage)
- Adversarial: Δ accuracy < 0 (confidence-aware is WORSE)

This is the HONEST finding: confidence-aware merge helps when confidence is
informative and hurts when confidence is adversarial. We report the crossover
point explicitly.

*Failure criterion:* The relationship is not monotonic, OR adversarial regime
still shows a positive Δ (would suggest a bug in our setup).

**H8.6d (Agreement Confidence Boosting — Calibration Quality):**
For properties where all sources agree, noisy-OR combined confidence is better
calibrated than single-source max confidence, measured by ECE (10 equal-width
bins). Target: ECE_noisy_or < ECE_max.

*Failure criterion:* ECE_noisy_or ≥ ECE_max.

**H8.6e (Diff Completeness and Correctness):**
`diff_graphs` achieves 100% precision and 100% recall in detecting added,
removed, and modified properties. Tested on 10 constructed graph pairs with
known diffs (500+ total diff operations spanning: node additions, node
removals, property modifications, property additions within existing nodes,
type changes, annotation-only changes).

*Failure criterion:* Any missed change or false positive.

**H8.6f (Audit Trail Fidelity):**
`MergeReport` records every conflict with correct node_id, property_name,
all candidate values, resolution strategy, and winner value — matching a
ground-truth oracle exactly. Report completeness = 100%.

*Failure criterion:* Any missing or incorrect conflict record.

**H8.6g (Throughput and Scaling):**
`merge_graphs` on 3 × 100-node graphs completes in < 50ms (median over 100
trials). Scaling: 3 × {100, 500, 1000, 5000} nodes. Scaling exponent ≤ 1.2
(near-linear). Report p50, p95, p99 latency.

*Failure criterion:* Scaling exponent > 1.5 (super-linear), OR > 500ms at
100-node scale.

### Baselines

| ID | Method | Description |
|----|--------|-------------|
| B1 | rdflib naive union | `g1 + g2 + g3` — triple set-union, last-added-wins for s/p conflicts |
| B2 | Random choice | For each conflict, pick uniformly at random among candidates |
| B3 | Majority vote | Pick the value asserted by the most sources (ties broken randomly) |
| B4 | Most-recent | Pick the value with the most recent `@extractedAt` timestamp |
| B5 | rdflib + confidence argmax | Custom rdflib merge that reads `@confidence` from triples and picks argmax (reimplements jsonld-ex's `highest` strategy using raw rdflib — same algorithm, different implementation) |

B5 is critical: it isolates whether the advantage is from the ALGORITHM
(confidence-aware selection) vs. the IMPLEMENTATION (jsonld-ex pipeline). If
B5 matches jsonld-ex accuracy, the contribution is ergonomics/audit trail/
integration, not algorithmic. This is an honest control.

### Metrics

| Metric | How Measured | Per Hypothesis |
|--------|-------------|----------------|
| Conflict resolution accuracy | % of conflicts where winner == ground truth | H8.6a, H8.6b, H8.6c |
| Strategy-specific accuracy | Accuracy broken out by conflict type (standard vs majority-correct) | H8.6b |
| ECE (Expected Calibration Error) | 10 equal-width bins, calibration of merged confidence | H8.6d |
| Diff precision/recall | TP/(TP+FP), TP/(TP+FN) for added/removed/modified | H8.6e |
| Audit completeness | % of true conflicts recorded in MergeReport | H8.6f |
| Throughput (p50/p95/p99) | Wall-clock ms, 100 trials, 5-trial warmup | H8.6g |
| Scaling exponent | Linear regression of log(time) vs log(n_nodes) | H8.6g |
| Bootstrap CIs (n=2000) | For all accuracy and ECE comparisons | All |

---

## Phase 2: Real-World Data (DBpedia × Wikidata)

### Rationale

A NeurIPS reviewer will rightfully question whether synthetic results
transfer to real-world conditions. We use the DBpedia–Wikidata overlap
(136M+ owl:sameAs links, CC-BY-SA licensed) as a natural multi-source
KG merge scenario.

### Protocol

1. **Entity Selection:** Query DBpedia SPARQL endpoint for two domains:
   - 500 `dbo:Scientist` entities (aligns with synthetic domain, stable data)
   - 500 `dbo:Company` entities (rapidly-changing data, more conflicts expected)
   Both filtered to entities with owl:sameAs links to Wikidata.

2. **Property Extraction:** For each entity, extract 8–10 comparable
   properties from BOTH sources via SPARQL:
   - DBpedia: `dbo:birthDate`, `dbo:nationality`, `dbo:almaMater`,
     `dbo:field`, `dbo:knownFor`, `dbo:award`, etc.
   - Wikidata: `wdt:P569` (birth date), `wdt:P27` (citizenship),
     `wdt:P69` (alma mater), `wdt:P101` (field), etc.
   Map to common property names.

3. **Conflict Identification:** Compare property values between sources.
   Classify each property as: agreed, conflicted, or missing-in-one.

4. **Ground Truth Curation:**
   - For factual properties (birth date, nationality): use the value
     with more supporting references. If tied, use a third source
     (YAGO, Google KG) as tiebreaker.
   - For subjective properties (field, knownFor): accept both as valid
     (mark as "soft conflict" — not evaluated for accuracy but included
     in merge analysis).
   - Randomly sample 100 hard conflicts for manual verification.
   - Document the curation process and any ambiguity.

5. **Confidence Assignment:** Assign confidence scores based on real
   metadata:
   - DBpedia: Number of Wikipedia references for the claim, recency
     of the Wikipedia edit
   - Wikidata: Number of reference statements (`P248`), rank (preferred/
     normal/deprecated)
   - Normalize to [0, 1] using a documented heuristic

6. **Convert to JSON-LD:** Build JSON-LD documents with `@id` (shared
   URI via sameAs), `@confidence`, `@source`, `@extractedAt`.

7. **Run Merge:** Apply all strategies (highest, weighted_vote, recency,
   union) + all baselines (B1–B5). Report accuracy against curated
   ground truth.

8. **Compare with Phase 1 predictions:** Do the calibration regime and
   accuracy trends from Phase 1 predict Phase 2 results? Measure the
   actual confidence-correctness correlation in the real data and
   compare to the closest synthetic regime.

### What We Expect (and What Would Be Interesting)

- Real-world confidence-correctness correlation is likely in the "noisy"
  regime (r ≈ 0.2–0.4). If Phase 1 predicts this gives Δ accuracy ≈ 5–10pp,
  and Phase 2 confirms, that validates the synthetic analysis.
- If real-world correlation is ≈ 0 (uncalibrated), the honest finding is
  that confidence-aware merge provides no accuracy advantage on this data,
  but still provides audit trails and structured conflict resolution that
  naive union does not.
- We report whichever result we get.

---

## Phase 3: Sensitivity & Robustness Analysis

### 3a. Parameter Sweep (Phase 1 configurations)

Full factorial design over:
- `corruption_rate` × `calibration_regime` × `n_sources`
- Fixed `graph_size=500` for the sweep (scaling tested separately in H8.6g)
- 4 × 4 × 3 = 48 configurations
- 5 random seeds per configuration = 240 runs
- Report: heatmap of Δ accuracy (jsonld-ex highest − majority vote)
  across the parameter space

### 3b. Scaling Analysis

- `graph_size` ∈ {100, 500, 1000, 5000}
- Fixed: `corruption_rate=0.10`, `calibration_regime=ideal`, `n_sources=3`
- Report: accuracy (should be size-independent) + throughput (should scale
  near-linearly)

### 3c. Confidence Distribution Robustness

- Test with non-Beta confidence distributions: uniform, bimodal, heavily
  skewed (0.9/0.1 spike)
- Verify merge accuracy is robust to distribution shape when rank-ordering
  is preserved

---

## Comparison with Related Work

| Tool | Confidence-Aware | Conflict Resolution | Audit Trail | JSON-LD Native |
|------|-----------------|--------------------|----|---|
| rdflib Graph union | No | Last-added-wins | No | No |
| Apache Jena | No | Named graphs | No | No |
| jsonmerge (Python) | No | Custom strategies | No | No (JSON only) |
| LIMES/Silk | Similarity-based | Threshold | Link log | No |
| jsonld-ex | Yes | 4 strategies | Full MergeReport | Yes |

---

## Directory Structure

```
experiments/EN8/
├── en8_6_core.py           # Data generation, baselines, evaluation
├── en8_6_real_world.py     # Phase 2: DBpedia × Wikidata pipeline
├── run_en8_6.py            # Full runner (Phase 1 + 2 + 3)
├── tests/
│   ├── test_en8_6.py       # RED-phase tests
│   └── conftest.py         # (existing)
└── results/
    └── en8_6_*.json        # Primary + timestamped archive
```

### Dependencies

```
# Already available
jsonld-ex (local editable)
numpy
rdflib

# Needed for Phase 2
SPARQLWrapper  # For DBpedia/Wikidata SPARQL queries
requests       # For fallback HTTP queries
```

---

## Execution Order

1. Write RED-phase tests (`test_en8_6.py`)
2. Implement `en8_6_core.py` (data gen, baselines, evaluation)
3. GREEN: verify tests pass
4. Run Phase 1 sweep (synthetic, ~48 configs × 5 seeds)
5. Implement `en8_6_real_world.py` (SPARQL extraction, conversion)
6. Run Phase 2 (real-world, ~500 entities)
7. Run Phase 3 (sensitivity analysis)
8. Save results, append to FINDINGS.md
9. Commit and push

---

## What Constitutes Failure

- H8.6a: jsonld-ex accuracy < 75% under ideal calibration → investigate
  bug in merge_graphs or data generation
- H8.6c: Adversarial regime shows POSITIVE Δ → bug in data gen (confidence
  should be anti-correlated with correctness)
- H8.6e: Any diff miss → bug in diff_graphs, file issue
- Phase 2: No conflicts found in real data → wrong property mapping,
  need different entity domain
- Phase 2: SPARQL endpoints down → cache responses, retry with fallback

---

## Honest Framing for Paper

The contribution is NOT "a new merge algorithm" — argmax-by-confidence is
trivial. The contribution IS:

1. **Integrated pipeline:** annotate → merge → diff → audit, all on JSON-LD
   graphs with W3C-compatible semantics
2. **Principled confidence boosting:** noisy-OR for agreed properties, with
   measured calibration quality
3. **Full audit trail:** MergeReport documents every conflict, every
   resolution, every candidate — essential for regulatory compliance and
   reproducibility
4. **Empirical characterization:** When does confidence-aware merge help?
   (Answer: when confidence is informative. When it's not, you get majority
   vote performance with better audit trails.)
5. **Code complexity:** jsonld-ex merge is 3 LoC (merge_graphs call +
   strategy selection). rdflib equivalent requires ~50+ LoC of custom
   conflict resolution logic.
