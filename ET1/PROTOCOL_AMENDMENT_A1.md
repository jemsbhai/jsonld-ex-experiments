# Protocol Amendment A1: Canonical Datasets

**Date:** April 27, 2026
**Amends:** PROTOCOL.md §4, §5, §8, §10, §12

---

## Rationale

The original protocol used only synthetic data (Meridian). A NeurIPS reviewer
will rightly ask: "Why not use an established benchmark?" Synthetic data
controls confounds but lacks ecological validity. We now adopt a three-dataset
design that provides both internal and ecological validity.

---

## Amended Design: Three-Dataset Triangulation

| Dataset | Primary Hypothesis | Why This Dataset |
|---------|--------------------|------------------|
| **FEVER** (Thorne et al., 2018) | H2 — Calibration Transfer | 3-way verification labels (SUPPORTED / REFUTED / NEI) map naturally to SL belief / disbelief / uncertainty. ~185K claims. ~3,500+ citations. |
| **SQuAD 2.0** (Rajpurkar et al., 2018) | H3 — Uncertainty Awareness | Answerable / unanswerable split is the gold standard for selective prediction. Multiple annotator answers provide continuous agreement signal. ~150K questions. ~5,000+ citations. |
| **Meridian** (synthetic) | All three (control) | Zero contamination, continuous SL values across full simplex, verifies mechanism under ideal conditions. |

If effects appear on FEVER and SQuAD but not Meridian → ecological validity
confirmed, but mechanism unclear. If effects appear on Meridian but not
FEVER/SQuAD → mechanism works under ideal conditions but doesn't generalize.
If effects appear on all three → strongest possible result.

---

## A1.1 FEVER: SL Opinion Derivation

FEVER provides claims paired with evidence sentences and a verification label.
We derive SL opinions from the label AND the evidence structure:

### Label-to-Opinion Base Mapping

| FEVER Label | Base SL Opinion | Confidence Tier |
|-------------|----------------|-----------------|
| SUPPORTED | b=0.90, d=0.02, u=0.08 | T1 (established) |
| REFUTED | b=0.05, d=0.88, u=0.07 | T5-like (high disbelief) |
| NOT ENOUGH INFO | b=0.20, d=0.10, u=0.70 | T3/T4 (high uncertainty) |

### Evidence-Modulated Refinement

The base opinions above are starting points. We refine them using evidence
metadata to create continuous variation within each label:

**For SUPPORTED claims:**
- Number of evidence sentences: more evidence → higher belief (b += 0.01 per extra sentence, capped)
- Evidence from multiple Wikipedia articles → higher belief
- Single-sentence evidence → slightly lower belief (b = 0.85)

**For REFUTED claims:**
- Direct contradiction evidence → higher disbelief (d = 0.90+)
- Implicit refutation → moderate disbelief (d = 0.75)

**For NOT ENOUGH INFO claims:**
- All assigned high uncertainty (u ≥ 0.60)
- Variation based on how many related entities exist in the evidence

### Provenance Mapping

- Source identifiers: Wikipedia article titles from evidence
- Source reliability: based on Wikipedia article length/edit count (proxy, fetched once)
- Method: "evidence_based_verification"
- Temporal: claims about dated events get valid_from metadata

### Concrete Example

**FEVER claim:** "The Colosseum is in Rome."
**Label:** SUPPORTED
**Evidence:** 1 sentence from "Colosseum" article.

**C1 (Plain Text):**
```
User: Is the following claim true? "The Colosseum is in Rome."
Assistant: The claim is supported. The Colosseum is indeed located in Rome.
```

**C4 (jsonld-ex Full):**
```
User: Is the following claim true? "The Colosseum is in Rome."
Assistant: {
  "@context": ["https://schema.org/", "https://jsonld-ex.org/context/v1"],
  "@type": "ClaimReview",
  "claimReviewed": "The Colosseum is in Rome.",
  "reviewRating": {
    "@type": "Rating",
    "ratingValue": "SUPPORTED",
    "@opinion": {"belief": 0.92, "disbelief": 0.02, "uncertainty": 0.06, "base_rate": 0.5},
    "@source": ["urn:wikipedia:Colosseum"],
    "@method": "evidence_based_verification"
  }
}
```

---

## A1.2 SQuAD 2.0: SL Opinion Derivation

SQuAD 2.0 provides questions, context paragraphs, and either answer spans
(answerable) or an empty answer list (unanswerable). Answerable questions
have multiple annotator answers.

### Answerable Questions: Agreement-Based Confidence

For answerable questions with N annotator answers:

1. **Compute annotator agreement:** pairwise F1 between all annotator answers.
   Mean pairwise F1 = agreement score ∈ [0, 1].
2. **Map to SL opinion:**
   - High agreement (F1 ≥ 0.8): b = 0.85 + 0.10 * F1, low u
   - Moderate agreement (0.5 ≤ F1 < 0.8): b = 0.5 + 0.3 * F1, moderate u
   - Low agreement (F1 < 0.5): b = 0.3 + 0.3 * F1, high u
3. **Uncertainty from answer diversity:** if annotators gave different answer
   spans, u increases. If all annotators agree on exact span, u is minimal.

### Unanswerable Questions: High Uncertainty

- b = 0.05, d = 0.15, u = 0.80
- These are questions where the context paragraph does NOT contain the answer.
- The "correct" model behavior is to abstain or express high uncertainty.
- This directly tests H3.

### Provenance Mapping

- Source: the context paragraph title/ID
- Method: "extractive_qa_annotation"
- Reliability: number of annotators who provided answers

### Concrete Example

**SQuAD 2.0 question:** "What is the capital of Normandy?"
**Context:** [paragraph about Normandy]
**Annotator answers:** ["Rouen", "Rouen", "Rouen"] (3/3 agree)
**Pairwise F1:** 1.0

**C4 (jsonld-ex Full):**
```
User: Based on the provided context, what is the capital of Normandy?
Assistant: {
  "@context": ["https://schema.org/", "https://jsonld-ex.org/context/v1"],
  "@type": "Answer",
  "text": "Rouen",
  "@opinion": {"belief": 0.95, "disbelief": 0.01, "uncertainty": 0.04, "base_rate": 0.5},
  "@source": ["urn:squad:56be4db0acb8001400a502ec"],
  "@method": "extractive_qa_annotation"
}
```

---

## A1.3 Meridian: Unchanged

The Meridian synthetic KB remains as specified in §4 of the original protocol.
It serves as the contamination-free control with continuous SL values.

---

## A1.4 Updated Project Structure

```
ET1/
├── src/
│   ├── knowledge_base.py          # Meridian synthetic generator (DONE)
│   ├── fever_loader.py            # FEVER dataset → SL-annotated facts
│   ├── squad_loader.py            # SQuAD 2.0 → SL-annotated facts
│   ├── fact.py                    # Common Fact dataclass used by all loaders
│   ├── data_formatter.py          # Fact → 7 conditions (C1-C7)
│   ├── train.py
│   ├── evaluate.py
│   ├── parse_response.py
│   ├── metrics.py
│   └── statistical_tests.py
├── tests/
│   ├── test_knowledge_base.py     # (DONE — 41 passing)
│   ├── test_fact.py               # Fact dataclass invariants
│   ├── test_fever_loader.py       # FEVER loading + SL derivation
│   ├── test_squad_loader.py       # SQuAD loading + SL derivation
│   ├── test_data_formatter.py     # 7-condition formatting
│   ├── ...
```

---

## A1.5 Updated Development Order (TDD)

1. ~~Knowledge base generator~~ (DONE — 41 tests passing)
2. **Fact dataclass** — common representation across all 3 data sources
3. **FEVER loader** — download, parse, derive SL opinions
4. **SQuAD 2.0 loader** — download, parse, derive SL opinions
5. **Data formatter** — render Facts into 7 conditions (C1-C7)
6. Response parser
7. Metrics module
8. Statistical tests
9. Training script
10. Evaluation script
11. End-to-end pipeline

---

## A1.6 Updated Pilot Run Count

3 datasets × 7 conditions × 3 models × 3 seeds = **189 fine-tuning runs**

At ~15 min per run on RTX 4090 (small models) = ~47 GPU-hours.
Still feasible in a single week. If too slow, we can start with 1 seed
for signal detection, then add seeds 2 and 3 for variance estimation.

---

## A1.7 Addressing the Contamination Objection

**"Base models were pre-trained on FEVER/SQuAD data. How do you know
the model isn't just retrieving memorized answers?"**

Response: This is why Meridian exists. If effects appear on Meridian (which
is provably absent from pre-training data), the mechanism is genuine.
FEVER/SQuAD results are then strengthened by the Meridian control.

Additionally, our fine-tuning changes the OUTPUT FORMAT, not the factual
content. The base model may "know" that the Colosseum is in Rome, but it
has never seen that fact expressed as a jsonld-ex document with SL opinions.
The question is not whether the model knows the fact, but whether the
format of training data affects how the model expresses confidence about
facts (including novel facts it doesn't know).

---

*End of Amendment A1*
