# EN3.2 — Confidence-Aware RAG: Rigorous Evaluation Design

## Motivation

EN3.1/3.1b revealed that SL as a **pre-filter** does not improve end-to-end
RAG for strong LLMs.  GPT-4o-mini achieves EM=0.64 with poisoned contexts,
and aggressive filtering destroys more useful context than it removes poison.

This was the wrong test.  jsonld-ex's value in RAG is not pre-filtering —
it's providing **structured epistemic metadata** that enables capabilities
impossible with scalar confidence alone.

## Theoretical Foundation

EA1.1 proved: scalar confidence destroys 98.4% of uncertainty information
and 61.4% of conflict information.  In RAG, this manifests as:

- Cosine similarity = 0.75 could mean "one highly relevant passage" OR
  "three moderately relevant passages that agree" OR "two passages that
  contradict each other."  Scalar methods treat all three identically.

- SL opinions preserve the distinction via the (b, d, u) triple:
  - High b, low u = strong evidence FOR relevance
  - High d, low u = strong evidence AGAINST
  - High u = insufficient evidence (should abstain)
  - High conflict between passages = sources disagree (should flag)

## Three Orthogonal Hypotheses

### H1: Calibrated Selective Answering (Abstention)

**Claim:** SL uncertainty enables better "know when you don't know."
At any given coverage level, SL-informed selective answering achieves
higher precision than scalar-threshold selective answering.

**Why SL should win:** Scalar abstention can only threshold on the
LLM's output probability or the max cosine score.  SL abstention
can threshold on:
  - Fused opinion uncertainty (u) — "do we have enough evidence?"
  - Pairwise conflict — "do sources agree?"
  - Belief-disbelief gap — "is the evidence one-sided?"

These are *mathematically distinct* signals that scalar methods collapse.

**Protocol:**
1. Dataset: Natural Questions (NQ) open-domain subset, 2,000 questions
   (1,000 answerable, 1,000 unanswerable from NQ)
2. Retrieve top-20 passages per question from Wikipedia (DPR or contriever)
3. For each question, compute:
   a. Scalar signals: max cosine sim, mean cosine sim, score variance
   b. SL signals: fused opinion (b, d, u), max pairwise conflict,
      inter-passage agreement ratio, fused uncertainty
4. LLM generates answer + self-reported confidence for ALL questions
5. Evaluate selective answering at coverage levels {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}:
   a. Scalar: rank by max_cosine, answer top-k%
   b. Scalar+LLM: rank by LLM self-confidence, answer top-k%
   c. SL uncertainty: rank by (1 - fused_u), answer top-k%
   d. SL composite: rank by fused_belief * (1 - conflict), answer top-k%
6. Metrics: Precision@k, AUC of precision-coverage curve, ECE

**Key advantage:** This tests SL's UNIQUE capability (uncertainty tracking)
on its NATURAL task (selective answering) with a fair comparison (same LLM,
same passages, different ranking signals).

### H2: Multi-Source Conflict Detection

**Claim:** When retrieved passages contain genuine contradictions, SL
pairwise_conflict correctly identifies contested questions, enabling
the system to flag them for human review or provide hedged answers.

**Why SL should win:** Scalar methods can detect that passages have
different cosine scores, but NOT that they contain contradictory claims.
SL conflict detection operates on the opinion-level agreement/disagreement,
which is a fundamentally different signal.

**Protocol:**
1. Dataset: ConflictQA or a curated subset of Natural Questions where
   Wikipedia revision history shows the answer changed (contested facts)
2. For each question: retrieve passages from MULTIPLE sources/timestamps
   that may genuinely disagree
3. Compute:
   a. Scalar: score variance, score range, answer extraction disagreement
   b. SL: pairwise_conflict matrix, max conflict, conflict_metric on fused
4. Binary classification task: "Is this question contested?"
   Ground truth from dataset labels or answer-extraction disagreement
5. Metrics: AUROC, F1 for conflict detection
6. Downstream: Show that flagging contested questions improves system
   reliability (precision of non-flagged answers)

### H3: Metadata-Enriched Prompting

**Claim:** When the LLM receives SL metadata IN the prompt (not just
passages), it produces better answers because it can reason about
source reliability.

**Why SL should win:** Instead of:
  "[Passage 1] Paris is the capital of France."
  "[Passage 2] Lyon is the capital of France."

jsonld-ex provides:
  "[Passage 1] (belief=0.92, uncertainty=0.05, 3 sources agree)
   Paris is the capital of France."
  "[Passage 2] (belief=0.31, uncertainty=0.45, 1 source, CONFLICT with P1)
   Lyon is the capital of France."
  "Fused assessment: HIGH CONFLICT detected. Best-supported answer: Paris."

The LLM can USE this metadata to make better decisions. This is the
unique jsonld-ex contribution — no other format can compute and provide
this structured epistemic metadata.

**Protocol:**
1. Dataset: SQuAD + poisoned passages (reuse v1b corpus) + NQ subset
2. Same retrieval, same passages for all conditions
3. Three prompt conditions (within-subject design):
   a. PLAIN: passages only (standard RAG)
   b. SCALAR: passages + cosine similarity scores
   c. JSONLD-EX: passages + full SL metadata (b, d, u per passage,
      pairwise conflict flags, fused opinion, source agreement count,
      confidence assessment in natural language)
4. LLM generates answers under each condition (same model, temp=0)
5. Metrics: EM, F1, + breakdown by question difficulty:
   - Easy (gold passage in top-3, no poison): expect no difference
   - Medium (gold passage in top-10, some irrelevant): expect small diff
   - Hard (gold passage low-ranked, poison present): expect large diff
6. Statistical test: paired McNemar's test per question

**Key insight:** The hypothesis is that SL metadata helps MORE on
hard questions where the LLM needs to reason about source quality.
On easy questions, all methods should be equivalent. The INTERACTION
between question difficulty and method is the scientifically
interesting finding.

## Why This Design Is Rigorous

1. **Fair baselines:** Same LLM, same passages, same retrieval.
   Only the SIGNALS differ between methods.

2. **Multiple independent hypotheses:** H1, H2, H3 can succeed or
   fail independently. We report all results honestly.

3. **Within-subject design:** Same questions across all conditions
   enables paired statistical tests with much higher power.

4. **Established datasets:** NQ and SQuAD are standard benchmarks.
   ConflictQA (if available) provides genuine contradictions.

5. **Graduated difficulty:** Breakdown by question difficulty reveals
   WHERE SL helps (hard questions) vs doesn't (easy questions).
   This is more informative than a single aggregate number.

6. **Theoretically grounded:** Each hypothesis maps to a specific
   mathematical property of SL (uncertainty preservation, conflict
   detection, evidence accumulation) that EA1.1 proved scalar
   methods cannot provide.

## Implementation Plan

Phase 1: H3 (Metadata-Enriched Prompting) — FASTEST, uses existing checkpoints
  - Reuses v1b SQuAD corpus + retrieval + QA extraction
  - Three prompt conditions via GPT-4o-mini API (~$3-4)
  - Can also use poisoned corpus from v1b for controlled comparison
  - Est: 2-3 hours

Phase 2: H1 (Calibrated Abstention) — STRONGEST theoretical argument
  - Needs NQ dataset download + DPR retrieval
  - Compute SL signals for all questions
  - Evaluate precision-coverage curves
  - Est: 4-6 hours

Phase 3: H2 (Conflict Detection) — MOST NOVEL
  - Needs ConflictQA or curated contested-facts dataset
  - Binary classification evaluation
  - Est: 4-6 hours

## Risk Assessment

- **H3 risk: LLM ignores metadata.** Strong LLMs may just read passages
  and ignore our annotations. Mitigation: test on hard questions where
  metadata provides decisive signal. Honest reporting if null.

- **H1 risk: LLM self-confidence is already well-calibrated.** Modern
  LLMs may not need external confidence signals. Mitigation: compare
  SL signals vs LLM-internal signals as complementary, not competing.

- **H2 risk: Real contradictions are rare in standard datasets.**
  Mitigation: use ConflictQA or construct adversarial contradictions.

- **Overall risk:** If all three are null, that's an honest finding
  about the limits of structured confidence in RAG. Still publishable
  as a negative result with strong methodology.
