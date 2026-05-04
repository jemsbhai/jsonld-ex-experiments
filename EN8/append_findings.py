"""Append EN8.2 findings to FINDINGS.md."""
from pathlib import Path

findings = """
---

## EN8.2 -- IoT Sensor Pipeline: Intel Lab Fusion Comparison

**Date:** 2026-05-04
**Status:** Phase A COMPLETE. Phase B (SSN/SOSA) and C (complexity) pending.

### Dataset

Intel Berkeley Research Lab (Bodik et al. 2004): 2,219,803 raw readings
from 54+ sensors over 37 days. After outlier filter (temperature in
[10, 45] C): 1,815,663 readings (81.8% retained). 18.2% were sensor
failures -- a real-world IoT data characteristic.

### Protocol

Leave-one-out (LOO) evaluation: hold out each sensor as ground truth,
fuse remaining sensors, measure RMSE. Per-sensor noise estimated from
data via MAD of deviations from consensus (robust to outliers).

Two clusters tested (replication):
- Primary [4, 6, 7, 9]: 7,684 time steps, 21,743 eval points
- Validation [22, 23, 24]: 9,876 time steps, 17,922 eval points

Six fusion methods: scalar average, inverse-variance weighted average,
1D Kalman filter, SL cumulative fusion, SL + decay, classical DS.

### H8.2c Results: Fusion Comparison

**Primary cluster [4, 6, 7, 9]:**

| Method | RMSE | Rank |
|--------|------|------|
| scalar_avg | **1.7334** | 1 |
| sl_fuse | 1.7349 | 2 |
| kalman | 1.8114 | 3 |
| weighted_avg | 1.8163 | 4 |
| ds_fuse | 2.1209 | 5 |

**Validation cluster [22, 23, 24]:**

| Method | RMSE | Rank |
|--------|------|------|
| weighted_avg | **1.5713** | 1 |
| sl_fuse | 1.6084 | 2 |
| scalar_avg | 1.6163 | 3 |
| kalman | 1.6305 | 4 |
| ds_fuse | 1.7414 | 5 |

**Bootstrap CIs (n=2000) -- SL vs baselines:**

Primary cluster:
- SL - scalar_avg: +0.0014 CI[+0.0009,+0.0020] SIG (SL slightly worse)
- SL - weighted_avg: -0.0812 CI[-0.103,-0.060] SIG (SL better)
- SL - kalman: -0.0762 CI[-0.096,-0.056] SIG (SL better)
- SL - ds_fuse: -0.3859 CI[-0.430,-0.341] SIG (SL much better)

Validation cluster:
- SL - scalar_avg: -0.0079 CI[-0.009,-0.007] SIG (SL better)
- SL - weighted_avg: +0.0371 CI[+0.023,+0.051] SIG (SL worse)
- SL - kalman: -0.0223 CI[-0.038,-0.005] SIG (SL better)
- SL - ds_fuse: -0.1326 CI[-0.155,-0.111] SIG (SL much better)

### Key Findings

**1. SL is consistently 2nd-best across both clusters.** It never ranks
worst, and always beats Kalman and DS. The winner alternates between
scalar_avg (primary) and weighted_avg (validation), but SL is the only
method in the top 2 on BOTH clusters.

**2. SL beats DS on both clusters.** Primary: -0.39 RMSE (18.2% reduction).
Validation: -0.13 RMSE (7.6% reduction). Both statistically significant.
Consistent with EN4.2 finding that DS's b1*b2 mutual reinforcement term
produces overconfident fusion.

**3. Misspecification robustness (shuffled sigma assignments):**

| Method | Primary degrad. | Validation degrad. |
|--------|----------------|-------------------|
| ds_fuse | +21.0% | +35.3% |
| scalar_avg | +30.9% | +40.0% |
| sl_fuse | +31.0% | +41.3% |
| weighted_avg | +33.4% | +57.4% |
| kalman | **+118.3%** | **+125.9%** |

**Kalman is catastrophically fragile** (+118-126%). Its process model
(random walk) does not match real temperature dynamics, and wrong
measurement noise makes it worse.

**weighted_avg degrades more than SL** on validation (+57% vs +41%).
When sigma estimates are wrong, inverse-variance weighting amplifies
the wrong sensor.

**SL and scalar_avg are equally robust** (~31-41%). SL's evidence-based
weighting degrades gracefully because opinions with wrong uncertainty
still contribute proportionally to evidence quality.

**4. sl_fuse_decay = sl_fuse** because LOO evaluation has no stale
readings (staleness=0 for all sensors at every time step). This is a
limitation -- decay testing requires artificial gap injection. Noted
for future work.

### Interpretation for Paper

The paper's claim is NOT "SL beats Kalman/GP on sensor fusion." That
would be overclaiming. The claim is:

"SL provides consistently competitive fusion (always top 2 across
clusters) while being significantly more robust than Kalman under
model misspecification, AND it integrates naturally with SSN/SOSA,
FHIR, PROV-O -- capabilities that no standalone fusion method provides."

The practical advantage: a developer using jsonld-ex gets fusion that
works well out of the box, degrades gracefully when noise estimates
are wrong, and produces standards-compliant SSN/SOSA output -- without
needing to implement and tune a Kalman filter.
"""

findings_path = Path(__file__).parent.parent / "FINDINGS.md"
with open(findings_path, "a", encoding="utf-8") as f:
    f.write(findings)
print(f"Appended to {findings_path}")
