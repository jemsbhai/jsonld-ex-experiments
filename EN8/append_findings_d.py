"""Append EN8.2 Phase D findings to FINDINGS.md."""
from pathlib import Path

findings = """
### H8.2d — Data Quality Detection: Detailed Results

**Bias-disagreement correlation:**
- Primary cluster (4 sensors): r=0.937 — **ACCEPTED** (>0.7 threshold)
- Validation cluster (5 sensors): r=0.508 — moderate, below threshold

**Drift detection:** Working. Sensors 7 and 9 flagged with drift scores
21.6 and 20.6 respectively, matching the 10-16x drift/sigma ratios
measured in Phase F diagnostics.

**Outlier detection — honest assessment:**

Cross-sensor consensus detects independent sensor failures with high
recall (93.2% for sensor 6 failing alone). However, the Intel Lab
dataset exhibits CORRELATED sensor failures where multiple co-located
sensors fail simultaneously (power issues, environmental events).

| Failure Type | Recall | Root Cause |
|-------------|--------|------------|
| Independent (1 sensor fails) | 93.2% | Neighbors provide valid consensus |
| Correlated (cluster-wide) | ~0% | No healthy neighbor for consensus |
| Mixed (primary cluster) | 52% ceiling | 48% are correlated failures |
| Mixed (validation cluster) | 16% ceiling | More correlated failures |

No threshold strategy breaks past these ceilings — this is a
fundamental limitation of all consensus-based methods, not specific
to jsonld-ex or SL.

**Principled conclusion:** The annotation model's quality value is in:
1. Drift detection (r=0.937 correlation with measured bias)
2. Calibration age tracking (monotonic, physically meaningful)
3. Independent failure detection (93% when applicable)

For correlated failures, domain-specific range checks (e.g.,
temperature in [10, 45] C) are necessary. The annotation model
supports combining both signals naturally.
"""

path = Path(__file__).parent.parent / "FINDINGS.md"
with open(path, "a", encoding="utf-8") as f:
    f.write(findings)
print(f"Appended to {path}")
