"""Diagnostic: investigate DS-SL divergence under low conflict."""
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "packages" / "python" / "src"))

import numpy as np
from jsonld_ex.confidence_algebra import Opinion, cumulative_fuse

rng = np.random.default_rng(42)
worst_diff = 0.0
worst_case = None
all_cases = []

for i in range(100):
    b1, b2 = rng.uniform(0.4, 0.8, size=2)
    u1, u2 = rng.uniform(0.1, 0.4, size=2)
    d1 = 1 - b1 - u1
    d2 = 1 - b2 - u2
    if d1 < 0 or d2 < 0:
        continue
    K = b1 * d2 + d1 * b2
    if K > 0.3:
        continue

    # DS
    norm = 1 - K
    ds_b = (b1 * b2 + b1 * u2 + u1 * b2) / norm

    # SL
    kappa = u1 + u2 - u1 * u2
    sl_b = (b1 * u2 + b2 * u1) / kappa

    diff = abs(ds_b - sl_b)
    b1b2_term = b1 * b2 / norm

    case = dict(b1=b1, d1=d1, u1=u1, b2=b2, d2=d2, u2=u2,
                K=K, ds_b=ds_b, sl_b=sl_b, diff=diff,
                b1b2_term=b1b2_term, kappa=kappa, norm=norm)
    all_cases.append(case)

    if diff > worst_diff:
        worst_diff = diff
        worst_case = case

print("=== WORST CASE ===")
for k, v in worst_case.items():
    print(f"  {k}: {v:.6f}")

print()
b1, b2 = worst_case["b1"], worst_case["b2"]
print(f"b1*b2 = {b1*b2:.6f}")
print(f"b1*b2 / (1-K) = {worst_case['b1b2_term']:.6f}  <-- divergence source")
print(f"DS has this term in belief numerator, SL does not")
print()
print(f"DS belief:  {worst_case['ds_b']:.6f}")
print(f"SL belief:  {worst_case['sl_b']:.6f}")
print(f"Divergence: {worst_case['diff']:.6f}")

# Characterize: how does divergence scale with K?
print()
print("=== DIVERGENCE vs CONFLICT LEVEL ===")
bins = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30)]
for lo, hi in bins:
    subset = [c for c in all_cases if lo <= c["K"] < hi]
    if subset:
        diffs = [c["diff"] for c in subset]
        b1b2s = [c["b1b2_term"] for c in subset]
        print(f"  K in [{lo:.2f}, {hi:.2f}): n={len(subset):3d}, "
              f"mean_diff={np.mean(diffs):.4f}, max_diff={np.max(diffs):.4f}, "
              f"mean_b1b2_term={np.mean(b1b2s):.4f}")

# What drives divergence: b1*b2 or normalization?
print()
print("=== CORRELATION: divergence vs b1*b2 term ===")
diffs = np.array([c["diff"] for c in all_cases])
b1b2s = np.array([c["b1b2_term"] for c in all_cases])
corr = np.corrcoef(diffs, b1b2s)[0, 1]
print(f"  Pearson r(diff, b1b2/(1-K)) = {corr:.4f}")
