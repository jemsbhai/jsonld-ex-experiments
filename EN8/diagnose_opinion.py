"""Diagnose: why is SL not outperforming weighted_avg?

Hypothesis: the sigma-to-uncertainty mapping is too compressed,
making all evidence weights nearly equal regardless of actual
sensor reliability.
"""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "python" / "src"))

from en8_2_core import _sensor_to_opinion

# Actual estimated sigmas from Phase A
sigmas = {4: 0.2567, 6: 0.1792, 7: 0.1937, 9: 0.3342}
value_range = (15.0, 35.0)
range_width = 20.0

print("=== Current opinion construction ===")
print(f"Formula: u = clamp(sigma / range_width * 4, 0.02, 0.95)")
print(f"Weight = 1 - u")
print()

# Show the mapping for each sensor at a typical value (20°C)
for sid, sig in sorted(sigmas.items()):
    op = _sensor_to_opinion(20.0, sig, value_range)
    weight = 1.0 - op.uncertainty
    print(f"  Sensor {sid}: sigma={sig:.4f}  u={op.uncertainty:.4f}  "
          f"weight={weight:.4f}  b={op.belief:.4f}  d={op.disbelief:.4f}")

# Compare weight ratios
weights = {sid: 1.0 - _sensor_to_opinion(20.0, sig, value_range).uncertainty
           for sid, sig in sigmas.items()}
max_w = max(weights.values())
min_w = min(weights.values())
print(f"\n  Weight ratio (max/min): {max_w/min_w:.2f}x")

# What weighted_avg uses
print("\n=== Weighted average (inverse-variance) ===")
inv_var_weights = {sid: 1.0/sig**2 for sid, sig in sigmas.items()}
max_iv = max(inv_var_weights.values())
min_iv = min(inv_var_weights.values())
for sid, w in sorted(inv_var_weights.items()):
    print(f"  Sensor {sid}: sigma={sigmas[sid]:.4f}  1/sigma^2={w:.2f}  "
          f"normalized={w/sum(inv_var_weights.values()):.3f}")
print(f"\n  Weight ratio (max/min): {max_iv/min_iv:.2f}x")

# The problem is clear: SL weights have ~1.05x ratio,
# weighted_avg has ~3.5x ratio. SL can't discriminate.

print("\n=== What the uncertainty SHOULD be ===")
print("For SL to match optimal Gaussian fusion, the evidence")
print("weight (1-u) should be proportional to 1/sigma^2.")
print()
print("Option 1: u = sigma^2 / (sigma^2 + k), k = reference variance")
print("  This maps sigma to uncertainty via a sigmoid-like function")
print()

# Try several mappings
ref_variance = np.mean([s**2 for s in sigmas.values()])
print(f"Reference variance: {ref_variance:.4f}")

for label, u_fn in [
    ("Current: u = sigma/range*4",
     lambda sig: max(0.02, min(0.95, sig / range_width * 4))),
    ("Quadratic: u = sigma^2 / (sigma^2 + ref_var)",
     lambda sig: sig**2 / (sig**2 + ref_variance)),
    ("Evidence: u = W/(W+N), N=1/sigma^2, W=2",
     lambda sig: 2.0 / (2.0 + 1.0/sig**2)),
    ("Evidence: u = W/(W+N), N=1/sigma^2, W=10",
     lambda sig: 10.0 / (10.0 + 1.0/sig**2)),
    ("Log: u = clamp(log(sigma/min_sig) / log(max_sig/min_sig), 0.02, 0.95)",
     lambda sig: max(0.02, min(0.95,
         np.log(sig/min(sigmas.values())) / np.log(max(sigmas.values())/min(sigmas.values()))))),
]:
    print(f"\n  {label}")
    ws = {}
    for sid, sig in sorted(sigmas.items()):
        u = u_fn(sig)
        w = 1.0 - u
        ws[sid] = w
        print(f"    Sensor {sid}: sigma={sig:.4f}  u={u:.4f}  weight={w:.4f}")
    ratio = max(ws.values()) / min(ws.values())
    print(f"    Weight ratio: {ratio:.2f}x")
