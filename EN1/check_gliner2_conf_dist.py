"""Quick confidence distribution analysis for GLiNER2 test predictions."""
import json
import numpy as np
from pathlib import Path

ckpt = Path(__file__).resolve().parent / "checkpoints" / "test_preds_gliner2.json"
data = json.load(open(ckpt, encoding="utf-8"))

ec = [p["confidence"] for s in data for p in s if p["tag"] != "O"]
a = np.array(ec)

print(f"Entity tokens: {len(ec)}")
print(f"min={a.min():.6f}  p5={np.percentile(a,5):.6f}  p25={np.percentile(a,25):.6f}  median={np.median(a):.6f}")
print(f"p75={np.percentile(a,75):.6f}  p95={np.percentile(a,95):.6f}  max={a.max():.6f}  std={a.std():.6f}")

bins = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 0.9999, 1.0001]
h, _ = np.histogram(a, bins=bins)
print("\nConfidence histogram:")
for i in range(len(h)):
    print(f"  [{bins[i]:.4f}, {bins[i+1]:.4f}): {h[i]:5d} ({100*h[i]/len(ec):5.1f}%)")
