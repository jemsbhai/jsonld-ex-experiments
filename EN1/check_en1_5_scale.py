"""Check scale: edges + multi-hop paths across ALL datasets."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from EN1.en1_5_core import load_all_bn_kbs, load_synthea_kb, extract_paths

print("=" * 70)
print("  EN1.5 Scale Check: Single-Edge + Multi-Hop")
print("=" * 70)

total_edges = 0
total_paths = {2: 0, 3: 0, 4: 0}

# BN models
kbs = load_all_bn_kbs()
for kb in kbs:
    paths = extract_paths(kb, max_length=4)
    by_len = {}
    for p in paths:
        by_len.setdefault(len(p), []).append(p)
    path_summary = ", ".join(f"L{k}={len(v)}" for k, v in sorted(by_len.items()))
    print(f"  {kb.name:>12}: {len(kb.edges):>4} edges, {len(paths):>5} paths  ({path_summary})")
    total_edges += len(kb.edges)
    for k, v in by_len.items():
        total_paths[k] = total_paths.get(k, 0) + len(v)

# Synthea
kb_s = load_synthea_kb()
paths_s = extract_paths(kb_s, max_length=4)
by_len_s = {}
for p in paths_s:
    by_len_s.setdefault(len(p), []).append(p)
path_summary_s = ", ".join(f"L{k}={len(v)}" for k, v in sorted(by_len_s.items()))
print(f"  {'synthea':>12}: {len(kb_s.edges):>4} edges, {len(paths_s):>5} paths  ({path_summary_s})")
total_edges += len(kb_s.edges)
for k, v in by_len_s.items():
    total_paths[k] = total_paths.get(k, 0) + len(v)

print(f"\n  TOTALS:")
print(f"    Single-edge:  {total_edges} edges")
for k in sorted(total_paths):
    print(f"    Length-{k} paths: {total_paths[k]}")
print(f"    Total paths:  {sum(total_paths.values())}")

# Estimate trial counts at 1000 reps, 5 N values
n_single = total_edges * 1000 * 5
n_multi = sum(min(v, 200) for v in total_paths.values()) * 1000 * 5
print(f"\n  Estimated trials (1000 reps, 5 N values):")
print(f"    Single-edge: {n_single:>12,}")
print(f"    Multi-hop:   {n_multi:>12,}  (capped at 200 paths/length)")
print(f"    Total:       {n_single + n_multi:>12,}")
