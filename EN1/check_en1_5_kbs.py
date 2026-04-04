"""Quick check: KB sizes for EN1.5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from EN1.en1_5_core import load_asia_kb, load_alarm_kb, load_synthea_kb

for loader, name in [(load_asia_kb, "ASIA"), (load_alarm_kb, "ALARM"), (load_synthea_kb, "Synthea")]:
    kb = loader()
    print(f"\n{name}: {len(kb.nodes)} nodes, {len(kb.edges)} edges")
    for k, v in kb.metadata.items():
        if k != "state_names_sample":
            print(f"  {k}: {v}")
    # Show first 3 edges
    for e in kb.edges[:3]:
        print(f"  edge: {e.parent} -> {e.child}  P(par)={e.p_parent:.4f}  P(ch|par)={e.p_child_given_parent:.4f}  P(ch|~par)={e.p_child_given_not_parent:.4f}")
