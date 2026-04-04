"""Check available pgmpy BN models."""
from pgmpy.utils import get_example_model

candidates = [
    "asia", "alarm", "sachs", "child", "insurance", "water",
    "mildew", "barley", "hailfinder", "hepar2", "win95pts",
    "andes", "cancer", "earthquake", "survey",
]
for name in candidates:
    try:
        m = get_example_model(name)
        print(f"{name:>12}: {len(m.nodes()):>3} nodes, {len(m.edges()):>3} edges")
    except Exception as e:
        print(f"{name:>12}: FAILED ({type(e).__name__}: {e})")
