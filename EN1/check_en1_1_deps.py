"""Quick dependency check for EN1.1."""
import sys

deps = {}

try:
    import spacy
    deps["spacy"] = spacy.__version__
except ImportError:
    deps["spacy"] = "MISSING"

try:
    import flair
    deps["flair"] = flair.__version__
except ImportError:
    deps["flair"] = "MISSING"

try:
    import stanza
    deps["stanza"] = stanza.__version__
except ImportError:
    deps["stanza"] = "MISSING"

try:
    import datasets
    deps["datasets"] = datasets.__version__
except ImportError:
    deps["datasets"] = "MISSING"

try:
    import seqeval
    try:
        from importlib.metadata import version as pkg_version
        deps["seqeval"] = pkg_version("seqeval")
    except Exception:
        deps["seqeval"] = "installed (version unknown)"
except ImportError:
    deps["seqeval"] = "MISSING"

try:
    import sklearn
    deps["sklearn"] = sklearn.__version__
except ImportError:
    deps["sklearn"] = "MISSING"

try:
    import transformers
    deps["transformers"] = transformers.__version__
except ImportError:
    deps["transformers"] = "MISSING"

try:
    import torch
    deps["torch"] = torch.__version__
    deps["cuda"] = str(torch.cuda.is_available())
    if torch.cuda.is_available():
        deps["gpu"] = torch.cuda.get_device_name(0)
except ImportError:
    deps["torch"] = "MISSING"

print("=== EN1.1 Dependency Check ===")
for k, v in deps.items():
    status = "OK" if v != "MISSING" else "MISSING"
    print(f"  {k:15s}: {v:30s} [{status}]")

# Check spaCy model
try:
    import spacy
    nlp = spacy.load("en_core_web_trf")
    print(f"\n  spaCy model: {nlp.meta['name']} v{nlp.meta['version']} [OK]")
except Exception as e:
    print(f"\n  spaCy model en_core_web_trf: {e}")

# Check stanza model
try:
    import stanza
    nlp_st = stanza.Pipeline("en", processors="tokenize,ner", use_gpu=True, verbose=False)
    print(f"  Stanza en NER pipeline: [OK]")
except Exception as e:
    print(f"  Stanza en NER pipeline: {e}")

missing = [k for k, v in deps.items() if v == "MISSING"]
if missing:
    print(f"\nMISSING: {', '.join(missing)}")
    sys.exit(1)
else:
    print("\nAll dependencies ready.")
