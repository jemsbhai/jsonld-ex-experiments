"""Quick dependency check for EN3.1 — RAG Pipeline with Confidence-Aware Retrieval.

Run: python experiments/EN3/check_en3_1_deps.py
"""
import sys

deps = {}

# Core ML
try:
    import torch
    deps["torch"] = torch.__version__
    deps["cuda_available"] = str(torch.cuda.is_available())
    if torch.cuda.is_available():
        deps["gpu_name"] = torch.cuda.get_device_name(0)
        deps["gpu_memory_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}"
except ImportError:
    deps["torch"] = "MISSING"

try:
    import transformers
    deps["transformers"] = transformers.__version__
except ImportError:
    deps["transformers"] = "MISSING"

# Sentence transformers for passage embedding
try:
    import sentence_transformers
    deps["sentence_transformers"] = sentence_transformers.__version__
except ImportError:
    deps["sentence_transformers"] = "MISSING"

# HuggingFace datasets for Natural Questions
try:
    import datasets
    deps["datasets"] = datasets.__version__
except ImportError:
    deps["datasets"] = "MISSING"

# FAISS for efficient retrieval
try:
    import faiss
    deps["faiss"] = faiss.__version__ if hasattr(faiss, "__version__") else "installed"
    deps["faiss_gpu"] = str(faiss.get_num_gpus() > 0) if hasattr(faiss, "get_num_gpus") else "cpu_only"
except ImportError:
    deps["faiss"] = "MISSING"

# bitsandbytes for 4-bit quantization
try:
    import bitsandbytes
    deps["bitsandbytes"] = bitsandbytes.__version__
except ImportError:
    deps["bitsandbytes"] = "MISSING"

# accelerate for device mapping
try:
    import accelerate
    deps["accelerate"] = accelerate.__version__
except ImportError:
    deps["accelerate"] = "MISSING"

# scipy / numpy / sklearn
try:
    import numpy
    deps["numpy"] = numpy.__version__
except ImportError:
    deps["numpy"] = "MISSING"

try:
    import scipy
    deps["scipy"] = scipy.__version__
except ImportError:
    deps["scipy"] = "MISSING"

try:
    import sklearn
    deps["sklearn"] = sklearn.__version__
except ImportError:
    deps["sklearn"] = "MISSING"

# jsonld-ex
try:
    import jsonld_ex
    deps["jsonld_ex"] = jsonld_ex.__version__
except ImportError:
    deps["jsonld_ex"] = "MISSING"

# Check specific jsonld-ex features needed
try:
    from jsonld_ex.confidence_algebra import (
        Opinion, cumulative_fuse, averaging_fuse, conflict_metric,
        pairwise_conflict,
    )
    deps["jsonld_ex_algebra"] = "OK"
except ImportError as e:
    deps["jsonld_ex_algebra"] = f"IMPORT ERROR: {e}"

try:
    from jsonld_ex.confidence_decay import decay_opinion, exponential_decay
    deps["jsonld_ex_decay"] = "OK"
except ImportError as e:
    deps["jsonld_ex_decay"] = f"IMPORT ERROR: {e}"


print("=" * 60)
print("EN3.1 — RAG Pipeline Dependency Check")
print("=" * 60)
for k, v in deps.items():
    status = "OK" if v not in ("MISSING",) and not v.startswith("IMPORT ERROR") else "MISSING"
    print(f"  {k:25s}: {v:30s} [{status}]")

missing = [k for k, v in deps.items()
           if v == "MISSING" or (isinstance(v, str) and v.startswith("IMPORT ERROR"))]

print()
if missing:
    print(f"MISSING: {', '.join(missing)}")
    print()
    print("Install commands:")
    if "sentence_transformers" in missing:
        print("  pip install sentence-transformers")
    if "faiss" in missing:
        print("  pip install faiss-cpu     # or: pip install faiss-gpu")
    if "bitsandbytes" in missing:
        print("  pip install bitsandbytes")
    if "accelerate" in missing:
        print("  pip install accelerate")
    if "datasets" in missing:
        print("  pip install datasets")
    sys.exit(1)
else:
    print("All dependencies ready for EN3.1.")
    print()
    print("NOTE: You will also need a local LLM. Recommended:")
    print("  - meta-llama/Meta-Llama-3.1-8B-Instruct (via HF, requires approval)")
    print("  - mistralai/Mistral-7B-Instruct-v0.3 (open weights)")
    print("  The experiment will use 4-bit quantization to fit on the 4090.")
