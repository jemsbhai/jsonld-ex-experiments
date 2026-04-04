#!/usr/bin/env python
"""EN8.4 -- Quantized Vector Retrieval in Knowledge Graphs.

NeurIPS 2026 D&B, Suite EN8 (Ecosystem Integration), Experiment 4.

Research Questions:
    RQ1: How does bit-width (2/3/4/8) affect retrieval quality?
    RQ2: Does TurboQuant (rotation + Lloyd-Max optimal scalar) preserve
         retrieval better than naive uniform scalar at same bit-width?
    RQ3: Does TurboQuantIP (QJL residual) improve inner-product-based
         search over MSE-only?
    RQ4: Does SL uncertainty-aware ranking improve retrieval under
         UNIFORM quantization?  (Expected: NO -- uniform uncertainty
         does not change ranking.)
    RQ5: Does SL uncertainty-aware ranking improve retrieval under
         MIXED-PRECISION quantization?  (Expected: possibly yes.)
    RQ6: Does hybrid symbolic+vector search in a KG improve precision
         over pure vector search?
    RQ7: What is the storage-vs-quality Pareto frontier?

Part A -- Synthetic Product Catalog (controlled):
    - 1000 products across 10 categories, 128-dim embeddings
    - Category cluster structure: category center + Gaussian noise
    - Quantize with 3 methods x 4 bit-widths
    - Ground truth = float32 cosine similarity ranking
    - 50 queries (5 per category)

Quantization methods:
    1. Naive scalar -- uniform grid quantization (our implementation)
    2. TurboQuantMSE -- rotation + Lloyd-Max optimal scalar (Stage 1)
    3. TurboQuantIP -- Stage 1 + QJL 1-bit residual (full TurboQuant)

Metrics:
    - Distortion: MSE between original and dequantized vectors
    - Inner product fidelity: Pearson r of cos_sim pairs
    - Ranking correlation: Spearman rho of full ranking
    - Recall@{1, 5, 10}: fraction of float32 top-k in quantized top-k
    - Storage: bytes per vector at each bit-width

Honest risk notes (pre-registered):
    - SL under uniform quantization will NOT change ranking -- all
      nodes get identical uncertainty.  Reported as correct theoretical
      prediction, not a failure.
    - TurboQuantIP optimizes inner products, not reconstructed
      vectors.  It may or may not help cosine-similarity search on
      dequantized vectors.  We report what we find.
    - If quantization barely degrades retrieval (which TurboQuant
      claims at 4-bit), then SL correction has little room to help.
      That is an honest finding about the regime where SL adds value.

Usage:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python experiments/EN8/en8_4_vector_retrieval.py

Output:
    experiments/EN8/results/en8_4a_results.json            (latest)
    experiments/EN8/results/en8_4a_results_YYYYMMDD_HHMMSS.json (archive)

References:
    Zandieh et al. (2025). TurboQuant: Online Vector Quantization
        with Near-optimal Distortion Rate. ICLR 2026. arXiv:2504.19874.
    Zandieh et al. (2025). PolarQuant. AISTATS 2026. arXiv:2502.02617.
    Zandieh et al. (2024). QJL: 1-Bit Quantized JL Transform.
        AAAI 2025. arXiv:2406.03482.
    Josang, A. (2016). Subjective Logic. Springer.
    Shannon, C.E. (1959). Coding theorems for a discrete source
        with a fidelity criterion.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

# -- Path setup -------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_PKG_SRC = _REPO_ROOT / "packages" / "python" / "src"
_EXPERIMENTS_ROOT = _REPO_ROOT / "experiments"

for p in [str(_REPO_ROOT), str(_PKG_SRC), str(_EXPERIMENTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# -- Imports: jsonld-ex ------------------------------------------------
from jsonld_ex.vector_search import (
    vector_search,
    hybrid_search,
    uncertainty_aware_search,
    SearchResult,
)
from jsonld_ex.vector import quantization_descriptor
from jsonld_ex.quantization_bridge import (
    quantization_distortion,
    DISTORTION_CONSTANTS,
)

# -- Imports: experiment infrastructure --------------------------------
from experiments.infra.results import ExperimentResult
from experiments.infra.env_log import log_environment
from experiments.infra.config import set_global_seed

# -- Imports: TurboQuant (optional) ------------------------------------
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

_HAS_TURBOQUANT = False
_HAS_TURBOQUANT_IP = False
_TURBOQUANT_SOURCE = "none"
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if _HAS_TORCH:
    try:
        from turboquant import TurboQuantMSE  # type: ignore[import]
        _HAS_TURBOQUANT = True
        _TURBOQUANT_SOURCE = "turboquant (PyPI)"
        try:
            from turboquant import TurboQuantIP  # type: ignore[import]
            _HAS_TURBOQUANT_IP = True
        except ImportError:
            _HAS_TURBOQUANT_IP = False
    except ImportError:
        _HAS_TURBOQUANT_IP = False


RESULTS_DIR = _SCRIPT_DIR / "results"
SEED = 42
N_PRODUCTS = 1000
N_CATEGORIES = 10
DIMS = 128
N_QUERIES_PER_CATEGORY = 5
N_QUERIES = N_CATEGORIES * N_QUERIES_PER_CATEGORY
BIT_WIDTHS = [2, 3, 4, 8]
K_VALUES = [1, 5, 10]
CATEGORY_NAMES = [
    "electronics", "clothing", "books", "home_garden",
    "sports", "toys", "automotive", "health", "food", "music",
]


# =====================================================================
# Section 1: Data Generation
# =====================================================================


def generate_category_centers(rng: np.random.Generator) -> np.ndarray:
    """Generate N_CATEGORIES random unit-vector cluster centers.

    These are well-separated in 128-dim space (random unit vectors
    in high dimensions are nearly orthogonal).

    Returns:
        (N_CATEGORIES, DIMS) array of unit vectors.
    """
    centers = rng.standard_normal((N_CATEGORIES, DIMS))
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    return centers / norms


def generate_product_embeddings(
    rng: np.random.Generator,
    centers: np.ndarray,
    noise_scale: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate product embeddings as noisy perturbations of category centers.

    Each product's embedding = normalize(center + noise * noise_scale).
    This creates meaningful cluster structure so within-category
    products are more similar than across-category products.

    Args:
        rng: NumPy random generator.
        centers: (N_CATEGORIES, DIMS) cluster centers.
        noise_scale: Standard deviation of Gaussian noise.

    Returns:
        embeddings: (N_PRODUCTS, DIMS) unit vectors.
        categories: (N_PRODUCTS,) integer category labels.
    """
    products_per_cat = N_PRODUCTS // N_CATEGORIES
    all_embeddings = []
    all_categories = []

    for cat_idx in range(N_CATEGORIES):
        noise = rng.standard_normal((products_per_cat, DIMS)) * noise_scale
        vecs = centers[cat_idx] + noise
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # avoid division by zero
        vecs = vecs / norms
        all_embeddings.append(vecs)
        all_categories.extend([cat_idx] * products_per_cat)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    categories = np.array(all_categories, dtype=np.int32)
    return embeddings, categories


def select_query_indices(
    rng: np.random.Generator, categories: np.ndarray
) -> np.ndarray:
    """Select N_QUERIES_PER_CATEGORY products per category as queries.

    Returns array of product indices.
    """
    query_indices = []
    for cat_idx in range(N_CATEGORIES):
        cat_mask = np.where(categories == cat_idx)[0]
        chosen = rng.choice(cat_mask, size=N_QUERIES_PER_CATEGORY, replace=False)
        query_indices.extend(chosen.tolist())
    return np.array(query_indices, dtype=np.int32)


# =====================================================================
# Section 2: Naive Scalar Quantization (our implementation)
# =====================================================================


def naive_scalar_quantize(
    vectors: np.ndarray, bit_width: int
) -> Tuple[np.ndarray, float, float]:
    """Uniform scalar quantization with symmetric range.

    Quantizes each coordinate independently to one of 2^bit_width
    uniformly spaced levels spanning [v_min, v_max] of the data.

    Args:
        vectors: (N, D) float32 array.
        bit_width: Bits per coordinate.

    Returns:
        indices: (N, D) integer indices in [0, 2^b - 1].
        v_min: Minimum value of the range.
        v_max: Maximum value of the range.
    """
    n_levels = 2 ** bit_width
    v_min = float(vectors.min())
    v_max = float(vectors.max())

    # Avoid degenerate range
    if v_max - v_min < 1e-10:
        v_max = v_min + 1e-10

    # Normalize to [0, 1], then quantize to [0, n_levels-1]
    normalized = (vectors - v_min) / (v_max - v_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    indices = np.round(normalized * (n_levels - 1)).astype(np.int32)
    return indices, v_min, v_max


def naive_scalar_dequantize(
    indices: np.ndarray, bit_width: int, v_min: float, v_max: float
) -> np.ndarray:
    """Dequantize by mapping indices back to bin centers.

    Args:
        indices: (N, D) integer indices.
        bit_width: Bits per coordinate.
        v_min, v_max: Original data range.

    Returns:
        (N, D) float32 dequantized vectors.
    """
    n_levels = 2 ** bit_width
    return (indices.astype(np.float32) / (n_levels - 1)) * (v_max - v_min) + v_min


def naive_scalar_roundtrip(
    vectors: np.ndarray, bit_width: int
) -> np.ndarray:
    """Quantize and dequantize vectors using naive scalar method.

    Returns dequantized vectors.
    """
    indices, v_min, v_max = naive_scalar_quantize(vectors, bit_width)
    return naive_scalar_dequantize(indices, bit_width, v_min, v_max)


# =====================================================================
# Section 3: TurboQuant Wrappers
# =====================================================================


def turboquant_mse_roundtrip(
    vectors: np.ndarray, bit_width: int
) -> Optional[np.ndarray]:
    """Quantize and dequantize using TurboQuantMSE (Stage 1).

    Returns dequantized vectors, or None if TurboQuant unavailable.
    """
    if not _HAS_TURBOQUANT:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tq = TurboQuantMSE(dim=vectors.shape[1], bits=bit_width, device=device)

    t = torch.from_numpy(vectors).float().to(device)
    indices, norms = tq.quantize(t)
    reconstructed = tq.dequantize(indices, norms)

    return reconstructed.cpu().numpy()


def turboquant_ip_roundtrip(
    vectors: np.ndarray, bit_width: int
) -> Optional[np.ndarray]:
    """Quantize and dequantize using TurboQuantIP (Stage 1 + QJL).

    TurboQuantIP provides unbiased inner product estimation but
    per-vector reconstruction error can be significant.  We test
    both reconstruction quality and inner-product quality.

    Returns dequantized vectors, or None if unavailable.
    """
    if not _HAS_TURBOQUANT or not _HAS_TURBOQUANT_IP:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tq = TurboQuantIP(dim=vectors.shape[1], bits=bit_width, device=device)

    t = torch.from_numpy(vectors).float().to(device)
    mse_indices, norms, qjl_signs, residual_norms = tq.quantize(t)
    reconstructed = tq.dequantize(mse_indices, norms, qjl_signs, residual_norms)

    return reconstructed.cpu().numpy()


# =====================================================================
# Section 4: Metrics
# =====================================================================


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean squared error between original and reconstructed vectors."""
    return float(np.mean((original - reconstructed) ** 2))


def compute_cosine_sims(
    query: np.ndarray, vectors: np.ndarray
) -> np.ndarray:
    """Compute cosine similarities between a query and all vectors.

    Args:
        query: (D,) query vector.
        vectors: (N, D) document vectors.

    Returns:
        (N,) cosine similarities.
    """
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-10:
        return np.zeros(vectors.shape[0])
    q = query / query_norm

    vec_norms = np.linalg.norm(vectors, axis=1)
    vec_norms = np.maximum(vec_norms, 1e-10)
    v_normed = vectors / vec_norms[:, np.newaxis]

    return v_normed @ q


def cosine_sim_correlation(
    query: np.ndarray,
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """Pearson correlation between cosine sims with original vs reconstructed.

    Measures how well quantized vectors preserve the similarity structure.
    """
    sims_orig = compute_cosine_sims(query, original)
    sims_recon = compute_cosine_sims(query, reconstructed)

    r, _ = scipy_stats.pearsonr(sims_orig, sims_recon)
    return float(r)


def spearman_rank_correlation(
    query: np.ndarray,
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """Spearman rho between rankings from original and reconstructed."""
    sims_orig = compute_cosine_sims(query, original)
    sims_recon = compute_cosine_sims(query, reconstructed)

    rho, _ = scipy_stats.spearmanr(sims_orig, sims_recon)
    return float(rho)


def recall_at_k(
    query: np.ndarray,
    original: np.ndarray,
    reconstructed: np.ndarray,
    k: int,
) -> float:
    """Fraction of float32 top-k that appear in quantized top-k."""
    sims_orig = compute_cosine_sims(query, original)
    sims_recon = compute_cosine_sims(query, reconstructed)

    top_k_orig = set(np.argsort(sims_orig)[-k:][::-1].tolist())
    top_k_recon = set(np.argsort(sims_recon)[-k:][::-1].tolist())

    overlap = len(top_k_orig & top_k_recon)
    return overlap / k


def storage_bytes_per_vector(dims: int, bit_width: int) -> float:
    """Bytes needed to store one vector at the given bit-width.

    For naive scalar: dims * bit_width / 8 (plus negligible range metadata).
    For TurboQuant: dims * bit_width / 8 + 4 (norm float32).
    """
    return dims * bit_width / 8.0


# =====================================================================
# Section 5: JSON-LD Document Construction
# =====================================================================


def build_jsonld_graph(
    embeddings: np.ndarray,
    categories: np.ndarray,
    quant_method: Optional[str] = None,
    quant_bit_width: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a JSON-LD @graph document from product embeddings.

    Args:
        embeddings: (N, D) vectors (already quantized+dequantized if applicable).
        categories: (N,) category indices.
        quant_method: Quantization method name (None for float32).
        quant_bit_width: Bit-width (None for float32).

    Returns:
        JSON-LD document with @graph containing product nodes.
    """
    nodes = []
    for i in range(len(embeddings)):
        cat_name = CATEGORY_NAMES[categories[i]]
        node: Dict[str, Any] = {
            "@id": f"product:{i:04d}",
            "@type": "Product",
            "category": cat_name,
            "name": f"{cat_name}_item_{i % (N_PRODUCTS // N_CATEGORIES):03d}",
        }

        vec_value: Dict[str, Any] = {
            "@value": embeddings[i].tolist(),
            "@container": "@vector",
        }

        if quant_method is not None and quant_bit_width is not None:
            vec_value["@quantization"] = {
                "method": quant_method,
                "bitWidth": quant_bit_width,
            }

        node["embedding"] = vec_value
        nodes.append(node)

    return {"@graph": nodes}


# =====================================================================
# Section 6: Experiment Runners
# =====================================================================


def run_distortion_and_ranking_eval(
    embeddings_orig: np.ndarray,
    query_indices: np.ndarray,
    method_name: str,
    bit_width: int,
    roundtrip_fn,
) -> Optional[Dict[str, Any]]:
    """Run distortion and ranking evaluation for one method x bit-width.

    Returns a dict of metrics, or None if the method is unavailable.
    """
    print(f"  {method_name} @ {bit_width}-bit ... ", end="", flush=True)
    t0 = time.perf_counter()

    reconstructed = roundtrip_fn(embeddings_orig, bit_width)
    if reconstructed is None:
        print("SKIPPED (not available)")
        return None

    # --- Distortion ---
    mse = compute_mse(embeddings_orig, reconstructed)

    # --- Per-query metrics ---
    pearson_rs = []
    spearman_rhos = []
    recalls = {k: [] for k in K_VALUES}

    for q_idx in query_indices:
        query = embeddings_orig[q_idx]

        pearson_rs.append(
            cosine_sim_correlation(query, embeddings_orig, reconstructed)
        )
        spearman_rhos.append(
            spearman_rank_correlation(query, embeddings_orig, reconstructed)
        )
        for k in K_VALUES:
            recalls[k].append(
                recall_at_k(query, embeddings_orig, reconstructed, k)
            )

    elapsed = time.perf_counter() - t0

    result = {
        "method": method_name,
        "bit_width": bit_width,
        "mse": mse,
        "pearson_r_mean": float(np.mean(pearson_rs)),
        "pearson_r_std": float(np.std(pearson_rs)),
        "spearman_rho_mean": float(np.mean(spearman_rhos)),
        "spearman_rho_std": float(np.std(spearman_rhos)),
        "storage_bytes_per_vector": storage_bytes_per_vector(DIMS, bit_width),
        "compression_ratio": (DIMS * 4) / storage_bytes_per_vector(DIMS, bit_width),
        "elapsed_seconds": elapsed,
    }

    for k in K_VALUES:
        result[f"recall_at_{k}_mean"] = float(np.mean(recalls[k]))
        result[f"recall_at_{k}_std"] = float(np.std(recalls[k]))

    print(
        f"MSE={mse:.6f}  "
        f"Spearman={result['spearman_rho_mean']:.4f}  "
        f"R@10={result['recall_at_10_mean']:.3f}  "
        f"({elapsed:.1f}s)"
    )
    return result


def run_rq1_rq2_rq3(
    embeddings: np.ndarray, query_indices: np.ndarray
) -> List[Dict[str, Any]]:
    """RQ1-RQ3: Distortion and ranking across methods and bit-widths."""
    print("\n" + "=" * 70)
    print("RQ1-RQ3: Quantization Impact on Retrieval Quality")
    print("=" * 70)

    all_results = []

    for bw in BIT_WIDTHS:
        print(f"\n--- Bit-width: {bw} ---")

        # Method 1: Naive scalar
        r = run_distortion_and_ranking_eval(
            embeddings, query_indices, "naive_scalar", bw,
            naive_scalar_roundtrip,
        )
        if r is not None:
            all_results.append(r)

        # Method 2: TurboQuantMSE
        r = run_distortion_and_ranking_eval(
            embeddings, query_indices, "turboquant_mse", bw,
            turboquant_mse_roundtrip,
        )
        if r is not None:
            all_results.append(r)

        # Method 3: TurboQuantIP
        r = run_distortion_and_ranking_eval(
            embeddings, query_indices, "turboquant_ip", bw,
            turboquant_ip_roundtrip,
        )
        if r is not None:
            all_results.append(r)

    return all_results


def run_rq4_uniform_sl(
    embeddings: np.ndarray,
    categories: np.ndarray,
    query_indices: np.ndarray,
) -> Dict[str, Any]:
    """RQ4: SL uncertainty-aware ranking under uniform quantization.

    Pre-registered expectation: NO improvement, because all nodes have
    identical quantization and therefore identical uncertainty.
    SL ranking degenerates to raw cosine ranking.
    """
    print("\n" + "=" * 70)
    print("RQ4: SL Under Uniform Quantization (Expected: No Change)")
    print("=" * 70)

    bit_width = 4
    method = "naive_scalar"
    reconstructed = naive_scalar_roundtrip(embeddings, bit_width)
    doc = build_jsonld_graph(reconstructed, categories, method, bit_width)

    # Compare vector_search vs uncertainty_aware_search rankings
    ranking_matches = 0
    total_queries = 0
    recall_diffs = []  # recall@10 difference (SL - raw)

    for q_idx in query_indices:
        query_vec = embeddings[q_idx].tolist()

        # Raw cosine ranking on quantized vectors
        raw_results = vector_search(doc, query_vec, "embedding", k=10)
        raw_ids = [r.node_id for r in raw_results]

        # SL uncertainty-aware ranking on same vectors
        sl_results = uncertainty_aware_search(
            doc, query_vec, "embedding", k=10
        )
        sl_ids = [r.node_id for r in sl_results]

        if raw_ids == sl_ids:
            ranking_matches += 1
        total_queries += 1

        # Recall vs float32 ground truth
        gt_sims = compute_cosine_sims(embeddings[q_idx], embeddings)
        gt_top10 = set(f"product:{i:04d}" for i in np.argsort(gt_sims)[-10:][::-1])

        raw_recall = len(gt_top10 & set(raw_ids)) / 10
        sl_recall = len(gt_top10 & set(sl_ids)) / 10
        recall_diffs.append(sl_recall - raw_recall)

    ranking_agreement = ranking_matches / total_queries

    result = {
        "bit_width": bit_width,
        "method": method,
        "ranking_agreement_pct": ranking_agreement * 100,
        "recall_at_10_diff_mean": float(np.mean(recall_diffs)),
        "recall_at_10_diff_std": float(np.std(recall_diffs)),
        "n_queries": total_queries,
        "expected_agreement": "~100% (identical uncertainty => identical ranking)",
        "confirmed": ranking_agreement > 0.95,
    }

    print(f"  Ranking agreement: {ranking_agreement * 100:.1f}%")
    print(f"  Recall@10 diff (SL - raw): {result['recall_at_10_diff_mean']:.4f}")
    print(f"  Prediction confirmed: {result['confirmed']}")

    return result


def run_rq5_mixed_precision_sl(
    embeddings: np.ndarray,
    categories: np.ndarray,
    query_indices: np.ndarray,
) -> Dict[str, Any]:
    """RQ5: SL uncertainty-aware ranking under mixed-precision quantization.

    50% of nodes are float32 (bit_width=32), 50% are 4-bit quantized.
    SL should prefer float32 results when cosine sims are close.
    """
    print("\n" + "=" * 70)
    print("RQ5: SL Under Mixed-Precision Quantization")
    print("=" * 70)

    rng = np.random.default_rng(SEED + 100)
    n = len(embeddings)

    # Randomly assign 50% to float32, 50% to 4-bit
    float32_mask = np.zeros(n, dtype=bool)
    float32_indices = rng.choice(n, size=n // 2, replace=False)
    float32_mask[float32_indices] = True

    # Create mixed-precision embeddings
    quant_4bit = naive_scalar_roundtrip(embeddings, bit_width=4)
    mixed_embeddings = np.where(
        float32_mask[:, np.newaxis], embeddings, quant_4bit
    )

    # Build JSON-LD doc with per-node quantization metadata
    nodes = []
    for i in range(n):
        cat_name = CATEGORY_NAMES[categories[i]]
        node: Dict[str, Any] = {
            "@id": f"product:{i:04d}",
            "@type": "Product",
            "category": cat_name,
            "name": f"{cat_name}_item_{i % (N_PRODUCTS // N_CATEGORIES):03d}",
        }

        vec_value: Dict[str, Any] = {
            "@value": mixed_embeddings[i].tolist(),
            "@container": "@vector",
        }

        if float32_mask[i]:
            # No quantization metadata => float32
            vec_value["@quantization"] = {"method": "scalar", "bitWidth": 32}
        else:
            vec_value["@quantization"] = {"method": "scalar", "bitWidth": 4}

        node["embedding"] = vec_value
        nodes.append(node)

    doc = {"@graph": nodes}

    # Evaluate
    recall_raw = []
    recall_sl = []
    sl_prefers_float32_correctly = 0
    sl_prefers_float32_total = 0

    for q_idx in query_indices:
        query_vec = embeddings[q_idx].tolist()

        # Ground truth from original float32
        gt_sims = compute_cosine_sims(embeddings[q_idx], embeddings)
        gt_top10 = set(f"product:{i:04d}" for i in np.argsort(gt_sims)[-10:][::-1])

        # Raw cosine search on mixed doc
        raw_results = vector_search(doc, query_vec, "embedding", k=10)
        raw_ids = set(r.node_id for r in raw_results)
        recall_raw.append(len(gt_top10 & raw_ids) / 10)

        # SL-aware search on mixed doc
        sl_results = uncertainty_aware_search(
            doc, query_vec, "embedding", k=10
        )
        sl_ids = set(r.node_id for r in sl_results)
        recall_sl.append(len(gt_top10 & sl_ids) / 10)

        # Check: among top-10 SL results, does SL prefer float32 nodes
        # over equally-scored quantized nodes?
        for r in sl_results[:5]:
            idx = int(r.node_id.split(":")[1])
            if float32_mask[idx]:
                sl_prefers_float32_correctly += 1
            sl_prefers_float32_total += 1

    recall_raw_mean = float(np.mean(recall_raw))
    recall_sl_mean = float(np.mean(recall_sl))
    improvement = recall_sl_mean - recall_raw_mean

    result = {
        "n_float32": int(float32_mask.sum()),
        "n_quantized_4bit": int((~float32_mask).sum()),
        "recall_at_10_raw_mean": recall_raw_mean,
        "recall_at_10_raw_std": float(np.std(recall_raw)),
        "recall_at_10_sl_mean": recall_sl_mean,
        "recall_at_10_sl_std": float(np.std(recall_sl)),
        "recall_improvement": improvement,
        "sl_float32_preference_pct": (
            sl_prefers_float32_correctly / sl_prefers_float32_total * 100
            if sl_prefers_float32_total > 0
            else 0.0
        ),
        "n_queries": len(query_indices),
    }

    print(f"  Recall@10 raw:  {recall_raw_mean:.4f} +/- {result['recall_at_10_raw_std']:.4f}")
    print(f"  Recall@10 SL:   {recall_sl_mean:.4f} +/- {result['recall_at_10_sl_std']:.4f}")
    print(f"  Improvement:    {improvement:+.4f}")
    print(f"  SL float32 preference: {result['sl_float32_preference_pct']:.1f}%")

    return result


def run_rq6_hybrid_search(
    embeddings: np.ndarray,
    categories: np.ndarray,
    query_indices: np.ndarray,
) -> Dict[str, Any]:
    """RQ6: Hybrid symbolic+vector search advantage.

    Filter by category before vector search.  Measures:
    - Nodes scanned (should be ~100 vs 1000)
    - Recall vs unfiltered (should be same or better for in-category)
    """
    print("\n" + "=" * 70)
    print("RQ6: Hybrid Symbolic+Vector Search")
    print("=" * 70)

    doc = build_jsonld_graph(embeddings, categories)

    # For each query, compare:
    # (a) Pure vector search top-10
    # (b) Hybrid search: filter to same category, then top-10
    intra_category_recall_pure = []
    intra_category_recall_hybrid = []
    nodes_scanned_pure = N_PRODUCTS  # always scans all
    nodes_scanned_hybrid = N_PRODUCTS // N_CATEGORIES  # ~100

    for q_idx in query_indices:
        query_vec = embeddings[q_idx].tolist()
        query_cat = CATEGORY_NAMES[categories[q_idx]]

        # Ground truth: top-10 within same category (float32)
        cat_mask = categories == categories[q_idx]
        cat_indices = np.where(cat_mask)[0]
        cat_sims = compute_cosine_sims(embeddings[q_idx], embeddings[cat_mask])
        top10_in_cat = set(
            f"product:{cat_indices[i]:04d}"
            for i in np.argsort(cat_sims)[-10:][::-1]
        )

        # Pure vector search (no filter)
        pure_results = vector_search(doc, query_vec, "embedding", k=10)
        pure_ids = set(r.node_id for r in pure_results)
        intra_category_recall_pure.append(
            len(top10_in_cat & pure_ids) / min(10, len(top10_in_cat))
        )

        # Hybrid search (filter to category)
        hybrid_results = hybrid_search(
            doc, query_vec, "embedding",
            filters={"category": query_cat},
            k=10,
        )
        hybrid_ids = set(r.node_id for r in hybrid_results)
        intra_category_recall_hybrid.append(
            len(top10_in_cat & hybrid_ids) / min(10, len(top10_in_cat))
        )

    result = {
        "nodes_scanned_pure": nodes_scanned_pure,
        "nodes_scanned_hybrid": nodes_scanned_hybrid,
        "scan_reduction_pct": (1 - nodes_scanned_hybrid / nodes_scanned_pure) * 100,
        "intra_cat_recall_pure_mean": float(np.mean(intra_category_recall_pure)),
        "intra_cat_recall_pure_std": float(np.std(intra_category_recall_pure)),
        "intra_cat_recall_hybrid_mean": float(np.mean(intra_category_recall_hybrid)),
        "intra_cat_recall_hybrid_std": float(np.std(intra_category_recall_hybrid)),
        "n_queries": len(query_indices),
    }

    print(f"  Nodes scanned:  pure={nodes_scanned_pure}, hybrid={nodes_scanned_hybrid}")
    print(f"  Scan reduction: {result['scan_reduction_pct']:.0f}%")
    print(f"  Intra-category recall (pure):   {result['intra_cat_recall_pure_mean']:.4f}")
    print(f"  Intra-category recall (hybrid): {result['intra_cat_recall_hybrid_mean']:.4f}")

    return result


def run_rq7_pareto(rq1_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """RQ7: Storage-vs-quality Pareto frontier.

    Extracts the Pareto-optimal configurations from RQ1-RQ3 results.
    A configuration is Pareto-optimal if no other configuration has
    BOTH higher recall@10 AND lower storage.
    """
    print("\n" + "=" * 70)
    print("RQ7: Storage-vs-Quality Pareto Frontier")
    print("=" * 70)

    # Also include float32 baseline
    float32_point = {
        "method": "float32",
        "bit_width": 32,
        "storage_bytes_per_vector": DIMS * 4,
        "compression_ratio": 1.0,
        "recall_at_10_mean": 1.0,
        "spearman_rho_mean": 1.0,
    }

    points = [float32_point] + rq1_results
    pareto = []

    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            # q dominates p if q has better recall AND less storage
            if (
                q.get("recall_at_10_mean", 0) >= p.get("recall_at_10_mean", 0)
                and q.get("storage_bytes_per_vector", 0) <= p.get("storage_bytes_per_vector", 0)
                and (
                    q.get("recall_at_10_mean", 0) > p.get("recall_at_10_mean", 0)
                    or q.get("storage_bytes_per_vector", 0) < p.get("storage_bytes_per_vector", 0)
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append({
                "method": p["method"],
                "bit_width": p["bit_width"],
                "storage_bytes": p["storage_bytes_per_vector"],
                "compression_ratio": p.get("compression_ratio", 1.0),
                "recall_at_10": p.get("recall_at_10_mean", 1.0),
                "spearman_rho": p.get("spearman_rho_mean", 1.0),
            })

    pareto.sort(key=lambda x: x["storage_bytes"])

    print(f"  Pareto-optimal configurations ({len(pareto)}):")
    for p in pareto:
        print(
            f"    {p['method']:20s} @ {p['bit_width']:2d}-bit: "
            f"{p['storage_bytes']:6.0f} B/vec, "
            f"R@10={p['recall_at_10']:.3f}, "
            f"rho={p['spearman_rho']:.4f}"
        )

    return pareto


# =====================================================================
# Section 7: Main Experiment
# =====================================================================


def run_experiment() -> ExperimentResult:
    """Run the complete EN8.4 Part A experiment."""
    print("=" * 70)
    print("EN8.4 Part A -- Quantized Vector Retrieval (Synthetic)")
    print("=" * 70)

    set_global_seed(SEED)
    env = log_environment()

    print(f"\nConfiguration:")
    print(f"  Products:    {N_PRODUCTS}")
    print(f"  Categories:  {N_CATEGORIES}")
    print(f"  Dimensions:  {DIMS}")
    print(f"  Queries:     {N_QUERIES} ({N_QUERIES_PER_CATEGORY}/category)")
    print(f"  Bit-widths:  {BIT_WIDTHS}")
    print(f"  Seed:        {SEED}")
    print(f"  TurboQuant:  {_TURBOQUANT_SOURCE}")
    if _HAS_TURBOQUANT:
        print(f"  TurboQuantIP: {'available' if _HAS_TURBOQUANT_IP else 'not available'}")
    print()

    # -- Data generation -----------------------------------------------
    print("Generating product embeddings ...", end=" ", flush=True)
    rng = np.random.default_rng(SEED)
    centers = generate_category_centers(rng)
    embeddings, categories = generate_product_embeddings(rng, centers)
    query_indices = select_query_indices(rng, categories)
    print(f"done. Shape: {embeddings.shape}")

    # Sanity check: within-category similarity > across-category similarity
    intra_sims = []
    inter_sims = []
    for cat_idx in range(N_CATEGORIES):
        cat_mask = categories == cat_idx
        cat_vecs = embeddings[cat_mask]
        if len(cat_vecs) > 1:
            sims = cat_vecs @ cat_vecs.T
            n = len(cat_vecs)
            for i in range(n):
                for j in range(i + 1, n):
                    intra_sims.append(sims[i, j])

        other_mask = categories != cat_idx
        other_vecs = embeddings[other_mask][:20]  # sample for speed
        cross_sims = cat_vecs[:20] @ other_vecs.T
        inter_sims.extend(cross_sims.ravel().tolist())

    print(f"  Intra-category cosine sim: {np.mean(intra_sims):.4f} +/- {np.std(intra_sims):.4f}")
    print(f"  Inter-category cosine sim: {np.mean(inter_sims):.4f} +/- {np.std(inter_sims):.4f}")

    t_start = time.perf_counter()

    # -- RQ1-RQ3: Distortion and ranking eval --------------------------
    rq1_results = run_rq1_rq2_rq3(embeddings, query_indices)

    # -- RQ4: SL under uniform quantization ----------------------------
    rq4_result = run_rq4_uniform_sl(embeddings, categories, query_indices)

    # -- RQ5: SL under mixed-precision ---------------------------------
    rq5_result = run_rq5_mixed_precision_sl(
        embeddings, categories, query_indices
    )

    # -- RQ6: Hybrid search advantage ----------------------------------
    rq6_result = run_rq6_hybrid_search(embeddings, categories, query_indices)

    # -- RQ7: Pareto frontier ------------------------------------------
    rq7_pareto = run_rq7_pareto(rq1_results)

    total_elapsed = time.perf_counter() - t_start

    # -- Assemble results ----------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Total elapsed: {total_elapsed:.1f}s")
    print(f"{'=' * 70}")

    metrics = {
        "rq1_rq2_rq3_results": rq1_results,
        "rq4_uniform_sl": rq4_result,
        "rq5_mixed_precision_sl": rq5_result,
        "rq6_hybrid_search": rq6_result,
        "rq7_pareto_frontier": rq7_pareto,
        "total_elapsed_seconds": total_elapsed,
        "turboquant_available": _HAS_TURBOQUANT,
        "turboquant_ip_available": _HAS_TURBOQUANT and _HAS_TURBOQUANT_IP,
        "turboquant_source": _TURBOQUANT_SOURCE,
    }

    # Summary table
    print("\n--- Summary Table (RQ1-RQ3) ---")
    print(f"{'Method':20s} {'Bits':>4s} {'MSE':>10s} {'Pearson r':>10s} "
          f"{'Spearman':>10s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} "
          f"{'Ratio':>6s}")
    print("-" * 90)
    for r in rq1_results:
        print(
            f"{r['method']:20s} {r['bit_width']:4d} "
            f"{r['mse']:10.6f} {r['pearson_r_mean']:10.6f} "
            f"{r['spearman_rho_mean']:10.6f} "
            f"{r['recall_at_1_mean']:6.3f} {r['recall_at_5_mean']:6.3f} "
            f"{r['recall_at_10_mean']:6.3f} "
            f"{r['compression_ratio']:6.1f}x"
        )

    result = ExperimentResult(
        experiment_id="EN8.4a",
        parameters={
            "n_products": N_PRODUCTS,
            "n_categories": N_CATEGORIES,
            "dims": DIMS,
            "n_queries": N_QUERIES,
            "n_queries_per_category": N_QUERIES_PER_CATEGORY,
            "bit_widths": BIT_WIDTHS,
            "k_values": K_VALUES,
            "seed": SEED,
            "noise_scale": 0.3,
            "category_names": CATEGORY_NAMES,
            "turboquant_source": _TURBOQUANT_SOURCE,
        },
        metrics=metrics,
        environment=env,
        notes=(
            "EN8.4 Part A: Synthetic product catalog with controlled "
            "cluster structure. Evaluates quantization impact on vector "
            "retrieval quality across naive scalar, TurboQuantMSE, and "
            "TurboQuantIP methods. Pre-registered prediction: SL under "
            "uniform quantization does not change ranking (RQ4). "
            "Mixed-precision graph is the honest interesting case (RQ5). "
            "Distortion constants in quantization_bridge.py are "
            "illustrative defaults, not calibrated."
        ),
    )

    return result


# =====================================================================
# Entry Point
# =====================================================================


def main() -> None:
    """Run EN8.4 Part A and save results."""
    result = run_experiment()

    # -- Save results --
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    latest_path = RESULTS_DIR / "en8_4a_results.json"
    result.save_json(str(latest_path))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = RESULTS_DIR / f"en8_4a_results_{ts}.json"
    result.save_json(str(archive_path))

    print(f"\nResults saved to:")
    print(f"  {latest_path}")
    print(f"  {archive_path}")


if __name__ == "__main__":
    main()
