#!/usr/bin/env python
"""EN8.4 Part B -- BEIR Benchmark Evaluation for Vector Retrieval.

NeurIPS 2026 D&B, Suite EN8 (Ecosystem Integration), Experiment 4B.

Pre-registered Research Questions:
    RQ-B1: How much NDCG@10 is lost at each bit-width (2/3/4/8) vs
           float32 on real IR tasks?
    RQ-B2: Does TurboQuant preserve more retrieval quality than naive
           scalar, consistent with Part A?
    RQ-B3: Does SL uncertainty-aware ranking help in mixed-precision
           BEIR retrieval?  (Pre-registered prediction: likely null
           or marginal, consistent with Part A RQ5 = -0.018.)
    RQ-B4: What is the storage-vs-NDCG@10 Pareto frontier on real data?

Datasets:
    - SciFact       (~5K docs, ~300 queries)   -- scientific claim verification
    - NFCorpus      (~3.6K docs, ~323 queries)  -- biomedical/nutrition
    - ArguAna       (~8.7K docs, ~1406 queries) -- argumentative retrieval
    - SCIDOCS       (~25K docs, ~1000 queries)  -- scientific documents
    - FiQA          (~57K docs, ~648 queries)   -- financial QA
    - TREC-COVID    (~171K docs, ~50 queries)   -- COVID biomedical
    - Touche-2020   (~382K docs, ~49 queries)   -- argument retrieval
    - CQADupStack   (~457K total, 12 sub-forums) -- StackExchange
    - Quora         (~523K docs, ~10K queries)  -- duplicate questions
    - NQ            (~2.7M docs, ~3.5K queries) -- open-domain QA
    - DBPedia-Entity (~4.6M docs, ~400 queries) -- entity retrieval

Encoder: sentence-transformers/all-MiniLM-L6-v2 (384-dim)

Metrics (standard IR):
    - NDCG@10 (primary), MRR@10, Recall@{1, 5, 10, 100}
    - Per-dataset AND aggregate
    - Bootstrap 95% CIs (n=1000) on NDCG@10

References:
    Thakur et al. (2021). BEIR: A Heterogeneous Benchmark for
        Zero-shot Evaluation of Information Retrieval Models.
    Jarvelin & Kekalainen (2002). Cumulated gain-based evaluation
        of IR techniques.
    Zandieh et al. (2025). TurboQuant. ICLR 2026.
    Josang, A. (2016). Subjective Logic. Springer.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np


# =====================================================================
# Section 1: IR Metrics
# =====================================================================


def ndcg_at_k(
    qrel: Dict[str, int],
    ranked_doc_ids: List[str],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    Uses the standard formula with exponential gain:
        DCG@k  = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)
        NDCG@k = DCG@k / IDCG@k

    where IDCG@k is DCG@k for the ideal ranking (sorted by relevance).

    Args:
        qrel: Dict mapping doc_id -> relevance grade (int >= 0).
              Only docs with relevance > 0 are considered relevant.
        ranked_doc_ids: System's ranked list of document IDs.
        k: Cutoff depth.

    Returns:
        NDCG@k in [0, 1]. Returns 0.0 if no relevant docs exist.
    """
    # Filter to relevance > 0 for IDCG computation
    relevant_grades = sorted(
        [rel for rel in qrel.values() if rel > 0], reverse=True
    )

    if not relevant_grades:
        return 0.0

    # DCG@k from system ranking
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        rel = qrel.get(doc_id, 0)
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 2)  # i+2 because i is 0-indexed

    # IDCG@k from ideal ranking
    idcg = 0.0
    for i, rel in enumerate(relevant_grades[:k]):
        idcg += (2.0 ** rel - 1.0) / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def mrr_at_k(
    qrel: Dict[str, int],
    ranked_doc_ids: List[str],
    k: int = 10,
) -> float:
    """Compute Reciprocal Rank at k.

    Returns 1/rank of the first relevant document (relevance > 0)
    within the top-k results, or 0 if none found.

    Args:
        qrel: Dict mapping doc_id -> relevance grade.
        ranked_doc_ids: System's ranked list of document IDs.
        k: Cutoff depth.

    Returns:
        RR in [0, 1].
    """
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        rel = qrel.get(doc_id, 0)
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(
    qrel: Dict[str, int],
    ranked_doc_ids: List[str],
    k: int = 10,
) -> float:
    """Compute Recall at k.

    Fraction of relevant documents (relevance > 0) that appear
    in the top-k results.

    Args:
        qrel: Dict mapping doc_id -> relevance grade.
        ranked_doc_ids: System's ranked list of document IDs.
        k: Cutoff depth.

    Returns:
        Recall in [0, 1]. Returns 0.0 if no relevant docs exist.
    """
    relevant = {doc_id for doc_id, rel in qrel.items() if rel > 0}

    if not relevant:
        return 0.0

    retrieved_relevant = relevant & set(ranked_doc_ids[:k])
    return len(retrieved_relevant) / len(relevant)


def compute_metrics_for_query(
    qrel: Dict[str, int],
    ranked_doc_ids: List[str],
) -> Dict[str, float]:
    """Compute all IR metrics for a single query.

    Args:
        qrel: Dict mapping doc_id -> relevance grade.
        ranked_doc_ids: System's ranked list of document IDs.

    Returns:
        Dict with keys: ndcg@10, mrr@10, recall@1, recall@5,
        recall@10, recall@100.
    """
    return {
        "ndcg@10": ndcg_at_k(qrel, ranked_doc_ids, k=10),
        "mrr@10": mrr_at_k(qrel, ranked_doc_ids, k=10),
        "recall@1": recall_at_k(qrel, ranked_doc_ids, k=1),
        "recall@5": recall_at_k(qrel, ranked_doc_ids, k=5),
        "recall@10": recall_at_k(qrel, ranked_doc_ids, k=10),
        "recall@100": recall_at_k(qrel, ranked_doc_ids, k=100),
    }


# =====================================================================
# Section 2: Bootstrap Confidence Intervals
# =====================================================================


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for the mean.

    Uses the percentile method: resample with replacement n_bootstrap
    times, compute the mean of each resample, then take the
    alpha/2 and 1-alpha/2 percentiles of the bootstrap distribution.

    Args:
        values: List of observed values (e.g., per-query NDCG scores).
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: mean, ci_lower, ci_upper, std.
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    sample_mean = float(np.mean(arr))
    sample_std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    if n <= 1:
        return {
            "mean": sample_mean,
            "ci_lower": sample_mean,
            "ci_upper": sample_mean,
            "std": sample_std,
        }

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        resample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(resample)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "mean": sample_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": sample_std,
    }



# =====================================================================
# Section 3: Constants and Configuration
# =====================================================================

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
from jsonld_ex.quantization_bridge import (
    quantization_to_opinion,
    quantization_distortion,
    distortion_to_uncertainty,
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
BEIR_DATA_DIR = Path("D:/cc/datasets/beir")

DATASETS = [
    "scifact",          # ~5K docs, ~300 queries    -- scientific claims
    "nfcorpus",         # ~3.6K docs, ~323 queries  -- biomedical/nutrition
    "arguana",          # ~8.7K docs, ~1406 queries  -- argumentative retrieval
    "scidocs",          # ~25K docs, ~1000 queries   -- scientific documents
    "fiqa",             # ~57K docs, ~648 queries    -- financial QA
    "trec-covid",       # ~171K docs, ~50 queries    -- COVID biomedical
    "webis-touche2020", # ~382K docs, ~49 queries    -- argument retrieval
    "cqadupstack",      # ~457K total, 12 sub-forums -- StackExchange
    "quora",            # ~523K docs, ~10000 queries -- duplicate questions
    "nq",               # ~2.7M docs, ~3452 queries  -- open-domain QA
    "dbpedia-entity",   # ~4.6M docs, ~400 queries   -- entity retrieval
]

CQADUPSTACK_SUBFORUMS = [
    "android", "english", "gaming", "gis", "mathematica",
    "physics", "programmers", "stats", "tex", "unix",
    "webmasters", "wordpress",
]

BIT_WIDTHS = [2, 3, 4, 8]
K_VALUES_RECALL = [1, 5, 10, 100]
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


# =====================================================================
# Section 4: BEIR Dataset Loading
# =====================================================================


def load_beir_dataset(
    dataset_name: str,
) -> tuple:
    """Load a BEIR dataset: corpus, queries, qrels.

    Uses beir's GenericDataLoader to download and parse datasets.

    Args:
        dataset_name: One of 'scifact', 'nfcorpus', 'fiqa'.

    Returns:
        (corpus, queries, qrels) where:
        - corpus: Dict[doc_id, {"title": str, "text": str}]
        - queries: Dict[query_id, str]
        - qrels: Dict[query_id, Dict[doc_id, int]]
    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    url = (
        f"https://public.ukp.informatik.tu-darmstadt.de/"
        f"thakur/BEIR/datasets/{dataset_name}.zip"
    )
    data_path = util.download_and_unzip(url, str(BEIR_DATA_DIR))
    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_path
    ).load(split="test")

    return corpus, queries, qrels


def load_cqadupstack_subforum(
    subforum: str,
) -> tuple:
    """Load a single CQADupStack sub-forum.

    Downloads the full CQADupStack dataset once, then loads the
    specified sub-forum.

    Args:
        subforum: Sub-forum name (e.g., 'android', 'physics').

    Returns:
        (corpus, queries, qrels) for the specified sub-forum.
    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/"
        "thakur/BEIR/datasets/cqadupstack.zip"
    )
    data_path = util.download_and_unzip(url, str(BEIR_DATA_DIR))
    subforum_path = os.path.join(data_path, subforum)
    corpus, queries, qrels = GenericDataLoader(
        data_folder=subforum_path
    ).load(split="test")

    return corpus, queries, qrels


# =====================================================================
# Section 5: Sentence-Transformer Encoding
# =====================================================================


def encode_corpus_and_queries(
    corpus: dict,
    queries: dict,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> tuple:
    """Encode corpus and queries with a sentence-transformer model.

    Args:
        corpus: Dict[doc_id, {"title": str, "text": str}].
        queries: Dict[query_id, str].
        model_name: HuggingFace model name.
        batch_size: Encoding batch size.

    Returns:
        (corpus_ids, corpus_embeddings, query_ids, query_embeddings)
        where embeddings are np.ndarray of shape (N, dim).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    # Prepare corpus texts: "title text" concatenation (BEIR standard)
    corpus_ids = list(corpus.keys())
    corpus_texts = [
        (corpus[cid].get("title", "") + " " + corpus[cid].get("text", "")).strip()
        for cid in corpus_ids
    ]

    # Prepare query texts
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print(f"    Encoding {len(corpus_texts)} corpus docs ...", end=" ", flush=True)
    t0 = time.perf_counter()
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Unit vectors for cosine sim = dot product
    )
    t_corpus = time.perf_counter() - t0
    print(f"done ({t_corpus:.1f}s)")

    print(f"    Encoding {len(query_texts)} queries ...", end=" ", flush=True)
    t0 = time.perf_counter()
    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    t_queries = time.perf_counter() - t0
    print(f"done ({t_queries:.1f}s)")

    return (
        corpus_ids,
        corpus_embeddings.astype(np.float32),
        query_ids,
        query_embeddings.astype(np.float32),
    )


# =====================================================================
# Section 6: Quantization (reuse from Part A)
# =====================================================================


def naive_scalar_roundtrip(
    vectors: np.ndarray, bit_width: int
) -> np.ndarray:
    """Quantize and dequantize using naive uniform scalar quantization."""
    n_levels = 2 ** bit_width
    v_min = float(vectors.min())
    v_max = float(vectors.max())
    if v_max - v_min < 1e-10:
        v_max = v_min + 1e-10
    normalized = (vectors - v_min) / (v_max - v_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    indices = np.round(normalized * (n_levels - 1)).astype(np.int32)
    return (indices.astype(np.float32) / (n_levels - 1)) * (v_max - v_min) + v_min


def turboquant_mse_roundtrip(
    vectors: np.ndarray, bit_width: int, batch_size: int = 8192
) -> "np.ndarray | None":
    """Quantize and dequantize using TurboQuantMSE.

    Processes in batches to avoid CUDA OOM on large corpora.
    """
    if not _HAS_TURBOQUANT:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tq = TurboQuantMSE(dim=vectors.shape[1], bits=bit_width, device=device)

    n = vectors.shape[0]
    reconstructed = np.empty_like(vectors)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        t = torch.from_numpy(vectors[start:end]).float().to(device)
        indices, norms = tq.quantize(t)
        recon = tq.dequantize(indices, norms)
        reconstructed[start:end] = recon.cpu().numpy()
        del t, indices, norms, recon
        if device == "cuda":
            torch.cuda.empty_cache()

    return reconstructed


def turboquant_ip_roundtrip(
    vectors: np.ndarray, bit_width: int, batch_size: int = 8192
) -> "np.ndarray | None":
    """Quantize and dequantize using TurboQuantIP.

    Processes in batches to avoid CUDA OOM on large corpora.
    """
    if not _HAS_TURBOQUANT or not _HAS_TURBOQUANT_IP:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tq = TurboQuantIP(dim=vectors.shape[1], bits=bit_width, device=device)

    n = vectors.shape[0]
    reconstructed = np.empty_like(vectors)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        t = torch.from_numpy(vectors[start:end]).float().to(device)
        mse_indices, norms, qjl_signs, residual_norms = tq.quantize(t)
        recon = tq.dequantize(mse_indices, norms, qjl_signs, residual_norms)
        reconstructed[start:end] = recon.cpu().numpy()
        del t, mse_indices, norms, qjl_signs, residual_norms, recon
        if device == "cuda":
            torch.cuda.empty_cache()

    return reconstructed


# =====================================================================
# Section 7: Vectorized Search
# =====================================================================


def search_all_queries(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_ids: List[str],
    top_k: int = 100,
) -> List[List[str]]:
    """Brute-force cosine similarity search for all queries.

    Since embeddings are L2-normalized, cosine similarity = dot product.

    Args:
        query_embeddings: (n_queries, dim) normalized vectors.
        corpus_embeddings: (n_corpus, dim) normalized vectors.
        corpus_ids: List of corpus document IDs.
        top_k: Number of results per query.

    Returns:
        List of ranked doc_id lists, one per query.
    """
    # (n_queries, n_corpus) similarity matrix
    # Process in chunks to avoid memory issues on large corpora
    n_queries = query_embeddings.shape[0]
    n_corpus = corpus_embeddings.shape[0]
    chunk_size = 64  # queries per chunk

    all_rankings = []

    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        q_batch = query_embeddings[start:end]

        # Similarity via dot product (vectors are normalized)
        sims = q_batch @ corpus_embeddings.T  # (chunk, n_corpus)

        for i in range(sims.shape[0]):
            # Get top-k indices (partial sort for efficiency)
            if top_k < n_corpus:
                top_indices = np.argpartition(sims[i], -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(sims[i][top_indices])[::-1]]
            else:
                top_indices = np.argsort(sims[i])[::-1][:top_k]

            all_rankings.append([corpus_ids[idx] for idx in top_indices])

    return all_rankings


# =====================================================================
# Section 8: Per-Method Evaluation
# =====================================================================


def evaluate_retrieval(
    query_ids: List[str],
    rankings: List[List[str]],
    qrels: dict,
) -> Dict[str, Any]:
    """Compute IR metrics across all queries.

    Args:
        query_ids: List of query IDs.
        rankings: List of ranked doc_id lists (one per query).
        qrels: Dict[query_id, Dict[doc_id, int]].

    Returns:
        Dict with per-metric mean, std, and bootstrap CI.
    """
    per_query_metrics: Dict[str, List[float]] = {
        "ndcg@10": [],
        "mrr@10": [],
        "recall@1": [],
        "recall@5": [],
        "recall@10": [],
        "recall@100": [],
    }

    n_evaluated = 0

    for qid, ranked in zip(query_ids, rankings):
        qrel = qrels.get(qid, {})
        # Skip queries with no relevant documents
        if not any(rel > 0 for rel in qrel.values()):
            continue

        metrics = compute_metrics_for_query(qrel, ranked)
        for key, val in metrics.items():
            per_query_metrics[key].append(val)
        n_evaluated += 1

    # Aggregate with bootstrap CIs
    result: Dict[str, Any] = {"n_queries_evaluated": n_evaluated}

    for metric_name, values in per_query_metrics.items():
        if values:
            ci = bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
            result[metric_name] = ci
        else:
            result[metric_name] = {
                "mean": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "std": 0.0,
            }

    return result


def run_method_evaluation(
    method_name: str,
    bit_width: int,
    corpus_embeddings_orig: np.ndarray,
    query_embeddings: np.ndarray,
    corpus_ids: List[str],
    query_ids: List[str],
    qrels: dict,
    roundtrip_fn,
) -> "Dict[str, Any] | None":
    """Evaluate one quantization method at one bit-width.

    Returns metrics dict or None if method unavailable.
    """
    print(f"  {method_name} @ {bit_width}-bit ... ", end="", flush=True)
    t0 = time.perf_counter()

    quantized = roundtrip_fn(corpus_embeddings_orig, bit_width)
    if quantized is None:
        print("SKIPPED (not available)")
        return None

    # Re-normalize after quantization (important for dot-product search)
    norms = np.linalg.norm(quantized, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    quantized_normed = quantized / norms

    # Search
    rankings = search_all_queries(
        query_embeddings, quantized_normed, corpus_ids, top_k=100
    )

    # Evaluate
    eval_result = evaluate_retrieval(query_ids, rankings, qrels)

    # MSE (on original, not re-normalized)
    mse = float(np.mean((corpus_embeddings_orig - quantized) ** 2))

    elapsed = time.perf_counter() - t0

    result = {
        "method": method_name,
        "bit_width": bit_width,
        "mse": mse,
        "elapsed_seconds": elapsed,
        **{k: v for k, v in eval_result.items()},
    }

    ndcg = eval_result["ndcg@10"]["mean"]
    print(
        f"NDCG@10={ndcg:.4f}  "
        f"MRR@10={eval_result['mrr@10']['mean']:.4f}  "
        f"R@10={eval_result['recall@10']['mean']:.4f}  "
        f"({elapsed:.1f}s)"
    )
    return result


# =====================================================================
# Section 9: RQ-B3 Mixed-Precision SL (Numpy-based)
# =====================================================================


def run_mixed_precision_sl(
    corpus_embeddings_orig: np.ndarray,
    query_embeddings: np.ndarray,
    corpus_ids: List[str],
    query_ids: List[str],
    qrels: dict,
    bit_width: int = 4,
) -> Dict[str, Any]:
    """RQ-B3: SL uncertainty-aware ranking under mixed precision.

    50% of corpus at float32, 50% at bit_width-bit naive_scalar.
    Compare raw cosine ranking vs SL-adjusted ranking.

    Pre-registered prediction: likely null or marginal improvement,
    consistent with Part A RQ5 = -0.018.
    """
    print("\n  RQ-B3: Mixed-Precision SL Evaluation")
    print(f"    50% float32 + 50% {bit_width}-bit naive_scalar")
    t0 = time.perf_counter()

    rng = np.random.default_rng(SEED + 200)
    n_corpus = len(corpus_ids)

    # Random 50/50 split
    float32_mask = np.zeros(n_corpus, dtype=bool)
    float32_indices = rng.choice(n_corpus, size=n_corpus // 2, replace=False)
    float32_mask[float32_indices] = True

    # Quantize the non-float32 portion
    quantized = naive_scalar_roundtrip(corpus_embeddings_orig, bit_width)
    mixed = np.where(float32_mask[:, np.newaxis], corpus_embeddings_orig, quantized)

    # Re-normalize
    norms = np.linalg.norm(mixed, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    mixed_normed = mixed / norms

    # Compute SL uncertainty for each corpus doc
    # float32 docs: bit_width=32, uncertainty ~0
    # quantized docs: bit_width=bit_width, uncertainty from distortion
    u_float32 = distortion_to_uncertainty(quantization_distortion(32, "scalar"))
    u_quantized = distortion_to_uncertainty(
        quantization_distortion(bit_width, "scalar")
    )

    uncertainties = np.where(float32_mask, u_float32, u_quantized)

    # --- Raw cosine rankings ---
    raw_rankings = search_all_queries(
        query_embeddings, mixed_normed, corpus_ids, top_k=100
    )
    raw_eval = evaluate_retrieval(query_ids, raw_rankings, qrels)

    # --- SL-adjusted rankings ---
    # For each query, compute dot-product similarities, then adjust
    # by SL projected probability: P = b + a*u where
    # confidence c = (sim + 1) / 2 (map [-1,1] to [0,1])
    # b = c * (1 - u), d = (1 - c) * (1 - u), a = 0.5
    # projected_prob = b + a * u = c*(1-u) + 0.5*u = c - c*u + 0.5*u
    sl_rankings = []

    for qi in range(query_embeddings.shape[0]):
        sims = query_embeddings[qi] @ mixed_normed.T  # (n_corpus,)
        confidences = (sims + 1.0) / 2.0  # map to [0, 1]

        # SL projected probability
        proj_probs = confidences * (1.0 - uncertainties) + 0.5 * uncertainties

        if 100 < n_corpus:
            top_indices = np.argpartition(proj_probs, -100)[-100:]
            top_indices = top_indices[np.argsort(proj_probs[top_indices])[::-1]]
        else:
            top_indices = np.argsort(proj_probs)[::-1][:100]

        sl_rankings.append([corpus_ids[idx] for idx in top_indices])

    sl_eval = evaluate_retrieval(query_ids, sl_rankings, qrels)

    elapsed = time.perf_counter() - t0

    raw_ndcg = raw_eval["ndcg@10"]["mean"]
    sl_ndcg = sl_eval["ndcg@10"]["mean"]
    diff = sl_ndcg - raw_ndcg

    result = {
        "bit_width": bit_width,
        "n_float32": int(float32_mask.sum()),
        "n_quantized": int((~float32_mask).sum()),
        "u_float32": u_float32,
        "u_quantized": u_quantized,
        "raw_eval": raw_eval,
        "sl_eval": sl_eval,
        "ndcg@10_diff": diff,
        "elapsed_seconds": elapsed,
        "pre_registered_prediction": (
            "Likely null or marginal, consistent with Part A RQ5 = -0.018"
        ),
    }

    print(f"    Raw NDCG@10:  {raw_ndcg:.4f} [{raw_eval['ndcg@10']['ci_lower']:.4f}, {raw_eval['ndcg@10']['ci_upper']:.4f}]")
    print(f"    SL  NDCG@10:  {sl_ndcg:.4f} [{sl_eval['ndcg@10']['ci_lower']:.4f}, {sl_eval['ndcg@10']['ci_upper']:.4f}]")
    print(f"    Difference:   {diff:+.4f}")
    print(f"    ({elapsed:.1f}s)")

    return result


# =====================================================================
# Section 10: RQ-B4 Pareto Frontier
# =====================================================================


def compute_pareto_frontier(
    method_results: List[Dict[str, Any]],
    dims: int,
) -> List[Dict[str, Any]]:
    """Extract Pareto-optimal configurations from method results.

    A configuration is Pareto-optimal if no other has BOTH higher
    NDCG@10 AND lower storage.
    """
    # Add float32 baseline
    points = []
    for r in method_results:
        ndcg = r["ndcg@10"]["mean"] if isinstance(r["ndcg@10"], dict) else r["ndcg@10"]
        points.append({
            "method": r["method"],
            "bit_width": r["bit_width"],
            "storage_bytes": dims * r["bit_width"] / 8.0,
            "compression_ratio": (dims * 4) / (dims * r["bit_width"] / 8.0),
            "ndcg@10": ndcg,
        })

    pareto = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if (
                q["ndcg@10"] >= p["ndcg@10"]
                and q["storage_bytes"] <= p["storage_bytes"]
                and (
                    q["ndcg@10"] > p["ndcg@10"]
                    or q["storage_bytes"] < p["storage_bytes"]
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(p)

    pareto.sort(key=lambda x: x["storage_bytes"])
    return pareto


# =====================================================================
# Section 11: Per-Dataset Experiment Runner
# =====================================================================


def run_dataset_experiment(
    dataset_name: str,
    subforum: str = "",
) -> Dict[str, Any]:
    """Run the full EN8.4B evaluation for one BEIR dataset.

    Args:
        dataset_name: BEIR dataset slug.
        subforum: If non-empty, load this CQADupStack sub-forum.
    """
    display_name = f"{dataset_name}/{subforum}" if subforum else dataset_name
    print(f"\n{'=' * 70}")
    print(f"Dataset: {display_name}")
    print(f"{'=' * 70}")

    # -- Load data -----------------------------------------------------
    print("  Loading dataset ...", end=" ", flush=True)
    t0 = time.perf_counter()
    if subforum:
        corpus, queries, qrels = load_cqadupstack_subforum(subforum)
    else:
        corpus, queries, qrels = load_beir_dataset(dataset_name)
    t_load = time.perf_counter() - t0
    print(
        f"done ({t_load:.1f}s). "
        f"Corpus: {len(corpus)}, Queries: {len(queries)}"
    )

    # -- Encode --------------------------------------------------------
    print("  Encoding with all-MiniLM-L6-v2:")
    corpus_ids, corpus_embs, query_ids, query_embs = encode_corpus_and_queries(
        corpus, queries
    )
    dims = corpus_embs.shape[1]
    print(f"    Embedding dimension: {dims}")

    # -- Float32 baseline ----------------------------------------------
    print("\n  Float32 baseline ...", end=" ", flush=True)
    t0 = time.perf_counter()
    float32_rankings = search_all_queries(
        query_embs, corpus_embs, corpus_ids, top_k=100
    )
    float32_eval = evaluate_retrieval(query_ids, float32_rankings, qrels)
    t_baseline = time.perf_counter() - t0
    f32_ndcg = float32_eval["ndcg@10"]["mean"]
    print(
        f"NDCG@10={f32_ndcg:.4f}  "
        f"MRR@10={float32_eval['mrr@10']['mean']:.4f}  "
        f"R@100={float32_eval['recall@100']['mean']:.4f}  "
        f"({t_baseline:.1f}s)"
    )

    # -- RQ-B1/B2: Method x bit-width ---------------------------------
    print("\n  RQ-B1/B2: Quantization impact on retrieval")
    method_results = []

    for bw in BIT_WIDTHS:
        print(f"\n  --- Bit-width: {bw} ---")

        r = run_method_evaluation(
            "naive_scalar", bw,
            corpus_embs, query_embs, corpus_ids, query_ids, qrels,
            naive_scalar_roundtrip,
        )
        if r is not None:
            method_results.append(r)

        r = run_method_evaluation(
            "turboquant_mse", bw,
            corpus_embs, query_embs, corpus_ids, query_ids, qrels,
            turboquant_mse_roundtrip,
        )
        if r is not None:
            method_results.append(r)

        r = run_method_evaluation(
            "turboquant_ip", bw,
            corpus_embs, query_embs, corpus_ids, query_ids, qrels,
            turboquant_ip_roundtrip,
        )
        if r is not None:
            method_results.append(r)

    # -- RQ-B3: Mixed-precision SL -------------------------------------
    rq_b3_result = run_mixed_precision_sl(
        corpus_embs, query_embs, corpus_ids, query_ids, qrels,
        bit_width=4,
    )

    # -- RQ-B4: Pareto frontier ----------------------------------------
    # Add float32 baseline to method results for Pareto computation
    float32_point = {
        "method": "float32",
        "bit_width": 32,
        "ndcg@10": float32_eval["ndcg@10"],
        "mse": 0.0,
    }
    pareto = compute_pareto_frontier(
        method_results + [float32_point], dims
    )

    print(f"\n  Pareto-optimal ({len(pareto)} configurations):")
    for p in pareto:
        print(
            f"    {p['method']:20s} @ {p['bit_width']:2d}-bit: "
            f"{p['storage_bytes']:6.0f} B/vec, "
            f"NDCG@10={p['ndcg@10']:.4f}"
        )

    return {
        "dataset": display_name,
        "n_corpus": len(corpus),
        "n_queries": len(queries),
        "embedding_dim": dims,
        "float32_baseline": float32_eval,
        "method_results": method_results,
        "rq_b3_mixed_precision": rq_b3_result,
        "pareto_frontier": pareto,
    }


# =====================================================================
# Section 12: Main Experiment
# =====================================================================


def run_experiment() -> ExperimentResult:
    """Run the complete EN8.4B experiment across all BEIR datasets."""
    print("=" * 70)
    print("EN8.4 Part B -- BEIR Benchmark Evaluation")
    print("=" * 70)

    set_global_seed(SEED)
    env = log_environment()

    print(f"\nConfiguration:")
    print(f"  Datasets:      {DATASETS}")
    print(f"  Encoder:       all-MiniLM-L6-v2 (384-dim)")
    print(f"  Bit-widths:    {BIT_WIDTHS}")
    print(f"  Bootstrap:     n={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}")
    print(f"  TurboQuant:    {_TURBOQUANT_SOURCE}")
    if _HAS_TURBOQUANT:
        print(f"  TurboQuantIP:  {'available' if _HAS_TURBOQUANT_IP else 'not available'}")
    print()

    t_start = time.perf_counter()
    all_dataset_results = {}

    # -- Load any existing checkpoints --------------------------------
    checkpoint_dir = RESULTS_DIR / "en8_4b_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in DATASETS:
        if ds_name == "cqadupstack":
            for subforum in CQADUPSTACK_SUBFORUMS:
                key = f"cqadupstack/{subforum}"
                cp_file = checkpoint_dir / f"en8_4b_{key.replace('/', '_')}.json"
                if cp_file.exists():
                    print(f"  [CHECKPOINT] Loading {key} from {cp_file.name}")
                    with open(cp_file, "r", encoding="utf-8") as f:
                        all_dataset_results[key] = json.load(f)
                else:
                    ds_result = run_dataset_experiment(ds_name, subforum=subforum)
                    all_dataset_results[key] = ds_result
                    with open(cp_file, "w", encoding="utf-8") as f:
                        json.dump(ds_result, f, indent=2, default=str)
                    print(f"  [CHECKPOINT] Saved {key} -> {cp_file.name}")
                gc.collect()
        else:
            cp_file = checkpoint_dir / f"en8_4b_{ds_name}.json"
            if cp_file.exists():
                print(f"  [CHECKPOINT] Loading {ds_name} from {cp_file.name}")
                with open(cp_file, "r", encoding="utf-8") as f:
                    all_dataset_results[ds_name] = json.load(f)
            else:
                ds_result = run_dataset_experiment(ds_name)
                all_dataset_results[ds_name] = ds_result
                with open(cp_file, "w", encoding="utf-8") as f:
                    json.dump(ds_result, f, indent=2, default=str)
                print(f"  [CHECKPOINT] Saved {ds_name} -> {cp_file.name}")
            gc.collect()

    total_elapsed = time.perf_counter() - t_start

    # -- Aggregate across datasets -------------------------------------
    print(f"\n{'=' * 70}")
    print("Aggregate Summary")
    print(f"{'=' * 70}")

    # Collect per-method aggregate NDCG@10 across datasets
    aggregate_by_config: Dict[str, List[float]] = {}
    all_keys = list(all_dataset_results.keys())

    for ds_key in all_keys:
        ds_result = all_dataset_results[ds_key]
        for mr in ds_result["method_results"]:
            key = f"{mr['method']}@{mr['bit_width']}"
            if key not in aggregate_by_config:
                aggregate_by_config[key] = []
            aggregate_by_config[key].append(mr["ndcg@10"]["mean"])

    print(f"\n  Mean NDCG@10 across {len(all_keys)} evaluation sets:")
    for config_key in sorted(aggregate_by_config.keys()):
        vals = aggregate_by_config[config_key]
        mean_ndcg = np.mean(vals)
        print(f"    {config_key:25s}: {mean_ndcg:.4f}")

    # -- RQ-B3 aggregate -----------------------------------------------
    rq_b3_diffs = {
        ds_key: all_dataset_results[ds_key]["rq_b3_mixed_precision"]["ndcg@10_diff"]
        for ds_key in all_keys
    }
    rq_b3_diff_values = list(rq_b3_diffs.values())
    print(f"\n  RQ-B3 NDCG@10 diff (SL - raw) across evaluation sets:")
    for ds_key, diff in rq_b3_diffs.items():
        print(f"    {ds_key:30s}: {diff:+.4f}")
    print(f"    {'mean':30s}: {np.mean(rq_b3_diff_values):+.4f}")

    # -- Summary table -------------------------------------------------
    print(f"\n  Summary Table:")
    print(f"  {'Dataset':30s} {'Method':20s} {'Bits':>4s} {'NDCG@10':>10s} "
          f"{'MRR@10':>10s} {'R@10':>10s} {'MSE':>10s}")
    print("  " + "-" * 100)

    for ds_key in all_keys:
        ds = all_dataset_results[ds_key]
        # Float32 baseline
        f32 = ds["float32_baseline"]
        print(
            f"  {ds_key:30s} {'float32':20s} {'32':>4s} "
            f"{f32['ndcg@10']['mean']:10.4f} "
            f"{f32['mrr@10']['mean']:10.4f} "
            f"{f32['recall@10']['mean']:10.4f} "
            f"{'0.0':>10s}"
        )
        for mr in ds["method_results"]:
            print(
                f"  {ds_key:30s} {mr['method']:20s} {mr['bit_width']:4d} "
                f"{mr['ndcg@10']['mean']:10.4f} "
                f"{mr['mrr@10']['mean']:10.4f} "
                f"{mr['recall@10']['mean']:10.4f} "
                f"{mr['mse']:10.6f}"
            )

    print(f"\nTotal elapsed: {total_elapsed:.1f}s")

    # -- Build ExperimentResult ----------------------------------------
    result = ExperimentResult(
        experiment_id="EN8.4b",
        parameters={
            "datasets": DATASETS,
            "cqadupstack_subforums": CQADUPSTACK_SUBFORUMS,
            "total_evaluation_sets": len(all_keys),
            "encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "bit_widths": BIT_WIDTHS,
            "k_values_recall": K_VALUES_RECALL,
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "seed": SEED,
            "turboquant_source": _TURBOQUANT_SOURCE,
        },
        metrics={
            "per_dataset": all_dataset_results,
            "rq_b3_diffs": rq_b3_diffs,
            "rq_b3_mean_diff": float(np.mean(rq_b3_diff_values)),
            "total_elapsed_seconds": total_elapsed,
            "turboquant_available": _HAS_TURBOQUANT,
            "turboquant_ip_available": _HAS_TURBOQUANT and _HAS_TURBOQUANT_IP,
        },
        environment=env,
        notes=(
            "EN8.4 Part B: BEIR benchmark evaluation of quantized vector "
            "retrieval. Pre-registered prediction for RQ-B3: SL mixed-precision "
            "improvement likely null or marginal (consistent with Part A "
            "RQ5 = -0.018). All metrics include bootstrap 95% CIs (n=1000). "
            "Distortion constants in quantization_bridge.py are illustrative "
            "defaults, not calibrated."
        ),
    )

    return result


# =====================================================================
# Entry Point
# =====================================================================


def main() -> None:
    """Run EN8.4B and save results."""
    result = run_experiment()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    latest_path = RESULTS_DIR / "en8_4b_results.json"
    result.save_json(str(latest_path))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = RESULTS_DIR / f"en8_4b_results_{ts}.json"
    result.save_json(str(archive_path))

    print(f"\nResults saved to:")
    print(f"  {latest_path}")
    print(f"  {archive_path}")


if __name__ == "__main__":
    main()
