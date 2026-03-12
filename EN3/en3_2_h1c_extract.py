"""EN3.2-H1c — Multi-Model QA Extraction for RAG Fusion.

Runs 3 additional QA models on existing v1b passages to create
independent extraction sources for SL fusion experiments.

Models:
  1. distilbert-base-cased-distilled-squad  (EXISTING — v1b checkpoints)
  2. deepset/roberta-base-squad2            (RoBERTa, ~125M params)
  3. deepset/electra-base-squad2            (ELECTRA, ~110M params)
  4. mrm8488/bert-tiny-finetuned-squadv2    (tiny BERT, ~4M params, weak)

Output: checkpoints/multimodel_qa_{model_tag}_{pr_tag}.json
  Format identical to v1b_qa_extraction: {qid: {pid: {answer, qa_score}}}

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex

    # All models, all poison rates (~10-15 min on 4090):
    python experiments/EN3/en3_2_h1c_extract.py

    # Single model (for testing):
    python experiments/EN3/en3_2_h1c_extract.py --model roberta

    # Dry run (10 questions, 1 poison rate, 1 model):
    python experiments/EN3/en3_2_h1c_extract.py --dry-run
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_ROOT))

CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

from infra.config import set_global_seed

GLOBAL_SEED = 42
POISON_RATES = [0.05, 0.10, 0.20, 0.30]

# Model registry: tag → HuggingFace model name
# distilbert already has extractions in v1b_qa_extraction checkpoints
MODELS = {
    "roberta": "deepset/roberta-base-squad2",
    "electra": "deepset/electra-base-squad2",
    "bert_tiny": "mrm8488/bert-tiny-finetuned-squadv2",
}


def _pr_tag(pr: float) -> str:
    return f"pr{int(pr * 100):02d}"


def load_v1b_data(pr_tag: str):
    """Load questions, corpus texts, and retrieval results."""
    paths = {
        "questions": CHECKPOINT_DIR / f"v1b_questions_{pr_tag}.json",
        "corpus_texts": CHECKPOINT_DIR / f"v1b_corpus_texts_{pr_tag}.json",
        "retrieval": CHECKPOINT_DIR / f"v1b_retrieval_{pr_tag}.json",
    }
    missing = [n for n, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing v1b checkpoints for {pr_tag}: {missing}")

    loaded = {}
    for name, path in paths.items():
        with open(str(path), "r") as f:
            loaded[name] = json.load(f)

    return loaded["questions"], loaded["corpus_texts"], loaded["retrieval"]


def extract_with_model(
    model_name: str,
    questions: List[Dict],
    retrieval_results: List[Dict],
    pid_to_text: Dict[str, str],
    max_questions: int | None = None,
) -> Dict[str, Dict[str, Dict]]:
    """Run extractive QA model on all (question, passage) pairs.

    Returns: {qid: {pid: {answer: str, qa_score: float}}}
    """
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    print(f"    Loading model: {model_name} (device={device})...")
    qa_pipe = pipeline(
        "question-answering",
        model=model_name,
        device=device,
    )

    qid_to_question = {q["id"]: q["question"] for q in questions}
    r_lookup = {r["question_id"]: r["retrieved"] for r in retrieval_results}

    question_ids = [q["id"] for q in questions]
    if max_questions:
        question_ids = question_ids[:max_questions]

    extractions: Dict[str, Dict[str, Dict]] = {}
    total_pairs = sum(
        len(r_lookup.get(qid, []))
        for qid in question_ids
    )
    done = 0
    t0 = time.time()

    for qid in question_ids:
        question_text = qid_to_question[qid]
        passages = r_lookup.get(qid, [])
        extractions[qid] = {}

        for p in passages:
            pid = p["passage_id"]
            passage_text = pid_to_text.get(pid, "")
            if not passage_text:
                continue

            try:
                result = qa_pipe(
                    question=question_text,
                    context=passage_text,
                    max_answer_len=50,
                )
                extractions[qid][pid] = {
                    "answer": result["answer"],
                    "qa_score": float(result["score"]),
                }
            except Exception as e:
                extractions[qid][pid] = {
                    "answer": "",
                    "qa_score": 0.0,
                }

            done += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_pairs - done) / rate if rate > 0 else 0
                print(f"      {done}/{total_pairs} ({100*done/total_pairs:.1f}%) "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    del qa_pipe
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return extractions


def main():
    parser = argparse.ArgumentParser(description="Multi-model QA extraction")
    parser.add_argument("--dry-run", action="store_true",
                        help="10 questions, 1 poison rate, 1 model")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODELS.keys()),
                        help="Run single model only")
    args = parser.parse_args()

    set_global_seed(GLOBAL_SEED)

    if args.dry_run:
        prs = [0.10]
        models_to_run = {"roberta": MODELS["roberta"]}
        max_q = 10
        print("=== DRY RUN: 10 questions, pr10, roberta only ===\n")
    elif args.model:
        prs = POISON_RATES
        models_to_run = {args.model: MODELS[args.model]}
        max_q = None
    else:
        prs = POISON_RATES
        models_to_run = dict(MODELS)
        max_q = None

    print(f"Configuration:")
    print(f"  Poison rates: {prs}")
    print(f"  Models: {list(models_to_run.keys())}")
    print(f"  Max questions: {max_q or 'all'}")

    t_start = time.time()

    for model_tag, model_name in models_to_run.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_tag} ({model_name})")
        print(f"{'='*60}")

        for pr in prs:
            tag = _pr_tag(pr)
            out_path = CHECKPOINT_DIR / f"multimodel_qa_{model_tag}_{tag}.json"

            if out_path.exists():
                print(f"\n  [{tag}] Checkpoint exists, skipping: {out_path.name}")
                continue

            print(f"\n  [{tag}] Loading v1b data...")
            questions, pid_to_text, retrieval_results = load_v1b_data(tag)

            print(f"  [{tag}] Extracting answers...")
            extractions = extract_with_model(
                model_name, questions, retrieval_results, pid_to_text,
                max_questions=max_q,
            )

            with open(str(out_path), "w") as f:
                json.dump(extractions, f)
            print(f"  [{tag}] Saved: {out_path.name} ({len(extractions)} questions)")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE — {elapsed:.1f}s total")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
