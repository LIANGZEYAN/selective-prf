#!/usr/bin/env python3
"""
Run all 24 base post-hoc configurations (8 models × 3 aggregations).

Each model runs in a separate subprocess for clean GPU memory management.

Usage:
    python scripts/run_all_posthoc.py                 # run all 8 models
    python scripts/run_all_posthoc.py --worker 0      # run model index 0 only
    python scripts/run_all_posthoc.py --start 4       # resume from model index 4
"""

import sys
import os
import json
import argparse
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.config import (
    QWEN_7B_MODEL, QWEN_14B_MODEL,
    QWEN3_8B_MODEL, QWEN3_14B_MODEL,
    LLAMA_8B_MODEL,
    MISTRAL_7B_MODEL, MISTRAL_24B_MODEL,
    RANKZEPHYR_7B_MODEL,
    QWEN_7B_CACHE_DIR, RANK_R1_CACHE_DIR,
    RANK_R1_V02_32B_BASE, RANK_R1_V02_32B_LORA,
    TOP_K_CANDIDATES,
)
from src.data_loader import load_labels, load_interleaved
from src.aggregation import compute_dcg_ratio, compute_mrr_ratio, compute_majority_at_k
from src.metrics import evaluate_all
from src.run_utils import setup_logging, append_to_summary

HF_CACHE = QWEN_7B_CACHE_DIR  # same cache dir for all models

# ── Model configurations ──────────────────────────────────────────────────────
MODEL_CONFIGS = [
    # ── Qwen 2.5 Pairwise (transformers-based) ───────────────────────────────
    {
        "name": "Pairwise-Qwen-7B",
        "method": "pairwise",
        "model": QWEN_7B_MODEL,
        "lora": None,
        "prompt": None,
    },
    {
        "name": "Pairwise-Qwen-14B",
        "method": "pairwise",
        "model": QWEN_14B_MODEL,
        "lora": None,
        "prompt": None,
    },
    # ── Qwen 2.5 Setwise base models (vLLM) ──────────────────────────────────
    {
        "name": "Setwise-Qwen-7B",
        "method": "setwise",
        "model": QWEN_7B_MODEL,
        "lora": None,
        "prompt": "base",
    },
    {
        "name": "Setwise-Qwen-14B",
        "method": "setwise",
        "model": QWEN_14B_MODEL,
        "lora": None,
        "prompt": "base",
    },
    # ── Qwen 2.5 Setwise Rank-R1 v0.1 (vLLM + LoRA, with <think>) ───────────
    {
        "name": "Setwise-R1-7B",
        "method": "setwise",
        "model": QWEN_7B_MODEL,
        "lora": "ielabgroup/Rank-R1-7B-v0.1",
        "prompt": "r1",
    },
    {
        "name": "Setwise-R1-14B",
        "method": "setwise",
        "model": QWEN_14B_MODEL,
        "lora": "ielabgroup/Rank-R1-14B-v0.1",
        "prompt": "r1",
    },
    # ── Qwen 2.5 Setwise SFT (vLLM + LoRA, no <think>) ──────────────────────
    {
        "name": "Setwise-SFT-7B",
        "method": "setwise",
        "model": QWEN_7B_MODEL,
        "lora": "ielabgroup/Setwise-SFT-7B-v0.1",
        "prompt": "base",
    },
    {
        "name": "Setwise-SFT-14B",
        "method": "setwise",
        "model": QWEN_14B_MODEL,
        "lora": "ielabgroup/Setwise-SFT-14B-v0.1",
        "prompt": "base",
    },
    # ── Qwen 3 Pairwise (transformers-based) ─────────────────────────────────
    {
        "name": "Pairwise-Qwen3-8B",
        "method": "pairwise",
        "model": QWEN3_8B_MODEL,
        "lora": None,
        "prompt": None,
    },
    {
        "name": "Pairwise-Qwen3-14B",
        "method": "pairwise",
        "model": QWEN3_14B_MODEL,
        "lora": None,
        "prompt": None,
    },
    # ── Qwen 3 Setwise base models (vLLM, thinking disabled) ─────────────────
    {
        "name": "Setwise-Qwen3-8B",
        "method": "setwise",
        "model": QWEN3_8B_MODEL,
        "lora": None,
        "prompt": "base",
    },
    {
        "name": "Setwise-Qwen3-14B",
        "method": "setwise",
        "model": QWEN3_14B_MODEL,
        "lora": None,
        "prompt": "base",
    },
    # ── Qwen 3 Setwise with thinking (vLLM, thinking enabled) ────────────────
    {
        "name": "Setwise-Qwen3-8B-Think",
        "method": "setwise",
        "model": QWEN3_8B_MODEL,
        "lora": None,
        "prompt": "r1",
    },
    {
        "name": "Setwise-Qwen3-14B-Think",
        "method": "setwise",
        "model": QWEN3_14B_MODEL,
        "lora": None,
        "prompt": "r1",
    },
    # ── Rank-R1 v0.2 32B (LoRA adapter on Qwen3-32B) ─────────────────────────
    {
        "name": "Setwise-R1v02-32B",
        "method": "setwise",
        "model": RANK_R1_V02_32B_BASE,
        "lora": RANK_R1_V02_32B_LORA,
        "prompt": "r1-v02",
    },
    # ── LLaMA 3.1 Pairwise (transformers-based) ──────────────────────────────
    {
        "name": "Pairwise-Llama-8B",
        "method": "pairwise",
        "model": LLAMA_8B_MODEL,
        "lora": None,
        "prompt": None,
    },
    # ── LLaMA 3.1 Setwise (vLLM) ─────────────────────────────────────────────
    {
        "name": "Setwise-Llama-8B",
        "method": "setwise",
        "model": LLAMA_8B_MODEL,
        "lora": None,
        "prompt": "base",
    },
    # ── Mistral 7B Pairwise (transformers-based) ─────────────────────────────
    {
        "name": "Pairwise-Mistral-7B",
        "method": "pairwise",
        "model": MISTRAL_7B_MODEL,
        "lora": None,
        "prompt": None,
    },
    # ── Mistral 7B Setwise (vLLM) ────────────────────────────────────────────
    {
        "name": "Setwise-Mistral-7B",
        "method": "setwise",
        "model": MISTRAL_7B_MODEL,
        "lora": None,
        "prompt": "base",
    },
    # ── Mistral-Small 24B Pairwise (transformers-based) ──────────────────────
    {
        "name": "Pairwise-Mistral-24B",
        "method": "pairwise",
        "model": MISTRAL_24B_MODEL,
        "lora": None,
        "prompt": None,
    },
    # ── Mistral-Small 24B Setwise (vLLM) ─────────────────────────────────────
    {
        "name": "Setwise-Mistral-24B",
        "method": "setwise",
        "model": MISTRAL_24B_MODEL,
        "lora": None,
        "prompt": "base",
    },
    # ── RankZephyr 7B Pairwise (Mistral-based, ranking-trained) ──────────────
    {
        "name": "Pairwise-RankZephyr-7B",
        "method": "pairwise",
        "model": RANKZEPHYR_7B_MODEL,
        "lora": None,
        "prompt": None,
    },
    # ── RankZephyr 7B Setwise (vLLM) ─────────────────────────────────────────
    {
        "name": "Setwise-RankZephyr-7B",
        "method": "setwise",
        "model": RANKZEPHYR_7B_MODEL,
        "lora": None,
        "prompt": "base",
    },
]

AGGREGATIONS = [
    ("DCG", "dcg_score", compute_dcg_ratio),
    ("MRR", "mrr_score", compute_mrr_ratio),
    ("Majority@k", "majority_at_k_score", compute_majority_at_k),
]


def run_single_model(config_idx: int):
    """Run one model config: rerank all queries, compute 3 aggregations, evaluate."""
    config = MODEL_CONFIGS[config_idx]
    name = config["name"]

    logger = setup_logging(f"posthoc_{name}")
    logger.info("=" * 60)
    logger.info(f"Model config: {name}")
    logger.info(f"  method : {config['method']}")
    logger.info(f"  model  : {config['model']}")
    logger.info(f"  lora   : {config['lora']}")
    logger.info(f"  prompt : {config['prompt']}")
    logger.info("=" * 60)

    # Load data
    labels = load_labels()
    inter = load_interleaved()

    # Use all queries that have labels AND interleaved data
    label_qids = set(labels["query_id"].tolist())
    inter_qids = set(inter["qid"].unique().tolist())
    common_qids = sorted(label_qids & inter_qids)
    logger.info(f"Queries: {len(common_qids)} (labels={len(label_qids)}, interleaved={len(inter_qids)})")

    # Create ranker
    t0 = time.time()
    if config["method"] == "pairwise":
        from src.posthoc.pairwise import PairwiseReranker
        ranker = PairwiseReranker(
            model_name_or_path=config["model"],
            cache_dir=HF_CACHE,
        )
    else:
        from src.posthoc.setwise import SetwiseReranker, PROMPT_R1, PROMPT_BASE, PROMPT_R1_V02
        prompt_map = {"base": PROMPT_BASE, "r1": PROMPT_R1, "r1-v02": PROMPT_R1_V02}
        prompt_file = str(prompt_map.get(config["prompt"], PROMPT_BASE))
        ranker = SetwiseReranker(
            model_name_or_path=config["model"],
            lora_path=config["lora"],
            prompt_file=prompt_file,
            cache_dir=HF_CACHE,
        )
    logger.info(f"Ranker created in {time.time() - t0:.1f}s")

    # Rerank all queries
    results = []
    for qid in tqdm(common_qids, desc=name):
        rows = inter[inter["qid"] == qid].head(TOP_K_CANDIDATES)
        if len(rows) == 0:
            logger.warning(f"No interleaved docs for qid={qid}, skipping")
            continue

        query_text = rows.iloc[0]["query_text"]
        doc_list = [
            {
                "docid": r["docno"],
                "text": str(r["passage_text"])[:400],
                "origin_label": r.get("origin_label", ""),
            }
            for _, r in rows.iterrows()
        ]
        origin_labels = {r["docno"]: r["origin_label"] for _, r in rows.iterrows()}

        reranked = ranker.rerank(query_text, doc_list)
        ranked_ids = [doc_id for doc_id, _ in reranked]

        # Compute all 3 aggregation scores from the same reranked list
        row = {"query_id": qid, "ranked_doc_ids": ranked_ids}
        for agg_name, col_name, agg_fn in AGGREGATIONS:
            row[col_name] = agg_fn(ranked_ids, origin_labels)

        # Capture reasoning traces for setwise models
        if config["method"] == "setwise":
            row["reasoning_trace"] = getattr(ranker, "last_reasoning_trace", "")
            row["all_completions"] = json.dumps(
                getattr(ranker._ranker, "all_completions", [])
            )
            row["completion_tokens"] = getattr(ranker, "total_completion_tokens", 0)
            row["prompt_tokens"] = getattr(ranker, "total_prompt_tokens", 0)

        results.append(row)
        logger.info(
            f"qid={qid}: DCG={row['dcg_score']:.4f}, "
            f"MRR={row['mrr_score']:.4f}, Maj@k={row['majority_at_k_score']:.4f}"
        )

    logger.info(f"Reranked {len(results)} queries in {time.time() - t0:.1f}s total")

    # ── Save per-query scores ─────────────────────────────────────────────────
    results_dir = PROJECT_ROOT / "results" / "posthoc"
    results_dir.mkdir(parents=True, exist_ok=True)

    scores_df = pd.DataFrame([
        {
            "query_id": r["query_id"],
            "dcg_score": r["dcg_score"],
            "mrr_score": r["mrr_score"],
            "majority_at_k_score": r["majority_at_k_score"],
            "ranked_doc_ids": json.dumps([str(x) for x in r["ranked_doc_ids"]]),
        }
        for r in results
    ])
    scores_path = results_dir / f"{name}_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    logger.info(f"Saved per-query scores → {scores_path}")

    # ── Save reasoning traces for setwise models ──────────────────────────────
    if config["method"] == "setwise" and any(r.get("reasoning_trace") for r in results):
        traces_df = pd.DataFrame([
            {
                "query_id": r["query_id"],
                "reasoning_trace": r.get("reasoning_trace", ""),
                "all_completions": r.get("all_completions", "[]"),
                "completion_tokens": r.get("completion_tokens", 0),
                "prompt_tokens": r.get("prompt_tokens", 0),
            }
            for r in results
        ])
        traces_path = results_dir / f"{name}_reasoning.csv"
        traces_df.to_csv(traces_path, index=False)
        logger.info(f"Saved reasoning traces → {traces_path}")

    # ── Evaluate all 3 aggregation methods ────────────────────────────────────
    for agg_name, col_name, _ in AGGREGATIONS:
        method_name = f"{name}-{agg_name}"
        scores_series = pd.Series(
            {r["query_id"]: r[col_name] for r in results}
        )
        scores_series.index.name = "query_id"

        eval_result = evaluate_all(
            labels_df=labels,
            scores_series=scores_series,
            method_name=method_name,
            stage="posthoc",
            predictor=f"{config['method']} {config['model'].split('/')[-1]} {agg_name}",
        )

        append_to_summary(eval_result)
        logger.info(
            f"  {method_name}: TPR_H={eval_result['tpr_hurt']}, "
            f"TPR_B={eval_result['tpr_benefit']}, Acc={eval_result['acc']}, "
            f"AUC={eval_result['auc']}, Gamma={eval_result['gamma']} "
            f"(p={eval_result['gamma_p']}), Rho={eval_result['rho']} "
            f"(p={eval_result['rho_p']})"
        )

    logger.info(f"=== {name} COMPLETE ===\n")


def main():
    parser = argparse.ArgumentParser(description="Run all 24 post-hoc configurations")
    parser.add_argument("--worker", type=int, default=-1,
                        help="Run a specific model config index (0-7)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start from this config index (inclusive)")
    parser.add_argument("--end", type=int, default=len(MODEL_CONFIGS),
                        help="End at this config index (exclusive)")
    args = parser.parse_args()

    if args.worker >= 0:
        # Worker mode: run one model in current process
        run_single_model(args.worker)
        return

    # Orchestrator mode: launch each model as a separate subprocess
    total = args.end - args.start
    print(f"\n{'='*70}")
    print(f"  Running {total} model configs × 3 aggregations = {total * 3} evaluations")
    print(f"{'='*70}\n")

    timings = {}
    failures = []

    for i in range(args.start, args.end):
        config = MODEL_CONFIGS[i]
        print(f"\n{'─'*70}")
        print(f"  [{i+1}/{len(MODEL_CONFIGS)}] {config['name']}")
        print(f"  Model: {config['model']}")
        print(f"  LoRA:  {config['lora'] or '(none)'}")
        print(f"{'─'*70}\n")

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(i)],
            cwd=str(PROJECT_ROOT),
        )
        elapsed = time.time() - t0
        timings[config["name"]] = elapsed

        if result.returncode != 0:
            failures.append(config["name"])
            print(f"\n  ✗ {config['name']} FAILED (exit code {result.returncode}) [{elapsed:.0f}s]")
        else:
            print(f"\n  ✓ {config['name']} done [{elapsed:.0f}s]")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  ALL MODELS COMPLETE")
    print(f"{'='*70}\n")

    print("Timings:")
    for name, t in timings.items():
        status = "FAIL" if name in failures else "OK"
        print(f"  {name:30s}  {t:7.0f}s  [{status}]")

    if failures:
        print(f"\nFailed models: {', '.join(failures)}")

    # Print summary table
    summary_path = PROJECT_ROOT / "results" / "summary_table.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        posthoc = df[df["stage"] == "posthoc"].copy()
        if len(posthoc) > 0:
            print(f"\n{'='*70}")
            print("  EVALUATION RESULTS (post-hoc)")
            print(f"{'='*70}\n")
            display_cols = [
                "method_name", "tpr_hurt", "tpr_benefit", "acc", "auc",
                "gamma", "gamma_p", "rho", "rho_p",
            ]
            print(posthoc[display_cols].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
