#!/usr/bin/env python3
"""
Run self-consistency (SC) experiments: 7 models × 5 runs × 3 aggregations.

Each model runs in a separate subprocess for clean GPU memory management.
Within each subprocess, the model is loaded once and reranking is performed
5 times with temperature=0.7. Per-query scores are averaged across runs,
then evaluated.

Usage:
    python scripts/run_all_sc.py                 # run all 7 models
    python scripts/run_all_sc.py --worker 0      # run model index 0 only
    python scripts/run_all_sc.py --start 2       # resume from model index 2
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
    TOP_K_CANDIDATES, SC_NUM_RUNS,
)
from src.data_loader import load_labels, load_interleaved
from src.aggregation import compute_dcg_ratio, compute_mrr_ratio, compute_majority_at_k
from src.metrics import evaluate_all
from src.run_utils import setup_logging, append_to_summary

HF_CACHE = QWEN_7B_CACHE_DIR

SC_TEMPERATURE = 0.7

# ── Model configurations ──────────────────────────────────────────────────────
MODEL_CONFIGS = [
    # ── Qwen 2.5 Pairwise ────────────────────────────────────────────────────
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
    # ── Qwen 2.5 Setwise base ────────────────────────────────────────────────
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
    # ── Qwen 2.5 Setwise Rank-R1 v0.1 ────────────────────────────────────────
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
    # ── Qwen 2.5 Setwise SFT ─────────────────────────────────────────────────
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
    # ── Qwen 3 Pairwise ──────────────────────────────────────────────────────
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
    # ── Qwen 3 Setwise base (thinking disabled) ──────────────────────────────
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
    # ── Qwen 3 Setwise with thinking ─────────────────────────────────────────
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
    # ── LLaMA 3.1 ────────────────────────────────────────────────────────────
    {
        "name": "Pairwise-Llama-8B",
        "method": "pairwise",
        "model": LLAMA_8B_MODEL,
        "lora": None,
        "prompt": None,
    },
    {
        "name": "Setwise-Llama-8B",
        "method": "setwise",
        "model": LLAMA_8B_MODEL,
        "lora": None,
        "prompt": "base",
    },
    # ── Mistral ──────────────────────────────────────────────────────────────
    {
        "name": "Pairwise-Mistral-7B",
        "method": "pairwise",
        "model": MISTRAL_7B_MODEL,
        "lora": None,
        "prompt": None,
    },
    {
        "name": "Setwise-Mistral-7B",
        "method": "setwise",
        "model": MISTRAL_7B_MODEL,
        "lora": None,
        "prompt": "base",
    },
    {
        "name": "Pairwise-Mistral-24B",
        "method": "pairwise",
        "model": MISTRAL_24B_MODEL,
        "lora": None,
        "prompt": None,
    },
    {
        "name": "Setwise-Mistral-24B",
        "method": "setwise",
        "model": MISTRAL_24B_MODEL,
        "lora": None,
        "prompt": "base",
    },
    # ── RankZephyr ───────────────────────────────────────────────────────────
    {
        "name": "Pairwise-RankZephyr-7B",
        "method": "pairwise",
        "model": RANKZEPHYR_7B_MODEL,
        "lora": None,
        "prompt": None,
    },
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
    """Run one model config: SC_NUM_RUNS reranking passes, average, evaluate."""
    config = MODEL_CONFIGS[config_idx]
    name = config["name"]
    sc_name = f"SC-{name}"

    logger = setup_logging(f"sc_{name}")
    logger.info("=" * 60)
    logger.info(f"Self-Consistency config: {sc_name}")
    logger.info(f"  method      : {config['method']}")
    logger.info(f"  model       : {config['model']}")
    logger.info(f"  lora        : {config['lora']}")
    logger.info(f"  prompt      : {config['prompt']}")
    logger.info(f"  temperature : {SC_TEMPERATURE}")
    logger.info(f"  num_runs    : {SC_NUM_RUNS}")
    logger.info("=" * 60)

    # Load data
    labels = load_labels()
    inter = load_interleaved()

    label_qids = set(labels["query_id"].tolist())
    inter_qids = set(inter["qid"].unique().tolist())
    common_qids = sorted(label_qids & inter_qids)
    logger.info(f"Queries: {len(common_qids)}")

    # Create ranker with temperature
    t0 = time.time()
    if config["method"] == "pairwise":
        from src.posthoc.pairwise import PairwiseReranker
        ranker = PairwiseReranker(
            model_name_or_path=config["model"],
            cache_dir=HF_CACHE,
            temperature=SC_TEMPERATURE,
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
            temperature=SC_TEMPERATURE,
        )
    logger.info(f"Ranker created in {time.time() - t0:.1f}s")

    # ── Run SC_NUM_RUNS passes ───────────────────────────────────────────────
    # all_scores[qid][run_idx] = {"dcg_score": ..., "mrr_score": ..., "majority_at_k_score": ...}
    all_scores = {qid: [] for qid in common_qids}
    all_rankings = {qid: [] for qid in common_qids}
    all_traces = {}

    for run_idx in range(SC_NUM_RUNS):
        logger.info(f"--- SC run {run_idx + 1}/{SC_NUM_RUNS} ---")
        for qid in tqdm(common_qids, desc=f"{sc_name} run {run_idx+1}"):
            rows = inter[inter["qid"] == qid].head(TOP_K_CANDIDATES)
            if len(rows) == 0:
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
            all_rankings[qid].append(ranked_ids)

            run_scores = {}
            for agg_name, col_name, agg_fn in AGGREGATIONS:
                run_scores[col_name] = agg_fn(ranked_ids, origin_labels)
            all_scores[qid].append(run_scores)

            # Capture reasoning traces for setwise models
            if config["method"] == "setwise":
                if qid not in all_traces:
                    all_traces[qid] = []
                all_traces[qid].append({
                    "run_idx": run_idx,
                    "reasoning_trace": getattr(ranker, "last_reasoning_trace", ""),
                    "all_completions": json.dumps(
                        getattr(ranker._ranker, "all_completions", [])
                    ),
                    "completion_tokens": getattr(ranker, "total_completion_tokens", 0),
                    "prompt_tokens": getattr(ranker, "total_prompt_tokens", 0),
                })

        logger.info(f"  Run {run_idx + 1} complete")

    logger.info(f"All {SC_NUM_RUNS} runs complete in {time.time() - t0:.1f}s")

    # ── Average scores across runs ───────────────────────────────────────────
    results = []
    for qid in common_qids:
        if not all_scores[qid]:
            continue
        row = {"query_id": qid}
        for _, col_name, _ in AGGREGATIONS:
            scores = [s[col_name] for s in all_scores[qid] if not np.isnan(s[col_name])]
            row[col_name] = np.mean(scores) if scores else float("nan")
        # Store all rankings for reference
        row["all_rankings"] = json.dumps(all_rankings[qid])
        results.append(row)

    # ── Save per-query averaged scores ───────────────────────────────────────
    results_dir = PROJECT_ROOT / "results" / "posthoc"
    results_dir.mkdir(parents=True, exist_ok=True)

    scores_df = pd.DataFrame([
        {
            "query_id": r["query_id"],
            "dcg_score": r["dcg_score"],
            "mrr_score": r["mrr_score"],
            "majority_at_k_score": r["majority_at_k_score"],
        }
        for r in results
    ])
    scores_path = results_dir / f"{sc_name}_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    logger.info(f"Saved per-query SC scores → {scores_path}")

    # ── Save reasoning traces for setwise models ──────────────────────────────
    if config["method"] == "setwise" and all_traces:
        trace_rows = []
        for qid, traces in all_traces.items():
            for t in traces:
                trace_rows.append({
                    "query_id": qid,
                    "run_idx": t["run_idx"],
                    "reasoning_trace": t["reasoning_trace"],
                    "all_completions": t["all_completions"],
                    "completion_tokens": t["completion_tokens"],
                    "prompt_tokens": t["prompt_tokens"],
                })
        if trace_rows:
            traces_df = pd.DataFrame(trace_rows)
            traces_path = results_dir / f"{sc_name}_reasoning.csv"
            traces_df.to_csv(traces_path, index=False)
            logger.info(f"Saved SC reasoning traces → {traces_path}")

    # ── Evaluate all 3 aggregation methods ───────────────────────────────────
    for agg_name, col_name, _ in AGGREGATIONS:
        method_name = f"{sc_name}-{agg_name}"
        scores_series = pd.Series(
            {r["query_id"]: r[col_name] for r in results}
        )
        scores_series.index.name = "query_id"

        eval_result = evaluate_all(
            labels_df=labels,
            scores_series=scores_series,
            method_name=method_name,
            stage="posthoc-sc",
            predictor=f"SC {config['method']} {config['model'].split('/')[-1]} {agg_name}",
        )

        append_to_summary(eval_result)
        logger.info(
            f"  {method_name}: TPR_H={eval_result['tpr_hurt']}, "
            f"TPR_B={eval_result['tpr_benefit']}, Acc={eval_result['acc']}, "
            f"AUC={eval_result['auc']}, Gamma={eval_result['gamma']} "
            f"(p={eval_result['gamma_p']}), Rho={eval_result['rho']} "
            f"(p={eval_result['rho_p']})"
        )

    logger.info(f"=== {sc_name} COMPLETE ===\n")


def main():
    parser = argparse.ArgumentParser(description="Run SC experiments")
    parser.add_argument("--worker", type=int, default=-1,
                        help="Run a specific model config index (0-6)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start from this config index (inclusive)")
    parser.add_argument("--end", type=int, default=len(MODEL_CONFIGS),
                        help="End at this config index (exclusive)")
    args = parser.parse_args()

    if args.worker >= 0:
        run_single_model(args.worker)
        return

    # Orchestrator mode
    total = args.end - args.start
    print(f"\n{'='*70}")
    print(f"  SC: Running {total} models × {SC_NUM_RUNS} runs × 3 aggregations")
    print(f"  Temperature: {SC_TEMPERATURE}")
    print(f"{'='*70}\n")

    timings = {}
    failures = []

    for i in range(args.start, args.end):
        config = MODEL_CONFIGS[i]
        print(f"\n{'─'*70}")
        print(f"  [{i+1}/{len(MODEL_CONFIGS)}] SC-{config['name']}")
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
            print(f"\n  ✗ SC-{config['name']} FAILED (exit code {result.returncode}) [{elapsed:.0f}s]")
        else:
            print(f"\n  ✓ SC-{config['name']} done [{elapsed:.0f}s]")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  ALL SC MODELS COMPLETE")
    print(f"{'='*70}\n")

    print("Timings:")
    for name, t in timings.items():
        status = "FAIL" if name in failures else "OK"
        print(f"  SC-{name:30s}  {t:7.0f}s  [{status}]")

    if failures:
        print(f"\nFailed models: {', '.join(failures)}")

    # Print summary table
    summary_path = PROJECT_ROOT / "results" / "summary_table.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        sc = df[df["stage"] == "posthoc-sc"].copy()
        if len(sc) > 0:
            print(f"\n{'='*70}")
            print("  EVALUATION RESULTS (self-consistency)")
            print(f"{'='*70}\n")
            display_cols = [
                "method_name", "tpr_hurt", "tpr_benefit", "acc", "auc",
                "gamma", "gamma_p", "rho", "rho_p",
            ]
            print(sc[display_cols].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
