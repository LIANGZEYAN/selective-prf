#!/usr/bin/env python3
"""
Run Llama-3.3-70B API reranking: pairwise + setwise, each with DCG/MRR/Majority@k.
Then run self-consistency (5 runs, temp=0.7) for both methods.

Uses the university OpenAI-compatible API endpoint.

Usage:
    python scripts/run_llama70b.py                      # run all 4 (pairwise, setwise, SC-pairwise, SC-setwise)
    python scripts/run_llama70b.py --method pairwise     # pairwise only
    python scripts/run_llama70b.py --method setwise      # setwise only
    python scripts/run_llama70b.py --method sc-pairwise  # SC pairwise only
    python scripts/run_llama70b.py --method sc-setwise   # SC setwise only
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROJECT_ROOT as ROOT,
    POSTHOC_RESULTS_DIR,
    TOP_K_CANDIDATES,
    SC_NUM_RUNS,
    QWEN_72B_API_URL,
)
from src.data_loader import load_labels, load_interleaved
from src.aggregation import compute_dcg_ratio, compute_mrr_ratio, compute_majority_at_k
from src.metrics import evaluate_all
from src.run_utils import setup_logging, append_to_summary

# ── Llama-3.3-70B API config ────────────────────────────────────────────────
LLAMA70B_API_URL = QWEN_72B_API_URL  # same university endpoint
LLAMA70B_API_MODEL = "llama-3.3-70b-instruct"
LLAMA70B_API_KEY = "ida_TbtBzRyj8HeV7kmxkeRsgImHnYelRIUMQ3vW9VIf"

SC_TEMPERATURE = 0.7

AGGREGATIONS = [
    ("DCG", "dcg_score", compute_dcg_ratio),
    ("MRR", "mrr_score", compute_mrr_ratio),
    ("Majority@k", "majority_at_k_score", compute_majority_at_k),
]


def _build_doc_list(q_df: pd.DataFrame, truncate_fn) -> tuple:
    doc_list = []
    origin_labels = {}
    for _, row in q_df.iterrows():
        text = truncate_fn(str(row["passage_text"]) if pd.notna(row["passage_text"]) else "")
        doc_list.append({"docid": row["docno"], "text": text, "origin_label": row["origin_label"]})
        origin_labels[row["docno"]] = row["origin_label"]
    return doc_list, origin_labels


def run_base(method: str, inter_df, labels_df, logger):
    """Run one method (pairwise or setwise) with all 3 aggregations."""
    query_ids = sorted(inter_df["qid"].unique().tolist())

    if method == "pairwise":
        from src.posthoc.pairwise_api import PairwiseApiReranker
        reranker = PairwiseApiReranker(
            api_key=LLAMA70B_API_KEY,
            base_url=LLAMA70B_API_URL,
            model=LLAMA70B_API_MODEL,
        )
        method_prefix = "Pairwise-Llama-70B"
    else:
        from src.posthoc.setwise_api import SetwiseApiReranker
        reranker = SetwiseApiReranker(
            api_key=LLAMA70B_API_KEY,
            base_url=LLAMA70B_API_URL,
            model=LLAMA70B_API_MODEL,
        )
        method_prefix = "Setwise-Llama-70B"

    logger.info("Running %s with %d queries", method_prefix, len(query_ids))

    results = []
    for qid in tqdm(query_ids, desc=method_prefix):
        q_df = inter_df[inter_df["qid"] == qid].sort_values("rank").head(TOP_K_CANDIDATES)
        if len(q_df) == 0:
            continue

        query = reranker.truncate_query(q_df["query_text"].iloc[0])
        doc_list, origin_labels = _build_doc_list(q_df, reranker.truncate_passage)

        reranked = reranker.rerank(query, doc_list)
        ranked_ids = [doc_id for doc_id, _ in reranked]

        row = {"query_id": qid}
        for agg_name, col_name, agg_fn in AGGREGATIONS:
            row[col_name] = agg_fn(ranked_ids, origin_labels)
        results.append(row)

        logger.info(
            "qid=%s  DCG=%.4f  MRR=%.4f  UW=%.4f  [%d cmps]",
            qid, row["dcg_score"], row["mrr_score"], row["majority_at_k_score"],
            reranker.total_compare,
        )

    # Save per-query scores
    results_dir = ROOT / POSTHOC_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    scores_df = pd.DataFrame(results)
    scores_path = results_dir / f"{method_prefix}_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    logger.info("Saved per-query scores → %s", scores_path)

    # Evaluate each aggregation
    for agg_name, col_name, _ in AGGREGATIONS:
        eval_name = f"{method_prefix}-{agg_name}"
        scores_series = pd.Series({r["query_id"]: r[col_name] for r in results})
        scores_series.index.name = "query_id"

        metrics = evaluate_all(
            labels_df=labels_df,
            scores_series=scores_series,
            method_name=eval_name,
            stage="posthoc",
            predictor=f"Llama-3.3-70B {method} {agg_name}",
        )
        append_to_summary(metrics)
        logger.info(
            "  %s: TPR_H=%s, TPR_B=%s, Acc=%s, AUC=%s, Gamma=%s (p=%s)",
            eval_name, metrics["tpr_hurt"], metrics["tpr_benefit"],
            metrics["acc"], metrics["auc"], metrics["gamma"], metrics["gamma_p"],
        )

    logger.info("%s complete.", method_prefix)


def run_sc(method: str, inter_df, labels_df, logger):
    """Run self-consistency for one method (pairwise or setwise)."""
    query_ids = sorted(inter_df["qid"].unique().tolist())

    if method == "pairwise":
        from src.posthoc.pairwise_api import PairwiseApiReranker
        reranker = PairwiseApiReranker(
            api_key=LLAMA70B_API_KEY,
            base_url=LLAMA70B_API_URL,
            model=LLAMA70B_API_MODEL,
            temperature=SC_TEMPERATURE,
        )
        method_prefix = "SC-Pairwise-Llama-70B"
    else:
        from src.posthoc.setwise_api import SetwiseApiReranker
        reranker = SetwiseApiReranker(
            api_key=LLAMA70B_API_KEY,
            base_url=LLAMA70B_API_URL,
            model=LLAMA70B_API_MODEL,
            temperature=SC_TEMPERATURE,
        )
        method_prefix = "SC-Setwise-Llama-70B"

    logger.info("Running %s with %d queries, R=%d, temp=%.1f",
                method_prefix, len(query_ids), SC_NUM_RUNS, SC_TEMPERATURE)

    all_scores = {qid: [] for qid in query_ids}

    t0 = time.time()
    for run_idx in range(SC_NUM_RUNS):
        logger.info("--- %s run %d/%d ---", method_prefix, run_idx + 1, SC_NUM_RUNS)
        for qid in tqdm(query_ids, desc=f"{method_prefix} run {run_idx+1}"):
            q_df = inter_df[inter_df["qid"] == qid].sort_values("rank").head(TOP_K_CANDIDATES)
            if len(q_df) == 0:
                continue

            query = reranker.truncate_query(q_df["query_text"].iloc[0])
            doc_list, origin_labels = _build_doc_list(q_df, reranker.truncate_passage)

            reranked = reranker.rerank(query, doc_list)
            ranked_ids = [doc_id for doc_id, _ in reranked]

            run_scores = {}
            for agg_name, col_name, agg_fn in AGGREGATIONS:
                run_scores[col_name] = agg_fn(ranked_ids, origin_labels)
            all_scores[qid].append(run_scores)

        logger.info("  Run %d complete", run_idx + 1)

    logger.info("All %d runs complete in %.1fs", SC_NUM_RUNS, time.time() - t0)

    # Average scores across runs
    results = []
    for qid in query_ids:
        if not all_scores[qid]:
            continue
        row = {"query_id": qid}
        for _, col_name, _ in AGGREGATIONS:
            scores = [s[col_name] for s in all_scores[qid] if not np.isnan(s[col_name])]
            row[col_name] = np.mean(scores) if scores else float("nan")
        results.append(row)

    # Save per-query averaged scores
    results_dir = ROOT / POSTHOC_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    scores_df = pd.DataFrame([
        {"query_id": r["query_id"], "dcg_score": r["dcg_score"],
         "mrr_score": r["mrr_score"], "majority_at_k_score": r["majority_at_k_score"]}
        for r in results
    ])
    scores_path = results_dir / f"{method_prefix}_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    logger.info("Saved per-query SC scores → %s", scores_path)

    # Evaluate all 3 aggregation methods
    for agg_name, col_name, _ in AGGREGATIONS:
        eval_name = f"{method_prefix}-{agg_name}"
        scores_series = pd.Series({r["query_id"]: r[col_name] for r in results})
        scores_series.index.name = "query_id"

        metrics = evaluate_all(
            labels_df=labels_df,
            scores_series=scores_series,
            method_name=eval_name,
            stage="posthoc-sc",
            predictor=f"SC Llama-3.3-70B {method} {agg_name}",
        )
        append_to_summary(metrics)
        logger.info(
            "  %s: TPR_H=%s, TPR_B=%s, Acc=%s, AUC=%s, Gamma=%s (p=%s)",
            eval_name, metrics["tpr_hurt"], metrics["tpr_benefit"],
            metrics["acc"], metrics["auc"], metrics["gamma"], metrics["gamma_p"],
        )

    logger.info("%s complete.", method_prefix)


def main():
    parser = argparse.ArgumentParser(description="Run Llama-3.3-70B API reranking")
    parser.add_argument(
        "--method",
        choices=["pairwise", "setwise", "sc-pairwise", "sc-setwise", "all"],
        default="all",
        help="Which method to run",
    )
    args = parser.parse_args()

    logger = setup_logging("run_llama70b")
    labels_df = load_labels()
    inter_df = load_interleaved()

    if args.method == "all":
        methods = ["pairwise", "setwise", "sc-pairwise", "sc-setwise"]
    else:
        methods = [args.method]

    for method in methods:
        if method == "pairwise":
            run_base("pairwise", inter_df, labels_df, logger)
        elif method == "setwise":
            run_base("setwise", inter_df, labels_df, logger)
        elif method == "sc-pairwise":
            run_sc("pairwise", inter_df, labels_df, logger)
        elif method == "sc-setwise":
            run_sc("setwise", inter_df, labels_df, logger)

    logger.info("All Llama-3.3-70B methods complete.")


if __name__ == "__main__":
    main()
