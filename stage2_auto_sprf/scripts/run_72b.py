#!/usr/bin/env python3
"""
Run Qwen-72B API reranking: pairwise + setwise, each with DCG/MRR/Majority@k.
Requires IDA_LLM_API_KEY environment variable.

Usage:
    python scripts/run_72b.py                  # run both pairwise and setwise
    python scripts/run_72b.py --method pairwise # pairwise only
    python scripts/run_72b.py --method setwise  # setwise only
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT as ROOT, POSTHOC_RESULTS_DIR, TOP_K_CANDIDATES
from src.data_loader import load_labels, load_interleaved
from src.aggregation import compute_dcg_ratio, compute_mrr_ratio, compute_majority_at_k
from src.metrics import evaluate_all
from src.run_utils import setup_logging, append_to_summary

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


def run_method(method: str, inter_df: pd.DataFrame, labels_df: pd.DataFrame, logger):
    """Run one method (pairwise or setwise) with all 3 aggregations."""
    query_ids = sorted(inter_df["qid"].unique().tolist())

    if method == "pairwise":
        from src.posthoc.pairwise_api import PairwiseApiReranker
        reranker = PairwiseApiReranker()
        method_prefix = "Pairwise-Qwen-72B"
    else:
        from src.posthoc.setwise_api import SetwiseApiReranker
        reranker = SetwiseApiReranker()
        method_prefix = "Setwise-Qwen-72B"

    logger.info("Running %s with %d queries", method_prefix, len(query_ids))

    # Collect per-query scores for all aggregations
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
            predictor=f"Qwen-72B {method} {agg_name}",
        )
        append_to_summary(metrics)
        logger.info(
            "  %s: TPR_H=%s, TPR_B=%s, Acc=%s, AUC=%s, Gamma=%s (p=%s)",
            eval_name, metrics["tpr_hurt"], metrics["tpr_benefit"],
            metrics["acc"], metrics["auc"], metrics["gamma"], metrics["gamma_p"],
        )

    logger.info("%s complete.", method_prefix)


def main():
    parser = argparse.ArgumentParser(description="Run Qwen-72B API reranking")
    parser.add_argument("--method", choices=["pairwise", "setwise", "both"],
                        default="both", help="Which method to run")
    args = parser.parse_args()

    logger = setup_logging("run_72b")
    labels_df = load_labels()
    inter_df = load_interleaved()

    methods = ["pairwise", "setwise"] if args.method == "both" else [args.method]

    for method in methods:
        run_method(method, inter_df, labels_df, logger)

    logger.info("All 72B methods complete.")


if __name__ == "__main__":
    main()
