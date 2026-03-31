#!/usr/bin/env python3
"""
Run Qwen-7B local pairwise reranking: DCG ratio, MRR ratio, Majority@k (top-5).
Requires CUDA GPU.
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT as ROOT, POSTHOC_RESULTS_DIR
from src.data_loader import load_labels, load_interleaved, get_query_texts
from src.aggregation import compute_dcg_ratio, compute_mrr_ratio, compute_majority_at_k
from src.metrics import evaluate_all, classify_scores
from src.run_utils import setup_logging, save_per_query_results, append_to_summary


def _build_doc_list(q_df: pd.DataFrame, truncate_fn) -> tuple:
    """Build doc_list and origin_labels from a query's interleaved DataFrame."""
    doc_list = []
    origin_labels = {}
    for _, row in q_df.iterrows():
        text = truncate_fn(str(row["passage_text"]) if pd.notna(row["passage_text"]) else "")
        doc_list.append({"docid": row["docno"], "text": text, "origin_label": row["origin_label"]})
        origin_labels[row["docno"]] = row["origin_label"]
    return doc_list, origin_labels


def main() -> None:
    logger = setup_logging("run_pairwise_7b")

    labels_df = load_labels()
    inter_df = load_interleaved()
    query_texts = get_query_texts(inter_df)
    query_ids = sorted(inter_df["qid"].unique().tolist())

    logger.info("Initialising Qwen-7B local reranker...")
    from src.posthoc.pairwise import PairwiseReranker
    reranker = PairwiseReranker()

    results_dcg = []
    results_mrr = []
    results_uw = []

    for qid in tqdm(query_ids, desc="Qwen-7B Pairwise"):
        q_df = inter_df[inter_df["qid"] == qid].sort_values("rank")
        query = reranker.truncate_query(q_df["query_text"].iloc[0])
        doc_list, origin_labels = _build_doc_list(q_df, reranker.truncate_passage)

        ranked_ids = reranker.rerank_return_ids(query, doc_list)

        dcg = compute_dcg_ratio(ranked_ids, origin_labels)
        mrr = compute_mrr_ratio(ranked_ids, origin_labels)
        uw = compute_majority_at_k(ranked_ids, origin_labels)

        pred_dcg = classify_scores(pd.Series([dcg])).iloc[0]
        pred_mrr = classify_scores(pd.Series([mrr])).iloc[0]
        pred_uw = classify_scores(pd.Series([uw])).iloc[0]

        results_dcg.append({"query_id": qid, "score": dcg, "prediction": pred_dcg})
        results_mrr.append({"query_id": qid, "score": mrr, "prediction": pred_mrr})
        results_uw.append({"query_id": qid, "score": uw, "prediction": pred_uw})

        logger.info("qid=%s  DCG=%.4f  MRR=%.4f  UW=%.4f  [%d cmps]",
                     qid, dcg, mrr, uw, reranker.total_compare)

    for name, results in [
        ("Qwen-7B-DCG", results_dcg),
        ("Qwen-7B-MRR", results_mrr),
        ("Qwen-7B-Majority@k", results_uw),
    ]:
        out_path = ROOT / POSTHOC_RESULTS_DIR / f"{name.lower().replace('-', '_')}.csv"
        save_per_query_results(results, out_path)

        scores = pd.Series({r["query_id"]: r["score"] for r in results})
        metrics = evaluate_all(labels_df, scores, name, "posthoc", f"Qwen-7B {name.split('-')[-1]}")
        append_to_summary(metrics)
        logger.info("%s — Acc=%.4f  AUC=%.4f", name, metrics["acc"], metrics["auc"])

    logger.info("Qwen-7B pairwise complete.")


if __name__ == "__main__":
    main()
