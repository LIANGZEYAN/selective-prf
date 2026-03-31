#!/usr/bin/env python3
"""
Run Rank-R1 setwise reranking for all 4 models.
Computes DCG ratio + reasoning trace mining features.
Requires CUDA GPU and vLLM.
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT as ROOT, POSTHOC_RESULTS_DIR, RANK_R1_MODELS
from src.data_loader import load_labels, load_interleaved
from src.aggregation import compute_dcg_ratio
from src.posthoc.ensemble import reasoning_drift_score, reasoning_length_ratio, load_drift_lexicon
from src.metrics import evaluate_all, classify_scores
from src.run_utils import setup_logging, save_per_query_results, append_to_summary


def _build_doc_list(q_df: pd.DataFrame, truncate_fn) -> tuple:
    doc_list = []
    origin_labels = {}
    for _, row in q_df.iterrows():
        text = truncate_fn(str(row["passage_text"]) if pd.notna(row["passage_text"]) else "")
        doc_list.append({"docid": row["docno"], "text": text, "origin_label": row["origin_label"]})
        origin_labels[row["docno"]] = row["origin_label"]
    return doc_list, origin_labels


def run_one_model(model_key: str, inter_df: pd.DataFrame, labels_df: pd.DataFrame, logger) -> None:
    """Run a single Rank-R1 model variant."""
    from src.posthoc.setwise import SetwiseReranker

    logger.info("Loading Rank-R1 model: %s (%s)", model_key, RANK_R1_MODELS[model_key])
    reranker = SetwiseReranker(model_key=model_key)

    drift_lexicon = load_drift_lexicon()
    query_ids = sorted(inter_df["qid"].unique().tolist())

    results_dcg = []
    results_drift = []
    results_len_ratio = []

    for qid in tqdm(query_ids, desc=f"R1-{model_key}"):
        q_df = inter_df[inter_df["qid"] == qid].sort_values("rank")
        query = reranker.truncate_query(q_df["query_text"].iloc[0])
        doc_list, origin_labels = _build_doc_list(q_df, reranker.truncate_passage)

        ranked_ids = reranker.rerank_return_ids(query, doc_list)
        dcg = compute_dcg_ratio(ranked_ids, origin_labels)
        pred = classify_scores(pd.Series([dcg])).iloc[0]

        drift = reasoning_drift_score(reranker.last_reasoning_trace, drift_lexicon)
        lr = reasoning_length_ratio(
            reranker.last_reasoning_tokens_prf,
            reranker.last_reasoning_tokens_orig,
        )

        results_dcg.append({"query_id": qid, "score": dcg, "prediction": pred})
        results_drift.append({"query_id": qid, "score": drift, "prediction": pred})
        results_len_ratio.append({"query_id": qid, "score": lr, "prediction": pred})

        logger.info("qid=%s  DCG=%.4f  drift=%.4f  len_ratio=%.4f",
                     qid, dcg, drift, lr)

    # Save DCG results
    method_name = f"R1-{model_key}-DCG"
    out_path = ROOT / POSTHOC_RESULTS_DIR / f"r1_{model_key}_dcg.csv"
    save_per_query_results(results_dcg, out_path)
    scores = pd.Series({r["query_id"]: r["score"] for r in results_dcg})
    metrics = evaluate_all(labels_df, scores, method_name, "posthoc", f"Rank-R1 {model_key} DCG")
    append_to_summary(metrics)
    logger.info("%s — Acc=%.4f  AUC=%.4f", method_name, metrics["acc"], metrics["auc"])

    # Save reasoning trace features
    for feat_name, results in [
        (f"R1-{model_key}-Drift", results_drift),
        (f"R1-{model_key}-LenRatio", results_len_ratio),
    ]:
        out_path = ROOT / POSTHOC_RESULTS_DIR / f"{feat_name.lower().replace('-', '_')}.csv"
        save_per_query_results(results, out_path)


def main() -> None:
    logger = setup_logging("run_setwise")

    labels_df = load_labels()
    inter_df = load_interleaved()

    logger.info("Running Rank-R1 setwise reranking for %d models", len(RANK_R1_MODELS))

    for model_key in RANK_R1_MODELS:
        run_one_model(model_key, inter_df, labels_df, logger)

    logger.info("All setwise methods complete.")


if __name__ == "__main__":
    main()
