#!/usr/bin/env python3
"""
Run self-consistency methods: SC-Mean, SC-SNR, DR ensemble.
Uses Qwen-72B API with R=5 independent runs per query.
Requires IDA_LLM_API_KEY environment variable.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT as ROOT, POSTHOC_RESULTS_DIR, SC_NUM_RUNS
from src.data_loader import load_labels, load_interleaved
from src.aggregation import compute_dcg_ratio
from src.posthoc.self_consistency import run_self_consistency, sc_mean, sc_snr
from src.posthoc.ensemble import doubly_robust
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


def main() -> None:
    logger = setup_logging("run_sc")

    labels_df = load_labels()
    inter_df = load_interleaved()
    query_ids = sorted(inter_df["qid"].unique().tolist())

    # ── Run both pairwise and setwise 72B rerankers ────────────────────────
    SC_TEMPERATURE = 0.7
    reranker_configs = [
        ("Pairwise-72B", "pairwise"),
        ("Setwise-72B", "setwise"),
    ]

    for reranker_label, reranker_type in reranker_configs:
        logger.info("Initialising %s API reranker for SC (R=%d, temp=%.1f)...",
                     reranker_label, SC_NUM_RUNS, SC_TEMPERATURE)

        if reranker_type == "pairwise":
            from src.posthoc.pairwise_api import PairwiseApiReranker
            reranker = PairwiseApiReranker(temperature=SC_TEMPERATURE)
        else:
            from src.posthoc.setwise_api import SetwiseApiReranker
            reranker = SetwiseApiReranker(temperature=SC_TEMPERATURE)

        results_mean = []
        results_snr = []
        results_dr = []

        for qid in tqdm(query_ids, desc=f"SC-{reranker_label}"):
            q_df = inter_df[inter_df["qid"] == qid].sort_values("rank")
            query = reranker.truncate_query(q_df["query_text"].iloc[0])
            doc_list, origin_labels = _build_doc_list(q_df, reranker.truncate_passage)

            scores = run_self_consistency(
                reranker, query, doc_list,
                n_runs=SC_NUM_RUNS,
                aggregation_fn=lambda ids, labels: compute_dcg_ratio(ids, labels),
            )

            mu = sc_mean(scores)
            snr = sc_snr(scores)
            sigma = float(np.std(scores)) if len(scores) > 1 else 0.0
            dr = doubly_robust(mu, sigma, lambda_=1.0)

            pred_mu = classify_scores(pd.Series([mu])).iloc[0]
            pred_snr = "Benefit" if snr > 0 else "Hurt"
            pred_dr = classify_scores(pd.Series([dr])).iloc[0]

            results_mean.append({"query_id": qid, "score": mu, "prediction": pred_mu})
            results_snr.append({"query_id": qid, "score": snr, "prediction": pred_snr})
            results_dr.append({"query_id": qid, "score": dr, "prediction": pred_dr})

            logger.info("qid=%s  %s  SC-Mean=%.4f  SC-SNR=%.4f  DR=%.4f",
                         qid, reranker_label, mu, snr, dr)

        for name, results in [
            (f"SC-Mean-{reranker_label}", results_mean),
            (f"SC-SNR-{reranker_label}", results_snr),
            (f"SC-DR-{reranker_label}", results_dr),
        ]:
            out_path = ROOT / POSTHOC_RESULTS_DIR / f"{name.lower().replace('-', '_')}.csv"
            save_per_query_results(results, out_path)

            scores_series = pd.Series({r["query_id"]: r["score"] for r in results})
            metrics = evaluate_all(labels_df, scores_series, name, "posthoc-sc",
                                   f"Self-Consistency {name}")
            append_to_summary(metrics)
            logger.info("%s — Acc=%.4f  AUC=%.4f", name, metrics["acc"], metrics["auc"])

        logger.info("%s SC methods complete.", reranker_label)

    logger.info("All self-consistency methods complete.")


if __name__ == "__main__":
    main()
