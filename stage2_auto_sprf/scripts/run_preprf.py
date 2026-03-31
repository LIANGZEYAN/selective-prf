#!/usr/bin/env python3
"""
Run pre-PRF methods: WIG, NQC, Clarity, SMV, Top-k Doc Coherence (TDC).

These methods use the initial retrieval scores and/or document texts
from the ColBERT ranking.  No LLM inference required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT as ROOT, QPP_RESULTS_DIR, PREHOC_RESULTS_DIR
from src.data_loader import load_labels, load_interleaved, load_colbert, get_query_texts
from src.metrics import evaluate_all
from src.qpp.baselines import wig, nqc, clarity, smv
from src.run_utils import setup_logging, save_per_query_results, append_to_summary


def _run_qpp_method(
    name: str,
    fn,
    colbert_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    query_ids: list,
    corpus_score: float,
    logger,
) -> None:
    """Run a single QPP method and save results."""
    results = []
    for qid in tqdm(query_ids, desc=name):
        q_df = colbert_df[colbert_df["qid"] == qid].sort_values("rank")
        scores_arr = q_df["score"].values.astype(float)

        if name in ("WIG", "NQC"):
            score = fn(scores_arr, corpus_score)
        else:
            score = fn(scores_arr)

        median_so_far = np.median([r["score"] for r in results]) if results else 0
        results.append({
            "query_id": qid,
            "score": score,
            "prediction": "Benefit" if score > median_so_far else "Hurt",
        })

    # Re-classify based on median split
    all_scores = [r["score"] for r in results]
    median_score = np.median(all_scores)
    for r in results:
        r["prediction"] = "Benefit" if r["score"] > median_score else "Hurt"

    out_path = ROOT / QPP_RESULTS_DIR / f"{name.lower()}.csv"
    save_per_query_results(results, out_path)

    scores_series = pd.Series({r["query_id"]: r["score"] for r in results})
    metrics = evaluate_all(labels_df, scores_series, name, "qpp", f"QPP-{name}")
    append_to_summary(metrics)
    logger.info("%s — Acc=%.4f  AUC=%.4f", name, metrics["acc"], metrics["auc"])


def run_tdc(query_ids: list, colbert_df: pd.DataFrame, labels_df: pd.DataFrame, logger) -> None:
    """Top-k Document Coherence."""
    from sentence_transformers import SentenceTransformer
    from src.prehoc.preprf import top_k_doc_coherence
    from src.config import SBERT_MODEL

    model = SentenceTransformer(SBERT_MODEL)

    results = []
    for qid in tqdm(query_ids, desc="TDC"):
        q_df = colbert_df[colbert_df["qid"] == qid].sort_values("rank").head(10)
        doc_texts = q_df["passage_text"].dropna().tolist()
        score = top_k_doc_coherence(doc_texts, model)
        results.append({"query_id": qid, "score": score, "prediction": "Benefit" if score > 0.5 else "Hurt"})

    # Re-classify based on median split
    all_scores = [r["score"] for r in results]
    median_score = np.median(all_scores)
    for r in results:
        r["prediction"] = "Benefit" if r["score"] > median_score else "Hurt"

    out_path = ROOT / PREHOC_RESULTS_DIR / "tdc.csv"
    save_per_query_results(results, out_path)

    scores_series = pd.Series({r["query_id"]: r["score"] for r in results})
    metrics = evaluate_all(labels_df, scores_series, "TDC", "prehoc", "Top-k Doc Coherence")
    append_to_summary(metrics)
    logger.info("TDC — Acc=%.4f  AUC=%.4f", metrics["acc"], metrics["auc"])


def main() -> None:
    logger = setup_logging("run_preprf")
    labels_df = load_labels()
    inter_df = load_interleaved()
    colbert_df = load_colbert()
    query_ids = sorted(inter_df["qid"].unique().tolist())

    # Corpus-level average score for WIG/NQC
    corpus_score = colbert_df["score"].mean()

    logger.info("Running pre-PRF methods on %d queries", len(query_ids))

    for name, fn in [("WIG", wig), ("NQC", nqc), ("Clarity", clarity), ("SMV", smv)]:
        _run_qpp_method(name, fn, colbert_df, labels_df, query_ids, corpus_score, logger)

    run_tdc(query_ids, colbert_df, labels_df, logger)

    logger.info("All pre-PRF methods complete.")


if __name__ == "__main__":
    main()
