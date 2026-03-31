#!/usr/bin/env python3
"""
Run pre-hoc methods: Named Entity Density (NED), IDF Coverage, Query Term Coherence (QTC).

These methods use only the query text — no LLM inference required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT as ROOT, PREHOC_RESULTS_DIR
from src.data_loader import load_labels, load_interleaved, load_colbert, get_query_texts
from src.metrics import evaluate_all
from src.run_utils import setup_logging, save_per_query_results, append_to_summary


def run_ned(query_texts: dict, labels_df: pd.DataFrame, logger) -> None:
    """Named Entity Density per query."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    from src.prehoc.lexical import named_entity_density

    results = []
    for qid, text in tqdm(query_texts.items(), desc="NED"):
        score = named_entity_density(text, nlp)
        results.append({"query_id": qid, "score": score, "prediction": "Benefit" if score > 0.5 else "Hurt"})

    out_path = ROOT / PREHOC_RESULTS_DIR / "ned.csv"
    save_per_query_results(results, out_path)

    scores = pd.Series({r["query_id"]: r["score"] for r in results})
    metrics = evaluate_all(labels_df, scores, "NED", "prehoc", "Named Entity Density")
    append_to_summary(metrics)
    logger.info("NED — Acc=%.4f  AUC=%.4f", metrics["acc"], metrics["auc"])


def run_idf_coverage(query_texts: dict, labels_df: pd.DataFrame, logger) -> None:
    """IDF Coverage per query."""
    from src.prehoc.lexical import idf_coverage, load_idf_table

    colbert_df = load_colbert()
    corpus_texts = colbert_df["passage_text"].dropna().tolist()
    idf_table = load_idf_table(corpus_texts)

    results = []
    for qid, text in tqdm(query_texts.items(), desc="IDF Coverage"):
        score = idf_coverage(text, idf_table)
        results.append({"query_id": qid, "score": score, "prediction": "Benefit" if score > 0.5 else "Hurt"})

    out_path = ROOT / PREHOC_RESULTS_DIR / "idf_coverage.csv"
    save_per_query_results(results, out_path)

    scores = pd.Series({r["query_id"]: r["score"] for r in results})
    metrics = evaluate_all(labels_df, scores, "IDF-Coverage", "prehoc", "IDF Coverage")
    append_to_summary(metrics)
    logger.info("IDF Coverage — Acc=%.4f  AUC=%.4f", metrics["acc"], metrics["auc"])


def run_qtc(query_texts: dict, labels_df: pd.DataFrame, logger) -> None:
    """Query Term Coherence per query."""
    from sentence_transformers import SentenceTransformer
    from src.prehoc.semantic import query_term_coherence
    from src.config import SBERT_MODEL

    model = SentenceTransformer(SBERT_MODEL)

    results = []
    for qid, text in tqdm(query_texts.items(), desc="QTC"):
        score = query_term_coherence(text, model)
        results.append({"query_id": qid, "score": score, "prediction": "Benefit" if score > 0.5 else "Hurt"})

    out_path = ROOT / PREHOC_RESULTS_DIR / "qtc.csv"
    save_per_query_results(results, out_path)

    scores = pd.Series({r["query_id"]: r["score"] for r in results})
    metrics = evaluate_all(labels_df, scores, "QTC", "prehoc", "Query Term Coherence")
    append_to_summary(metrics)
    logger.info("QTC — Acc=%.4f  AUC=%.4f", metrics["acc"], metrics["auc"])


def main() -> None:
    logger = setup_logging("run_prehoc")
    labels_df = load_labels()
    inter_df = load_interleaved()
    query_texts = get_query_texts(inter_df)

    logger.info("Running pre-hoc methods on %d queries", len(query_texts))

    run_ned(query_texts, labels_df, logger)
    run_idf_coverage(query_texts, labels_df, logger)
    run_qtc(query_texts, labels_df, logger)

    logger.info("All pre-hoc methods complete.")


if __name__ == "__main__":
    main()
