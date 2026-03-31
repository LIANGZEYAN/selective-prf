#!/usr/bin/env python3
"""
Run ensemble DR combinations from cached score CSVs.

No LLM calls — reads results/posthoc/ files and computes
doubly-robust combinations mathematically.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT as ROOT, POSTHOC_RESULTS_DIR, RANK_R1_MODELS
from src.data_loader import load_labels
from src.posthoc.ensemble import setwise_dr
from src.metrics import evaluate_all, classify_scores
from src.run_utils import setup_logging, save_per_query_results, append_to_summary


def _load_cached(filename: str) -> pd.DataFrame:
    """Load a cached per-query result CSV from results/posthoc/."""
    path = ROOT / POSTHOC_RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Cached results not found: {path}. "
            f"Run the corresponding method first."
        )
    return pd.read_csv(path)


def main() -> None:
    logger = setup_logging("run_ensemble")
    labels_df = load_labels()

    for model_key in RANK_R1_MODELS:
        # Drift/lenratio files use underscores (from feat_name.lower().replace('-','_'))
        safe_key = model_key.replace("-", "_")
        dcg_file = f"r1_{model_key}_dcg.csv"
        drift_file = f"r1_{safe_key}_drift.csv"
        lenratio_file = f"r1_{safe_key}_lenratio.csv"

        try:
            dcg_df = _load_cached(dcg_file)
            drift_df = _load_cached(drift_file)
            lr_df = _load_cached(lenratio_file)
        except FileNotFoundError as e:
            logger.warning("Skipping R1-%s ensemble: %s", model_key, e)
            continue

        # Merge on query_id
        merged = dcg_df[["query_id", "score"]].rename(columns={"score": "dcg"})
        merged = merged.merge(
            drift_df[["query_id", "score"]].rename(columns={"score": "drift"}),
            on="query_id",
        )
        merged = merged.merge(
            lr_df[["query_id", "score"]].rename(columns={"score": "len_ratio"}),
            on="query_id",
        )

        results = []
        for _, row in tqdm(merged.iterrows(), total=len(merged), desc=f"DR-R1-{model_key}"):
            dr_score = setwise_dr(
                dcg_ratio=row["dcg"],
                delta_reason=row["drift"],
                lambda_ratio=row["len_ratio"],
                lambda1=1.0,
                lambda2=0.5,
            )
            pred = classify_scores(pd.Series([dr_score])).iloc[0]
            results.append({
                "query_id": row["query_id"],
                "score": dr_score,
                "prediction": pred,
            })

        method_name = f"DR-R1-{model_key}"
        out_path = ROOT / POSTHOC_RESULTS_DIR / f"dr_r1_{model_key}.csv"
        save_per_query_results(results, out_path)

        scores = pd.Series({r["query_id"]: r["score"] for r in results})
        metrics = evaluate_all(labels_df, scores, method_name, "posthoc", f"DR ensemble R1-{model_key}")
        append_to_summary(metrics)
        logger.info("%s — Acc=%.4f  AUC=%.4f", method_name, metrics["acc"], metrics["auc"])

    logger.info("Ensemble methods complete.")


if __name__ == "__main__":
    main()
