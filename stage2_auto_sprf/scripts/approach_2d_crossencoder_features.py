#!/usr/bin/env python3
"""
Approach 2d: Cross-Encoder Confidence Gap.

Use Qwen3-Reranker-8B pointwise scores on the initial ColBERT list to derive
QPP-style features. No LLM prompting — just numerical analysis of score distributions.

Also computes features from the raw ColBERT scores as a baseline.

Outputs:
  results/classifier/approach_2d_features.csv  – per-query feature matrix
  results/classifier/approach_2d_results.csv   – per-feature evaluation metrics
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROJECT_ROOT as ROOT,
    QWEN3_RERANKER_8B_MODEL,
    RANK_R1_CACHE_DIR,
    MS_MARCO_CROSS_ENCODER,
    QWEN_7B_CACHE_DIR,
)
from src.metrics import evaluate_all
from src.run_utils import setup_logging, append_to_summary

OUT_DIR = ROOT / "results" / "classifier"
DATA_DIR = ROOT / "data" / "classifier"
RANKINGS_DIR = DATA_DIR / "initial_rankings"
TOP_K = 10  # number of docs to compute features over


# ── Data loading ────────────────────────────────────────────────────────────

def load_queries() -> dict:
    df = pd.read_csv(DATA_DIR / "queries.tsv", sep="\t")
    return dict(zip(df["qid"], df["query_text"]))


def load_labels() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "labels.tsv", sep="\t")


def load_initial_ranking(qid: int, top_k: int = TOP_K) -> pd.DataFrame:
    """Load top-k documents from initial ColBERT ranking."""
    fpath = RANKINGS_DIR / f"{qid}.csv"
    df = pd.read_csv(fpath).head(top_k)
    return df


# ── Feature computation from score distributions ───────────────────────────

def compute_score_features(scores: np.ndarray, prefix: str) -> dict:
    """Compute statistical features from a score distribution.

    Parameters
    ----------
    scores : array of relevance scores, ordered by rank (highest first).
    prefix : string prefix for feature names (e.g., 'colbert_' or 'reranker_').
    """
    features = {}
    n = len(scores)

    features[f"{prefix}score_mean"] = np.mean(scores)
    features[f"{prefix}score_std"] = np.std(scores) if n > 1 else 0.0
    features[f"{prefix}score_max"] = np.max(scores)
    features[f"{prefix}score_min"] = np.min(scores)

    # Gap features
    if n >= 2:
        features[f"{prefix}score_gap_1_2"] = scores[0] - scores[1]
    else:
        features[f"{prefix}score_gap_1_2"] = 0.0

    features[f"{prefix}score_gap_1_k"] = scores[0] - scores[-1]

    # Normalised entropy
    if n > 1:
        shifted = scores - scores.min()
        total = shifted.sum()
        if total > 0:
            probs = shifted / total
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            max_entropy = np.log(n)
            features[f"{prefix}score_entropy"] = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            features[f"{prefix}score_entropy"] = 1.0
    else:
        features[f"{prefix}score_entropy"] = 0.0

    # Top-3 vs rest gap
    if n >= 4:
        top3_mean = np.mean(scores[:3])
        rest_mean = np.mean(scores[3:])
        features[f"{prefix}top3_vs_rest"] = top3_mean - rest_mean
    else:
        features[f"{prefix}top3_vs_rest"] = 0.0

    # Score decay rate (linear fit slope)
    if n >= 3:
        ranks = np.arange(n)
        slope = np.polyfit(ranks, scores, 1)[0]
        features[f"{prefix}score_slope"] = slope
    else:
        features[f"{prefix}score_slope"] = 0.0

    return features


# ── Qwen3-Reranker scoring ─────────────────────────────────────────────────

def score_with_reranker(queries: dict, logger) -> dict:
    """Score all query-document pairs with Qwen3-Reranker-8B.

    Returns dict[qid] -> np.array of scores (ordered by ColBERT rank).
    """
    from src.posthoc.pointwise import PointwiseReranker

    logger.info(f"Loading Qwen3-Reranker-8B...")
    reranker = PointwiseReranker(
        model_name_or_path=QWEN3_RERANKER_8B_MODEL,
        cache_dir=RANK_R1_CACHE_DIR,
    )

    all_scores = {}
    for i, (qid, query_text) in enumerate(queries.items()):
        df = load_initial_ranking(qid, TOP_K)
        passages = df["passage_text"].tolist()

        # Truncate passages
        passages_trunc = [reranker.truncate_passage(str(p)) for p in passages]
        query_trunc = reranker.truncate_query(query_text)

        scores = reranker.score_batch(query_trunc, passages_trunc)
        all_scores[qid] = np.array(scores)

        if (i + 1) % 10 == 0:
            logger.info(f"  Scored {i+1}/{len(queries)} queries")

    logger.info(f"  Total comparisons: {reranker.total_compare}, "
                f"tokens: {reranker.total_prompt_tokens}")

    del reranker
    gc.collect()
    torch.cuda.empty_cache()

    return all_scores


def score_with_cross_encoder(queries: dict, logger) -> dict:
    """Score all query-document pairs with ms-marco cross-encoder.

    Returns dict[qid] -> np.array of scores (ordered by ColBERT rank).
    """
    from sentence_transformers import CrossEncoder

    logger.info(f"Loading cross-encoder: {MS_MARCO_CROSS_ENCODER}")
    model = CrossEncoder(
        MS_MARCO_CROSS_ENCODER,
        max_length=512,
        device="cuda",
    )

    all_scores = {}
    for i, (qid, query_text) in enumerate(queries.items()):
        df = load_initial_ranking(qid, TOP_K)
        passages = df["passage_text"].tolist()

        pairs = [(query_text, str(p)) for p in passages]
        scores = model.predict(pairs)
        all_scores[qid] = np.array(scores, dtype=float)

        if (i + 1) % 10 == 0:
            logger.info(f"  Scored {i+1}/{len(queries)} queries")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return all_scores


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("approach_2d")

    queries = load_queries()
    labels = load_labels()
    logger.info(f"Loaded {len(queries)} queries, {len(labels)} labelled")

    # ── Step 1: Compute ColBERT score features (free, no GPU) ────────
    logger.info("Computing ColBERT score features...")
    colbert_features = {}
    for qid in queries:
        df = load_initial_ranking(qid, TOP_K)
        scores = df["score"].values.astype(float)
        colbert_features[qid] = compute_score_features(scores, "colbert_")

    # ── Step 2: Score with Qwen3-Reranker-8B ─────────────────────────
    logger.info("Scoring with Qwen3-Reranker-8B...")
    reranker_scores = score_with_reranker(queries, logger)

    reranker_features = {}
    for qid, scores in reranker_scores.items():
        reranker_features[qid] = compute_score_features(scores, "reranker_")

    # ── Step 2b: Score with ms-marco cross-encoder ───────────────────
    logger.info("Scoring with ms-marco-MiniLM-L-12-v2...")
    ce_scores = score_with_cross_encoder(queries, logger)

    ce_features = {}
    for qid, scores in ce_scores.items():
        ce_features[qid] = compute_score_features(scores, "msmarco_ce_")

    # ── Step 3: Merge into feature matrix ────────────────────────────
    rows = []
    for qid in queries:
        row = {"qid": qid}
        row.update(colbert_features.get(qid, {}))
        row.update(reranker_features.get(qid, {}))
        row.update(ce_features.get(qid, {}))
        rows.append(row)

    features_df = pd.DataFrame(rows)
    features_df.to_csv(OUT_DIR / "approach_2d_features.csv", index=False)
    logger.info(f"Saved features → {OUT_DIR / 'approach_2d_features.csv'}")
    logger.info(f"Feature columns: {list(features_df.columns)}")

    # ── Step 4: Evaluate each feature independently ──────────────────
    feature_cols = [c for c in features_df.columns if c != "qid"]

    # For each feature, determine direction: higher → Benefit or higher → Hurt
    # We try both directions and pick the one with better AUC
    all_results = []

    for col in feature_cols:
        scores_raw = features_df.set_index("qid")[col].astype(float)
        scores_raw.index.name = "query_id"

        # Try: higher score → more likely Benefit
        scores_pos = scores_raw.copy()
        s_min, s_max = scores_pos.min(), scores_pos.max()
        if s_max > s_min:
            scores_norm_pos = (scores_pos - s_min) / (s_max - s_min)
        else:
            scores_norm_pos = scores_pos * 0 + 0.5

        result_pos = evaluate_all(labels, scores_norm_pos, f"2d-{col}(+)",
                                  "classifier-2d", col)

        # Try: higher score → more likely Hurt (invert)
        scores_norm_neg = 1.0 - scores_norm_pos

        result_neg = evaluate_all(labels, scores_norm_neg, f"2d-{col}(-)",
                                  "classifier-2d", col)

        # Pick direction with better AUC
        auc_pos = result_pos["auc"] if result_pos["auc"] is not None else 0.5
        auc_neg = result_neg["auc"] if result_neg["auc"] is not None else 0.5

        if auc_pos >= auc_neg:
            result = result_pos
            result["direction"] = "higher→Benefit"
        else:
            result = result_neg
            result["direction"] = "higher→Hurt"

        all_results.append(result)
        logger.info(f"  {col}: AUC={result['auc']}, gamma={result['gamma']}, "
                    f"acc={result['acc']}, dir={result['direction']}")

    # ── Step 5: Save results ─────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "approach_2d_results.csv", index=False)

    for r in all_results:
        append_to_summary(r)

    logger.info(f"\n{'='*60}\nApproach 2d Summary (sorted by AUC)\n{'='*60}")
    results_df_sorted = results_df.sort_values("auc", ascending=False)
    cols = ["method_name", "direction", "tpr_hurt", "tpr_benefit", "acc", "auc", "gamma", "rho"]
    logger.info("\n" + results_df_sorted[cols].to_string(index=False))


if __name__ == "__main__":
    main()
