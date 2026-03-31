#!/usr/bin/env python3
"""
Re-evaluate all classifier approaches (1a, 1b, 2a, 2d) with different h thresholds.

h=0.00  → b_ratio > 0.50 = Benefit, < 0.50 = Hurt (strict 50/50)
h=0.05  → b_ratio > 0.55 = Benefit, < 0.45 = Hurt
h=0.061 → original threshold (MAD-derived)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT as ROOT
from src.metrics import compute_auc, compute_tpr, compute_accuracy, compute_gamma, compute_rho

OUT_DIR = ROOT / "results" / "classifier"
DATA_DIR = ROOT / "data" / "classifier"

H_VALUES = [0.00, 0.05, 0.061]


def make_labels(h: float) -> pd.DataFrame:
    """Re-derive Benefit/Hurt labels from raw b_ratio with given h threshold."""
    labels_all = pd.read_csv(ROOT / "data" / "labels.csv")
    # Exclude Insufficient-Data
    labels_all = labels_all[labels_all["label"] != "Insufficient-Data"].copy()

    def classify(ratio):
        if pd.isna(ratio):
            return "Insufficient-Data"
        if ratio > 0.5 + h:
            return "Benefit"
        if ratio < 0.5 - h:
            return "Hurt"
        return "Neutral"

    labels_all["label_h"] = labels_all["b_ratio"].apply(classify)
    bh = labels_all[labels_all["label_h"].isin(["Benefit", "Hurt"])].copy()
    bh["label"] = bh["label_h"]
    return bh


def best_threshold_acc(y_true, scores):
    """Find the threshold on scores that maximises accuracy.

    Predictions: score >= threshold → "Benefit", else "Hurt".
    Returns (best_threshold, best_acc, best_y_pred).
    """
    thresholds = np.unique(scores)
    # Add midpoints between consecutive unique scores
    if len(thresholds) > 1:
        midpoints = (thresholds[:-1] + thresholds[1:]) / 2
        thresholds = np.concatenate([thresholds, midpoints])
    best_acc, best_t, best_pred = -1, 0.5, None
    for t in thresholds:
        y_pred = np.where(scores >= t, "Benefit", "Hurt")
        acc = (y_true == y_pred).sum() / len(y_true)
        if acc > best_acc:
            best_acc, best_t, best_pred = acc, t, y_pred
    return best_t, best_acc, best_pred


def evaluate_scores(labels_df, scores_series, method_name):
    """Evaluate a score series against labels."""
    merged = labels_df.set_index("query_id").join(
        scores_series.rename("score"), how="inner"
    )
    if len(merged) == 0:
        return None

    y_true = merged["label"].values
    scores = merged["score"].values
    b_ratios = merged["b_ratio"].values.astype(float)

    n_benefit = (y_true == "Benefit").sum()
    n_hurt = (y_true == "Hurt").sum()

    auc = compute_auc(y_true, scores)
    gamma, gamma_p = compute_gamma(y_true, scores)
    rho, rho_p = compute_rho(b_ratios, scores)

    # Optimal-threshold classification metrics
    _thresh, acc, y_pred = best_threshold_acc(y_true, scores)
    tpr_hurt = compute_tpr(y_true, y_pred, "Hurt")
    tpr_benefit = compute_tpr(y_true, y_pred, "Benefit")

    return {
        "method_name": method_name,
        "n_queries": n_benefit + n_hurt,
        "n_benefit": n_benefit,
        "n_hurt": n_hurt,
        "tpr_hurt": round(tpr_hurt, 4) if not np.isnan(tpr_hurt) else None,
        "tpr_benefit": round(tpr_benefit, 4) if not np.isnan(tpr_benefit) else None,
        "acc": round(acc, 4),
        "auc": round(auc, 4) if not np.isnan(auc) else None,
        "gamma": round(gamma, 4),
        "gamma_p": round(gamma_p, 4),
        "rho": round(rho, 4) if not np.isnan(rho) else None,
        "rho_p": round(rho_p, 4) if not np.isnan(rho_p) else None,
    }


def normalise_01(scores_raw):
    """Normalise a score series to [0,1]."""
    s_min, s_max = scores_raw.min(), scores_raw.max()
    if s_max > s_min:
        return (scores_raw - s_min) / (s_max - s_min)
    return scores_raw * 0 + 0.5


def main():
    # ── Load all data sources ────────────────────────────────────────
    features_1a = pd.read_csv(OUT_DIR / "approach_1a_features.csv")
    scores_1b = pd.read_csv(OUT_DIR / "approach_1b_scores.csv")
    scores_2a = pd.read_csv(OUT_DIR / "approach_2a_scores.csv")
    features_2d = pd.read_csv(OUT_DIR / "approach_2d_features.csv")

    all_results = []

    for h in H_VALUES:
        labels = make_labels(h)
        n_ben = (labels["label"] == "Benefit").sum()
        n_hurt = (labels["label"] == "Hurt").sum()
        print(f"\n{'='*60}")
        print(f"h={h:.3f}: {len(labels)} queries ({n_ben} Benefit, {n_hurt} Hurt)")
        print(f"{'='*60}")

        # ── 1a: Evaluate each feature ──────────────────────────────
        feature_cols_1a = {
            "nli_bart-large-mnli_benefit": ("1a-BART-MNLI", True),
            "nli_DeBERTa-v3-base-mnli-fever-anli_benefit": ("1a-DeBERTa-MNLI", True),
            "emb_norm": ("1a-Emb-norm", True),
            "term_sim_mean": ("1a-TermSimMean", True),
            "term_sim_std": ("1a-TermSimStd", False),
            "n_terms": ("1a-n_terms", False),
            "n_entities": ("1a-NER-count", True),
            "entity_ratio": ("1a-NER-ratio", True),
        }

        for col, (name, higher_benefit) in feature_cols_1a.items():
            scores_raw = features_1a.set_index("qid")[col]
            if not higher_benefit:
                scores_raw = -scores_raw
            scores_norm = normalise_01(scores_raw)
            scores_norm.index.name = "query_id"

            result = evaluate_scores(labels, scores_norm, name)
            if result:
                result["h"] = h
                all_results.append(result)

        # ── 1b: Evaluate each model × prompt ──────────────────────
        for (model, prompt_name), grp in scores_1b.groupby(["model", "prompt_name"]):
            score_series = pd.Series(grp["score"].values, index=grp["qid"].values.astype(int))
            score_series.index.name = "query_id"

            name = f"1b-{model}-{prompt_name}"
            result = evaluate_scores(labels, score_series, name)
            if result:
                result["h"] = h
                all_results.append(result)

        # ── 2a: Evaluate each model × prompt × top_k ─────────────
        for (model, prompt_name, top_k), grp in scores_2a.groupby(["model", "prompt_name", "top_k"]):
            score_series = pd.Series(grp["score"].values, index=grp["qid"].values.astype(int))
            score_series.index.name = "query_id"

            name = f"2a-{model}-{prompt_name}-k{top_k}"
            result = evaluate_scores(labels, score_series, name)
            if result:
                result["h"] = h
                all_results.append(result)

        # ── 2d: Evaluate each feature (try both directions) ──────
        feature_cols_2d = [c for c in features_2d.columns if c != "qid"]
        for col in feature_cols_2d:
            scores_raw = features_2d.set_index("qid")[col].astype(float)
            scores_raw.index.name = "query_id"

            # Try higher → Benefit
            scores_pos = normalise_01(scores_raw)
            result_pos = evaluate_scores(labels, scores_pos, f"2d-{col}(+)")

            # Try higher → Hurt (invert)
            scores_neg = 1.0 - scores_pos
            result_neg = evaluate_scores(labels, scores_neg, f"2d-{col}(-)")

            # Pick direction with better AUC
            auc_pos = result_pos["auc"] if result_pos and result_pos["auc"] is not None else 0.5
            auc_neg = result_neg["auc"] if result_neg and result_neg["auc"] is not None else 0.5

            if auc_pos >= auc_neg:
                result = result_pos
            else:
                result = result_neg

            if result:
                result["h"] = h
                all_results.append(result)

    # ── Save and print ──────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    out_path = OUT_DIR / "threshold_sweep_all.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    # Print summary table per h value
    for h in H_VALUES:
        sub = results_df[results_df["h"] == h].copy()
        sub = sub.sort_values("auc", ascending=False)
        print(f"\n--- h={h:.3f} (n={sub.iloc[0]['n_queries'] if len(sub) > 0 else 0}) ---")
        cols = ["method_name", "tpr_hurt", "tpr_benefit", "acc", "auc", "gamma", "gamma_p", "rho", "rho_p"]
        print(sub[cols].to_string(index=False))


if __name__ == "__main__":
    main()
