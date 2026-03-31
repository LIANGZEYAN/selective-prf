#!/usr/bin/env python3
"""
Re-evaluate all posthoc models with different prediction thresholds.

Produces three summary tables:
  1. summary_table.csv          — original (h=0.061 for both GT and predictions)
  2. summary_table_optA.csv     — h_gt=0.061, h_pred=0 (separate thresholds, binary prediction)
  3. summary_table_optB.csv     — h_gt=0.061, h_pred=MAD per model (MAD-derived)

Pre-hoc and QPP rows are copied as-is (threshold doesn't apply to their scores).
AUC, Gamma, Rho are threshold-free and unchanged across all three tables.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import H_THRESHOLD, GAMMA_N_PERMUTATIONS
from src.data_loader import load_labels
from src.metrics import (
    compute_tpr, compute_accuracy, compute_auc,
    compute_gamma, compute_rho,
)

RESULTS_DIR = PROJECT_ROOT / "results"
POSTHOC_DIR = RESULTS_DIR / "posthoc"

AGGREGATIONS = [
    ("DCG", "dcg_score"),
    ("MRR", "mrr_score"),
    ("Majority@k", "majority_at_k_score"),
]


def classify_with_threshold(scores: pd.Series, h: float) -> pd.Series:
    """Classify continuous scores into Hurt/Benefit/Neutral using given threshold."""
    def _classify(ratio):
        if np.isnan(ratio):
            return "Insufficient-Data"
        if ratio > 0.5 + h:
            return "Benefit"
        if ratio < 0.5 - h:
            return "Hurt"
        return "Neutral"
    return scores.apply(_classify)


def evaluate_with_threshold(labels_df, scores_series, h_pred, original_row):
    """Re-evaluate a single method with a given prediction threshold.

    Returns a dict with updated TPR/Acc and a note about h_pred.
    AUC, Gamma, Rho are copied from original (threshold-free).
    """
    merged = labels_df.set_index("query_id").join(
        scores_series.rename("score"), how="inner"
    )
    merged = merged[merged["label"] != "Insufficient-Data"].copy()

    y_true = merged["label"].values
    scores = merged["score"].values
    y_pred = classify_with_threshold(merged["score"], h_pred).values

    tpr_h = compute_tpr(y_true, y_pred, "Hurt")
    tpr_b = compute_tpr(y_true, y_pred, "Benefit")
    acc = compute_accuracy(y_true, y_pred)

    row = dict(original_row)
    row["tpr_hurt"] = round(tpr_h, 4) if not np.isnan(tpr_h) else None
    row["tpr_benefit"] = round(tpr_b, 4) if not np.isnan(tpr_b) else None
    row["acc"] = round(acc, 4) if not np.isnan(acc) else None
    row["h_pred"] = round(h_pred, 4)
    return row


def compute_mad_threshold(scores: np.ndarray) -> float:
    """Compute MAD-derived threshold from a score distribution."""
    clean = scores[~np.isnan(scores)]
    if len(clean) == 0:
        return 0.0
    return float(np.median(np.abs(clean - np.median(clean))))


def main():
    labels_df = load_labels()
    original = pd.read_csv(RESULTS_DIR / "summary_table.csv")

    # Identify posthoc score files
    score_files = sorted(POSTHOC_DIR.glob("*_scores.csv"))
    print(f"Found {len(score_files)} posthoc score files")

    # Build a map: method_prefix -> score DataFrame
    # e.g., "Pairwise-Qwen-14B" -> DataFrame with dcg_score, mrr_score, majority_at_k_score
    score_map = {}
    for f in score_files:
        prefix = f.stem.replace("_scores", "")
        score_map[prefix] = pd.read_csv(f)

    rows_a = []  # Option A: h_pred=0
    rows_b = []  # Option B: h_pred=MAD

    for _, orig_row in original.iterrows():
        method_name = orig_row["method_name"]
        stage = orig_row["stage"]

        # Non-posthoc: copy as-is
        if stage not in ("posthoc", "posthoc-sc"):
            row_a = dict(orig_row)
            row_a["h_pred"] = "N/A"
            rows_a.append(row_a)
            row_b = dict(orig_row)
            row_b["h_pred"] = "N/A"
            rows_b.append(row_b)
            continue

        # Parse method name: e.g., "Pairwise-Qwen-14B-DCG" -> prefix="Pairwise-Qwen-14B", agg="DCG"
        parts = method_name.rsplit("-", 1)
        if len(parts) != 2:
            print(f"  WARNING: can't parse {method_name}, copying as-is")
            rows_a.append(dict(orig_row))
            rows_b.append(dict(orig_row))
            continue

        prefix, agg_name = parts

        if prefix not in score_map:
            print(f"  WARNING: no score file for {prefix}, copying as-is")
            rows_a.append(dict(orig_row))
            rows_b.append(dict(orig_row))
            continue

        # Find the score column
        agg_col = None
        for a_name, a_col in AGGREGATIONS:
            if a_name == agg_name:
                agg_col = a_col
                break

        if agg_col is None or agg_col not in score_map[prefix].columns:
            print(f"  WARNING: column {agg_col} not in {prefix}, copying as-is")
            rows_a.append(dict(orig_row))
            rows_b.append(dict(orig_row))
            continue

        df = score_map[prefix]
        scores_series = pd.Series(
            df[agg_col].values, index=df["query_id"].values
        )
        scores_series.index.name = "query_id"

        # Option A: h_pred = 0
        row_a = evaluate_with_threshold(labels_df, scores_series, 0.0, orig_row)
        rows_a.append(row_a)

        # Option B: h_pred = MAD of this model's scores
        mad = compute_mad_threshold(scores_series.values)
        row_b = evaluate_with_threshold(labels_df, scores_series, mad, orig_row)
        rows_b.append(row_b)

        print(f"  {method_name}: MAD={mad:.4f} | OptA(h=0): Acc={row_a['acc']} | OptB(h={mad:.3f}): Acc={row_b['acc']}")

    # Save
    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)

    path_a = RESULTS_DIR / "summary_table_optA.csv"
    path_b = RESULTS_DIR / "summary_table_optB.csv"

    df_a.to_csv(path_a, index=False)
    df_b.to_csv(path_b, index=False)

    print(f"\nSaved Option A (separate thresholds, h_pred=0) → {path_a}")
    print(f"Saved Option B (MAD-derived h_pred per model) → {path_b}")

    # Print comparison for key models
    print("\n=== Comparison: Pairwise-Qwen-72B-Majority@k ===")
    for label, df in [("Original (h=0.061)", original), ("OptA (h_pred=0)", df_a), ("OptB (MAD)", df_b)]:
        row = df[df["method_name"] == "Pairwise-Qwen-72B-Majority@k"]
        if not row.empty:
            r = row.iloc[0]
            h_note = f" [h_pred={r['h_pred']}]" if "h_pred" in r and pd.notna(r.get("h_pred")) else ""
            print(f"  {label}{h_note}: TPR_H={r['tpr_hurt']}, TPR_B={r['tpr_benefit']}, Acc={r['acc']}, AUC={r['auc']}")

    print("\n=== Comparison: SC-Setwise-SFT-7B-Majority@k ===")
    for label, df in [("Original (h=0.061)", original), ("OptA (h_pred=0)", df_a), ("OptB (MAD)", df_b)]:
        row = df[df["method_name"] == "SC-Setwise-SFT-7B-Majority@k"]
        if not row.empty:
            r = row.iloc[0]
            h_note = f" [h_pred={r['h_pred']}]" if "h_pred" in r and pd.notna(r.get("h_pred")) else ""
            print(f"  {label}{h_note}: TPR_H={r['tpr_hurt']}, TPR_B={r['tpr_benefit']}, Acc={r['acc']}, AUC={r['auc']}")

    # Summary statistics
    print("\n=== Overall Posthoc Accuracy Comparison ===")
    posthoc_mask_orig = original["stage"].isin(["posthoc", "posthoc-sc"])
    posthoc_mask_a = df_a["stage"].isin(["posthoc", "posthoc-sc"])
    posthoc_mask_b = df_b["stage"].isin(["posthoc", "posthoc-sc"])

    print(f"  Original mean Acc: {original.loc[posthoc_mask_orig, 'acc'].mean():.4f}")
    print(f"  OptA mean Acc:     {df_a.loc[posthoc_mask_a, 'acc'].astype(float).mean():.4f}")
    print(f"  OptB mean Acc:     {df_b.loc[posthoc_mask_b, 'acc'].astype(float).mean():.4f}")


if __name__ == "__main__":
    main()
