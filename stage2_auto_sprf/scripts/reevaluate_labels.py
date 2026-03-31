#!/usr/bin/env python3
"""
Re-evaluate ALL methods with different ground-truth label thresholds (h_label).

Uses summary_table.csv as the method template so every model that has been
run is included.  Raw scores are read from the existing result files — no
models are re-run.

Produces:
  results/summary_table_h000.csv   — h_label=0.00  (no neutral band)
  results/summary_table_h005.csv   — h_label=0.05
  results/summary_table_h010.csv   — h_label=0.10
  results/summary_table_h015.csv   — h_label=0.15

Usage:
    python scripts/reevaluate_labels.py                          # default set
    python scripts/reevaluate_labels.py --h 0.00 0.05 0.10 0.15 # custom set
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREFERENCE_CSV, GAMMA_N_PERMUTATIONS
from src.metrics import (
    compute_tpr, compute_accuracy, compute_auc,
    compute_gamma, compute_rho, classify_scores,
)

RESULTS_DIR = PROJECT_ROOT / "results"
POSTHOC_DIR = RESULTS_DIR / "posthoc"
PREHOC_DIR = RESULTS_DIR / "prehoc"
QPP_DIR = RESULTS_DIR / "qpp"

# Aggregation columns in multi-score files
AGGREGATIONS = {
    "DCG": "dcg_score",
    "MRR": "mrr_score",
    "Majority@k": "majority_at_k_score",
}

# Mapping: method_name → (score_file_path,) for single-score files
INDIVIDUAL_SCORE_MAP = {
    "R1-7b-DCG": POSTHOC_DIR / "r1_7b_dcg.csv",
    "R1-14b-DCG": POSTHOC_DIR / "r1_14b_dcg.csv",
    "R1-7b-sft-DCG": POSTHOC_DIR / "r1_7b-sft_dcg.csv",
    "R1-14b-sft-DCG": POSTHOC_DIR / "r1_14b-sft_dcg.csv",
    "DR-R1-7b": POSTHOC_DIR / "dr_r1_7b.csv",
    "DR-R1-14b": POSTHOC_DIR / "dr_r1_14b.csv",
    "DR-R1-7b-sft": POSTHOC_DIR / "dr_r1_7b-sft.csv",
    "DR-R1-14b-sft": POSTHOC_DIR / "dr_r1_14b-sft.csv",
    "SC-Mean-Pairwise-72B": POSTHOC_DIR / "sc_mean_pairwise_72b.csv",
    "SC-SNR-Pairwise-72B": POSTHOC_DIR / "sc_snr_pairwise_72b.csv",
    "SC-DR-Pairwise-72B": POSTHOC_DIR / "sc_dr_pairwise_72b.csv",
    "SC-Mean-Setwise-72B": POSTHOC_DIR / "sc_mean_setwise_72b.csv",
    "SC-SNR-Setwise-72B": POSTHOC_DIR / "sc_snr_setwise_72b.csv",
    "SC-DR-Setwise-72B": POSTHOC_DIR / "sc_dr_setwise_72b.csv",
    # Prehoc
    "NED": PREHOC_DIR / "ned.csv",
    "IDF-Coverage": PREHOC_DIR / "idf_coverage.csv",
    "QTC": PREHOC_DIR / "qtc.csv",
    "TDC": PREHOC_DIR / "tdc.csv",
    # QPP
    "WIG": QPP_DIR / "wig.csv",
    "NQC": QPP_DIR / "nqc.csv",
    "Clarity": QPP_DIR / "clarity.csv",
    "SMV": QPP_DIR / "smv.csv",
}


def build_labels(h: float) -> pd.DataFrame:
    """Build ground-truth labels from preference.csv with given threshold h."""
    pref_df = pd.read_csv(PROJECT_ROOT / PREFERENCE_CSV)
    labels = []
    for _, row in pref_df.iterrows():
        qid = row["qid"]
        b_ratio = row["b_preference_ratio"]
        if pd.isna(b_ratio):
            label = "Insufficient-Data"
        elif b_ratio > 0.5 + h:
            label = "Benefit"
        elif b_ratio < 0.5 - h:
            label = "Hurt"
        else:
            label = "Neutral"
        labels.append({"query_id": qid, "b_ratio": b_ratio, "label": label})
    return pd.DataFrame(labels)


def evaluate_method(labels_df, scores_series, h_pred, method_name, stage, predictor,
                    n_permutations=1000):
    """Evaluate one method against labels, using h_pred for classification."""
    merged = labels_df.set_index("query_id").join(
        scores_series.rename("score"), how="inner"
    )
    merged = merged[merged["label"] != "Insufficient-Data"].copy()

    y_true = merged["label"].values
    scores = merged["score"].values
    y_pred = classify_scores(merged["score"], h=h_pred).values

    n_hurt = (y_true == "Hurt").sum()
    n_benefit = (y_true == "Benefit").sum()
    non_neutral = (y_true == "Hurt") | (y_true == "Benefit")

    tpr_h = compute_tpr(y_true, y_pred, "Hurt")
    tpr_b = compute_tpr(y_true, y_pred, "Benefit")
    acc = compute_accuracy(y_true, y_pred)
    auc = compute_auc(y_true, scores)
    gamma, gamma_p = compute_gamma(y_true, scores, n_permutations=n_permutations)
    b_ratios = merged["b_ratio"].values.astype(float)
    rho, rho_p = compute_rho(b_ratios, scores)

    return {
        "method_name": method_name,
        "stage": stage,
        "predictor": predictor,
        "tpr_hurt": round(tpr_h, 4) if not np.isnan(tpr_h) else None,
        "tpr_benefit": round(tpr_b, 4) if not np.isnan(tpr_b) else None,
        "acc": round(acc, 4) if not np.isnan(acc) else None,
        "auc": round(auc, 4) if not np.isnan(auc) else None,
        "gamma": round(gamma, 4),
        "gamma_p": round(gamma_p, 4),
        "rho": round(rho, 4),
        "rho_p": round(rho_p, 4),
        "n_queries": int(non_neutral.sum()),
        "n_hurt": int(n_hurt),
        "n_benefit": int(n_benefit),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "h_label": None,  # filled by caller
        "h_pred": round(h_pred, 4),
    }


# Cache loaded score DataFrames to avoid re-reading files
_score_cache = {}


def _load_csv_cached(fpath: Path) -> pd.DataFrame:
    """Load a CSV and cache the DataFrame."""
    if fpath not in _score_cache:
        _score_cache[fpath] = pd.read_csv(fpath)
    return _score_cache[fpath]


def load_scores_for_method(method_name, stage):
    """Load raw scores for a method.

    Returns a pd.Series indexed by query_id, or None if not found.
    """
    # 1. Individual score map (single "score" column files)
    if method_name in INDIVIDUAL_SCORE_MAP:
        fpath = INDIVIDUAL_SCORE_MAP[method_name]
        if not fpath.exists():
            return None
        df = _load_csv_cached(fpath)
        df["query_id"] = df["query_id"].astype(int)
        return pd.Series(df["score"].values, index=df["query_id"].values)

    # 2. Multi-column _scores.csv files (posthoc/posthoc-sc)
    #    Method name format: "Prefix-AggName" where AggName ∈ {DCG, MRR, Majority@k}
    for agg_name, agg_col in AGGREGATIONS.items():
        suffix = f"-{agg_name}"
        if method_name.endswith(suffix):
            prefix = method_name[: -len(suffix)]
            score_file = POSTHOC_DIR / f"{prefix}_scores.csv"
            if score_file.exists():
                df = _load_csv_cached(score_file)
                if agg_col in df.columns:
                    return pd.Series(
                        df[agg_col].values, index=df["query_id"].values
                    )
            break  # matched suffix but file missing

    return None


def run_evaluation(h_label, h_pred, output_name, template_df, n_permutations=1000):
    """Run full evaluation for a given label threshold."""
    print(f"\n{'='*70}")
    print(f"  Re-evaluating with h_label={h_label}, h_pred={h_pred}, n_perm={n_permutations}")
    print(f"{'='*70}")

    # Build labels
    labels_df = build_labels(h_label)
    counts = labels_df["label"].value_counts()
    n_hurt = counts.get("Hurt", 0)
    n_benefit = counts.get("Benefit", 0)
    n_neutral = counts.get("Neutral", 0)
    n_insuff = counts.get("Insufficient-Data", 0)
    print(f"  Labels: Hurt={n_hurt}, Benefit={n_benefit}, Neutral={n_neutral}, Insuff={n_insuff}")
    print(f"  Non-neutral queries: {n_hurt + n_benefit}")

    rows = []
    n_skipped = 0
    for idx, (_, orig_row) in enumerate(template_df.iterrows()):
        method_name = orig_row["method_name"]
        stage = orig_row["stage"]
        predictor = orig_row["predictor"]

        scores_series = load_scores_for_method(method_name, stage)

        if scores_series is None:
            n_skipped += 1
            continue

        row = evaluate_method(
            labels_df, scores_series, h_pred,
            method_name, stage, predictor,
            n_permutations=n_permutations,
        )
        row["h_label"] = h_label
        rows.append(row)

        if (idx + 1) % 20 == 0:
            print(f"  ... {idx + 1}/{len(template_df)} methods evaluated")

    # Save
    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / output_name
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    print(f"  Total methods evaluated: {len(df)}  (skipped: {n_skipped})")

    # Summary stats
    posthoc_mask = df["stage"].isin(["posthoc", "posthoc-sc"])
    if posthoc_mask.any():
        mean_acc = df.loc[posthoc_mask, "acc"].astype(float).mean()
        mean_auc = df.loc[posthoc_mask, "auc"].astype(float).mean()
        print(f"  Posthoc mean Acc: {mean_acc:.4f}")
        print(f"  Posthoc mean AUC: {mean_auc:.4f}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate all methods with different h_label thresholds"
    )
    parser.add_argument(
        "--h", type=float, nargs="+",
        default=[0.00, 0.05, 0.10, 0.15],
        help="h_label values to evaluate (default: 0.00 0.05 0.10 0.15)",
    )
    parser.add_argument(
        "--h-pred", type=float, default=0.0,
        help="Prediction threshold for classification (default: 0.0)",
    )
    parser.add_argument(
        "--permutations", type=int, default=1000,
        help="Number of permutations for Gamma p-value (default: 1000)",
    )
    args = parser.parse_args()

    # Use current summary_table.csv as the template (all models)
    template_path = RESULTS_DIR / "summary_table.csv"
    if not template_path.exists():
        print(f"ERROR: {template_path} not found. Run models first.")
        sys.exit(1)

    template_df = pd.read_csv(template_path)
    # De-duplicate: keep only unique method_name rows (first occurrence)
    template_df = template_df.drop_duplicates(subset="method_name", keep="first")
    print(f"Template: {len(template_df)} unique methods from {template_path.name}")

    all_results = {}
    for h_val in args.h:
        # Output file name: summary_table_h000.csv, summary_table_h005.csv, etc.
        h_tag = f"h{int(h_val * 100):03d}"
        output_name = f"summary_table_{h_tag}.csv"
        df = run_evaluation(h_val, args.h_pred, output_name, template_df,
                           n_permutations=args.permutations)
        all_results[h_val] = df

    # ── Comparison across all h values ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  COMPARISON ACROSS h_label VALUES")
    print(f"{'='*70}")

    # Also include h=0.061 (original) if available
    orig_path = RESULTS_DIR / "summary_table.csv"
    if orig_path.exists():
        df_orig = pd.read_csv(orig_path)
        all_results[0.061] = df_orig

    for h_val in sorted(all_results.keys()):
        df = all_results[h_val]
        posthoc = df[df["stage"].isin(["posthoc", "posthoc-sc"])]
        n_q = int(df["n_queries"].iloc[0]) if len(df) > 0 and "n_queries" in df.columns else "?"
        nh = int(df["n_hurt"].iloc[0]) if len(df) > 0 and "n_hurt" in df.columns else "?"
        nb = int(df["n_benefit"].iloc[0]) if len(df) > 0 and "n_benefit" in df.columns else "?"
        print(f"\n  h_label={h_val:.3f}  (N={n_q} non-neutral: {nh} Hurt, {nb} Benefit)")
        if len(posthoc) > 0:
            print(f"    Posthoc mean Acc : {posthoc['acc'].astype(float).mean():.4f}")
            print(f"    Posthoc mean AUC : {posthoc['auc'].astype(float).mean():.4f}")
        print(f"    All methods Acc  : {df['acc'].astype(float).mean():.4f}")
        print(f"    All methods AUC  : {df['auc'].astype(float).mean():.4f}")

    # Top 5 by AUC in each setting
    print(f"\n{'='*70}")
    print("  TOP 5 METHODS BY AUC (each h_label)")
    print(f"{'='*70}")
    for h_val in sorted(all_results.keys()):
        df = all_results[h_val]
        top5 = df.nlargest(5, "auc")
        print(f"\n  h_label={h_val:.3f}:")
        for _, r in top5.iterrows():
            print(f"    {r['method_name']:<45} AUC={r['auc']:.4f}  Acc={r['acc']}")

    # Top 5 by Accuracy in each setting
    print(f"\n{'='*70}")
    print("  TOP 5 METHODS BY ACCURACY (each h_label)")
    print(f"{'='*70}")
    for h_val in sorted(all_results.keys()):
        df = all_results[h_val]
        top5 = df.nlargest(5, "acc")
        print(f"\n  h_label={h_val:.3f}:")
        for _, r in top5.iterrows():
            print(f"    {r['method_name']:<45} Acc={r['acc']:.4f}  AUC={r['auc']}")


if __name__ == "__main__":
    main()
