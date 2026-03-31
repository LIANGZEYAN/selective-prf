#!/usr/bin/env python3
"""
Build ground-truth labels from preference.csv using h=0.061 (MAD-derived).

Reads the raw user-study preference file and produces data/labels.csv,
the single source of truth for all downstream evaluation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import H_THRESHOLD, PREFERENCE_CSV, LABELS_PATH


def build_labels(preference_path: Path, h: float) -> pd.DataFrame:
    """Read preference.csv, apply threshold h, return labelled DataFrame."""
    pref_df = pd.read_csv(preference_path)

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


def main() -> None:
    preference_path = PROJECT_ROOT / PREFERENCE_CSV
    labels_df = build_labels(preference_path, H_THRESHOLD)

    # --- save ----------------------------------------------------------------
    out_path = PROJECT_ROOT / LABELS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(out_path, index=False)

    # --- summary -------------------------------------------------------------
    total = len(labels_df)
    valid = labels_df[labels_df["label"] != "Insufficient-Data"]
    valid_ratios = valid["b_ratio"].dropna()
    mad = np.median(np.abs(valid_ratios - np.median(valid_ratios)))

    counts = labels_df["label"].value_counts()
    print("=" * 60)
    print("  build_labels.py — Ground-Truth Label Summary")
    print("=" * 60)
    print(f"  h threshold      : {H_THRESHOLD}")
    print(f"  MAD of b_ratio   : {mad:.4f}")
    print(f"  Total queries    : {total}")
    print()
    for lbl in ["Hurt", "Benefit", "Neutral", "Insufficient-Data"]:
        c = counts.get(lbl, 0)
        print(f"  {lbl:<20}: {c}")
    print()
    non_neutral = counts.get("Hurt", 0) + counts.get("Benefit", 0)
    print(f"  Non-neutral (Hurt + Benefit): {non_neutral}")
    print(f"  Saved -> {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
