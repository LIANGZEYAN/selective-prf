#!/usr/bin/env python3
"""
Print a formatted summary table of all method results.

Reads results/summary_table.csv and prints to stdout, sorted by AUC descending,
with significance stars next to gamma and rho.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT as ROOT, SUMMARY_TABLE_PATH


def _sig_stars(p_value) -> str:
    """Return significance stars based on p-value."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def main() -> None:
    summary_path = ROOT / SUMMARY_TABLE_PATH
    if not summary_path.exists():
        print(f"No summary table found at {summary_path}")
        print("Run some methods first, then try again.")
        sys.exit(1)

    df = pd.read_csv(summary_path)
    if df.empty:
        print("Summary table is empty.")
        sys.exit(0)

    # Sort by AUC descending
    df = df.sort_values("auc", ascending=False, na_position="last")

    # Format display
    print()
    print("=" * 120)
    print("  RESULTS SUMMARY — sorted by AUC descending")
    print("=" * 120)
    print()

    header = (
        f"{'Method':<28} {'Stage':<9} {'TPR_H':>6} {'TPR_B':>6} "
        f"{'Acc':>6} {'AUC':>6} {'Gamma':>9} {'Rho':>9} "
        f"{'N':>4} {'#H':>3} {'#B':>3}"
    )
    print(header)
    print("-" * 120)

    for _, row in df.iterrows():
        tpr_h = f"{row['tpr_hurt']:.3f}" if pd.notna(row["tpr_hurt"]) else "  -  "
        tpr_b = f"{row['tpr_benefit']:.3f}" if pd.notna(row["tpr_benefit"]) else "  -  "
        acc = f"{row['acc']:.3f}" if pd.notna(row["acc"]) else "  -  "
        auc = f"{row['auc']:.3f}" if pd.notna(row["auc"]) else "  -  "

        gamma_str = f"{row['gamma']:.3f}" if pd.notna(row["gamma"]) else "  -  "
        gamma_str += _sig_stars(row.get("gamma_p"))

        rho_str = f"{row['rho']:.3f}" if pd.notna(row["rho"]) else "  -  "
        rho_str += _sig_stars(row.get("rho_p"))

        n_q = int(row["n_queries"]) if pd.notna(row["n_queries"]) else "-"
        n_h = int(row["n_hurt"]) if pd.notna(row["n_hurt"]) else "-"
        n_b = int(row["n_benefit"]) if pd.notna(row["n_benefit"]) else "-"

        line = (
            f"{row['method_name']:<28} {row['stage']:<9} {tpr_h:>6} {tpr_b:>6} "
            f"{acc:>6} {auc:>6} {gamma_str:>9} {rho_str:>9} "
            f"{str(n_q):>4} {str(n_h):>3} {str(n_b):>3}"
        )
        print(line)

    print("-" * 120)
    print(f"  Total methods: {len(df)}")
    print(f"  Significance: * p<0.05, ** p<0.01")
    print("=" * 120)
    print()


if __name__ == "__main__":
    main()
