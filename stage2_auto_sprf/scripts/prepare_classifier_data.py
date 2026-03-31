"""
Step 0: Prepare data for the sPRF classifier experiments.

Outputs:
  data/classifier/queries.tsv        – qid \t query_text (43 queries)
  data/classifier/labels.tsv         – qid \t b_ratio \t label (Benefit/Hurt only, 26 queries)
  data/classifier/initial_rankings/  – one CSV per query with top-k ColBERT passages
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "data" / "classifier"
RANKINGS_DIR = OUT_DIR / "initial_rankings"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RANKINGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Queries ────────────────────────────────────────────────────
    interleaved = pd.read_csv(PROJECT_ROOT / "data" / "result_interleaved_with_text.csv")
    queries = interleaved.groupby("qid")["query_text"].first().reset_index()
    queries.to_csv(OUT_DIR / "queries.tsv", sep="\t", index=False)
    print(f"Wrote {len(queries)} queries → {OUT_DIR / 'queries.tsv'}")

    # ── 2. Labels (Benefit / Hurt only) ───────────────────────────────
    labels = pd.read_csv(PROJECT_ROOT / "data" / "labels.csv")
    labels_bh = labels[labels["label"].isin(["Benefit", "Hurt"])].copy()
    labels_bh.to_csv(OUT_DIR / "labels.tsv", sep="\t", index=False)
    print(f"Wrote {len(labels_bh)} labelled queries → {OUT_DIR / 'labels.tsv'}")
    print(f"  Benefit: {(labels_bh.label == 'Benefit').sum()}, Hurt: {(labels_bh.label == 'Hurt').sum()}")

    # ── 3. Initial ColBERT rankings (per query) ──────────────────────
    colbert = pd.read_csv(PROJECT_ROOT / "data" / "df_colbert_deduped.csv")
    for qid, grp in colbert.groupby("qid"):
        grp.to_csv(RANKINGS_DIR / f"{qid}.csv", index=False)
    print(f"Wrote per-query rankings for {colbert['qid'].nunique()} queries → {RANKINGS_DIR}")


if __name__ == "__main__":
    main()
