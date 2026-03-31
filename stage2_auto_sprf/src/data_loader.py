"""
Data loading and validation.

Single module responsible for reading all input CSVs and returning
clean, validated DataFrames with consistent column names.
"""

from pathlib import Path

import pandas as pd

from src.config import (
    PROJECT_ROOT,
    LABELS_PATH,
    INTERLEAVED_PATH,
    COLBERT_PATH,
    COLBERT_PRF_PATH,
)


def _resolve(rel_path: Path) -> Path:
    return PROJECT_ROOT / rel_path


def load_labels() -> pd.DataFrame:
    """Load data/labels.csv (ground truth).

    Returns DataFrame with columns: query_id, b_ratio, label.
    Raises FileNotFoundError with instructions if file is missing.
    """
    p = _resolve(LABELS_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run 'python scripts/build_labels.py' first."
        )
    df = pd.read_csv(p)
    required = {"query_id", "b_ratio", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv is missing columns: {missing}")
    return df


def load_interleaved() -> pd.DataFrame:
    """Load the interleaved top-9 candidate list.

    Returns DataFrame with columns: qid, docno, rank, origin_label,
    query_text, passage_text.
    """
    p = _resolve(INTERLEAVED_PATH)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found.")
    df = pd.read_csv(p)
    required = {"qid", "docno", "rank", "origin_label", "query_text", "passage_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"interleaved CSV is missing columns: {missing}")
    return df


def load_colbert() -> pd.DataFrame:
    """Load original ColBERT ranking (no PRF)."""
    p = _resolve(COLBERT_PATH)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found.")
    return pd.read_csv(p)


def load_colbert_prf() -> pd.DataFrame:
    """Load ColBERT + PRF ranking."""
    p = _resolve(COLBERT_PRF_PATH)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found.")
    return pd.read_csv(p)


def load_all() -> dict:
    """Load and cross-validate all datasets.

    Returns dict with keys: labels, interleaved, colbert, colbert_prf.
    Validates that query IDs are consistent across files.
    """
    labels_df = load_labels()
    inter_df = load_interleaved()
    colbert_df = load_colbert()
    colbert_prf_df = load_colbert_prf()

    # Cross-validate query IDs
    label_qids = set(labels_df["query_id"])
    inter_qids = set(inter_df["qid"])

    # Every interleaved query must appear in labels
    missing_in_labels = inter_qids - label_qids
    if missing_in_labels:
        raise ValueError(
            f"Queries in interleaved but not in labels: {missing_in_labels}"
        )

    return {
        "labels": labels_df,
        "interleaved": inter_df,
        "colbert": colbert_df,
        "colbert_prf": colbert_prf_df,
    }


def get_query_texts(inter_df: pd.DataFrame) -> dict:
    """Return {qid: query_text} mapping from interleaved DataFrame."""
    return (
        inter_df.groupby("qid")["query_text"]
        .first()
        .to_dict()
    )
