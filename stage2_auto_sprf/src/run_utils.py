"""
Shared utilities for run scripts: logging setup and summary table management.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, SUMMARY_TABLE_PATH, SUMMARY_COLUMNS, LOGS_DIR


def setup_logging(script_name: str) -> logging.Logger:
    """Configure dual logging to stdout and a log file in results/logs/."""
    logs_dir = PROJECT_ROOT / LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{script_name}_{timestamp}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("Log file: %s", log_file)
    return logger


def save_per_query_results(
    results: list,
    output_path: Path,
) -> pd.DataFrame:
    """Save per-query scores to CSV.

    Parameters
    ----------
    results : list of dicts, each with at least query_id, score, prediction.
    output_path : path to write the CSV.

    Returns
    -------
    The saved DataFrame.
    """
    df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def append_to_summary(row: dict) -> None:
    """Append one row to results/summary_table.csv.

    If the method_name already exists, overwrite that row.
    """
    summary_path = PROJECT_ROOT / SUMMARY_TABLE_PATH
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if summary_path.exists():
        df = pd.read_csv(summary_path)
        # Remove existing row for this method
        df = df[df["method_name"] != row["method_name"]]
    else:
        df = pd.DataFrame(columns=SUMMARY_COLUMNS)

    new_row = pd.DataFrame([row], columns=SUMMARY_COLUMNS)
    df = pd.concat([df, new_row], ignore_index=True)
    df = df[SUMMARY_COLUMNS]

    df.to_csv(summary_path, index=False)
