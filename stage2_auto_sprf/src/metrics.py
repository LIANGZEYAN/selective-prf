"""
Evaluation metrics for selective PRF prediction.

All classification and agreement metrics in one place:
TPR_H, TPR_B, Accuracy, AUC, Goodman-Kruskal Gamma, Spearman rho.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from src.config import GAMMA_N_PERMUTATIONS, H_THRESHOLD


# ── Classification metrics ───────────────────────────────────────────────────

def compute_tpr(y_true: np.ndarray, y_pred: np.ndarray, pos_label: str) -> float:
    """True positive rate for a specific label (Hurt or Benefit)."""
    mask = y_true == pos_label
    n = mask.sum()
    if n == 0:
        return float("nan")
    return ((y_pred[mask] == pos_label).sum()) / n


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy on non-neutral queries only (Hurt + Benefit)."""
    mask = (y_true == "Hurt") | (y_true == "Benefit")
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if len(y_t) == 0:
        return float("nan")
    return (y_t == y_p).sum() / len(y_t)


# ── Agreement metrics ────────────────────────────────────────────────────────

def compute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC (c-statistic) — Benefit=1 vs Hurt=0, others excluded."""
    mask = (y_true == "Hurt") | (y_true == "Benefit")
    y_bin = (y_true[mask] == "Benefit").astype(int)
    s = scores[mask]
    if len(np.unique(y_bin)) < 2:
        return float("nan")
    # Manual Mann-Whitney AUC
    pos = s[y_bin == 1]
    neg = s[y_bin == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u_stat = 0.0
    for p in pos:
        u_stat += (neg < p).sum() + 0.5 * (neg == p).sum()
    return u_stat / (n_pos * n_neg)


def _goodman_kruskal_gamma(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Goodman-Kruskal Gamma between two ordinal arrays."""
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            # ties (prod == 0) are excluded
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def compute_gamma(
    labels: np.ndarray,
    scores: np.ndarray,
    n_permutations: int = GAMMA_N_PERMUTATIONS,
) -> tuple:
    """Goodman-Kruskal Gamma + permutation p-value.

    labels: ordinal ground truth (Hurt < Neutral < Benefit), encoded as int.
    scores: continuous predictor (e.g., LLM ratio).
    Returns (gamma, p_value).
    """
    # Encode labels to ordinal integers
    label_order = {"Hurt": 0, "Neutral": 1, "Benefit": 2}
    if isinstance(labels[0], str):
        x = np.array([label_order.get(l, 1) for l in labels], dtype=float)
    else:
        x = labels.astype(float)
    y = scores.astype(float)

    # Remove NaN pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    observed = _goodman_kruskal_gamma(x, y)

    rng = np.random.default_rng(42)
    count_extreme = 0
    for _ in range(n_permutations):
        perm_y = rng.permutation(y)
        if abs(_goodman_kruskal_gamma(x, perm_y)) >= abs(observed):
            count_extreme += 1
    p_value = (count_extreme + 1) / (n_permutations + 1)

    return observed, p_value


def compute_rho(b_ratios: np.ndarray, scores: np.ndarray) -> tuple:
    """Spearman rho + p-value between human b_ratio and LLM scores."""
    valid = ~(np.isnan(b_ratios) | np.isnan(scores))
    rho, p = stats.spearmanr(b_ratios[valid], scores[valid])
    return rho, p


# ── Unified evaluation ──────────────────────────────────────────────────────

def classify_scores(scores: pd.Series, h: float = H_THRESHOLD) -> pd.Series:
    """Classify continuous scores into Hurt/Benefit/Neutral."""

    def _classify(ratio: float) -> str:
        if np.isnan(ratio):
            return "Insufficient-Data"
        if ratio > 0.5 + h:
            return "Benefit"
        if ratio < 0.5 - h:
            return "Hurt"
        return "Neutral"

    return scores.apply(_classify)


def evaluate_all(
    labels_df: pd.DataFrame,
    scores_series: pd.Series,
    method_name: str,
    stage: str,
    predictor: str,
) -> dict:
    """Compute all metrics and return a dict ready for summary_table.csv.

    Parameters
    ----------
    labels_df : DataFrame with columns query_id, b_ratio, label.
    scores_series : Series indexed by query_id with continuous scores.
    method_name : e.g. "Qwen-72B-DCG"
    stage : "prehoc" or "posthoc" or "qpp"
    predictor : short description of the predictor
    """
    # Align
    merged = labels_df.set_index("query_id").join(
        scores_series.rename("score"), how="inner"
    )

    # Exclude Insufficient-Data
    merged = merged[merged["label"] != "Insufficient-Data"].copy()

    y_true = merged["label"].values
    scores = merged["score"].values
    y_pred = classify_scores(merged["score"]).values

    # Non-neutral mask
    non_neutral = (y_true == "Hurt") | (y_true == "Benefit")

    n_hurt = (y_true == "Hurt").sum()
    n_benefit = (y_true == "Benefit").sum()

    tpr_h = compute_tpr(y_true, y_pred, "Hurt")
    tpr_b = compute_tpr(y_true, y_pred, "Benefit")
    acc = compute_accuracy(y_true, y_pred)
    auc = compute_auc(y_true, scores)

    gamma, gamma_p = compute_gamma(y_true, scores)
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
    }
