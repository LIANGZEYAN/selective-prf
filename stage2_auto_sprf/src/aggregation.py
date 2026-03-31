"""
Aggregation functions: DCG ratio, MRR ratio, Majority@k ratio.

Pure math — no model loading, no API calls. Takes a ranked list and
sets of PRF / original document IDs as input, returns a float score.
"""

import math

from src.config import TOP_K_MAJORITY, PRF_LABEL, BASE_LABEL


def dcg_weight(rank: int) -> float:
    """Standard DCG gain: 1 / log2(rank + 1). Rank is 1-based."""
    if rank < 1:
        raise ValueError(f"Rank must be >= 1, got {rank}")
    return 1.0 / math.log2(rank + 1)


def mrr_weight(rank: int) -> float:
    """MRR-style weight: 1 / rank. Rank is 1-based."""
    if rank < 1:
        raise ValueError(f"Rank must be >= 1, got {rank}")
    return 1.0 / rank


def compute_dcg_ratio(
    ranked_doc_ids: list,
    origin_labels: dict,
) -> float:
    """DCG-weighted PRF preference ratio.

    Parameters
    ----------
    ranked_doc_ids : list of doc IDs in rank order (index 0 = rank 1).
    origin_labels : dict mapping doc_id -> origin_label string.

    Returns
    -------
    float in [0, 1], or NaN if no discriminative documents.
    """
    g_prf = 0.0
    g_base = 0.0
    for i, doc_id in enumerate(ranked_doc_ids):
        rank = i + 1
        label = origin_labels.get(doc_id, "")
        gain = dcg_weight(rank)
        if label == PRF_LABEL:
            g_prf += gain
        elif label == BASE_LABEL:
            g_base += gain

    total = g_prf + g_base
    if total == 0:
        return float("nan")
    return g_prf / total


def compute_mrr_ratio(
    ranked_doc_ids: list,
    origin_labels: dict,
) -> float:
    """MRR-weighted PRF preference ratio.

    Same logic as DCG ratio but uses 1/rank instead of 1/log2(rank+1).
    """
    g_prf = 0.0
    g_base = 0.0
    for i, doc_id in enumerate(ranked_doc_ids):
        rank = i + 1
        label = origin_labels.get(doc_id, "")
        gain = mrr_weight(rank)
        if label == PRF_LABEL:
            g_prf += gain
        elif label == BASE_LABEL:
            g_base += gain

    total = g_prf + g_base
    if total == 0:
        return float("nan")
    return g_prf / total


def compute_majority_at_k(
    ranked_doc_ids: list,
    origin_labels: dict,
    top_k: int = TOP_K_MAJORITY,
) -> float:
    """Majority@k ratio: fraction of discriminative documents in top-k that are PRF.

    Position-agnostic vote among top-k documents — PRF wins if it holds
    the majority of discriminative slots.
    """
    n_prf = 0
    n_base = 0
    for doc_id in ranked_doc_ids[:top_k]:
        label = origin_labels.get(doc_id, "")
        if label == PRF_LABEL:
            n_prf += 1
        elif label == BASE_LABEL:
            n_base += 1

    total = n_prf + n_base
    if total == 0:
        return float("nan")
    return n_prf / total
