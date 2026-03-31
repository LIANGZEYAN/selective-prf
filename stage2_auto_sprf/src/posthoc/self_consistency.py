"""
Self-Consistency methods: SC-Mean and SC-SNR.

Runs a reranker multiple times and aggregates the per-run scores.
"""

import numpy as np

from src.config import SC_NUM_RUNS


def run_self_consistency(
    reranker,
    query: str,
    doc_list: list,
    n_runs: int = SC_NUM_RUNS,
    aggregation_fn=None,
) -> list:
    """Run the reranker n_runs times and collect per-run ratio scores.

    Parameters
    ----------
    reranker : object with rerank_return_ids(query, doc_list) method.
    query : query text string.
    doc_list : list of dicts with 'docid' and 'text'.
    n_runs : number of independent runs.
    aggregation_fn : callable(ranked_doc_ids, origin_labels) -> float.
        If provided, computes a score per run using this function.

    Returns
    -------
    list of float scores, one per run.
    """
    scores = []
    origin_labels = {d["docid"]: d.get("origin_label", "") for d in doc_list}

    for _ in range(n_runs):
        ranked_ids = reranker.rerank_return_ids(query, doc_list)
        if aggregation_fn is not None:
            score = aggregation_fn(ranked_ids, origin_labels)
        else:
            score = 0.0
        scores.append(score)

    return scores


def sc_mean(scores: list) -> float:
    """Mean of self-consistency scores."""
    if not scores:
        return 0.0
    return float(np.mean(scores))


def sc_snr(scores: list) -> float:
    """Signal-to-noise ratio: mean / std. Returns 0 if std == 0."""
    if not scores:
        return 0.0
    std = np.std(scores)
    if std == 0:
        return 0.0
    return float(np.mean(scores) / std)
