"""
QPP baseline wrappers: WIG, NQC, Clarity, SMV.

Thin wrappers that compute query performance prediction scores from
initial ranking scores.  These do not require any LLM — they are
statistical predictors based on retrieval score distributions.
"""

import math

import numpy as np


def wig(scores: np.ndarray, corpus_score: float, k: int = 10) -> float:
    """Weighted Information Gain.

    WIG = (1/k) * sum_{i=1}^{k} (score_i - corpus_score)

    Parameters
    ----------
    scores : retrieval scores for the top-k documents, in rank order.
    corpus_score : average retrieval score across the entire corpus.
    k : number of top documents to consider.
    """
    top_k = scores[:k]
    if len(top_k) == 0:
        return 0.0
    return float(np.mean(top_k) - corpus_score)


def nqc(scores: np.ndarray, corpus_score: float, k: int = 10) -> float:
    """Normalised Query Commitment.

    NQC = std(top-k scores) / corpus_score

    Parameters
    ----------
    scores : retrieval scores for the top-k documents.
    corpus_score : average retrieval score across the entire corpus.
    k : number of top documents to consider.
    """
    top_k = scores[:k]
    if len(top_k) == 0 or corpus_score == 0:
        return 0.0
    return float(np.std(top_k) / abs(corpus_score))


def clarity(scores: np.ndarray, k: int = 10) -> float:
    """Simplified Clarity Score.

    Approximated as the KL-divergence between the top-k score distribution
    (normalised to a probability) and a uniform distribution.

    Parameters
    ----------
    scores : retrieval scores for the top-k documents.
    k : number of top documents to consider.
    """
    top_k = scores[:k]
    if len(top_k) == 0:
        return 0.0
    # Shift to positive if needed and normalise
    shifted = top_k - top_k.min() + 1e-10
    probs = shifted / shifted.sum()
    uniform = 1.0 / len(probs)
    kl = 0.0
    for p in probs:
        if p > 0:
            kl += p * math.log(p / uniform)
    return float(kl)


def smv(scores: np.ndarray, k: int = 10) -> float:
    """Score Magnitude and Variance.

    SMV = mean(top-k) * (1 / (1 + std(top-k)))

    Higher when top-k scores are high and tightly clustered.

    Parameters
    ----------
    scores : retrieval scores for the top-k documents.
    k : number of top documents to consider.
    """
    top_k = scores[:k]
    if len(top_k) == 0:
        return 0.0
    return float(np.mean(top_k) / (1.0 + np.std(top_k)))
