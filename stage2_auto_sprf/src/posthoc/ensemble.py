"""
Ensemble methods: Doubly-Robust combinations and reasoning trace mining.

No LLM calls — these are pure mathematical combinations of cached scores.
"""

import math
from pathlib import Path

from src.config import PROJECT_ROOT, DRIFT_LEXICON_PATH


def doubly_robust(mu: float, sigma: float, lambda_: float = 1.0) -> float:
    """DR score: mu * exp(-lambda * sigma).

    Downweights high-variance predictions.
    """
    return mu * math.exp(-lambda_ * sigma)


def setwise_dr(
    dcg_ratio: float,
    delta_reason: float,
    lambda_ratio: float,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
) -> float:
    """Setwise doubly-robust combination.

    Combines DCG ratio with reasoning trace signals.
    """
    base = dcg_ratio
    penalty = lambda1 * delta_reason + lambda2 * lambda_ratio
    return base * math.exp(-penalty)


def load_drift_lexicon(path: Path = None) -> set:
    """Load drift-indicative terms from a plain text file.

    Returns a set of lowercase strings.
    """
    if path is None:
        path = PROJECT_ROOT / DRIFT_LEXICON_PATH
    if not path.exists():
        return set()
    terms = set()
    with open(path) as f:
        for line in f:
            term = line.strip().lower()
            if term:
                terms.add(term)
    return terms


def reasoning_drift_score(reasoning_trace: str, drift_lexicon: set = None) -> float:
    """Count normalised drift-indicative terms in a reasoning trace.

    Parameters
    ----------
    reasoning_trace : raw text from the <think> block.
    drift_lexicon : set of drift-indicative strings.

    Returns
    -------
    float — normalised count (hits / total_words). 0.0 if trace is empty.
    """
    if drift_lexicon is None:
        drift_lexicon = load_drift_lexicon()

    if not reasoning_trace:
        return 0.0

    trace_lower = reasoning_trace.lower()
    words = trace_lower.split()
    if not words:
        return 0.0

    hits = sum(1 for term in drift_lexicon if term in trace_lower)
    return hits / len(words)


def reasoning_length_ratio(tokens_prf: int, tokens_orig: int) -> float:
    """Ratio of reasoning tokens for PRF docs vs original docs.

    Returns tokens_prf / tokens_orig, or 0.0 if tokens_orig == 0.
    """
    if tokens_orig == 0:
        return 0.0
    return tokens_prf / tokens_orig
