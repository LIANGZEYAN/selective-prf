"""
Semantic pre-hoc feature: Query Term Coherence.

Uses sentence-transformers to measure how coherent the query terms are
with each other in embedding space.
"""

import numpy as np


def query_term_coherence(query_text: str, model) -> float:
    """Average pairwise cosine similarity between query term embeddings.

    Parameters
    ----------
    query_text : raw query string.
    model : a loaded SentenceTransformer model.  Passed in so the model
            is loaded once at the caller level.

    Returns
    -------
    float in [-1, 1].  Returns 0.0 for queries with fewer than 2 tokens.
    """
    tokens = query_text.split()
    if len(tokens) < 2:
        return 0.0

    embeddings = model.encode(tokens, convert_to_numpy=True)
    # Normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms

    # Pairwise cosine similarity (upper triangle only)
    sim_matrix = embeddings @ embeddings.T
    n = len(tokens)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sim_matrix[i, j]
            count += 1

    if count == 0:
        return 0.0
    return total / count
