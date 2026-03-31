"""
Pre-PRF features: Top-k Document Coherence, Pre-PRF Self-Consistency,
and QPP wrappers (WIG, NQC, Clarity, SMV).
"""

import random

import numpy as np


def top_k_doc_coherence(doc_texts: list, model) -> float:
    """Average pairwise cosine similarity between top-k document embeddings.

    Parameters
    ----------
    doc_texts : list of document text strings.
    model : a loaded SentenceTransformer model.

    Returns
    -------
    float in [-1, 1].  Returns 0.0 if fewer than 2 documents.
    """
    if len(doc_texts) < 2:
        return 0.0

    embeddings = model.encode(doc_texts, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms

    sim_matrix = embeddings @ embeddings.T
    n = len(doc_texts)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sim_matrix[i, j]
            count += 1

    if count == 0:
        return 0.0
    return total / count


def preprf_self_consistency(
    query_id,
    initial_ranking_df,
    reranker_fn,
    n_samples: int = 20,
    seed: int = 42,
) -> float:
    """Pre-PRF self-consistency via sampled pairwise comparisons.

    Samples random document pairs from the initial top-k, calls the
    reranker on each pair, and checks transitive consistency.

    Parameters
    ----------
    query_id : query identifier (for filtering).
    initial_ranking_df : DataFrame with columns [docno, passage_text]
        for this query, sorted by rank.
    reranker_fn : callable(doc_a_text, doc_b_text, query_text) -> str
        Returns the docno of the preferred document.
    n_samples : number of random pairs to sample.
    seed : random seed for reproducibility.

    Returns
    -------
    float in [0, 1] — fraction of consistent transitive judgements.
    """
    docs = list(initial_ranking_df.itertuples(index=False))
    if len(docs) < 3:
        return 0.0

    rng = random.Random(seed)
    preferences = {}  # (doc_a, doc_b) -> winner

    # Sample pairs
    all_pairs = [(i, j) for i in range(len(docs)) for j in range(i + 1, len(docs))]
    sampled = rng.sample(all_pairs, min(n_samples, len(all_pairs)))

    for i, j in sampled:
        winner = reranker_fn(docs[i].passage_text, docs[j].passage_text)
        preferences[(i, j)] = winner

    # Check transitive consistency
    consistent = 0
    total_triples = 0
    indices = list(set(idx for pair in sampled for idx in pair))

    for a in indices:
        for b in indices:
            for c in indices:
                if a >= b or b >= c:
                    continue
                ab = preferences.get((a, b))
                bc = preferences.get((b, c))
                ac = preferences.get((a, c))
                if ab is None or bc is None or ac is None:
                    continue
                total_triples += 1
                # If a > b and b > c, then a should > c
                if ab == a and bc == b and ac == a:
                    consistent += 1
                elif ab == b and bc == c and ac == c:
                    consistent += 1
                elif ab == a and bc == c:
                    consistent += 1  # a > b, c > b — no constraint on a vs c
                elif ab == b and bc == b:
                    consistent += 1  # b > a, b > c — no constraint on a vs c

    if total_triples == 0:
        return 0.0
    return consistent / total_triples
