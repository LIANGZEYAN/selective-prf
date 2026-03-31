"""
Lexical pre-hoc features: Named Entity Density and IDF Coverage.
"""

import math
from collections import Counter


def named_entity_density(query_text: str, nlp) -> float:
    """Fraction of query tokens that are part of a named entity.

    Parameters
    ----------
    query_text : raw query string.
    nlp : a loaded spaCy model (e.g. en_core_web_sm).  Passed in so
          the model is loaded once at the caller level.

    Returns
    -------
    float in [0, 1].  Returns 0.0 for empty queries.
    """
    doc = nlp(query_text)
    if len(doc) == 0:
        return 0.0
    ent_tokens = sum(len(ent) for ent in doc.ents)
    return ent_tokens / len(doc)


def idf_coverage(query_text: str, idf_table: dict) -> float:
    """Fraction of query terms that appear in the IDF table,
    weighted by their IDF scores.

    Parameters
    ----------
    query_text : raw query string (will be lowercased and split on whitespace).
    idf_table : dict mapping term -> IDF score.

    Returns
    -------
    float.  Mean IDF of covered terms, or 0.0 if no terms are covered.
    """
    tokens = query_text.lower().split()
    if not tokens:
        return 0.0
    covered_idfs = [idf_table[t] for t in tokens if t in idf_table]
    if not covered_idfs:
        return 0.0
    return sum(covered_idfs) / len(tokens)


def load_idf_table(corpus_texts: list) -> dict:
    """Build a simple IDF table from a corpus of document texts.

    Parameters
    ----------
    corpus_texts : list of document text strings.

    Returns
    -------
    dict mapping term (str) -> IDF score (float).
    """
    n_docs = len(corpus_texts)
    if n_docs == 0:
        return {}
    doc_freq = Counter()
    for text in corpus_texts:
        terms = set(str(text).lower().split())
        doc_freq.update(terms)
    return {
        term: math.log((n_docs + 1) / (df + 1))
        for term, df in doc_freq.items()
    }
