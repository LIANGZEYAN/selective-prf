#!/usr/bin/env python3
"""
Approach 1a: Off-the-shelf Query Encoder as Classifier.

Uses pre-trained models (zero-shot NLI, sentence embeddings, NER) to predict
whether PRF will help or hurt — no fine-tuning, query text only.

Outputs:
  results/classifier/approach_1a_scores.csv   – per-query scores
  results/classifier/approach_1a_results.csv  – metrics summary
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROJECT_ROOT as ROOT, QWEN_7B_CACHE_DIR,
    BGE_LARGE_MODEL, GTE_LARGE_MODEL, COLBERT_V2_MODEL,
)
from src.metrics import evaluate_all
from src.run_utils import setup_logging, append_to_summary

HF_CACHE = QWEN_7B_CACHE_DIR  # shared HF cache
OUT_DIR = ROOT / "results" / "classifier"
DATA_DIR = ROOT / "data" / "classifier"


# ── Data loading ────────────────────────────────────────────────────────────

def load_queries() -> dict:
    """Return {qid: query_text}."""
    df = pd.read_csv(DATA_DIR / "queries.tsv", sep="\t")
    return dict(zip(df["qid"], df["query_text"]))


def load_labels() -> pd.DataFrame:
    """Return labels DataFrame (Benefit/Hurt only)."""
    return pd.read_csv(DATA_DIR / "labels.tsv", sep="\t")


# ── Feature 1: Zero-shot NLI classification ─────────────────────────────────

def zero_shot_classify(queries: dict, model_name: str, logger) -> pd.DataFrame:
    """Use an NLI model as zero-shot classifier.

    Returns DataFrame with columns: qid, benefit_prob, hurt_prob.
    """
    from transformers import pipeline

    logger.info(f"Loading zero-shot pipeline: {model_name}")
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0,
        model_kwargs={"cache_dir": HF_CACHE},
    )

    candidate_labels = [
        "this query will benefit from query expansion",
        "this query will be hurt by query expansion",
    ]

    results = []
    for qid, text in queries.items():
        out = classifier(text, candidate_labels)
        label_scores = dict(zip(out["labels"], out["scores"]))
        results.append({
            "qid": qid,
            "benefit_prob": label_scores[candidate_labels[0]],
            "hurt_prob": label_scores[candidate_labels[1]],
        })
        logger.info(f"  qid={qid}: benefit={label_scores[candidate_labels[0]]:.3f}, "
                     f"hurt={label_scores[candidate_labels[1]]:.3f}")

    return pd.DataFrame(results)


# ── Feature 2: Sentence embedding features ──────────────────────────────────

def compute_embedding_features(
    queries: dict, model_name: str, logger, query_prefix: str = "",
) -> pd.DataFrame:
    """Compute query embedding norm and term-spread features."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading sentence-transformer: {model_name}")
    model = SentenceTransformer(model_name, cache_folder=HF_CACHE)

    results = []
    for qid, text in queries.items():
        # Full query embedding (with optional prefix for BGE etc.)
        q_emb = model.encode(query_prefix + text, normalize_embeddings=False)
        q_norm = np.linalg.norm(q_emb)

        # Per-term embeddings
        terms = text.strip().split()
        if len(terms) > 1:
            term_texts = [query_prefix + t for t in terms]
            term_embs = model.encode(term_texts, normalize_embeddings=True)
            q_emb_normed = q_emb / (q_norm + 1e-9)
            # Cosine similarity between query and each term
            term_sims = term_embs @ q_emb_normed
            term_sim_mean = float(np.mean(term_sims))
            term_sim_std = float(np.std(term_sims))
        else:
            term_sim_mean = 1.0
            term_sim_std = 0.0

        results.append({
            "qid": qid,
            "emb_norm": float(q_norm),
            "term_sim_mean": term_sim_mean,
            "term_sim_std": term_sim_std,
            "n_terms": len(terms),
        })

    return pd.DataFrame(results)


def compute_colbert_features(queries: dict, model_name: str, logger) -> pd.DataFrame:
    """Compute query features using ColBERTv2 token-level embeddings.

    ColBERT produces per-token embeddings, making it a natural fit for
    measuring per-term similarity within a query.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    logger.info(f"Loading ColBERT model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE)
    model = AutoModel.from_pretrained(model_name, cache_dir=HF_CACHE)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = []
    for qid, text in queries.items():
        terms = text.strip().split()

        # Tokenize with offset mapping to align tokens to terms
        enc = tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True,
            padding=True, truncation=True, max_length=128,
        )
        offsets = enc.pop("offset_mapping")[0]  # (seq_len, 2)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
        token_embs = out.last_hidden_state[0].cpu().numpy()  # (seq_len, dim)

        # CLS embedding as query representation
        cls_emb = token_embs[0]
        q_norm = float(np.linalg.norm(cls_emb))

        # Map each whitespace term to its token indices via character offsets
        term_char_spans = []
        pos = 0
        for term in terms:
            start = text.index(term, pos)
            end = start + len(term)
            term_char_spans.append((start, end))
            pos = end

        term_embeddings = []
        for t_start, t_end in term_char_spans:
            tok_indices = []
            for idx, (o_start, o_end) in enumerate(offsets.tolist()):
                if o_start == 0 and o_end == 0:
                    continue  # special tokens
                if o_start >= t_start and o_end <= t_end:
                    tok_indices.append(idx)
            if tok_indices:
                term_emb = token_embs[tok_indices].mean(axis=0)
                term_embeddings.append(term_emb)

        if len(term_embeddings) > 1:
            term_embs_arr = np.array(term_embeddings)
            # Normalise
            norms = np.linalg.norm(term_embs_arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            term_embs_arr = term_embs_arr / norms
            cls_normed = cls_emb / (q_norm + 1e-9)
            term_sims = term_embs_arr @ cls_normed
            term_sim_mean = float(np.mean(term_sims))
            term_sim_std = float(np.std(term_sims))
        else:
            term_sim_mean = 1.0
            term_sim_std = 0.0

        results.append({
            "qid": qid,
            "emb_norm": q_norm,
            "term_sim_mean": term_sim_mean,
            "term_sim_std": term_sim_std,
            "n_terms": len(terms),
        })

    # Free GPU
    import gc
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return pd.DataFrame(results)


# ── Feature 3: Named Entity Density ─────────────────────────────────────────

def compute_ner_features(queries: dict, model_name: str, logger) -> pd.DataFrame:
    """Count named entities per query using spaCy."""
    import spacy
    logger.info(f"Loading spaCy {model_name}")
    nlp = spacy.load(model_name)

    results = []
    for qid, text in queries.items():
        doc = nlp(text)
        results.append({
            "qid": qid,
            "n_entities": len(doc.ents),
            "entity_ratio": len(doc.ents) / max(len(text.split()), 1),
        })

    return pd.DataFrame(results)


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_feature(
    feature_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    score_col: str,
    method_name: str,
    higher_means_benefit: bool = True,
    logger=None,
) -> dict:
    """Evaluate a single feature as a predictor using the project's metrics."""
    merged = labels_df.merge(feature_df[["qid", score_col]], left_on="query_id", right_on="qid")
    scores = merged[score_col].values
    if not higher_means_benefit:
        scores = -scores  # flip so higher = more likely benefit

    scores_series = pd.Series(scores, index=merged["query_id"])
    # Normalise to [0, 1] for AUC computation
    s_min, s_max = scores_series.min(), scores_series.max()
    if s_max > s_min:
        scores_norm = (scores_series - s_min) / (s_max - s_min)
    else:
        scores_norm = scores_series * 0.0 + 0.5

    result = evaluate_all(labels_df, scores_norm, method_name, "classifier-1a", score_col)
    if logger:
        logger.info(f"  {method_name}: acc={result['acc']}, auc={result['auc']}, "
                     f"gamma={result['gamma']}, rho={result['rho']}")
    return result


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("approach_1a")

    queries = load_queries()
    labels = load_labels()
    logger.info(f"Loaded {len(queries)} queries, {len(labels)} labelled")

    # ── Run all feature extractors ──────────────────────────────────
    nli_models = [
        "facebook/bart-large-mnli",
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    ]

    all_nli_dfs = []
    all_emb_dfs = {}
    all_ner_dfs = {}
    all_results = []

    # ── NLI classifiers ─────────────────────────────────────────────
    for nli_model in nli_models:
        short_name = nli_model.split("/")[-1]
        logger.info(f"\n{'='*60}\nZero-shot NLI: {nli_model}\n{'='*60}")
        nli_df = zero_shot_classify(queries, nli_model, logger)

        result = evaluate_feature(
            nli_df, labels, "benefit_prob",
            f"ZeroShot-{short_name}", higher_means_benefit=True, logger=logger,
        )
        all_results.append(result)
        nli_df["nli_model"] = short_name
        all_nli_dfs.append(nli_df)

    # ── Embedding features (multiple models) ────────────────────────
    # (short_tag, model_id, is_colbert, query_prefix)
    embedding_models = [
        ("MiniLM", "all-MiniLM-L6-v2", False, ""),
        ("BGE", BGE_LARGE_MODEL, False, "Represent this sentence: "),
        ("GTE", GTE_LARGE_MODEL, False, ""),
        ("ColBERT", COLBERT_V2_MODEL, True, ""),
    ]

    for tag, model_id, is_colbert, prefix in embedding_models:
        logger.info(f"\n{'='*60}\nEmbedding features: {tag} ({model_id})\n{'='*60}")

        if is_colbert:
            emb_df = compute_colbert_features(queries, model_id, logger)
        else:
            emb_df = compute_embedding_features(queries, model_id, logger, query_prefix=prefix)

        all_emb_dfs[tag] = emb_df

        # n_terms is model-independent; only evaluate it once (for MiniLM)
        feature_cols = [
            ("emb_norm", True),
            ("term_sim_mean", True),
            ("term_sim_std", False),
        ]
        if tag == "MiniLM":
            feature_cols.append(("n_terms", False))

        for col, higher_benefit in feature_cols:
            method_name = f"Emb-{tag}-{col}" if tag != "MiniLM" else f"Emb-{col}"
            result = evaluate_feature(
                emb_df, labels, col,
                method_name, higher_means_benefit=higher_benefit, logger=logger,
            )
            all_results.append(result)

    # ── NER features (multiple spaCy models) ────────────────────────
    ner_models = [
        ("sm", "en_core_web_sm"),
        ("trf", "en_core_web_trf"),
    ]

    for tag, model_name in ner_models:
        logger.info(f"\n{'='*60}\nNER features: {tag} ({model_name})\n{'='*60}")
        ner_df = compute_ner_features(queries, model_name, logger)
        all_ner_dfs[tag] = ner_df

        for col, higher_benefit in [("n_entities", True), ("entity_ratio", True)]:
            method_name = f"NER-{tag}-{col}" if tag != "sm" else f"NER-{col}"
            result = evaluate_feature(
                ner_df, labels, col,
                method_name, higher_means_benefit=higher_benefit, logger=logger,
            )
            all_results.append(result)

    # ── Save results ────────────────────────────────────────────────
    # Merge all features into one CSV
    features_merged = pd.DataFrame({"qid": list(queries.keys())})

    for tag, emb_df in all_emb_dfs.items():
        prefix = tag.lower()
        renamed = emb_df.rename(
            columns={c: f"{prefix}_{c}" for c in emb_df.columns if c != "qid"}
        )
        features_merged = features_merged.merge(renamed, on="qid")

    for tag, ner_df in all_ner_dfs.items():
        renamed = ner_df.rename(
            columns={c: f"ner_{tag}_{c}" for c in ner_df.columns if c != "qid"}
        )
        features_merged = features_merged.merge(renamed, on="qid")

    # Add NLI model scores
    for nli_df in all_nli_dfs:
        short = nli_df["nli_model"].iloc[0]
        nli_cols = nli_df[["qid", "benefit_prob", "hurt_prob"]].copy()
        nli_cols = nli_cols.rename(columns={
            "benefit_prob": f"nli_{short}_benefit",
            "hurt_prob": f"nli_{short}_hurt",
        })
        features_merged = features_merged.merge(nli_cols, on="qid")

    features_merged.to_csv(OUT_DIR / "approach_1a_features.csv", index=False)
    logger.info(f"Saved features → {OUT_DIR / 'approach_1a_features.csv'}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "approach_1a_results.csv", index=False)
    logger.info(f"Saved results → {OUT_DIR / 'approach_1a_results.csv'}")

    # Append to project summary table
    for r in all_results:
        append_to_summary(r)

    # ── Print summary ───────────────────────────────────────────────
    logger.info(f"\n{'='*60}\nApproach 1a Summary\n{'='*60}")
    cols = ["method_name", "acc", "auc", "gamma", "rho"]
    logger.info("\n" + results_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
