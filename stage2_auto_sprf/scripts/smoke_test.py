#!/usr/bin/env python3
"""
Smoke test: verify every approach can run end-to-end on a 3-query mini dataset.

Exit code 0 if all pass (or skip), 1 if any fail.
"""

import os
import sys
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROJECT_ROOT as ROOT,
    QWEN_72B_API_KEY_ENV,
    H_THRESHOLD,
)
from src.data_loader import load_labels, load_interleaved, load_colbert, get_query_texts
from src.aggregation import compute_dcg_ratio, compute_mrr_ratio, compute_majority_at_k
from src.metrics import classify_scores, evaluate_all


# ── Helpers ──────────────────────────────────────────────────────────────────

class Result:
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


def _check_output(results: list, name: str) -> tuple:
    """Validate output format. Returns (status, reason)."""
    if not results:
        return Result.FAIL, "empty results"

    for r in results:
        if "query_id" not in r or "score" not in r or "prediction" not in r:
            return Result.FAIL, f"missing columns in result: {list(r.keys())}"

        score = r["score"]
        if score is None or (isinstance(score, float) and (math.isnan(score) or math.isinf(score))):
            # NaN is acceptable for Insufficient-Data queries, but not for our test set
            return Result.FAIL, f"non-finite score for qid={r['query_id']}: {score}"

        pred = r["prediction"]
        if pred not in ("Hurt", "Benefit"):
            return Result.FAIL, f"invalid prediction '{pred}' for qid={r['query_id']}"

    return Result.PASS, "all checks passed"


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _api_key_set() -> bool:
    return bool(os.environ.get(QWEN_72B_API_KEY_ENV))


# ── Test runners ─────────────────────────────────────────────────────────────

def test_build_labels() -> tuple:
    """Test that build_labels produces correct output."""
    try:
        labels_df = load_labels()
        n_hurt = (labels_df["label"] == "Hurt").sum()
        n_benefit = (labels_df["label"] == "Benefit").sum()
        if n_hurt != 14 or n_benefit != 12:
            return Result.FAIL, f"expected 14H/12B, got {n_hurt}H/{n_benefit}B"
        return Result.PASS, f"14 Hurt, 12 Benefit, h={H_THRESHOLD}"
    except Exception as e:
        return Result.FAIL, str(e)


def test_ned(query_texts: dict, test_qids: list) -> tuple:
    """Named Entity Density."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        from src.prehoc.lexical import named_entity_density

        results = []
        for qid in test_qids:
            score = named_entity_density(query_texts[qid], nlp)
            pred = "Benefit" if score > 0.5 else "Hurt"
            results.append({"query_id": qid, "score": score, "prediction": pred})

        return _check_output(results, "NED")
    except ImportError:
        return Result.SKIP, "spacy not installed"
    except OSError:
        return Result.SKIP, "en_core_web_sm model not downloaded"
    except Exception as e:
        return Result.FAIL, str(e)


def test_idf_coverage(query_texts: dict, test_qids: list) -> tuple:
    """IDF Coverage."""
    try:
        from src.prehoc.lexical import idf_coverage, load_idf_table
        colbert_df = load_colbert()
        corpus = colbert_df["passage_text"].dropna().tolist()[:1000]  # subset for speed
        idf_table = load_idf_table(corpus)

        results = []
        for qid in test_qids:
            score = idf_coverage(query_texts[qid], idf_table)
            pred = "Benefit" if score > 0.5 else "Hurt"
            results.append({"query_id": qid, "score": score, "prediction": pred})

        return _check_output(results, "IDF-Coverage")
    except Exception as e:
        return Result.FAIL, str(e)


def test_qtc(query_texts: dict, test_qids: list) -> tuple:
    """Query Term Coherence."""
    try:
        from sentence_transformers import SentenceTransformer
        from src.prehoc.semantic import query_term_coherence
        from src.config import SBERT_MODEL

        model = SentenceTransformer(SBERT_MODEL)
        results = []
        for qid in test_qids:
            score = query_term_coherence(query_texts[qid], model)
            pred = "Benefit" if score > 0.5 else "Hurt"
            results.append({"query_id": qid, "score": score, "prediction": pred})

        return _check_output(results, "QTC")
    except ImportError:
        return Result.SKIP, "sentence-transformers not installed"
    except Exception as e:
        return Result.FAIL, str(e)


def test_qpp(test_qids: list) -> tuple:
    """QPP baselines (WIG, NQC, Clarity, SMV)."""
    try:
        from src.qpp.baselines import wig, nqc, clarity, smv
        colbert_df = load_colbert()
        corpus_score = colbert_df["score"].mean()

        for fn_name, fn in [("WIG", wig), ("NQC", nqc), ("Clarity", clarity), ("SMV", smv)]:
            results = []
            for qid in test_qids:
                q_df = colbert_df[colbert_df["qid"] == qid].sort_values("rank")
                scores_arr = q_df["score"].values.astype(float)
                if fn_name in ("WIG", "NQC"):
                    score = fn(scores_arr, corpus_score)
                else:
                    score = fn(scores_arr)
                pred = "Benefit" if score > 0 else "Hurt"
                results.append({"query_id": qid, "score": score, "prediction": pred})

            status, reason = _check_output(results, fn_name)
            if status == Result.FAIL:
                return status, f"{fn_name}: {reason}"

        return Result.PASS, "all 4 QPP methods pass"
    except Exception as e:
        return Result.FAIL, str(e)


def test_tdc(test_qids: list) -> tuple:
    """Top-k Document Coherence."""
    try:
        from sentence_transformers import SentenceTransformer
        from src.prehoc.preprf import top_k_doc_coherence
        from src.config import SBERT_MODEL

        model = SentenceTransformer(SBERT_MODEL)
        colbert_df = load_colbert()

        results = []
        for qid in test_qids:
            q_df = colbert_df[colbert_df["qid"] == qid].sort_values("rank").head(10)
            doc_texts = q_df["passage_text"].dropna().tolist()
            score = top_k_doc_coherence(doc_texts, model)
            pred = "Benefit" if score > 0.5 else "Hurt"
            results.append({"query_id": qid, "score": score, "prediction": pred})

        return _check_output(results, "TDC")
    except ImportError:
        return Result.SKIP, "sentence-transformers not installed"
    except Exception as e:
        return Result.FAIL, str(e)


def test_aggregation(inter_df: pd.DataFrame, test_qids: list) -> tuple:
    """Aggregation functions (DCG, MRR, Majority@k)."""
    try:
        results = []
        for qid in test_qids:
            q_df = inter_df[inter_df["qid"] == qid].sort_values("rank")
            ranked_ids = q_df["docno"].tolist()
            origin_labels = dict(zip(q_df["docno"], q_df["origin_label"]))

            dcg = compute_dcg_ratio(ranked_ids, origin_labels)
            mrr = compute_mrr_ratio(ranked_ids, origin_labels)
            uw = compute_majority_at_k(ranked_ids, origin_labels)

            pred = classify_scores(pd.Series([dcg])).iloc[0]
            if pred not in ("Hurt", "Benefit"):
                pred = "Hurt"
            results.append({"query_id": qid, "score": dcg, "prediction": pred})

        return _check_output(results, "Aggregation")
    except Exception as e:
        return Result.FAIL, str(e)


def test_pairwise_7b(inter_df: pd.DataFrame, test_qids: list) -> tuple:
    """Qwen-7B local pairwise."""
    if not _gpu_available():
        return Result.SKIP, "CUDA not available"
    try:
        from src.posthoc.pairwise import PairwiseReranker
        reranker = PairwiseReranker()

        results = []
        for qid in test_qids:
            q_df = inter_df[inter_df["qid"] == qid].sort_values("rank")
            query = q_df["query_text"].iloc[0]
            doc_list = [
                {"docid": row["docno"], "text": str(row["passage_text"]) if pd.notna(row["passage_text"]) else ""}
                for _, row in q_df.iterrows()
            ]
            origin_labels = dict(zip(q_df["docno"], q_df["origin_label"]))
            ranked_ids = reranker.rerank_return_ids(query, doc_list)
            dcg = compute_dcg_ratio(ranked_ids, origin_labels)
            pred = classify_scores(pd.Series([dcg])).iloc[0]
            if pred not in ("Hurt", "Benefit"):
                pred = "Hurt"
            results.append({"query_id": qid, "score": dcg, "prediction": pred})

        return _check_output(results, "Qwen-7B")
    except Exception as e:
        return Result.FAIL, str(e)


def test_pairwise_72b(inter_df: pd.DataFrame, test_qids: list) -> tuple:
    """Qwen-72B API pairwise."""
    if not _api_key_set():
        return Result.SKIP, f"{QWEN_72B_API_KEY_ENV} not set"
    try:
        from src.posthoc.pairwise_api import PairwiseApiReranker
        reranker = PairwiseApiReranker()

        results = []
        for qid in test_qids:
            q_df = inter_df[inter_df["qid"] == qid].sort_values("rank")
            query = reranker.truncate_query(q_df["query_text"].iloc[0])
            doc_list = [
                {"docid": row["docno"], "text": reranker.truncate_passage(str(row["passage_text"]) if pd.notna(row["passage_text"]) else "")}
                for _, row in q_df.iterrows()
            ]
            origin_labels = dict(zip(q_df["docno"], q_df["origin_label"]))
            ranked_ids = reranker.rerank_return_ids(query, doc_list)
            dcg = compute_dcg_ratio(ranked_ids, origin_labels)
            pred = classify_scores(pd.Series([dcg])).iloc[0]
            if pred not in ("Hurt", "Benefit"):
                pred = "Hurt"
            results.append({"query_id": qid, "score": dcg, "prediction": pred})

        return _check_output(results, "Qwen-72B")
    except Exception as e:
        return Result.FAIL, str(e)


def test_setwise(inter_df: pd.DataFrame, test_qids: list) -> tuple:
    """Setwise reranking with Qwen-7B-Instruct base model (via vLLM)."""
    if not _gpu_available():
        return Result.SKIP, "CUDA not available"
    try:
        from src.posthoc.setwise import SetwiseReranker
        # Use base Qwen-7B (already in cache) — no Rank-R1 LoRA
        reranker = SetwiseReranker()

        results = []
        for qid in test_qids:
            q_df = inter_df[inter_df["qid"] == qid].sort_values("rank")
            query = q_df["query_text"].iloc[0]
            doc_list = [
                {"docid": row["docno"], "text": str(row["passage_text"]) if pd.notna(row["passage_text"]) else "", "origin_label": row["origin_label"]}
                for _, row in q_df.iterrows()
            ]
            origin_labels = dict(zip(q_df["docno"], q_df["origin_label"]))
            ranked_ids = reranker.rerank_return_ids(query, doc_list)
            dcg = compute_dcg_ratio(ranked_ids, origin_labels)
            pred = classify_scores(pd.Series([dcg])).iloc[0]
            if pred not in ("Hurt", "Benefit"):
                pred = "Hurt"
            results.append({"query_id": qid, "score": dcg, "prediction": pred})

        return _check_output(results, "Rank-R1")
    except ImportError:
        return Result.SKIP, "vllm not installed"
    except Exception as e:
        return Result.FAIL, str(e)


def test_self_consistency() -> tuple:
    """SC-Mean and SC-SNR functions."""
    try:
        from src.posthoc.self_consistency import sc_mean, sc_snr
        scores = [0.4, 0.5, 0.6, 0.55, 0.45]
        m = sc_mean(scores)
        s = sc_snr(scores)
        if not (0 < m < 1):
            return Result.FAIL, f"sc_mean out of range: {m}"
        if not isinstance(s, float):
            return Result.FAIL, f"sc_snr not float: {type(s)}"
        return Result.PASS, f"mean={m:.4f}, snr={s:.4f}"
    except Exception as e:
        return Result.FAIL, str(e)


def test_ensemble() -> tuple:
    """Ensemble DR functions."""
    try:
        from src.posthoc.ensemble import (
            doubly_robust, setwise_dr, reasoning_drift_score,
            reasoning_length_ratio, load_drift_lexicon,
        )

        dr = doubly_robust(0.6, 0.1, 1.0)
        if not isinstance(dr, float):
            return Result.FAIL, f"doubly_robust not float: {type(dr)}"

        sdr = setwise_dr(0.55, 0.05, 1.2, 1.0, 0.5)
        if not isinstance(sdr, float):
            return Result.FAIL, f"setwise_dr not float: {type(sdr)}"

        lexicon = load_drift_lexicon()
        drift = reasoning_drift_score("This passage is off-topic and irrelevant", lexicon)
        if drift <= 0:
            return Result.FAIL, f"drift score should be > 0, got {drift}"

        lr = reasoning_length_ratio(100, 50)
        if lr != 2.0:
            return Result.FAIL, f"length ratio should be 2.0, got {lr}"

        return Result.PASS, f"DR={dr:.4f}, SDR={sdr:.4f}, drift={drift:.4f}"
    except Exception as e:
        return Result.FAIL, str(e)


def test_metrics(labels_df: pd.DataFrame) -> tuple:
    """Metrics module."""
    try:
        # Create fake scores
        valid = labels_df[labels_df["label"] != "Insufficient-Data"]
        scores = pd.Series(
            np.random.default_rng(42).uniform(0, 1, len(valid)),
            index=valid["query_id"],
        )
        result = evaluate_all(labels_df, scores, "test", "test", "test")
        required = {"method_name", "tpr_hurt", "tpr_benefit", "acc", "auc", "gamma", "rho"}
        missing = required - set(result.keys())
        if missing:
            return Result.FAIL, f"missing keys: {missing}"
        return Result.PASS, f"all metric keys present"
    except Exception as e:
        return Result.FAIL, str(e)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  SMOKE TEST — Selective PRF Prediction Pipeline")
    print("=" * 70)
    print()

    labels_df = load_labels()
    inter_df = load_interleaved()
    query_texts = get_query_texts(inter_df)

    # First 3 non-neutral queries
    non_neutral = labels_df[labels_df["label"].isin(["Hurt", "Benefit"])]
    test_qids = sorted(non_neutral["query_id"].tolist())[:3]
    print(f"Test queries: {test_qids}")
    print()

    tests = [
        ("build_labels", lambda: test_build_labels()),
        ("metrics", lambda: test_metrics(labels_df)),
        ("aggregation", lambda: test_aggregation(inter_df, test_qids)),
        ("NED", lambda: test_ned(query_texts, test_qids)),
        ("IDF-Coverage", lambda: test_idf_coverage(query_texts, test_qids)),
        ("QTC", lambda: test_qtc(query_texts, test_qids)),
        ("QPP-baselines", lambda: test_qpp(test_qids)),
        ("TDC", lambda: test_tdc(test_qids)),
        ("SC-functions", lambda: test_self_consistency()),
        ("ensemble-functions", lambda: test_ensemble()),
        ("Qwen-7B", lambda: test_pairwise_7b(inter_df, test_qids)),
        ("Qwen-72B-API", lambda: test_pairwise_72b(inter_df, test_qids)),
        ("Rank-R1", lambda: test_setwise(inter_df, test_qids)),
    ]

    n_pass = 0
    n_fail = 0
    n_skip = 0

    for name, test_fn in tests:
        try:
            status, reason = test_fn()
        except Exception as e:
            status = Result.FAIL
            reason = f"unhandled exception: {e}"

        if status == Result.PASS:
            n_pass += 1
            icon = "PASS"
        elif status == Result.SKIP:
            n_skip += 1
            icon = "SKIP"
        else:
            n_fail += 1
            icon = "FAIL"

        print(f"  [{icon}] {name:<20}  {reason}")

    print()
    print("-" * 70)
    print(f"  PASS={n_pass}  SKIP={n_skip}  FAIL={n_fail}")

    if n_fail > 0:
        print("  STATUS: SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("  STATUS: ALL TESTS PASSED (or skipped)")
        sys.exit(0)


if __name__ == "__main__":
    main()
