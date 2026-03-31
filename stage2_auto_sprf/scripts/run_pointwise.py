#!/usr/bin/env python3
"""
Run Qwen3-Reranker pointwise scoring: DCG ratio, MRR ratio, Majority@k.

Each model scores all query-document pairs independently, then documents
are ranked by relevance probability. Standard aggregations are computed
from the resulting ranked lists.

Usage:
    python scripts/run_pointwise.py                 # run both models
    python scripts/run_pointwise.py --worker 0      # run model index 0 only
"""

import sys
import json
import argparse
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.config import (
    PROJECT_ROOT as ROOT,
    QWEN3_RERANKER_4B_MODEL,
    QWEN3_RERANKER_8B_MODEL,
    RANK_R1_CACHE_DIR,
    TOP_K_CANDIDATES,
)
from src.data_loader import load_labels, load_interleaved
from src.aggregation import compute_dcg_ratio, compute_mrr_ratio, compute_majority_at_k
from src.metrics import evaluate_all
from src.run_utils import setup_logging, append_to_summary

HF_CACHE = RANK_R1_CACHE_DIR

MODEL_CONFIGS = [
    {
        "name": "Pointwise-Qwen3-Reranker-4B",
        "model": QWEN3_RERANKER_4B_MODEL,
    },
    {
        "name": "Pointwise-Qwen3-Reranker-8B",
        "model": QWEN3_RERANKER_8B_MODEL,
    },
]

AGGREGATIONS = [
    ("DCG", "dcg_score", compute_dcg_ratio),
    ("MRR", "mrr_score", compute_mrr_ratio),
    ("Majority@k", "majority_at_k_score", compute_majority_at_k),
]


def run_single_model(config_idx: int):
    config = MODEL_CONFIGS[config_idx]
    name = config["name"]

    logger = setup_logging(f"pointwise_{name}")
    logger.info("=" * 60)
    logger.info(f"Pointwise config: {name}")
    logger.info(f"  model  : {config['model']}")
    logger.info("=" * 60)

    labels = load_labels()
    inter = load_interleaved()

    label_qids = set(labels["query_id"].tolist())
    inter_qids = set(inter["qid"].unique().tolist())
    common_qids = sorted(label_qids & inter_qids)
    logger.info(f"Queries: {len(common_qids)}")

    t0 = time.time()
    from src.posthoc.pointwise import PointwiseReranker
    ranker = PointwiseReranker(
        model_name_or_path=config["model"],
        cache_dir=HF_CACHE,
    )
    logger.info(f"Ranker created in {time.time() - t0:.1f}s")

    results = []
    for qid in tqdm(common_qids, desc=name):
        rows = inter[inter["qid"] == qid].head(TOP_K_CANDIDATES)
        if len(rows) == 0:
            logger.warning(f"No interleaved docs for qid={qid}, skipping")
            continue

        query_text = ranker.truncate_query(rows.iloc[0]["query_text"])
        doc_list = [
            {
                "docid": r["docno"],
                "text": ranker.truncate_passage(
                    str(r["passage_text"]) if pd.notna(r["passage_text"]) else ""
                ),
                "origin_label": r.get("origin_label", ""),
            }
            for _, r in rows.iterrows()
        ]
        origin_labels = {r["docno"]: r["origin_label"] for _, r in rows.iterrows()}

        reranked = ranker.rerank(query_text, doc_list)
        ranked_ids = [doc_id for doc_id, _ in reranked]

        row = {"query_id": qid, "ranked_doc_ids": ranked_ids}
        for agg_name, col_name, agg_fn in AGGREGATIONS:
            row[col_name] = agg_fn(ranked_ids, origin_labels)

        results.append(row)
        logger.info(
            f"qid={qid}: DCG={row['dcg_score']:.4f}, "
            f"MRR={row['mrr_score']:.4f}, Maj@k={row['majority_at_k_score']:.4f}"
        )

    logger.info(f"Scored {len(results)} queries in {time.time() - t0:.1f}s total")

    # Save per-query scores
    results_dir = ROOT / "results" / "posthoc"
    results_dir.mkdir(parents=True, exist_ok=True)

    scores_df = pd.DataFrame([
        {
            "query_id": r["query_id"],
            "dcg_score": r["dcg_score"],
            "mrr_score": r["mrr_score"],
            "majority_at_k_score": r["majority_at_k_score"],
            "ranked_doc_ids": json.dumps([str(x) for x in r["ranked_doc_ids"]]),
        }
        for r in results
    ])
    scores_path = results_dir / f"{name}_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    logger.info(f"Saved per-query scores -> {scores_path}")

    # Evaluate all 3 aggregations
    for agg_name, col_name, _ in AGGREGATIONS:
        method_name = f"{name}-{agg_name}"
        scores_series = pd.Series(
            {r["query_id"]: r[col_name] for r in results}
        )
        scores_series.index.name = "query_id"

        eval_result = evaluate_all(
            labels_df=labels,
            scores_series=scores_series,
            method_name=method_name,
            stage="posthoc",
            predictor=f"pointwise {config['model'].split('/')[-1]} {agg_name}",
        )

        append_to_summary(eval_result)
        logger.info(
            f"  {method_name}: TPR_H={eval_result['tpr_hurt']}, "
            f"TPR_B={eval_result['tpr_benefit']}, Acc={eval_result['acc']}, "
            f"AUC={eval_result['auc']}, Gamma={eval_result['gamma']} "
            f"(p={eval_result['gamma_p']})"
        )

    logger.info(f"=== {name} COMPLETE ===\n")


def main():
    parser = argparse.ArgumentParser(description="Run pointwise Qwen3-Reranker")
    parser.add_argument("--worker", type=int, default=-1,
                        help="Run a specific model config index (0-1)")
    args = parser.parse_args()

    if args.worker >= 0:
        run_single_model(args.worker)
        return

    # Orchestrator mode
    print(f"\n{'='*70}")
    print(f"  Pointwise: Running {len(MODEL_CONFIGS)} models x 3 aggregations")
    print(f"{'='*70}\n")

    timings = {}
    failures = []

    for i in range(len(MODEL_CONFIGS)):
        config = MODEL_CONFIGS[i]
        print(f"\n{'─'*70}")
        print(f"  [{i+1}/{len(MODEL_CONFIGS)}] {config['name']}")
        print(f"  Model: {config['model']}")
        print(f"{'─'*70}\n")

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(i)],
            cwd=str(PROJECT_ROOT),
        )
        elapsed = time.time() - t0
        timings[config["name"]] = elapsed

        if result.returncode != 0:
            failures.append(config["name"])
            print(f"\n  x {config['name']} FAILED [{elapsed:.0f}s]")
        else:
            print(f"\n  ok {config['name']} done [{elapsed:.0f}s]")

    print(f"\n{'='*70}")
    print("  ALL POINTWISE MODELS COMPLETE")
    print(f"{'='*70}\n")

    print("Timings:")
    for name, t in timings.items():
        status = "FAIL" if name in failures else "OK"
        print(f"  {name:40s}  {t:7.0f}s  [{status}]")


if __name__ == "__main__":
    main()
