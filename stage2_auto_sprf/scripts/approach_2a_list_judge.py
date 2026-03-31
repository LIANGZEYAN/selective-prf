#!/usr/bin/env python3
"""
Approach 2a: LLM as List-Level Quality Judge.

Present the LLM with query + top-k initial ColBERT results.
Ask directly: should we expand this query?

Models: Qwen3-8B, Qwen3-14B, Qwen3-14B-Think (local), Qwen-2.5-72B (API)
2 prompt variants × top-k ∈ {3, 5, 10} × 5 runs × self-consistency.

Outputs:
  results/classifier/approach_2a_raw.csv
  results/classifier/approach_2a_scores.csv
  results/classifier/approach_2a_results.csv
"""

import gc
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROJECT_ROOT as ROOT,
    QWEN3_8B_MODEL, QWEN3_14B_MODEL,
    QWEN_7B_MODEL, QWEN_14B_MODEL,
    QWEN_72B_API_URL, QWEN_72B_API_MODEL, QWEN_72B_API_KEY_ENV,
    QWEN_7B_CACHE_DIR, SC_NUM_RUNS,
    RANK_R1_MODELS,
)
from src.metrics import evaluate_all
from src.run_utils import setup_logging, append_to_summary

HF_CACHE = QWEN_7B_CACHE_DIR
OUT_DIR = ROOT / "results" / "classifier"
DATA_DIR = ROOT / "data" / "classifier"
RANKINGS_DIR = DATA_DIR / "initial_rankings"
N_RUNS = SC_NUM_RUNS  # 5
TOP_K_VALUES = [3, 5, 10]

# ── Prompt templates ────────────────────────────────────────────────────────

PROMPTS = {
    "coverage": (
        'You are evaluating search results for the query: "{query}"\n\n'
        "Here are the top {k} results:\n"
        "{passages}\n\n"
        "Do these results adequately answer the user's query? Consider:\n"
        "1. Are the results relevant to the query?\n"
        "2. Do they cover different aspects of the information need?\n"
        "3. Is there important information missing that query expansion could find?\n\n"
        "If the results are already good, expansion risks introducing noise.\n"
        "If the results are weak or incomplete, expansion could help find better content.\n\n"
        "Answer with exactly one word: Adequate or Expand"
    ),
    "comparative": (
        'You are an information retrieval expert. A user searched for: "{query}"\n\n'
        "The search system returned these top {k} passages:\n"
        "{passages}\n\n"
        "We are considering applying pseudo-relevance feedback (PRF), which "
        "extracts key terms from these results and adds them to the query to "
        "find additional relevant content.\n\n"
        "Think step by step:\n"
        "1. Are these results relevant to the query?\n"
        "2. If we extract terms from these passages, will they be good expansion "
        "terms or will they cause drift?\n"
        "3. Is there evidence that important aspects of the query are missing "
        "from these results?\n\n"
        "Based on your analysis, should we apply PRF?\n"
        "Answer: Apply or Skip"
    ),
}

# Map model responses to Benefit/Hurt direction
RESPONSE_MAP = {
    "coverage": {"adequate": "Hurt", "expand": "Benefit"},
    "comparative": {"apply": "Benefit", "skip": "Hurt"},
}


# ── Data loading ────────────────────────────────────────────────────────────

def load_queries() -> dict:
    df = pd.read_csv(DATA_DIR / "queries.tsv", sep="\t")
    return dict(zip(df["qid"], df["query_text"]))


def load_labels() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "labels.tsv", sep="\t")


def load_passages(qid: int, top_k: int) -> str:
    """Load top-k passages for a query, formatted for prompt."""
    fpath = RANKINGS_DIR / f"{qid}.csv"
    df = pd.read_csv(fpath).head(top_k)
    parts = []
    for i, row in df.iterrows():
        text = str(row["passage_text"])[:400]  # truncate long passages
        parts.append(f"[{len(parts)+1}] {text}")
    return "\n".join(parts)


# ── Response parsing ────────────────────────────────────────────────────────

def parse_response(text: str, prompt_name: str) -> str:
    """Extract the answer from model output, map to Benefit/Hurt."""
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text_lower = text_clean.lower()
    last_line = text_clean.strip().split("\n")[-1].lower()

    mapping = RESPONSE_MAP[prompt_name]

    # Check last line first
    for keyword, label in mapping.items():
        if keyword in last_line:
            return label

    # Fallback: last occurrence in full text
    all_keywords = list(mapping.keys())
    matches = []
    for keyword in all_keywords:
        idx = text_lower.rfind(keyword)
        if idx >= 0:
            matches.append((idx, keyword))
    if matches:
        matches.sort()
        return mapping[matches[-1][1]]

    return "Unknown"


def self_consistency(labels: list) -> tuple:
    """Majority vote. Returns (label, confidence)."""
    valid = [l for l in labels if l != "Unknown"]
    if not valid:
        return "Unknown", 0.0
    benefit_count = sum(1 for l in valid if l == "Benefit")
    hurt_count = sum(1 for l in valid if l == "Hurt")
    total = len(valid)
    if benefit_count >= hurt_count:
        return "Benefit", benefit_count / total
    else:
        return "Hurt", hurt_count / total


# ── Local model inference ───────────────────────────────────────────────────

def run_local_model(
    queries: dict,
    model_name: str,
    prompts: dict,
    top_k_values: list,
    n_runs: int,
    logger,
    enable_thinking: bool = False,
    lora_path: str = None,
) -> list:
    """Run inference with a local model via vLLM."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    logger.info(f"Loading model: {model_name} (thinking={enable_thinking}, lora={lora_path})")

    llm_kwargs = dict(
        model=model_name,
        download_dir=HF_CACHE,
        gpu_memory_utilization=0.92,
        enforce_eager=True,
        max_num_seqs=32,
        trust_remote_code=True,
    )
    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 32

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    max_tok = 2048 if enable_thinking else 512
    sampling = SamplingParams(temperature=0.7, max_tokens=max_tok, top_p=0.9)

    lora_request = None
    if lora_path:
        lora_request = LoRARequest("r1_adapter", 1, lora_path)

    results = []

    for top_k in top_k_values:
        for prompt_name, prompt_template in prompts.items():
            batch_prompts = []
            batch_meta = []

            for qid, query_text in queries.items():
                passages_str = load_passages(qid, top_k)
                user_msg = prompt_template.format(
                    query=query_text, k=top_k, passages=passages_str
                )
                for run_idx in range(n_runs):
                    messages = [{"role": "user", "content": user_msg}]
                    if enable_thinking:
                        formatted = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=True,
                        )
                    else:
                        # Qwen3 needs explicit enable_thinking=False to avoid garbage
                        try:
                            formatted = tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False,
                            )
                        except TypeError:
                            # Non-Qwen3 tokenizers don't support enable_thinking
                            formatted = tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True,
                            )
                    batch_prompts.append(formatted)
                    batch_meta.append({
                        "qid": qid,
                        "prompt_name": prompt_name,
                        "top_k": top_k,
                        "run_idx": run_idx,
                    })

            logger.info(f"  top_k={top_k}, prompt='{prompt_name}': {len(batch_prompts)} generations")
            if lora_request:
                outputs = llm.generate(batch_prompts, sampling, lora_request=lora_request)
            else:
                outputs = llm.generate(batch_prompts, sampling)

            for meta, out in zip(batch_meta, outputs):
                text = out.outputs[0].text
                meta["response"] = text[:500]
                meta["parsed"] = parse_response(text, prompt_name)
                results.append(meta)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ── API inference ───────────────────────────────────────────────────────────

def run_api_model(
    queries: dict,
    prompts: dict,
    top_k_values: list,
    n_runs: int,
    logger,
    api_key: str,
) -> list:
    """Run inference with Qwen-72B via university API."""
    from openai import OpenAI, RateLimitError, APIError, APIConnectionError

    client = OpenAI(base_url=QWEN_72B_API_URL, api_key=api_key)
    results = []

    for top_k in top_k_values:
        for prompt_name, prompt_template in prompts.items():
            count = 0
            for qid, query_text in queries.items():
                passages_str = load_passages(qid, top_k)
                user_msg = prompt_template.format(
                    query=query_text, k=top_k, passages=passages_str
                )
                for run_idx in range(n_runs):
                    text = "ERROR"
                    for attempt in range(5):
                        try:
                            resp = client.chat.completions.create(
                                model=QWEN_72B_API_MODEL,
                                messages=[{"role": "user", "content": user_msg}],
                                temperature=0.7,
                                max_tokens=512,
                                timeout=60,
                            )
                            text = resp.choices[0].message.content
                            break
                        except (RateLimitError, APIError, APIConnectionError) as e:
                            wait = 2 ** attempt
                            logger.warning(f"  API error ({e}), retry in {wait}s")
                            time.sleep(wait)
                        except Exception as e:
                            logger.warning(f"  Unexpected error ({e}), retry in {2**attempt}s")
                            time.sleep(2 ** attempt)

                    results.append({
                        "qid": qid,
                        "prompt_name": prompt_name,
                        "top_k": top_k,
                        "run_idx": run_idx,
                        "response": text[:500],
                        "parsed": parse_response(text, prompt_name),
                    })
                    count += 1

            logger.info(f"  top_k={top_k}, prompt='{prompt_name}': {count} API calls done")

    return results


# ── Aggregate and evaluate ──────────────────────────────────────────────────

def aggregate_results(raw_results: list, model_tag: str) -> pd.DataFrame:
    df = pd.DataFrame(raw_results)
    rows = []
    for (qid, prompt_name, top_k), grp in df.groupby(["qid", "prompt_name", "top_k"]):
        labels = grp["parsed"].tolist()
        label, confidence = self_consistency(labels)
        score = confidence if label == "Benefit" else (1.0 - confidence)
        rows.append({
            "qid": qid,
            "prompt_name": prompt_name,
            "top_k": top_k,
            "model": model_tag,
            "sc_label": label,
            "sc_confidence": confidence,
            "score": score,
            "n_benefit": sum(1 for l in labels if l == "Benefit"),
            "n_hurt": sum(1 for l in labels if l == "Hurt"),
            "n_unknown": sum(1 for l in labels if l == "Unknown"),
        })
    return pd.DataFrame(rows)


def evaluate_model_prompt_k(agg_df, labels_df, model_tag, prompt_name, top_k, logger):
    sub = agg_df[(agg_df["prompt_name"] == prompt_name) & (agg_df["top_k"] == top_k)].copy()
    scores_series = pd.Series(sub["score"].values, index=sub["qid"].values.astype(int))
    scores_series.index.name = "query_id"
    scores_series = scores_series.reindex(labels_df["query_id"])

    method = f"2a-{model_tag}-{prompt_name}-k{top_k}"
    result = evaluate_all(labels_df, scores_series, method, "classifier-2a",
                          f"{model_tag} {prompt_name} top-{top_k}")
    logger.info(f"  {method}: acc={result['acc']}, auc={result['auc']}, "
                f"gamma={result['gamma']}, rho={result['rho']}")
    return result


# ── Main ────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    # (tag, model_name, thinking, lora_path, is_api)
    ("Qwen3-8B", QWEN3_8B_MODEL, False, None, False),
    ("Qwen3-14B", QWEN3_14B_MODEL, False, None, False),
    ("Qwen3-14B-Think", QWEN3_14B_MODEL, True, None, False),
    ("R1-7B", QWEN_7B_MODEL, False, RANK_R1_MODELS["7b"], False),
    ("R1-14B", QWEN_14B_MODEL, False, RANK_R1_MODELS["14b"], False),
    # ("Qwen-72B", None, False, None, True),  # API — skipped if unreliable
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("approach_2a")

    queries = load_queries()
    labels = load_labels()
    logger.info(f"Loaded {len(queries)} queries, {len(labels)} labelled")

    api_key = os.environ.get(QWEN_72B_API_KEY_ENV, "")
    if not api_key:
        api_key = os.environ.get("IDA_LLM_API_KEY", "")

    all_raw = []
    all_agg = []
    all_results = []

    for tag, model_name, thinking, lora_path, is_api in MODEL_CONFIGS:
        logger.info(f"\n{'='*60}\nModel: {tag}\n{'='*60}")

        if is_api:
            if not api_key:
                logger.warning(f"Skipping {tag}: no API key")
                continue
            raw = run_api_model(queries, PROMPTS, TOP_K_VALUES, N_RUNS, logger, api_key)
        else:
            # Resolve LoRA path to actual snapshot dir
            resolved_lora = None
            if lora_path:
                from huggingface_hub import snapshot_download
                resolved_lora = snapshot_download(
                    lora_path, cache_dir=HF_CACHE,
                    local_files_only=True,
                )
            raw = run_local_model(
                queries, model_name, PROMPTS, TOP_K_VALUES, N_RUNS, logger,
                enable_thinking=thinking,
                lora_path=resolved_lora,
            )

        for r in raw:
            r["model"] = tag
        all_raw.extend(raw)

        agg = aggregate_results(raw, tag)
        all_agg.append(agg)

        for prompt_name in PROMPTS:
            for top_k in TOP_K_VALUES:
                result = evaluate_model_prompt_k(agg, labels, tag, prompt_name, top_k, logger)
                all_results.append(result)

    # ── Save outputs ────────────────────────────────────────────────
    raw_df = pd.DataFrame(all_raw)
    raw_df.to_csv(OUT_DIR / "approach_2a_raw.csv", index=False, escapechar="\\")
    logger.info(f"Saved raw → {OUT_DIR / 'approach_2a_raw.csv'}")

    agg_df = pd.concat(all_agg, ignore_index=True)
    agg_df.to_csv(OUT_DIR / "approach_2a_scores.csv", index=False)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "approach_2a_results.csv", index=False)

    for r in all_results:
        append_to_summary(r)

    logger.info(f"\n{'='*60}\nApproach 2a Summary\n{'='*60}")
    cols = ["method_name", "acc", "auc", "gamma", "rho"]
    logger.info("\n" + results_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
