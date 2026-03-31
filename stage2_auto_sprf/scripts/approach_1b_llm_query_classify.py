#!/usr/bin/env python3
"""
Approach 1b: LLM Zero-Shot Query Classification.

Prompt LLMs to predict PRF outcome from query text alone.
Models: Qwen3-8B, Qwen3-14B, Qwen3-14B-Think, Rank-R1-7B, Rank-R1-14B (local)
        Qwen-2.5-72B (university API)

3 prompt variants × 5 runs (temperature=0.7) × self-consistency majority vote.

Outputs:
  results/classifier/approach_1b_raw.csv      – all individual runs
  results/classifier/approach_1b_scores.csv   – per-query aggregated scores
  results/classifier/approach_1b_results.csv  – metrics summary
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
    QWEN_72B_API_URL, QWEN_72B_API_MODEL, QWEN_72B_API_KEY_ENV,
    RANK_R1_MODELS, QWEN_7B_CACHE_DIR,
    QWEN_14B_MODEL, QWEN_7B_MODEL,
    SC_NUM_RUNS,
)
from src.metrics import evaluate_all
from src.run_utils import setup_logging, append_to_summary

HF_CACHE = QWEN_7B_CACHE_DIR
OUT_DIR = ROOT / "results" / "classifier"
DATA_DIR = ROOT / "data" / "classifier"
N_RUNS = SC_NUM_RUNS  # 5

# ── Prompt templates ────────────────────────────────────────────────────────

PROMPTS = {
    "direct": (
        'Given the search query: "{query}"\n\n'
        "Do you think expanding this query using terms from the top-ranked search "
        "results (pseudo-relevance feedback) would improve or degrade the search "
        "quality?\n\n"
        "Consider:\n"
        "- Is the query specific and unambiguous?\n"
        "- Are the top search results likely to be on-topic?\n"
        "- Could adding related terms cause the search to drift away from the "
        "user's intent?\n\n"
        "Answer with exactly one word: Help or Hurt"
    ),
    "cot": (
        'Given the search query: "{query}"\n\n'
        "I want to decide whether to apply pseudo-relevance feedback (PRF) to "
        "this query. PRF expands the query using terms extracted from the "
        "top-ranked documents.\n\n"
        "Think step by step:\n"
        "1. What is the user's likely information need?\n"
        "2. How specific or ambiguous is this query?\n"
        "3. Are the top search results for this query likely to be relevant?\n"
        "4. If we extract expansion terms from those results, will they stay "
        "on-topic or drift?\n\n"
        "Based on your reasoning, answer: Help or Hurt"
    ),
    "expert": (
        "You are an expert information retrieval researcher. You are evaluating "
        "whether pseudo-relevance feedback (PRF) should be applied to a search "
        "query. PRF works by taking the top-ranked documents, extracting key "
        "terms, and adding them to the query. This helps when the initial results "
        "are relevant (good expansion terms), but hurts when they are off-topic "
        "(query drift).\n\n"
        'Query: "{query}"\n\n'
        "Based on your expertise, will PRF help or hurt this query?\n"
        "Answer: Help or Hurt"
    ),
}


# ── Data loading ────────────────────────────────────────────────────────────

def load_queries() -> dict:
    df = pd.read_csv(DATA_DIR / "queries.tsv", sep="\t")
    return dict(zip(df["qid"], df["query_text"]))


def load_labels() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "labels.tsv", sep="\t")


# ── Response parsing ────────────────────────────────────────────────────────

def parse_response(text: str) -> str:
    """Extract Help or Hurt from model output."""
    # Strip thinking traces
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Look for Help/Hurt (case-insensitive)
    text_lower = text_clean.lower()
    # Check last line first (where the answer usually is)
    last_line = text_clean.strip().split("\n")[-1].lower()
    if "help" in last_line and "hurt" not in last_line:
        return "Help"
    if "hurt" in last_line and "help" not in last_line:
        return "Hurt"
    # Fallback: check whole text, prefer the last occurrence
    matches = re.findall(r"\b(help|hurt)\b", text_lower)
    if matches:
        return matches[-1].capitalize()
    return "Unknown"


def self_consistency(responses: list) -> tuple:
    """Majority vote and confidence from multiple runs.

    Returns (label, confidence) where confidence = proportion of majority.
    """
    parsed = [parse_response(r) for r in responses]
    valid = [p for p in parsed if p != "Unknown"]
    if not valid:
        return "Unknown", 0.0
    help_count = sum(1 for p in valid if p == "Help")
    hurt_count = sum(1 for p in valid if p == "Hurt")
    total = len(valid)
    if help_count >= hurt_count:
        return "Help", help_count / total
    else:
        return "Hurt", hurt_count / total


# ── Local model inference via vLLM ──────────────────────────────────────────

def run_local_model(
    queries: dict,
    model_name: str,
    prompts: dict,
    n_runs: int,
    logger,
    enable_thinking: bool = False,
    lora_path: str = None,
) -> list:
    """Run inference with a local model via vLLM. Returns list of raw result dicts."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    logger.info(f"Loading model: {model_name} (thinking={enable_thinking}, lora={lora_path})")

    llm_kwargs = dict(
        model=model_name,
        download_dir=HF_CACHE,
        gpu_memory_utilization=0.92,
        enforce_eager=True,
        max_num_seqs=64,
        trust_remote_code=True,
    )
    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 32

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        top_p=0.9,
    )

    results = []
    lora_request = None
    if lora_path:
        lora_request = LoRARequest("r1_adapter", 1, lora_path)

    for prompt_name, prompt_template in prompts.items():
        # Build all prompts for batch inference
        batch_prompts = []
        batch_meta = []

        for qid, query_text in queries.items():
            user_msg = prompt_template.format(query=query_text)
            for run_idx in range(n_runs):
                messages = [{"role": "user", "content": user_msg}]
                if enable_thinking:
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=True,
                    )
                else:
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                batch_prompts.append(formatted)
                batch_meta.append({
                    "qid": qid,
                    "prompt_name": prompt_name,
                    "run_idx": run_idx,
                })

        logger.info(f"  Prompt '{prompt_name}': {len(batch_prompts)} generations")

        if lora_request:
            outputs = llm.generate(batch_prompts, sampling, lora_request=lora_request)
        else:
            outputs = llm.generate(batch_prompts, sampling)

        for meta, out in zip(batch_meta, outputs):
            text = out.outputs[0].text
            meta["response"] = text
            meta["parsed"] = parse_response(text)
            results.append(meta)

    # Free GPU memory
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ── API inference for Qwen-72B ──────────────────────────────────────────────

def run_api_model(
    queries: dict,
    prompts: dict,
    n_runs: int,
    logger,
    api_key: str,
) -> list:
    """Run inference with Qwen-72B via university API."""
    from openai import OpenAI, RateLimitError, APIError, APIConnectionError

    client = OpenAI(base_url=QWEN_72B_API_URL, api_key=api_key)
    results = []

    for prompt_name, prompt_template in prompts.items():
        for qid, query_text in queries.items():
            user_msg = prompt_template.format(query=query_text)
            for run_idx in range(n_runs):
                for attempt in range(5):
                    try:
                        resp = client.chat.completions.create(
                            model=QWEN_72B_API_MODEL,
                            messages=[{"role": "user", "content": user_msg}],
                            temperature=0.7,
                            max_tokens=512,
                        )
                        text = resp.choices[0].message.content
                        break
                    except (RateLimitError, APIError, APIConnectionError) as e:
                        wait = 2 ** attempt
                        logger.warning(f"  API error ({e}), retrying in {wait}s...")
                        time.sleep(wait)
                else:
                    text = "ERROR"

                results.append({
                    "qid": qid,
                    "prompt_name": prompt_name,
                    "run_idx": run_idx,
                    "response": text,
                    "parsed": parse_response(text),
                })

        logger.info(f"  Prompt '{prompt_name}': {len(queries) * n_runs} API calls done")

    return results


# ── Aggregate and evaluate ──────────────────────────────────────────────────

def aggregate_results(raw_results: list, model_tag: str) -> pd.DataFrame:
    """Aggregate runs into self-consistency votes per query × prompt."""
    df = pd.DataFrame(raw_results)
    rows = []
    for (qid, prompt_name), grp in df.groupby(["qid", "prompt_name"]):
        responses = grp["response"].tolist()
        label, confidence = self_consistency(responses)
        # Map Help → Benefit direction for scoring (higher = more likely benefit)
        score = confidence if label == "Help" else (1.0 - confidence)
        rows.append({
            "qid": qid,
            "prompt_name": prompt_name,
            "model": model_tag,
            "sc_label": label,
            "sc_confidence": confidence,
            "score": score,
            "n_help": sum(1 for r in grp["parsed"] if r == "Help"),
            "n_hurt": sum(1 for r in grp["parsed"] if r == "Hurt"),
            "n_unknown": sum(1 for r in grp["parsed"] if r == "Unknown"),
        })
    return pd.DataFrame(rows)


def evaluate_model_prompt(
    agg_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    model_tag: str,
    prompt_name: str,
    logger,
) -> dict:
    """Evaluate one model × prompt combination."""
    sub = agg_df[agg_df["prompt_name"] == prompt_name].copy()
    scores_series = pd.Series(sub["score"].values, index=sub["qid"].values)
    scores_series.index.name = None
    scores_series = scores_series.rename_axis("query_id")
    # Re-index to match labels
    scores_series.index = scores_series.index.astype(int)
    scores_series = scores_series.reindex(labels_df["query_id"])
    scores_series.index = labels_df["query_id"]

    method = f"1b-{model_tag}-{prompt_name}"
    result = evaluate_all(labels_df, scores_series, method, "classifier-1b", f"{model_tag} {prompt_name}")
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
    # ("Qwen-72B", None, False, None, True),  # API — skipped, unreliable
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("approach_1b")

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
            raw = run_api_model(queries, PROMPTS, N_RUNS, logger, api_key)
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
                queries, model_name, PROMPTS, N_RUNS, logger,
                enable_thinking=thinking,
                lora_path=resolved_lora,
            )

        # Tag results
        for r in raw:
            r["model"] = tag
        all_raw.extend(raw)

        # Aggregate
        agg = aggregate_results(raw, tag)
        all_agg.append(agg)

        # Evaluate per prompt
        for prompt_name in PROMPTS:
            result = evaluate_model_prompt(agg, labels, tag, prompt_name, logger)
            all_results.append(result)

    # ── Save all outputs ────────────────────────────────────────────
    raw_df = pd.DataFrame(all_raw)
    # Truncate long responses and escape special chars for CSV safety
    if "response" in raw_df.columns:
        raw_df["response"] = raw_df["response"].astype(str).str[:500]
    raw_df.to_csv(OUT_DIR / "approach_1b_raw.csv", index=False, escapechar="\\")
    logger.info(f"Saved raw → {OUT_DIR / 'approach_1b_raw.csv'}")

    agg_df = pd.concat(all_agg, ignore_index=True)
    agg_df.to_csv(OUT_DIR / "approach_1b_scores.csv", index=False)
    logger.info(f"Saved scores → {OUT_DIR / 'approach_1b_scores.csv'}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "approach_1b_results.csv", index=False)

    for r in all_results:
        append_to_summary(r)

    # ── Print summary ───────────────────────────────────────────────
    logger.info(f"\n{'='*60}\nApproach 1b Summary\n{'='*60}")
    cols = ["method_name", "acc", "auc", "gamma", "rho"]
    logger.info("\n" + results_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
