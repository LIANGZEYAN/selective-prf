#!/usr/bin/env python3
"""
DEPRECATED — This file is the legacy entry point.
Use the new modular scripts instead:
  - scripts/run_pairwise_72b.py  (Qwen-72B API, with h=0.061)
  - scripts/run_pairwise_7b.py   (Qwen-7B local GPU, with h=0.061)
See ARCHITECTURE.md for the full restructured project layout.

Original description:
Pairwise LLM Ranker for PRF System Preference Prediction
=========================================================
Uses Qwen2.5-72B-Instruct via OpenAI-compatible API to pairwise-rank
interleaved TDI documents, then derives a DCG-weighted B-preference ratio
and compares it against human preference labels.

Origin labels:
  A-Only        -> Baseline-only document  (discriminative for human pref)
  B-Only        -> PRF-only document        (discriminative for human pref)
  Both          -> Retrieved by both systems (non-discriminative)
  Easy-Negative -> Neither system ranked highly (non-discriminative)

Human labels:
  PRF-Benefit         -> b_preference_ratio > 0.5
  PRF-Hurt            -> b_preference_ratio < 0.5
  Neutral             -> exactly 0.5
  Insufficient-Data   -> zero discriminative clicks  (excluded from eval)
"""

import sys
import os
import re
import math
import copy
import time
import argparse
import warnings
import concurrent.futures
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
from openai import OpenAI, RateLimitError, APIError, APIConnectionError

sys.path.insert(0, '/mnt/primary/QE audit/rankllm/llm-rankers')
from llmrankers.rankers import SearchResult, LlmRanker

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR        = '/mnt/primary/QE audit/rankllm'
INTERLEAVED_CSV = f'{DATA_DIR}/result_interleaved_with_text.csv'
PREFERENCE_CSV  = f'{DATA_DIR}/preference.csv'
OUTPUT_CSV      = f'{DATA_DIR}/pairwise_results.csv'

API_BASE_URL    = 'http://api.llm.apps.os.dcs.gla.ac.uk/v1'
API_MODEL       = 'qwen-2.5-72b-instruct'

PREF_THRESHOLD  = 0.0   # exactly 0.5 = Neutral; >0.5 = Benefit; <0.5 = Hurt
PRF_LABEL       = 'B-Only'
BASE_LABEL      = 'A-Only'

SYSTEM_PROMPT = (
    'You are RankGPT, an intelligent assistant that selects the most '
    'relevant passage from a pair of passages based on their relevance '
    'to a given query. Output exactly "Passage A" or "Passage B".'
)

PAIRWISE_PROMPT = (
    'Given a query "{query}", which of the following two passages is more '
    'relevant to the query?\n\n'
    'Passage A: "{doc1}"\n\n'
    'Passage B: "{doc2}"\n\n'
    'Output Passage A or Passage B:'
)


# ──────────────────────────────────────────────────────────────────────────────
# API-based pairwise ranker (Qwen 72B via OpenAI-compatible endpoint)
# ──────────────────────────────────────────────────────────────────────────────
class QwenApiPairwiseLlmRanker(LlmRanker):
    """
    Pairwise LLM ranker backed by an OpenAI-compatible REST API.
    Forward and reverse prompts are fired in parallel to halve latency.
    Retries on rate-limit / transient errors with exponential back-off.
    """

    def __init__(self, api_key: str,
                 base_url: str   = API_BASE_URL,
                 model: str      = API_MODEL,
                 method: str     = 'heapsort',
                 k: int          = 10,
                 max_retries: int = 5):
        self.model       = model
        self.method      = method
        self.k           = k
        self.max_retries = max_retries
        self.client      = OpenAI(base_url=base_url, api_key=api_key)

        self.total_compare           = 0
        self.total_prompt_tokens     = 0
        self.total_completion_tokens = 0

        print(f'API ranker ready  ->  {base_url}  model={model}\n')

    def _call_api(self, prompt: str) -> tuple:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user',   'content': prompt},
                    ],
                    temperature=0.0,
                    max_tokens=5,
                )
                content = resp.choices[0].message.content.strip()
                p_tok   = resp.usage.prompt_tokens     if resp.usage else 0
                c_tok   = resp.usage.completion_tokens if resp.usage else 0

                matches = re.findall(r'Passage\s+([AB])', content, re.IGNORECASE)
                if matches:
                    return f'Passage {matches[0].upper()}', p_tok, c_tok
                if content.upper().startswith('A'):
                    return 'Passage A', p_tok, c_tok
                if content.upper().startswith('B'):
                    return 'Passage B', p_tok, c_tok

                print(f'  [warn] unexpected output: {repr(content)} -> defaulting to Passage A')
                return 'Passage A', p_tok, c_tok

            except RateLimitError:
                wait = 2 ** attempt
                print(f'  [rate-limit] sleeping {wait}s ...')
                time.sleep(wait)
            except (APIError, APIConnectionError) as e:
                wait = 2 ** attempt
                print(f'  [api-error] {e}  sleeping {wait}s ...')
                time.sleep(wait)

        raise RuntimeError(f'API call failed after {self.max_retries} retries.')

    def compare(self, query: str, docs: list) -> list:
        self.total_compare += 1
        doc1_text, doc2_text = docs[0], docs[1]

        fwd_prompt = PAIRWISE_PROMPT.format(query=query, doc1=doc1_text, doc2=doc2_text)
        rev_prompt = PAIRWISE_PROMPT.format(query=query, doc1=doc2_text, doc2=doc1_text)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fwd_fut = ex.submit(self._call_api, fwd_prompt)
            rev_fut = ex.submit(self._call_api, rev_prompt)
            fwd_ans, fp, fc = fwd_fut.result()
            rev_ans, rp, rc = rev_fut.result()

        self.total_prompt_tokens     += fp + rp
        self.total_completion_tokens += fc + rc

        return [fwd_ans, rev_ans]

    def truncate(self, text: str, length: int) -> str:
        char_limit = length * 4
        return text[:char_limit] if len(text) > char_limit else text

    def heapify(self, arr, n, i):
        largest = i
        l, r    = 2 * i + 1, 2 * i + 2
        if l < n and arr[l] > arr[i]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.heapify(arr, n, largest)

    def heapSort(self, arr, k):
        n, ranked = len(arr), 0
        for i in range(n // 2, -1, -1):
            self.heapify(arr, n, i)
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            self.heapify(arr, i, 0)

    def rerank(self, query: str, ranking: list) -> list:
        original_ranking = copy.deepcopy(ranking)
        self.total_compare           = 0
        self.total_prompt_tokens     = 0
        self.total_completion_tokens = 0

        if self.method != 'heapsort':
            raise NotImplementedError(f'Method {self.method} not implemented.')

        ranker_ref = self

        class ComparableDoc:
            def __init__(self, docid, text):
                self.docid = docid
                self.text  = text

            def __gt__(self, other):
                out = ranker_ref.compare(query, [self.text, other.text])
                return out[0] == 'Passage A' and out[1] == 'Passage B'

        arr = [ComparableDoc(docid=d.docid, text=d.text) for d in ranking]
        self.heapSort(arr, self.k)

        ranking_sorted = [
            SearchResult(docid=doc.docid, score=-idx, text=None)
            for idx, doc in enumerate(reversed(arr))
        ]

        results, top_ids, rank = [], set(), 1
        for doc in ranking_sorted[:self.k]:
            top_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1
        return results


# ──────────────────────────────────────────────────────────────────────────────
# DCG-weighted preference ratio
# ──────────────────────────────────────────────────────────────────────────────
def dcg_gain(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def compute_preference_ratio(reranked: list, label_map: dict) -> float:
    g_prf = g_base = 0.0
    for result in reranked:
        rank  = -result.score
        label = label_map.get(result.docid, '')
        gain  = dcg_gain(rank)
        if label == PRF_LABEL:
            g_prf  += gain
        elif label == BASE_LABEL:
            g_base += gain

    total = g_prf + g_base
    return float('nan') if total == 0 else g_prf / total


# ──────────────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────────────
def classify(ratio: float, threshold: float = PREF_THRESHOLD) -> str:
    if math.isnan(ratio):
        return 'Insufficient-Data'
    if ratio > 0.5 + threshold:
        return 'PRF-Benefit'
    if ratio < 0.5 - threshold:
        return 'PRF-Hurt'
    return 'Neutral'


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(results_df: pd.DataFrame):
    eval_df = results_df[
        results_df['human_preference'] != 'Insufficient-Data'
    ].dropna(subset=['llm_ratio', 'human_ratio']).copy()

    print(f'\nEvaluation on {len(eval_df)} queries '
          f'(excluded Insufficient-Data: {len(results_df) - len(eval_df)})')

    pearson_r,   pearson_p  = stats.pearsonr(eval_df['llm_ratio'], eval_df['human_ratio'])
    kendall_tau, kendall_p  = stats.kendalltau(eval_df['llm_ratio'], eval_df['human_ratio'])

    correct  = (eval_df['llm_pred'] == eval_df['human_preference']).sum()
    accuracy = correct / len(eval_df)

    hurt_mask    = eval_df['human_preference'] == 'PRF-Hurt'
    n_hurt       = hurt_mask.sum()
    hurt_correct = ((eval_df['llm_pred'] == 'PRF-Hurt') & hurt_mask).sum()
    hurt_recall  = hurt_correct / n_hurt if n_hurt > 0 else float('nan')

    print('\n' + '='*62)
    print('  METRICS')
    print('='*62)
    print(f'  Pearson  r     = {pearson_r:+.4f}  (p={pearson_p:.4f})')
    print(f'  Kendall  t     = {kendall_tau:+.4f}  (p={kendall_p:.4f})')
    print(f'  3-way Accuracy = {accuracy:.4f}  ({correct}/{len(eval_df)})')
    print(f'  Hurt Recall    = {hurt_recall:.4f}  ({hurt_correct}/{n_hurt})  <- KEY METRIC')
    print('='*62)

    print('\n  Per-class recall:')
    for label, mask in [
        ('PRF-Benefit', eval_df['human_preference'] == 'PRF-Benefit'),
        ('PRF-Hurt',    hurt_mask),
        ('Neutral',     eval_df['human_preference'] == 'Neutral'),
    ]:
        n    = mask.sum()
        n_ok = ((eval_df['llm_pred'] == label) & mask).sum()
        r    = n_ok / n if n > 0 else float('nan')
        print(f'    {label:<14}: {n_ok}/{n}  recall={r:.3f}')

    return dict(pearson_r=pearson_r, pearson_p=pearson_p,
                kendall_tau=kendall_tau, kendall_p=kendall_p,
                accuracy_3way=accuracy, hurt_recall=float(hurt_recall))


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(api_key: str,
                 test_qids: list      = None,
                 passage_length: int  = 128,
                 query_length: int    = 32,
                 k_per_query: int     = 9):

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print('='*62)
    print(' STEP 1 — Loading data')
    print('='*62)
    inter_df = pd.read_csv(INTERLEAVED_CSV)
    pref_df  = pd.read_csv(PREFERENCE_CSV)

    print(f'Interleaved docs : {inter_df.shape[0]} rows, '
          f'{inter_df["qid"].nunique()} queries')
    print(f'Preference labels: {pref_df.shape[0]} queries')
    print(f'Label distribution:\n{pref_df["preference"].value_counts().to_string()}\n')

    pref_map = pref_df.set_index('qid')[
        ['b_preference_ratio', 'preference']].to_dict('index')

    all_qids = sorted(inter_df['qid'].unique().tolist())
    run_qids = test_qids if test_qids is not None else all_qids
    print(f'Running on {len(run_qids)} quer{"y" if len(run_qids)==1 else "ies"}: '
          f'{run_qids[:5]}{"..." if len(run_qids) > 5 else ""}')

    # ── Step 2: Init ranker ──────────────────────────────────────────────────
    print('\n' + '='*62)
    print(' STEP 2 — Initialising API ranker')
    print('='*62)
    ranker = QwenApiPairwiseLlmRanker(
        api_key=api_key,
        method='heapsort',
        k=k_per_query,
    )

    # ── Step 3: Rerank & compute ratios ─────────────────────────────────────
    print('='*62)
    print(' STEP 3 — Pairwise reranking')
    print('='*62)
    rows = []

    for qid in tqdm(run_qids, desc='Queries'):
        q_df       = inter_df[inter_df['qid'] == qid].sort_values('rank')
        query_text = ranker.truncate(q_df['query_text'].iloc[0], query_length)

        docs, label_map = [], {}
        for _, row in q_df.iterrows():
            text = ranker.truncate(
                str(row['passage_text']) if pd.notna(row['passage_text']) else '',
                passage_length)
            docs.append(SearchResult(docid=row['docno'],
                                     score=-row['rank'], text=text))
            label_map[row['docno']] = row['origin_label']

        reranked  = ranker.rerank(query_text, docs)
        llm_ratio = compute_preference_ratio(reranked, label_map)
        llm_pred  = classify(llm_ratio)

        human_info  = pref_map.get(qid, {})
        human_ratio = human_info.get('b_preference_ratio', float('nan'))
        human_pref  = human_info.get('preference', 'Unknown')

        correct_flag = llm_pred == human_pref
        rows.append(dict(
            qid=qid,
            query=q_df['query_text'].iloc[0],
            llm_ratio=round(llm_ratio, 4) if not math.isnan(llm_ratio) else float('nan'),
            human_ratio=round(human_ratio, 4) if not math.isnan(human_ratio) else float('nan'),
            llm_pred=llm_pred,
            human_preference=human_pref,
            correct=correct_flag,
            n_comparisons=ranker.total_compare,
        ))

        tick = 'Y' if correct_flag else 'N'
        hurt = ' <- HURT' if human_pref == 'PRF-Hurt' else ''
        print(f'  qid={qid:<10}  '
              f'llm={llm_ratio:.3f} ({llm_pred:<12})  '
              f'human={human_ratio:.3f} ({human_pref:<18}) '
              f'{tick}{hurt}  [{ranker.total_compare} cmps]')

    results_df = pd.DataFrame(rows)

    # ── Step 4: Hurt-query summary ───────────────────────────────────────────
    print('\n' + '='*62)
    print(' STEP 4 — Hurt query summary')
    print('='*62)
    for _, r in results_df[results_df['human_preference'] == 'PRF-Hurt'].iterrows():
        status = ('CORRECTLY IDENTIFIED'
                  if r['llm_pred'] == 'PRF-Hurt'
                  else f'MISSED  (predicted: {r["llm_pred"]})')
        print(f'  qid={r["qid"]:<10}  llm_ratio={r["llm_ratio"]:.3f}  {status}')

    # ── Step 5: Evaluate ─────────────────────────────────────────────────────
    metrics = {}
    if len(run_qids) > 1:
        metrics = evaluate(results_df)

    # ── Step 6: Save ─────────────────────────────────────────────────────────
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f'\nResults saved -> {OUTPUT_CSV}')
    print(results_df[[
        'qid', 'query', 'llm_ratio', 'human_ratio',
        'llm_pred', 'human_preference', 'correct',
    ]].to_string(index=False))

    return results_df, metrics


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true',
                        help='Run on first 3 queries to verify pipeline')
    parser.add_argument('--test_qids', type=int, nargs='+', default=None,
                        help='Specific qids, e.g. --test_qids 1037798 104861')
    parser.add_argument('--passage_length', type=int, default=128)
    parser.add_argument('--query_length',   type=int, default=32)
    args = parser.parse_args()

    api_key = os.environ.get('IDA_LLM_API_KEY')
    if not api_key:
        sys.exit('ERROR: IDA_LLM_API_KEY environment variable is not set.')

    if args.test:
        _df       = pd.read_csv(INTERLEAVED_CSV)
        test_qids = sorted(_df['qid'].unique().tolist())[:3]
    elif args.test_qids:
        test_qids = args.test_qids
    else:
        test_qids = None   # all 43

    run_pipeline(
        api_key        = api_key,
        test_qids      = test_qids,
        passage_length = args.passage_length,
        query_length   = args.query_length,
        k_per_query    = 9,
    )
