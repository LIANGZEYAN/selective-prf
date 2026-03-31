# Architecture Plan — Selective PRF Codebase Restructuring

## What Each Existing File Does

### Main Script
- **`run_pairwise_prf.py`** — The primary pipeline. Uses Qwen-2.5-72B via a university
  OpenAI-compatible API (`http://api.llm.apps.os.dcs.gla.ac.uk/v1`) to pairwise-rerank
  interleaved TDI documents per query. Implements `QwenApiPairwiseLlmRanker` (extends
  `LlmRanker` from llm-rankers). Computes a DCG-weighted B-preference ratio, classifies
  queries (Benefit/Hurt/Neutral), and evaluates against human labels.
  **Key issue**: hardcodes `PREF_THRESHOLD = 0.0` but the paper uses h=0.061.

### Data Files
- **`preference.csv`** — 43 queries. Columns: `qid, a_only_clicks, b_only_clicks,
  both_clicks, total_clicks, discriminative_clicks, b_preference_ratio, preference`.
  2 queries are Insufficient-Data (zero discriminative clicks).
- **`result_interleaved_with_text.csv`** — 387 rows (43 queries x 9 docs). Columns:
  `qid, docno, rank, origin_label, query_text, passage_text`. Origin labels: A-Only,
  B-Only, Both, Easy-Negative.
- **`df_colbert_deduped.csv`** — Full ColBERT (no PRF) rankings. Columns: `qid, docno,
  rank, score, name, passage_text`. Many docs per query (full retrieval run).
- **`df_colbert_prf_deduped.csv`** — Full ColBERT+PRF rankings. Same schema.
- **`trec_eval_perquery.csv`** — NDCG@10 for both systems per query, with h=0.061 labels.
  Columns: `qid, ndcg10_colbert, ndcg10_prf, preference, b_preference_ratio, delta, label_h061`.

### Output/Result Files (already computed, will be regenerated)
- **`pairwise_results.csv`** — Identical to `pairwise_results_72b.csv` (duplicate).
- **`pairwise_results_72b.csv`** — Per-query 72B results. Columns: `qid, query, llm_ratio,
  human_ratio, llm_pred, human_preference, correct, n_comparisons`. Uses h=0.0 threshold
  for classification (produces many false Benefit/Hurt on near-0.5 ratios).
- **`pairwise_results_7b.csv`** — Per-query 7B results. Same schema but ratios are tightly
  clustered near 0.5 (MAD=0.021), indicating poor discriminative power.
- **`analysis_results.csv`** — Summary metrics for both models at h=0.061.
- **`threshold_sweep_72b.csv`** — Threshold sensitivity analysis (h=0.00 to 0.12).
- **`threshold_sweep_results.csv`** — Alternate threshold sweep with Pearson/Kendall stats.
- **`agreement_metrics.csv`** — AUC, Gamma, Spearman rho for both models.

### Supporting Code
- **`download.py`** — Downloads Qwen2.5-14B from HuggingFace (note: 14B, not 7B or 72B).
- **`llm-rankers/`** — External library (ielab/llm-rankers). Key files:
  - `llmrankers/rankers.py` — `SearchResult` dataclass + `LlmRanker` base class.
  - `llmrankers/pairwise.py` — `PairwiseLlmRanker` (local GPU; supports T5, Llama, Qwen2
    model types; heapsort/allpair/bubblesort), `DuoT5LlmRanker`, `OpenAiPairwiseLlmRanker`.
  - `llmrankers/setwise.py` — `SetwiseLlmRanker`, `OpenAiSetwiseLlmRanker`,
    `RankR1SetwiseLlmRanker` (uses vLLM + LoRA).
  - `Rank-R1/run_setwise.py` — Standalone runner for Rank-R1 setwise reranking on
    standard IR benchmarks (Pyserini index + TREC run files). Contains `R1SetwiseLlmRanker`.

### Documentation
- **`experiment_details.txt`** — Comprehensive: research question, dataset, method,
  threshold derivation (h=0.061 = MAD of 72B ratio distribution), evaluation protocol
  (Neutrals excluded), main results tables, agreement metrics, implementation notes.

---

## Data Flow

```
preference.csv ──────────────────────────────────────┐
                                                      ▼
result_interleaved_with_text.csv ──► run_pairwise_prf.py ──► pairwise_results_72b.csv
                                       │                         │
                                       │ (loads docs,            ▼
                                       │  calls QwenApi          Post-hoc analysis
                                       │  PairwiseRanker,        (threshold sweep,
                                       │  computes DCG ratio,    agreement metrics
                                       │  classifies,            — done externally,
                                       │  evaluates)             probably in notebook)
                                       │
                                       ▼
                             llm-rankers/llmrankers/
                             (SearchResult, LlmRanker base)

df_colbert_deduped.csv ─────────► trec_eval_perquery.csv
df_colbert_prf_deduped.csv ─────┘  (NDCG@10 comparison — computed externally)
```

The two ColBERT CSVs provide full retrieval rankings used to compute NDCG@10 per query
(in `trec_eval_perquery.csv`). They are NOT used by `run_pairwise_prf.py` which only
reads the interleaved list.

---

## Redundancy and Duplication Found

1. **`pairwise_results.csv` is a byte-for-byte duplicate of `pairwise_results_72b.csv`.**
   One should be removed; the restructured code will only produce named outputs.

2. **Heapsort appears 4+ times**: in `run_pairwise_prf.py` (QwenApiPairwiseLlmRanker),
   `llmrankers/pairwise.py` (PairwiseLlmRanker), `llmrankers/setwise.py`
   (SetwiseLlmRanker), and `Rank-R1/run_setwise.py` (R1SetwiseLlmRanker). The new code
   should NOT touch the llm-rankers library (treat as external), but the custom
   `QwenApiPairwiseLlmRanker` in run_pairwise_prf.py duplicates the llm-rankers heapsort.

3. **The `compare()` → `heapSort()` → `rerank()` pattern** is reimplemented in the main
   script's `QwenApiPairwiseLlmRanker` when it could extend the library version.

4. **Threshold h=0.0 in code vs h=0.061 in analysis.** The code's `classify()` uses h=0.0;
   the experiment_details.txt and result CSVs show h=0.061 was applied in a separate
   analysis step. This creates confusion — the single pipeline should use h=0.061.

5. **No modular metrics code.** Evaluation metrics (Pearson, Kendall, accuracy, per-class
   recall) are embedded in `evaluate()` inside `run_pairwise_prf.py`. The agreement metrics
   (AUC, Gamma, Spearman) are not in any Python file at all — they were likely computed in
   an ad-hoc notebook. These need to be consolidated into `src/metrics.py`.

6. **`download.py` downloads the wrong model** (14B instead of 7B). The restructured
   `scripts/download_models.py` will download the correct Rank-R1 models.

---

## Proposed New Structure

```
project/
├── data/
│   ├── labels.csv                          # Generated by build_labels.py (h=0.061)
│   ├── result_interleaved_with_text.csv    # Existing input (symlink or copy)
│   ├── df_colbert_deduped.csv              # Existing input
│   ├── df_colbert_prf_deduped.csv          # Existing input
│   └── drift_lexicon.txt                   # Drift-indicative terms for ensemble
│
├── scripts/
│   ├── build_labels.py         # Reads preference.csv → data/labels.csv (h=0.061)
│   ├── run_prehoc.py           # NED, IDF Coverage, QTC
│   ├── run_preprf.py           # WIG, NQC, Clarity, SMV, TDC, Pre-PRF SC
│   ├── run_pairwise_7b.py      # Qwen-7B local: DCG, MRR, Majority@k
│   ├── run_pairwise_72b.py     # Qwen-72B API: DCG, MRR, Majority@k
│   ├── run_sc.py               # Self-consistency (R=5 runs via 72B)
│   ├── run_setwise.py          # Rank-R1 (4 models): DCG + reasoning trace
│   ├── run_ensemble.py         # DR combinations (no LLM, reads cached scores)
│   ├── print_results_table.py  # Pretty-print summary_table.csv
│   ├── download_models.py      # Download Rank-R1 models from HuggingFace
│   └── smoke_test.py           # End-to-end verification on 3 queries
│
├── src/
│   ├── __init__.py
│   ├── config.py               # All constants, paths, model configs, h=0.061
│   ├── data_loader.py          # Load & validate all CSVs
│   ├── metrics.py              # TPR_H, TPR_B, Acc, AUC, Gamma, Rho
│   ├── aggregation.py          # DCG ratio, MRR ratio, Majority@k ratio
│   │
│   ├── prehoc/
│   │   ├── __init__.py
│   │   ├── lexical.py          # NED, IDF Coverage
│   │   ├── semantic.py         # Query Term Coherence
│   │   └── preprf.py           # Top-k Doc Coherence, Pre-PRF SC, QPP wrappers
│   │
│   ├── posthoc/
│   │   ├── __init__.py
│   │   ├── pairwise.py         # Qwen 7B local reranker
│   │   ├── pairwise_api.py     # Qwen 72B API reranker
│   │   ├── setwise.py          # Rank-R1 setwise reranker (4 models)
│   │   ├── self_consistency.py # SC-Mean, SC-SNR (R=5)
│   │   └── ensemble.py         # DR combinations, reasoning trace mining
│   │
│   └── qpp/
│       ├── __init__.py
│       └── baselines.py        # WIG, NQC, Clarity, SMV via pyterrier-prf
│
├── results/
│   ├── prehoc/                 # One CSV per method
│   ├── posthoc/                # One CSV per method
│   ├── qpp/                    # One CSV per QPP method
│   ├── logs/                   # Log files from each run
│   └── summary_table.csv       # Aggregated metrics
│
├── notebooks/
│   └── analysis.ipynb          # Optional exploration
│
├── ARCHITECTURE.md             # This file
├── requirements.txt
├── README.md
└── run_pairwise_prf.py         # Legacy entry point (deprecation notice)
```

### Key design decisions:
1. **h=0.061 everywhere.** Defined once in `src/config.py`, used by `build_labels.py` to
   create ground truth, and by all classification code. No other script re-implements labelling.
2. **`data/labels.csv` is the single source of truth** for ground-truth labels.
3. **`src/aggregation.py` is pure math** — takes ranked lists + doc ID sets, returns ratios.
   No model loading, no API calls.
4. **Ranker classes share the `rerank(query, doc_list) → [(doc_id, rank)]` interface.**
5. **llm-rankers/ is treated as external** — no modifications. We import `SearchResult` and
   `LlmRanker` from it. The custom `QwenApiPairwiseLlmRanker` is restructured into
   `src/posthoc/pairwise_api.py`.
6. **All file I/O uses `pathlib.Path`**, all constants in `config.py`.
