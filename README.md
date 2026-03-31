# Selective Pseudo-Relevance Feedback (sPRF)

This repository contains the code and data for two related research papers on **Selective Pseudo-Relevance Feedback**:

1. **Stage 1 — Participatory Audit (SIGIR 2026):** A user study with 108 participants demonstrating that PRF hurts 25.6% of queries while only benefiting 20.9%, using Team Draft Interleaving to collect implicit user preferences.

2. **Stage 2 — Automated Selective PRF (TOIS):** Automated methods to predict per-query PRF outcomes using LLM-based document rerankers as system preference predictors, achieving AUC=0.808 (single-run) and AUC=0.832 (self-consistency), substantially outperforming QPP baselines (AUC=0.583).

## Project Structure

```
selective-prf/
├── stage1_user_study/                 # Participatory audit (SIGIR paper)
│   ├── initial_ranking/               # ColBERT indexing & ranking generation
│   ├── data_preparation/              # Data deduplication, text extraction, overlap analysis
│   ├── analysis/                      # User behaviour analysis, fairness, QPP correlation
│   │   └── qpp_analysis/             # NQC, WIG, Clarity, SMV analysis
│   └── colbert_prf_appendix/          # ColBERT-PRF baseline results (TREC format)
│
├── stage2_auto_sprf/                  # Automated sPRF prediction (TOIS paper)
│   ├── src/                           # Core library
│   │   ├── config.py                 # All constants, paths, model configs, h=0.061
│   │   ├── data_loader.py            # CSV loading and validation
│   │   ├── metrics.py                # TPR, Accuracy, AUC, Gamma, Spearman
│   │   ├── aggregation.py            # DCG ratio, MRR ratio, Majority@k
│   │   ├── prehoc/                   # Pre-hoc features (NED, IDF, QTC)
│   │   ├── posthoc/                  # Post-hoc rerankers (Pairwise, Setwise, Pointwise, SC, Ensemble)
│   │   └── qpp/                      # QPP baselines (WIG, NQC, Clarity, SMV)
│   ├── scripts/                       # Run scripts for each method family
│   ├── llm-rankers/                   # External library (ielab/llm-rankers + Rank-R1)
│   └── data/                          # Labels, interleaved list, drift lexicon
│
├── results/                           # Computed results (CSVs, logs)
│   ├── prehoc/                       # Pre-hoc method results
│   ├── posthoc/                      # Post-hoc reranker results
│   ├── qpp/                          # QPP baseline results
│   ├── classifier/                   # Direct classification approach results
│   └── logs/                         # Execution logs
│
├── notebooks/                         # LLM inference experiments (zero-shot, few-shot, CoT)
├── figures/                           # Generated visualisations (heatmaps, scatter plots)
├── ARCHITECTURE.md                    # Detailed design document
└── requirements.txt                   # Python dependencies
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Generate ground-truth labels (h=0.061)
cd stage2_auto_sprf
python scripts/build_labels.py

# 3. Run smoke test (3 queries, verifies setup)
python scripts/smoke_test.py

# 4. Run individual method families
python scripts/run_prehoc.py          # NED, IDF Coverage, QTC
python scripts/run_preprf.py          # WIG, NQC, Clarity, SMV, TDC
python scripts/run_pairwise_72b.py    # Qwen-72B API (needs IDA_LLM_API_KEY)
python scripts/run_pairwise_7b.py     # Qwen-7B local (needs CUDA GPU)
python scripts/run_setwise.py         # Rank-R1 (needs GPU + vLLM)
python scripts/run_sc.py              # Self-consistency (5 runs)
python scripts/run_ensemble.py        # DR combinations (reads cached results)

# 5. Run direct classification approaches (RQ4)
python scripts/approach_1a_query_classifier.py   # Query encoder features + NLI
python scripts/approach_1b_llm_query_classify.py # LLM zero-shot query classification
python scripts/approach_2a_list_judge.py         # LLM list-level quality judge
python scripts/approach_2d_crossencoder_features.py  # Cross-encoder score features

# 6. View results
python scripts/print_results_table.py
```

## Data

The dataset consists of 43 queries from TREC Deep Learning 2019, with ground-truth preference labels collected from a participatory audit with 108 users via Team Draft Interleaving.

- At h=0.061 (MAD threshold): 14 Hurt, 12 Benefit, 15 Neutral, 2 Insufficient-Data
- At h=0.00 (binary): 20 Hurt, 18 Benefit (38 queries with clear preference)

**Large data files** (`df_colbert_deduped.csv`, `df_colbert_prf_deduped.csv`) are excluded from this repository due to size. These contain the full ColBERT and ColBERT-PRF retrieval runs and can be regenerated using the scripts in `stage1_user_study/initial_ranking/`.

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| h (neutral band) | 0.061 | MAD of 72B ratio distribution |
| Top-k candidates | 9 | Interleaved list size |
| SC runs | 5 | Self-consistency repetitions |
| Passage length | 128 tokens | Truncation limit |
| Query length | 32 tokens | Truncation limit |

## Models

The following LLM families are evaluated:

| Model | Scale | Access | Use |
|-------|-------|--------|-----|
| Qwen-2.5 | 7B, 14B, 72B | Local GPU / API | Pairwise reranking |
| Qwen-3 | 8B, 14B | Local GPU | Pairwise + Setwise |
| Rank-R1 | 7B, 14B | Local GPU + vLLM | Setwise (reasoning) |
| Rank-R1 SFT | 7B, 14B | Local GPU + vLLM | Setwise (SFT) |
| Qwen-3 Reranker | 4B, 8B | Local GPU | Pointwise scoring |
| LLaMA-3.1 | 8B | Local GPU | Pairwise |
| RankZephyr | 7B | Local GPU | Pairwise/Setwise |

## Citation

If you use this code, please cite:

```
@inproceedings{liang2026auditing,
  title={Auditing Query Drift: Do Users Actually Benefit from Pseudo-Relevance Feedback?},
  author={Liang, Zeyan and McDonald, Graham and Ounis, Iadh},
  booktitle={Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '26)},
  year={2026}
}
```

## License

This project is licensed under the MIT License.
