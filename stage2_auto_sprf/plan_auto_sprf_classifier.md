# Implementation Plan: Query-Level sPRF Classifiers

## Overview

We build classifiers that predict whether PRF will help or hurt a query, using two input levels:
- **Query-only** (1a, 1b): just the query text
- **Query + initial list** (2a, 2d, 2e): query text + top-k ColBERT results

All 43 queries (38 with clear preference: 20 Hurt, 18 Benefit) serve as the **test set only** — no training on these queries. We use off-the-shelf models throughout.

---

## Approach 1a: Off-the-Shelf Query Encoder as Classifier

### What
Use a pre-trained cross-encoder or sentence model to compute a query "difficulty" score without any fine-tuning.

### How

**Step 1: Load query data**
- Read the 43 TREC DL'19 queries and their ground truth labels (b_ratio from the user study)
- Binary label: b_ratio >= 0.55 → Benefit, b_ratio <= 0.45 → Hurt, else Neutral (exclude)

**Step 2: Compute query features from pre-trained models**
- Use a sentence-transformer model (e.g., `all-MiniLM-L6-v2` or `BAAI/bge-large-en-v1.5`) to embed each query
- Compute self-similarity features:
  - Query embedding norm (proxy for specificity)
  - Cosine similarity between the query and its individual terms (term importance spread)
- Use an NER model (spaCy `en_core_web_sm`) to count named entities
- Compute IDF statistics of query terms against the MS MARCO corpus (you likely already have this from the NED/IDF Coverage features)

**Step 3: Zero-shot classification with NLI model**
- Use `facebook/bart-large-mnli` or `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` as a zero-shot classifier
- Candidate labels: `["this query will benefit from query expansion", "this query will be hurt by query expansion"]`
- For each query, get the probability of each label → this is the prediction score
- No training, just inference

**Step 4: Evaluate**
- Compute Accuracy, AUC, gamma, rho against the 38 labelled queries
- Compare to your existing pre-hoc baselines (NED, IDF Coverage, QTC)

### Code structure
```
scripts/approach_1a_query_classifier.py
  - load_queries(path) → dict[qid, query_text]
  - load_labels(path) → dict[qid, label]  # from user study b_ratio
  - compute_embedding_features(queries, model_name) → DataFrame
  - zero_shot_classify(queries, model_name, candidate_labels) → DataFrame
  - evaluate(predictions, labels) → metrics dict
  - main(): run all, print results table
```

### Dependencies
- `transformers`, `sentence-transformers`, `spacy`, `sklearn`

---

## Approach 1b: LLM Zero-Shot Query Classification

### What
Prompt an LLM to predict PRF outcome from the query text alone.

### How

**Step 1: Design prompts**

Three prompt variants to test:

**Prompt A (direct):**
```
Given the search query: "{query}"

Do you think expanding this query using terms from the top-ranked search
results (pseudo-relevance feedback) would improve or degrade the search
quality?

Consider:
- Is the query specific and unambiguous?
- Are the top search results likely to be on-topic?
- Could adding related terms cause the search to drift away from the
  user's intent?

Answer with exactly one word: Help or Hurt
```

**Prompt B (chain-of-thought):**
```
Given the search query: "{query}"

I want to decide whether to apply pseudo-relevance feedback (PRF) to
this query. PRF expands the query using terms extracted from the
top-ranked documents.

Think step by step:
1. What is the user's likely information need?
2. How specific or ambiguous is this query?
3. Are the top search results for this query likely to be relevant?
4. If we extract expansion terms from those results, will they stay
   on-topic or drift?

Based on your reasoning, answer: Help or Hurt
```

**Prompt C (expert persona):**
```
You are an expert information retrieval researcher. You are evaluating
whether pseudo-relevance feedback (PRF) should be applied to a search
query. PRF works by taking the top-ranked documents, extracting key
terms, and adding them to the query. This helps when the initial results
are relevant (good expansion terms), but hurts when they are off-topic
(query drift).

Query: "{query}"

Based on your expertise, will PRF help or hurt this query?
Answer: Help or Hurt
```

**Step 2: Run inference**
- Models to test:
  - **Standard:** Qwen3-14B-Instruct (best model), Qwen3-8B-Instruct, Qwen-2.5-72B-Instruct
  - **Reasoning/R1:** Qwen3-14B-Think (thinking mode enabled), Rank-R1-14B, Rank-R1-7B
- For reasoning models, extract the `<think>` trace and final answer separately; use the final answer for classification
- For each query × prompt × model: collect the answer and any confidence signal
- Run each 5 times with temperature=0.7 for self-consistency (majority vote)

**Step 3: Parse and evaluate**
- Extract Help/Hurt from model output
- Compute Accuracy, AUC (using self-consistency vote proportion as score), gamma, rho

### Code structure
```
scripts/approach_1b_llm_query_classify.py
  - load_queries(path) → dict
  - load_labels(path) → dict
  - build_prompt(query, template) → str
  - run_llm(prompt, model_name, temperature, n_runs) → list[str]
  - parse_response(response) → "Help" | "Hurt" | "Unknown"
  - self_consistency(responses) → (label, confidence_score)
  - evaluate(predictions, labels) → metrics
  - main(): loop over queries × prompts × models, save results CSV
```

### Dependencies
- `vllm` or `transformers` for local inference (models already downloaded from Paper 2)

---

## Approach 2a: LLM as List-Level Quality Judge

### What
Present the LLM with the query AND the top-k initial ColBERT results. Ask it directly: "Does this list need expansion?"

### How

**Step 1: Prepare inputs**
- For each of the 43 queries, extract the top-5 passages from the initial ColBERT ranking (before PRF)
- Format each passage as: title/docid + passage text (truncated to ~200 tokens)

**Step 2: Design prompts**

**Prompt A (coverage assessment):**
```
You are evaluating search results for the query: "{query}"

Here are the top 5 results:
[1] {passage_1}
[2] {passage_2}
[3] {passage_3}
[4] {passage_4}
[5] {passage_5}

Do these results adequately answer the user's query? Consider:
1. Are the results relevant to the query?
2. Do they cover different aspects of the information need?
3. Is there important information missing that query expansion could find?

If the results are already good, expansion risks introducing noise.
If the results are weak or incomplete, expansion could help find
better content.

Answer with exactly one word: Adequate or Expand
```
(Map: Adequate → Hurt, Expand → Benefit)

**Prompt B (comparative reasoning):**
```
You are an information retrieval expert. A user searched for: "{query}"

The search system returned these top 5 passages:
[1] {passage_1}
[2] {passage_2}
[3] {passage_3}
[4] {passage_4}
[5] {passage_5}

We are considering applying pseudo-relevance feedback (PRF), which
extracts key terms from these results and adds them to the query to
find additional relevant content.

Think step by step:
1. Are these results relevant to the query?
2. If we extract terms from these passages, will they be good expansion
   terms or will they cause drift?
3. Is there evidence that important aspects of the query are missing
   from these results?

Based on your analysis, should we apply PRF?
Answer: Apply or Skip
```
(Map: Apply → Benefit, Skip → Hurt)

**Step 3: Run inference**
- Same models as 1b:
  - **Standard:** Qwen3-14B, Qwen3-8B, Qwen-2.5-72B
  - **Reasoning/R1:** Qwen3-14B-Think, Rank-R1-14B, Rank-R1-7B
- For reasoning models, extract `<think>` trace and final answer separately
- Self-consistency: 5 runs per query, majority vote
- Also test with top-3 and top-10 passages to see sensitivity to k

**Step 4: Evaluate**
- Accuracy, AUC, gamma, rho against 38 labelled queries
- Compare to your existing QPP baselines AND to the full reranking approach (AUC=0.808)

### Code structure
```
scripts/approach_2a_list_judge.py
  - load_queries(path) → dict
  - load_labels(path) → dict
  - load_initial_rankings(path) → dict[qid, list[passage_text]]
  - build_prompt(query, passages, template, top_k) → str
  - run_llm(prompt, model_name, temperature, n_runs) → list[str]
  - parse_response(response) → "Benefit" | "Hurt"
  - self_consistency(responses) → (label, confidence_score)
  - evaluate(predictions, labels) → metrics
  - main(): loop over queries × prompts × models × top_k, save results
```

### Key difference from Paper 2
Paper 2 presents the LLM with an interleaved document set and infers preference from the ranking. Approach 2a skips the ranking step entirely — it asks the LLM a direct yes/no question about the initial list. Much simpler, and tests a fundamentally different mechanism.

---

## Approach 2d: Cross-Encoder Confidence Gap

### What
Use the existing Qwen3-Reranker scores on the initial ColBERT list to derive QPP-style features. No LLM prompting — just numerical analysis of scores you likely already have.

### How

**Step 1: Score documents**
- For each query, take the top-10 passages from the initial ColBERT ranking
- Score each passage with Qwen3-Reranker-8B (pointwise cross-encoder)
- You may already have these scores from the Paper 2 experiments

**Step 2: Compute features from the score distribution**
For each query, compute:

| Feature | Description | Intuition |
|---|---|---|
| `score_mean` | Mean reranker score of top-k | High mean → good initial list → PRF risky |
| `score_std` | Std of reranker scores | Low std → all docs similar quality → PRF may not help |
| `score_max` | Max reranker score | Very high → at least one great result → PRF risky |
| `score_gap_1_2` | Score[rank1] - Score[rank2] | Large gap → clear best result → expansion may dilute |
| `score_gap_1_k` | Score[rank1] - Score[rank_k] | Score range across the list |
| `score_entropy` | Entropy of normalised scores | High entropy → uncertain ranking → PRF may help |
| `top3_coherence` | Mean pairwise cosine sim of top-3 passages (using ColBERT embeddings) | High coherence → on-topic → safe expansion |
| `top3_vs_rest` | Mean sim(top-3, docs 4-10) | Low → clear relevance drop-off → good list |

**Step 3: Predict**
- Since N=38 is too small for training, use each feature independently as a predictor:
  - Threshold sweep → best accuracy per feature
  - Rank correlation (rho, gamma) with b_ratio
- Report the best single feature and its direction

**Step 4: Save features for Approach 2e**
- Output a CSV: `qid, feature_1, feature_2, ..., label`

### Code structure
```
scripts/approach_2d_crossencoder_features.py
  - load_queries(path) → dict
  - load_labels(path) → dict
  - load_reranker_scores(path) → dict[qid, list[(docid, score)]]
  - load_colbert_embeddings(path) → dict[docid, embedding]
  - compute_score_features(scores) → dict of features
  - compute_coherence_features(embeddings, top_k) → dict of features
  - evaluate_single_features(features_df, labels) → per-feature metrics
  - main(): compute all features, evaluate, save CSV
```

### Dependencies
- `numpy`, `scipy`, `sklearn`
- Reranker scores from Paper 2 experiments (check `results/` directory)
- ColBERT embeddings (from the retrieval index)

---

## Approach 2e: Combined Feature Classifier

### What
Combine ALL signals from 1a, 1b, 2a, 2d into a single classifier. Since N=38, use leave-one-out cross-validation with a simple model.

### How

**Step 1: Assemble feature matrix**
For each query, collect:

| Source | Features |
|---|---|
| 1a: Query encoder | Zero-shot NLI probability, query embedding norm, entity count, IDF stats |
| 1b: LLM query-only | Self-consistency vote proportion (3 prompts × 6 models incl. R1/Think) |
| 2a: LLM list judge | Self-consistency vote proportion (2 prompts × 6 models incl. R1/Think × top_k) |
| 2d: Cross-encoder | score_mean, score_std, score_max, score_gap_1_2, score_entropy, top3_coherence |
| Existing (Paper 2) | Best single-run LLM reranker score (Setwise-Qwen3-14B DCG/MRR score) |

**Step 2: Feature selection**
- With N=38 and potentially 20+ features, overfitting is a serious risk
- Use forward feature selection with LOO-CV: start with the best single feature, greedily add features that improve LOO accuracy
- Cap at 3-5 features maximum

**Step 3: Train classifier**
- **Logistic Regression** (L2 regularised) — most appropriate for small N
- **Random Forest** (max_depth=2, n_estimators=50) — handles non-linear interactions
- **XGBoost** (max_depth=2, n_estimators=20, high regularisation) — if RF works
- All evaluated with **leave-one-out cross-validation** (LOO-CV) on the 38 queries

**Step 4: Evaluate**
- LOO-CV Accuracy, AUC (using predicted probability as score), gamma, rho
- Compare to:
  - Best single feature (from 2d)
  - Best LLM approach (from 2a)
  - Best reranking approach from Paper 2 (AUC=0.808)
  - TREC nDCG baseline (AUC=0.681)

**Step 5: Ablation**
- Which feature group contributes most? (query-only vs list-level vs reranker score)
- Does the LLM list judge (2a) add value beyond the cross-encoder features (2d)?
- Does query-only (1a, 1b) add anything on top of list-level features?

### Code structure
```
scripts/approach_2e_combined_classifier.py
  - load_all_features(paths) → DataFrame  # merge outputs from 1a, 1b, 2a, 2d
  - load_labels(path) → Series
  - forward_feature_selection(X, y, max_features, model_class) → list[str]
  - loo_cv_evaluate(X, y, model_class) → metrics dict
  - run_ablation(X, y, feature_groups) → ablation table
  - main(): load features, select, train, evaluate, ablation
```

### Dependencies
- `sklearn`, `xgboost`, `pandas`

---

## Execution Order

| Step | Script | Depends on | Time estimate |
|---|---|---|---|
| 0 | Prepare data: extract queries, labels, initial rankings, reranker scores into clean CSVs | Existing data from Paper 2 | 30 min |
| 1 | `approach_1a_query_classifier.py` | Step 0 | 30 min (inference) |
| 2 | `approach_1b_llm_query_classify.py` | Step 0 | 1-2 hrs (LLM inference × 3 prompts × 3 models × 5 runs) |
| 3 | `approach_2a_list_judge.py` | Step 0 | 2-3 hrs (LLM inference with longer prompts) |
| 4 | `approach_2d_crossencoder_features.py` | Step 0 | 30 min (may already have scores) |
| 5 | `approach_2e_combined_classifier.py` | Steps 1-4 | 15 min (just sklearn) |

Steps 1, 2, 3, 4 are independent and can run in parallel.

---

## Data Files Needed

Before coding, locate or create these files:

```
data/
  queries.tsv              # qid \t query_text (43 queries)
  labels.tsv               # qid \t b_ratio \t label (Benefit/Hurt/Neutral)
  initial_rankings/        # qid → top-k passage texts from ColBERT (before PRF)
  reranker_scores/         # qid → [(docid, qwen3_reranker_score)] from Paper 2
  colbert_embeddings/      # docid → embedding vector (if available)
```

### Step 0: Data preparation script
```
scripts/prepare_data.py
  - Read from existing Paper 2 data files (check results/, analysis/ directories)
  - Extract and standardise into the above format
  - Verify: 43 queries, 38 with clear labels, top-k passages per query
```

---

## Expected Outcome

| Approach | Input | AUC target | Comparison |
|---|---|---|---|
| 1a: Query encoder | Query only | 0.55-0.65 | Beat random (0.50), probably below QPP baselines |
| 1b: LLM query-only | Query only | 0.60-0.70 | Beat pre-hoc baselines (best=0.556) |
| 2a: LLM list judge | Query + list | 0.70-0.80 | Beat QPP baselines (best=0.583), approach Paper 2 |
| 2d: Cross-encoder features | Query + list | 0.65-0.75 | Complement to 2a |
| 2e: Combined | All | 0.80-0.85 | Beat Paper 2 single-run (0.808) |

The key question is whether **2a (direct LLM judgment on the list)** can match Paper 2's approach of **full reranking + preference inference** — if it can, that's a much simpler and cheaper method. And if **2e** pushes past 0.808, that demonstrates the value of combining complementary signals.
