"""
Centralised configuration — all constants, paths, and hyperparameters.

No other file should hardcode paths or threshold values.
"""

from pathlib import Path

# ── Project root (one level above src/) ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Threshold ────────────────────────────────────────────────────────────────
H_THRESHOLD = 0.061  # MAD-derived neutral band half-width

# ── Paths (relative to PROJECT_ROOT) ────────────────────────────────────────
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

# Input data
PREFERENCE_CSV = Path("preference.csv")  # raw user-study file (project root)
INTERLEAVED_PATH = DATA_DIR / "result_interleaved_with_text.csv"
COLBERT_PATH = DATA_DIR / "df_colbert_deduped.csv"
COLBERT_PRF_PATH = DATA_DIR / "df_colbert_prf_deduped.csv"
DRIFT_LEXICON_PATH = DATA_DIR / "drift_lexicon.txt"

# Generated ground truth
LABELS_PATH = DATA_DIR / "labels.csv"

# Result sub-directories
PREHOC_RESULTS_DIR = RESULTS_DIR / "prehoc"
POSTHOC_RESULTS_DIR = RESULTS_DIR / "posthoc"
QPP_RESULTS_DIR = RESULTS_DIR / "qpp"
LOGS_DIR = RESULTS_DIR / "logs"
SUMMARY_TABLE_PATH = RESULTS_DIR / "summary_table.csv"

# ── Document origin labels ───────────────────────────────────────────────────
PRF_LABEL = "B-Only"
BASE_LABEL = "A-Only"

# ── Model configuration ─────────────────────────────────────────────────────
# Qwen 2.5 — local GPU
QWEN_7B_MODEL = "Qwen/Qwen2.5-7B-Instruct"
QWEN_14B_MODEL = "Qwen/Qwen2.5-14B-Instruct"
QWEN_7B_CACHE_DIR = "/mnt/primary/QE audit/hf_cache"

# Qwen 3 — local GPU
QWEN3_8B_MODEL = "Qwen/Qwen3-8B"
QWEN3_14B_MODEL = "Qwen/Qwen3-14B"

# Qwen 3 Reranker — pointwise (local GPU)
QWEN3_RERANKER_4B_MODEL = "Qwen/Qwen3-Reranker-4B"
QWEN3_RERANKER_8B_MODEL = "Qwen/Qwen3-Reranker-8B"

# LLaMA 3 — local GPU
LLAMA_8B_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Mistral — local GPU
MISTRAL_7B_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_24B_MODEL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

# RankZephyr — listwise reranker (Mistral-based, for pairwise/setwise zero-shot)
RANKZEPHYR_7B_MODEL = "castorini/rank_zephyr_7b_v1_full"

# Qwen 72B — university API
QWEN_72B_API_URL = "http://api.llm.apps.os.dcs.gla.ac.uk/v1"
QWEN_72B_API_MODEL = "qwen-2.5-72b-instruct"
QWEN_72B_API_KEY_ENV = "IDA_LLM_API_KEY"

# Rank-R1 models (HuggingFace repo IDs — ielabgroup)
RANK_R1_MODELS = {
    "7b": "ielabgroup/Rank-R1-7B-v0.1",
    "14b": "ielabgroup/Rank-R1-14B-v0.1",
    "7b-sft": "ielabgroup/Setwise-SFT-7B-v0.1",
    "14b-sft": "ielabgroup/Setwise-SFT-14B-v0.1",
}
# Rank-R1 v0.2 — LoRA adapter on Qwen3-32B
RANK_R1_V02_32B_BASE = "Qwen/Qwen3-32B"
RANK_R1_V02_32B_LORA = "ielabgroup/Rank-R1-32B-v0.2"

RANK_R1_CACHE_DIR = "/mnt/primary/QE audit/hf_cache"

# Sentence transformer for semantic features
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Stronger embedding models for approach 1a
BGE_LARGE_MODEL = "BAAI/bge-large-en-v1.5"
GTE_LARGE_MODEL = "thenlper/gte-large"
COLBERT_V2_MODEL = "colbert-ir/colbertv2.0"

# Cross-encoder for approach 2d
MS_MARCO_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# ── Reranking hyperparameters ────────────────────────────────────────────────
TOP_K_CANDIDATES = 9       # total candidates per query in interleaved list
TOP_K_MAJORITY = 5         # top-k cutoff for Majority@k count ratio
SC_NUM_RUNS = 5            # number of self-consistency runs
PASSAGE_LENGTH = 128       # max passage tokens for truncation
QUERY_LENGTH = 32          # max query tokens for truncation

# ── Prompts ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are RankGPT, an intelligent assistant that selects the most "
    "relevant passage from a pair of passages based on their relevance "
    "to a given query. Output exactly \"Passage A\" or \"Passage B\"."
)

PAIRWISE_PROMPT = (
    'Given a query "{query}", which of the following two passages is more '
    'relevant to the query?\n\n'
    'Passage A: "{doc1}"\n\n'
    'Passage B: "{doc2}"\n\n'
    'Output Passage A or Passage B:'
)

# ── Summary table schema ────────────────────────────────────────────────────
SUMMARY_COLUMNS = [
    "method_name", "stage", "predictor",
    "tpr_hurt", "tpr_benefit", "acc", "auc",
    "gamma", "gamma_p", "rho", "rho_p",
    "n_queries", "n_hurt", "n_benefit", "timestamp",
]

# ── Permutation test settings ───────────────────────────────────────────────
GAMMA_N_PERMUTATIONS = 5000

# ── llm-rankers library path ────────────────────────────────────────────────
LLM_RANKERS_PATH = PROJECT_ROOT / "llm-rankers"
