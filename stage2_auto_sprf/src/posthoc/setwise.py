"""
Setwise reranker using vLLM (Rank-R1 fine-tuned models or base models).

Uses RankR1SetwiseLlmRanker from llm-rankers for inference.
Stores the raw <think> reasoning trace and reasoning token counts
after each call for downstream mining.
"""

import sys
import re
from pathlib import Path

from src.config import (
    LLM_RANKERS_PATH,
    RANK_R1_MODELS,
    RANK_R1_CACHE_DIR,
    QWEN_7B_MODEL,
    QWEN_14B_MODEL,
    QWEN_7B_CACHE_DIR,
    TOP_K_CANDIDATES,
    PASSAGE_LENGTH,
    QUERY_LENGTH,
)

# Add llm-rankers to path
sys.path.insert(0, str(LLM_RANKERS_PATH))
from llmrankers.rankers import SearchResult

# Prompt files for different model types
PROMPTS_DIR = LLM_RANKERS_PATH / "Rank-R1" / "prompts"
PROMPT_BASE = PROMPTS_DIR / "prompt_setwise.toml"         # base models (no <think>)
PROMPT_R1 = PROMPTS_DIR / "prompt_setwise-R1.toml"        # Rank-R1 v0.1 fine-tuned (with <think>)
PROMPT_R1_V02 = PROMPTS_DIR / "prompt_setwise-R1-v0.2.toml"  # Rank-R1 v0.2 (Qwen3-based)


class SetwiseReranker:
    """Setwise reranker with reasoning trace capture.

    Supports both base models (e.g. Qwen-7B-Instruct) and Rank-R1 fine-tuned
    models. Shares the same rerank(query, doc_list) interface as pairwise classes.

    After each call to rerank(), the following attributes are populated:
      - last_reasoning_trace : str — raw <think> block text
      - last_reasoning_tokens_prf : int — reasoning tokens for PRF docs
      - last_reasoning_tokens_orig : int — reasoning tokens for orig docs
    """

    def __init__(
        self,
        model_name_or_path: str = None,
        model_key: str = None,
        lora_path: str = None,
        prompt_file: str = None,
        k: int = TOP_K_CANDIDATES,
        num_child: int = None,
        cache_dir: str = None,
        temperature: float = 0.0,
    ):
        """Initialise the setwise reranker.

        Parameters
        ----------
        model_name_or_path : HuggingFace model ID or local path.
            If None and model_key is given, looks up RANK_R1_MODELS.
            If both None, defaults to Qwen-7B-Instruct.
        model_key : shorthand key into RANK_R1_MODELS (e.g. "7b-sft").
        lora_path : path to LoRA adapter (None for base model).
        prompt_file : path to .toml prompt file. If None, auto-selects:
            prompt_setwise-R1.toml for Rank-R1 models, prompt_setwise.toml
            for base models.
        k : top-k documents to rerank.
        num_child : number of children in heap (default: k-1, i.e. all docs
            compared in one setwise call).
        cache_dir : HuggingFace cache directory.
        """
        # Resolve model
        if model_name_or_path is not None:
            self.model_name = model_name_or_path
        elif model_key is not None:
            if model_key not in RANK_R1_MODELS:
                raise ValueError(
                    f"Unknown model key '{model_key}'. "
                    f"Available: {list(RANK_R1_MODELS.keys())}"
                )
            # model_key maps to a LoRA adapter — use the base Qwen model
            self.model_name = QWEN_14B_MODEL if "14b" in model_key else QWEN_7B_MODEL
            if lora_path is None:
                lora_path = RANK_R1_MODELS[model_key]
        else:
            self.model_name = QWEN_7B_MODEL

        self.model_key = model_key
        self.lora_path = lora_path
        self.k = k
        self.num_child = num_child if num_child is not None else max(k - 1, 2)
        self.cache_dir = cache_dir or (QWEN_7B_CACHE_DIR if model_key is None else RANK_R1_CACHE_DIR)
        self.temperature = temperature

        # Auto-select prompt file
        if prompt_file is not None:
            self.prompt_file = str(prompt_file)
        elif lora_path is not None or (model_key is not None and "sft" not in (model_key or "")):
            # Rank-R1 model with reasoning
            self.prompt_file = str(PROMPT_R1) if PROMPT_R1.exists() else str(PROMPT_BASE)
        else:
            # Base model or SFT without reasoning
            self.prompt_file = str(PROMPT_BASE)

        self.last_reasoning_trace = ""
        self.last_reasoning_tokens_prf = 0
        self.last_reasoning_tokens_orig = 0

        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        self._ranker = None  # lazy-loaded

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._ranker is not None:
            return

        # Use HF transformers backend for large models that can't use vLLM TP
        use_hf = any(s in self.model_name.lower()
                     for s in ['32b', '30b', '33b', '34b'])
        if use_hf:
            from llmrankers.setwise import HFSetwiseLlmRanker
            self._ranker = HFSetwiseLlmRanker(
                model_name_or_path=self.model_name,
                prompt_file=self.prompt_file,
                lora_name_or_path=self.lora_path,
                tokenizer_name_or_path=self.model_name,
                num_child=self.num_child,
                k=self.k,
                scoring="generation",
                method="heapsort",
                num_permutation=1,
                cache_dir=self.cache_dir,
                verbose=True,
                temperature=self.temperature,
            )
        else:
            from llmrankers.setwise import RankR1SetwiseLlmRanker
            self._ranker = RankR1SetwiseLlmRanker(
                model_name_or_path=self.model_name,
                prompt_file=self.prompt_file,
                lora_name_or_path=self.lora_path,
                tokenizer_name_or_path=self.model_name,
                num_child=self.num_child,
                k=self.k,
                scoring="generation",
                method="heapsort",
                num_permutation=1,
                cache_dir=self.cache_dir,
                verbose=True,
                temperature=self.temperature,
            )

    def truncate_passage(self, text: str) -> str:
        self._ensure_loaded()
        return self._ranker.truncate(text, PASSAGE_LENGTH)

    def truncate_query(self, text: str) -> str:
        self._ensure_loaded()
        return self._ranker.truncate(text, QUERY_LENGTH)

    def rerank(self, query: str, doc_list: list) -> list:
        """Rerank documents using setwise comparisons.

        Parameters
        ----------
        query : query text string.
        doc_list : list of dicts with keys 'docid', 'text',
                   and optionally 'origin_label'.

        Returns
        -------
        list of (doc_id, rank) tuples, 1-based rank.
        """
        self._ensure_loaded()

        ranking = [
            SearchResult(docid=d["docid"], score=-i, text=d["text"])
            for i, d in enumerate(doc_list)
        ]

        self._ranker.all_completions = []
        reranked = self._ranker.rerank(query, ranking)

        self.total_compare = self._ranker.total_compare
        self.total_prompt_tokens = self._ranker.total_prompt_tokens
        self.total_completion_tokens = self._ranker.total_completion_tokens

        # Extract reasoning trace from last completion (if available)
        self._extract_reasoning_trace(doc_list)

        return [
            (r.docid, idx + 1)
            for idx, r in enumerate(reranked)
        ]

    def _extract_reasoning_trace(self, doc_list: list):
        """Extract <think> blocks from stored completions and compute per-origin token counts."""
        self.last_reasoning_trace = ""
        self.last_reasoning_tokens_prf = 0
        self.last_reasoning_tokens_orig = 0

        # Concatenate all <think> blocks from completions
        think_blocks = []
        for completion in getattr(self._ranker, "all_completions", []):
            blocks = re.findall(r"<think>(.*?)</think>", completion, re.DOTALL)
            think_blocks.extend(blocks)
        self.last_reasoning_trace = "\n".join(think_blocks)

        # Build origin lookup
        origin_map = {d["docid"]: d.get("origin_label", "") for d in doc_list}

        # Estimate per-origin token counts proportionally
        total = self.total_completion_tokens
        n_prf = sum(1 for d in doc_list if origin_map.get(d["docid"]) == "B-Only")
        n_orig = sum(1 for d in doc_list if origin_map.get(d["docid"]) == "A-Only")
        n_total = n_prf + n_orig
        if n_total > 0:
            self.last_reasoning_tokens_prf = int(total * n_prf / n_total)
            self.last_reasoning_tokens_orig = int(total * n_orig / n_total)

    def rerank_return_ids(self, query: str, doc_list: list) -> list:
        """Rerank and return just the ordered list of doc IDs."""
        pairs = self.rerank(query, doc_list)
        return [doc_id for doc_id, _ in pairs]
