"""
Qwen 7B local GPU pairwise reranker.

Wraps the PairwiseLlmRanker from llm-rankers for local inference.
Shares the same rerank() interface as pairwise_api.py.
"""

import sys
from pathlib import Path

from src.config import (
    LLM_RANKERS_PATH,
    QWEN_7B_MODEL,
    QWEN_7B_CACHE_DIR,
    PASSAGE_LENGTH,
    QUERY_LENGTH,
    TOP_K_CANDIDATES,
)

# Add llm-rankers to path so we can import it
sys.path.insert(0, str(LLM_RANKERS_PATH))
from llmrankers.rankers import SearchResult
from llmrankers.pairwise import PairwiseLlmRanker


class PairwiseReranker:
    """Qwen 7B local pairwise reranker using heapsort.

    Model is loaded once on __init__, not on every call.
    """

    def __init__(
        self,
        model_name_or_path: str = QWEN_7B_MODEL,
        cache_dir: str = QWEN_7B_CACHE_DIR,
        device: str = "cuda",
        k: int = TOP_K_CANDIDATES,
        temperature: float = 0.0,
    ):
        self.k = k
        self._ranker = PairwiseLlmRanker(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=model_name_or_path,
            device=device,
            method="heapsort",
            batch_size=2,
            k=k,
            cache_dir=cache_dir,
            temperature=temperature,
        )

    def truncate_passage(self, text: str) -> str:
        return self._ranker.truncate(text, PASSAGE_LENGTH)

    def truncate_query(self, text: str) -> str:
        return self._ranker.truncate(text, QUERY_LENGTH)

    def rerank(self, query: str, doc_list: list) -> list:
        """Rerank documents via pairwise comparisons.

        Parameters
        ----------
        query : query text string.
        doc_list : list of dicts with keys 'docid' and 'text'.

        Returns
        -------
        list of (doc_id, rank) tuples, 1-based rank.
        """
        ranking = [
            SearchResult(docid=d["docid"], score=-i, text=d["text"])
            for i, d in enumerate(doc_list)
        ]

        reranked = self._ranker.rerank(query, ranking)

        return [
            (r.docid, idx + 1)
            for idx, r in enumerate(reranked)
        ]

    def rerank_return_ids(self, query: str, doc_list: list) -> list:
        """Rerank and return just the ordered list of doc IDs."""
        pairs = self.rerank(query, doc_list)
        return [doc_id for doc_id, _ in pairs]

    @property
    def total_compare(self) -> int:
        return self._ranker.total_compare

    @property
    def total_prompt_tokens(self) -> int:
        return self._ranker.total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self._ranker.total_completion_tokens
