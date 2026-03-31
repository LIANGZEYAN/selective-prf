"""
Qwen 72B university API pairwise reranker.

Same rerank() interface as pairwise.py but calls the university
OpenAI-compatible API instead of running locally.
"""

import os
import re
import copy
import math
import time
import concurrent.futures

from openai import OpenAI, RateLimitError, APIError, APIConnectionError

from src.config import (
    QWEN_72B_API_URL,
    QWEN_72B_API_MODEL,
    QWEN_72B_API_KEY_ENV,
    SYSTEM_PROMPT,
    PAIRWISE_PROMPT,
    PASSAGE_LENGTH,
    QUERY_LENGTH,
    TOP_K_CANDIDATES,
)


class PairwiseApiReranker:
    """Qwen 72B API pairwise reranker using heapsort.

    Model calls go to the university API endpoint.
    Implements retry logic with exponential backoff (3 retries, 2s base).
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = QWEN_72B_API_URL,
        model: str = QWEN_72B_API_MODEL,
        k: int = TOP_K_CANDIDATES,
        max_retries: int = 3,
        temperature: float = 0.0,
    ):
        if api_key is None:
            api_key = os.environ.get(QWEN_72B_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"API key not set. Set the {QWEN_72B_API_KEY_ENV} environment "
                f"variable or pass api_key to the constructor."
            )

        self.model = model
        self.k = k
        self.max_retries = max_retries
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _call_api(self, prompt: str) -> tuple:
        """Single API call with retry logic."""
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=5,
                )
                content = resp.choices[0].message.content.strip()
                p_tok = resp.usage.prompt_tokens if resp.usage else 0
                c_tok = resp.usage.completion_tokens if resp.usage else 0

                matches = re.findall(r"Passage\s+([AB])", content, re.IGNORECASE)
                if matches:
                    return f"Passage {matches[0].upper()}", p_tok, c_tok
                if content.upper().startswith("A"):
                    return "Passage A", p_tok, c_tok
                if content.upper().startswith("B"):
                    return "Passage B", p_tok, c_tok

                return "Passage A", p_tok, c_tok

            except RateLimitError:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            except (APIError, APIConnectionError) as e:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)

        raise RuntimeError(f"API call failed after {self.max_retries} retries.")

    def _compare(self, query: str, doc1_text: str, doc2_text: str) -> list:
        """Compare two documents, firing forward and reverse prompts in parallel."""
        self.total_compare += 1
        fwd_prompt = PAIRWISE_PROMPT.format(query=query, doc1=doc1_text, doc2=doc2_text)
        rev_prompt = PAIRWISE_PROMPT.format(query=query, doc1=doc2_text, doc2=doc1_text)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fwd_fut = ex.submit(self._call_api, fwd_prompt)
            rev_fut = ex.submit(self._call_api, rev_prompt)
            fwd_ans, fp, fc = fwd_fut.result()
            rev_ans, rp, rc = rev_fut.result()

        self.total_prompt_tokens += fp + rp
        self.total_completion_tokens += fc + rc

        return [fwd_ans, rev_ans]

    def truncate_passage(self, text: str) -> str:
        char_limit = PASSAGE_LENGTH * 4
        return text[:char_limit] if len(text) > char_limit else text

    def truncate_query(self, text: str) -> str:
        char_limit = QUERY_LENGTH * 4
        return text[:char_limit] if len(text) > char_limit else text

    def _heapify(self, arr, n, i, query):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n:
            out = self._compare(query, arr[l]["text"], arr[i]["text"])
            if out[0] == "Passage A" and out[1] == "Passage B":
                largest = l
        if r < n:
            out = self._compare(query, arr[r]["text"], arr[largest]["text"])
            if out[0] == "Passage A" and out[1] == "Passage B":
                largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self._heapify(arr, n, largest, query)

    def _heap_sort(self, arr, query, k):
        n = len(arr)
        ranked = 0
        for i in range(n // 2, -1, -1):
            self._heapify(arr, n, i, query)
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            self._heapify(arr, i, 0, query)

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
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        arr = [{"docid": d["docid"], "text": d["text"]} for d in doc_list]
        original_order = [d["docid"] for d in doc_list]

        self._heap_sort(arr, query, self.k)

        # Build result: top-k from sorted portion, then remaining in original order
        sorted_ids = list(reversed([d["docid"] for d in arr]))
        top_ids = set(sorted_ids[: self.k])

        results = []
        rank = 1
        for doc_id in sorted_ids[: self.k]:
            results.append((doc_id, rank))
            rank += 1
        for doc_id in original_order:
            if doc_id not in top_ids:
                results.append((doc_id, rank))
                rank += 1

        return results

    def rerank_return_ids(self, query: str, doc_list: list) -> list:
        """Rerank and return just the ordered list of doc IDs."""
        pairs = self.rerank(query, doc_list)
        return [doc_id for doc_id, _ in pairs]
