"""
Qwen 72B university API setwise reranker.

Same rerank() interface as setwise.py but calls the university
OpenAI-compatible API instead of running vLLM locally.
Uses the same prompt template and heapsort logic as RankR1SetwiseLlmRanker.
"""

import os
import re
import time
import random
from collections import Counter
from dataclasses import dataclass

from openai import OpenAI, RateLimitError, APIError, APIConnectionError

from src.config import (
    QWEN_72B_API_URL,
    QWEN_72B_API_MODEL,
    QWEN_72B_API_KEY_ENV,
    PASSAGE_LENGTH,
    QUERY_LENGTH,
    TOP_K_CANDIDATES,
)

random.seed(929)


@dataclass
class _SearchResult:
    docid: str
    score: float
    text: str


# Same prompt as prompt_setwise.toml (base models, no <think>)
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant provides the user with the "
    "answer enclosed within <answer> </answer> tags, i.e., "
    "<answer> answer here </answer>."
)

USER_PROMPT = (
    'Given the query: "{query}", which of the following documents is most relevant?\n'
    '{docs}\n'
    'Please provide only the label of the most relevant document to the query, '
    'enclosed in square brackets, within the answer tags. For example, if the '
    'third document is the most relevant, the answer should be: '
    '<answer>[3]</answer>.'
)

ANSWER_PATTERN = r'<answer>(.*?)</answer>'
CHARACTERS = [f'[{i+1}]' for i in range(20)]


class SetwiseApiReranker:
    """Qwen 72B API setwise reranker using heapsort.

    Model calls go to the university API endpoint.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = QWEN_72B_API_URL,
        model: str = QWEN_72B_API_MODEL,
        k: int = TOP_K_CANDIDATES,
        num_child: int = None,
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
        self.num_child = num_child if num_child is not None else max(k - 1, 2)
        self.max_retries = max_retries
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _call_api(self, messages: list) -> tuple:
        """Single API call with retry logic. Returns (content, prompt_tokens, completion_tokens)."""
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=100,
                )
                content = resp.choices[0].message.content.strip()
                p_tok = resp.usage.prompt_tokens if resp.usage else 0
                c_tok = resp.usage.completion_tokens if resp.usage else 0
                return content, p_tok, c_tok

            except RateLimitError:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            except (APIError, APIConnectionError) as e:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)

        raise RuntimeError(f"API call failed after {self.max_retries} retries.")

    def compare(self, query: str, docs: list) -> str:
        """Compare documents in a single setwise call. Returns winning label like '[1]'."""
        self.total_compare += 1

        # Build document list with labels [1], [2], ...
        passages = "\n".join(
            f'{CHARACTERS[i]} {doc.text}' for i, doc in enumerate(docs)
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(query=query, docs=passages)},
        ]

        content, p_tok, c_tok = self._call_api(messages)
        self.total_prompt_tokens += p_tok
        self.total_completion_tokens += c_tok

        # Parse answer
        match = re.search(ANSWER_PATTERN, content.lower(), re.DOTALL)
        if match:
            result = match.group(1).strip()
        else:
            # Fallback: look for [N] pattern
            bracket_match = re.search(r'\[(\d+)\]', content)
            if bracket_match:
                result = f'[{bracket_match.group(1)}]'
            else:
                result = CHARACTERS[0]  # default to first

        if result in CHARACTERS[:len(docs)]:
            return result
        else:
            return CHARACTERS[0]

    def heapify(self, arr, n, i, query):
        """Heapify subtree rooted at index i."""
        if self.num_child * i + 1 < n:
            docs = [arr[i]] + arr[self.num_child * i + 1: min((self.num_child * (i + 1) + 1), n)]
            inds = [i] + list(range(self.num_child * i + 1, min((self.num_child * (i + 1) + 1), n)))
            output = self.compare(query, docs)
            try:
                best_ind = CHARACTERS.index(output)
            except ValueError:
                best_ind = 0
            try:
                largest = inds[best_ind]
            except IndexError:
                largest = i
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.heapify(arr, n, largest, query)

    def heap_sort(self, arr, query, k):
        """Heapsort to find top-k."""
        n = len(arr)
        ranked = 0
        for i in range(n // self.num_child, -1, -1):
            self.heapify(arr, n, i, query)
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            self.heapify(arr, i, 0, query)

    def truncate_passage(self, text: str) -> str:
        char_limit = PASSAGE_LENGTH * 4
        return text[:char_limit] if len(text) > char_limit else text

    def truncate_query(self, text: str) -> str:
        char_limit = QUERY_LENGTH * 4
        return text[:char_limit] if len(text) > char_limit else text

    def rerank(self, query: str, doc_list: list) -> list:
        """Rerank documents via setwise comparisons.

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

        ranking = [
            _SearchResult(docid=d["docid"], score=-i, text=d["text"])
            for i, d in enumerate(doc_list)
        ]

        self.heap_sort(ranking, query, self.k)
        ranking = list(reversed(ranking))

        return [
            (r.docid, idx + 1)
            for idx, r in enumerate(ranking)
        ]

    def rerank_return_ids(self, query: str, doc_list: list) -> list:
        """Rerank and return just the ordered list of doc IDs."""
        pairs = self.rerank(query, doc_list)
        return [doc_id for doc_id, _ in pairs]
