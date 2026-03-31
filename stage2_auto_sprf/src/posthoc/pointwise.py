"""
Qwen3-Reranker pointwise reranker.

Scores each query-document pair independently using yes/no logits.
Documents are then ranked by their relevance probability and the
standard DCG/MRR/Majority@k aggregations are applied.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    QWEN3_RERANKER_8B_MODEL,
    RANK_R1_CACHE_DIR,
    PASSAGE_LENGTH,
    QUERY_LENGTH,
)


# Template following the official Qwen3-Reranker format
_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the "
    "Query and the Instruct provided. Note that the answer can only "
    'be "yes" or "no".'
)

_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class PointwiseReranker:
    """Qwen3-Reranker pointwise scorer.

    Scores each document independently via P(yes) / (P(yes) + P(no)).
    Returns a ranked list compatible with the DCG/MRR/Majority@k pipeline.
    """

    def __init__(
        self,
        model_name_or_path: str = QWEN3_RERANKER_8B_MODEL,
        cache_dir: str = RANK_R1_CACHE_DIR,
        device: str = "cuda",
        instruction: str = _DEFAULT_INSTRUCTION,
        max_length: int = 8192,
    ):
        self.device = device
        self.instruction = instruction
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        ).to(device).eval()

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        self._prefix = (
            f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
        )
        self._suffix = (
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

        self.total_compare = 0
        self.total_prompt_tokens = 0

    def _format_input(self, query: str, doc_text: str) -> str:
        content = (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc_text}"
        )
        return self._prefix + content + self._suffix

    def truncate_passage(self, text: str) -> str:
        tokens = self.tokenizer.tokenize(text)[:PASSAGE_LENGTH]
        return self.tokenizer.convert_tokens_to_string(tokens)

    def truncate_query(self, text: str) -> str:
        tokens = self.tokenizer.tokenize(text)[:QUERY_LENGTH]
        return self.tokenizer.convert_tokens_to_string(tokens)

    @torch.no_grad()
    def score_pair(self, query: str, doc_text: str) -> float:
        """Score a single query-document pair. Returns P(yes) in [0, 1]."""
        formatted = self._format_input(query, doc_text)
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        self.total_prompt_tokens += inputs["input_ids"].shape[1]
        self.total_compare += 1

        logits = self.model(**inputs).logits[:, -1, :]
        true_logit = logits[:, self.token_true_id]
        false_logit = logits[:, self.token_false_id]
        scores = torch.stack([false_logit, true_logit], dim=1)
        prob_yes = torch.nn.functional.log_softmax(scores, dim=1)[:, 1].exp()
        return prob_yes.item()

    def score_batch(self, query: str, doc_texts: list) -> list:
        """Score multiple documents for the same query. Returns list of P(yes)."""
        formatted = [self._format_input(query, doc) for doc in doc_texts]
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        self.total_prompt_tokens += inputs["input_ids"].numel()
        self.total_compare += len(doc_texts)

        logits = self.model(**inputs).logits[:, -1, :]
        true_logits = logits[:, self.token_true_id]
        false_logits = logits[:, self.token_false_id]
        scores = torch.stack([false_logits, true_logits], dim=1)
        probs = torch.nn.functional.log_softmax(scores, dim=1)[:, 1].exp()
        return probs.tolist()

    def rerank(self, query: str, doc_list: list) -> list:
        """Rerank documents by pointwise relevance score.

        Parameters
        ----------
        query : query text string.
        doc_list : list of dicts with keys 'docid' and 'text'.

        Returns
        -------
        list of (doc_id, rank) tuples, 1-based rank (highest score = rank 1).
        """
        doc_texts = [d["text"] for d in doc_list]
        scores = self.score_batch(query, doc_texts)

        # Pair doc IDs with scores, sort descending
        scored = [(doc_list[i]["docid"], scores[i]) for i in range(len(doc_list))]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [(doc_id, rank + 1) for rank, (doc_id, _) in enumerate(scored)]

    def rerank_return_ids(self, query: str, doc_list: list) -> list:
        """Rerank and return just the ordered list of doc IDs."""
        pairs = self.rerank(query, doc_list)
        return [doc_id for doc_id, _ in pairs]
