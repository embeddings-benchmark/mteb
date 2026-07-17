"""LLM-powered retriever wrappers.

Each wraps a base retriever (bm25, dense, late-interaction) with an LLM
transformation of the retrieval step: query rewrite, HyDE, or listwise rerank.
They implement the same index/search contract as any retriever, so they compose
into rag via retriever= and are scorable on ranking metrics like any retriever.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mteb.agentic.interface import ChatModel
    from mteb.models.models_protocols import SearchProtocol

_REWRITE = (
    "Rewrite the question into a concise keyword search query. "
    "Reply with only the query.\n\nQuestion: {q}"
)
_HYDE = (
    "Write a short passage that plausibly answers the question. "
    "Reply with only the passage.\n\nQuestion: {q}"
)
_RERANK = (
    "Rank the documents by relevance to the question. Reply with a JSON array "
    "of document ids, best first.\n\nQuestion: {q}\n\nDocuments:\n{docs}"
)


def _transform_queries(queries: Any, model: ChatModel, template: str) -> Any:
    """Rewrite each query's text with the LLM, keeping its id."""
    from datasets import Dataset

    rows = []
    for row in queries:
        out = model.generate(
            [{"role": "user", "content": template.format(q=row["text"])}]
        )
        rows.append({"id": row["id"], "text": out.text.strip() or row["text"]})
    return Dataset.from_list(rows)


class _QueryTransformRetriever:
    """Base for retrievers that LLM-transform the query text before searching."""

    _template: str

    def __init__(self, base: SearchProtocol, model: ChatModel) -> None:
        self.base = base
        self.model = model

    def index(self, **kwargs: Any) -> None:
        """Index the base retriever."""
        self.base.index(**kwargs)

    def search(self, *, queries: Any, **kwargs: Any) -> dict:
        """Transform each query with the LLM, then search the base retriever."""
        return self.base.search(
            queries=_transform_queries(queries, self.model, self._template), **kwargs
        )


class QueryRewriteRetriever(_QueryTransformRetriever):
    """LLM rewrites each query into a keyword query, then the base searches."""

    _template = _REWRITE


class HyDERetriever(_QueryTransformRetriever):
    """LLM writes a hypothetical answer passage; the base retrieves with it."""

    _template = _HYDE


def _parse_ids(text: str) -> list[str] | None:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return [str(x) for x in value] if isinstance(value, list) else None


class RerankRetriever:
    """Base retrieves a candidate pool; the LLM listwise reranks it to top_k."""

    def __init__(
        self, base: SearchProtocol, model: ChatModel, *, pool_size: int = 20
    ) -> None:
        self.base = base
        self.model = model
        self.pool_size = pool_size

    def index(self, *, corpus: Any, **kwargs: Any) -> None:
        """Index the base retriever and cache document text for reranking."""
        self._text = {row["id"]: row.get("text", "") for row in corpus}
        self.base.index(corpus=corpus, **kwargs)

    def search(self, *, queries: Any, top_k: int, **kwargs: Any) -> dict:
        """Retrieve a candidate pool, then LLM listwise rerank it to top_k."""
        query_text = {row["id"]: row["text"] for row in queries}
        pool = self.base.search(queries=queries, top_k=self.pool_size, **kwargs)
        results: dict[str, dict[str, float]] = {}
        for qid, ranking in pool.items():
            candidates = list(ranking)
            docs = "\n".join(f"[{d}] {self._text.get(d, '')[:500]}" for d in candidates)
            out = self.model.generate(
                [
                    {
                        "role": "user",
                        "content": _RERANK.format(q=query_text[qid], docs=docs),
                    }
                ]
            )
            ranked = [d for d in (_parse_ids(out.text) or []) if d in ranking]
            ranked = (ranked or candidates)[:top_k]
            # Descending scores preserve the reranked order for downstream top-k.
            results[qid] = {d: float(len(ranked) - i) for i, d in enumerate(ranked)}
        return results
