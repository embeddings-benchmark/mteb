"""LLM retriever wrappers: query rewrite, HyDE, and listwise rerank.

Each wraps a base retriever (BM25, dense, late-interaction) and implements
SearchProtocol itself, so the wrappers compose with each other and run on
ordinary retrieval tasks.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from mteb.models.model_meta import ModelMeta

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.chat_models import ChatModelProtocol
    from mteb.models.models_protocols import SearchProtocol
    from mteb.types import (
        CorpusDatasetType,
        EncodeKwargs,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )

logger = logging.getLogger(__name__)

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


def _wrapper_meta(kind: str, base: SearchProtocol) -> ModelMeta:
    # Composed-retriever identity, mirroring HybridSearch naming.
    base_meta = getattr(base, "mteb_model_meta", None)
    base_name = (getattr(base_meta, "name", None) or "unknown").rsplit("/", 1)[-1]
    return ModelMeta.create_empty(
        overwrites={
            "name": f"mteb/baseline-{kind}-{base_name}",
            "model_type": ["hybrid"],
        }
    )


def _transform_queries(
    queries: QueryDatasetType, model: ChatModelProtocol, template: str
) -> QueryDatasetType:
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
    _kind: str

    def __init__(self, base: SearchProtocol, model: ChatModelProtocol) -> None:
        self.base = base
        self.model = model
        self.mteb_model_meta = _wrapper_meta(self._kind, base)

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None,
    ) -> None:
        """Index the base retriever."""
        self.base.index(
            corpus,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
        )

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None = None,
        num_proc: int | None,
    ) -> RetrievalOutputType:
        """Transform each query with the LLM, then search the base retriever."""
        return self.base.search(
            _transform_queries(queries, self.model, self._template),
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            top_k=top_k,
            encode_kwargs=encode_kwargs,
            top_ranked=top_ranked,
            num_proc=num_proc,
        )


class QueryRewriteRetriever(_QueryTransformRetriever):
    """LLM rewrites each query into a keyword query, then the base searches."""

    _template = _REWRITE
    _kind = "query-rewrite"


class HyDERetriever(_QueryTransformRetriever):
    """LLM writes a hypothetical answer passage; the base retrieves with it."""

    _template = _HYDE
    _kind = "hyde"


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
        self,
        base: SearchProtocol,
        model: ChatModelProtocol,
        *,
        pool_size: int = 20,
        snippet_chars: int = 500,
    ) -> None:
        self.base = base
        self.model = model
        self.pool_size = pool_size
        self.snippet_chars = snippet_chars
        self.mteb_model_meta = _wrapper_meta("llm-rerank", base)
        self._text: dict[str, str] = {}

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None,
    ) -> None:
        """Index the base retriever and cache document text for reranking."""
        self._text = {row["id"]: row.get("text", "") for row in corpus}
        self.base.index(
            corpus,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
        )

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None = None,
        num_proc: int | None,
    ) -> RetrievalOutputType:
        """Retrieve a candidate pool, then LLM listwise rerank it to top_k."""
        query_text = {row["id"]: row["text"] for row in queries}
        pool = self.base.search(
            queries,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            top_k=self.pool_size,
            encode_kwargs=encode_kwargs,
            top_ranked=top_ranked,
            num_proc=num_proc,
        )
        results: dict[str, dict[str, float]] = {}
        for qid, ranking in pool.items():
            candidates = sorted(ranking, key=lambda d: -ranking[d])
            docs = "\n".join(
                f"[{d}] {self._text.get(d, '')[: self.snippet_chars]}"
                for d in candidates
            )
            out = self.model.generate(
                [
                    {
                        "role": "user",
                        "content": _RERANK.format(q=query_text[qid], docs=docs),
                    }
                ]
            )
            ranked = [d for d in (_parse_ids(out.text) or []) if d in ranking]
            if not ranked:
                logger.warning(
                    "rerank parse failed for query %s; keeping base order", qid
                )
            ranked = (ranked or candidates)[:top_k]
            # Descending scores preserve the reranked order for downstream top-k.
            results[qid] = {d: float(len(ranked) - i) for i, d in enumerate(ranked)}
        return results
