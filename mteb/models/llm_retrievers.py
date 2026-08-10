"""LLM retriever wrappers: query rewrite, HyDE, rerank, and multi-hop search.

Each wraps a base retriever (BM25, dense, late-interaction) and implements
SearchProtocol itself, so the wrappers compose with each other and run on
ordinary retrieval tasks. The multi-hop agent and tournament reranker follow
the OBLIQ-Bench setups (arXiv 2605.06235, section 5 and appendix C).
"""

from __future__ import annotations

import json
import logging
import random
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
_HOP_QUERY = (
    "You are searching a corpus for documents relevant to a question. Given "
    "the question and notes from previous searches, reply with only the "
    "single most useful next search query.\n\nQuestion: {q}\n\nNotes:\n{notes}"
)
_HOP_READ = (
    "Identify which of the retrieved documents below are relevant to the "
    "question. Reply with a JSON array of the relevant document ids (may be "
    "empty), then on a new line one short observation to guide the next "
    "search.\n\nQuestion: {q}\n\nDocuments:\n{docs}"
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


def _parse_ids(text: str) -> list[str] | None:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return [str(x) for x in value] if isinstance(value, list) else None


def _to_scores(ordered: list[str], top_k: int) -> dict[str, float]:
    # Descending scores preserve the order for downstream top-k.
    ordered = ordered[:top_k]
    return {d: float(len(ordered) - i) for i, d in enumerate(ordered)}


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


class _TextCachingRetriever:
    """Base for retrievers whose LLM reads document text at search time."""

    def __init__(
        self,
        base: SearchProtocol,
        model: ChatModelProtocol,
        kind: str,
        snippet_chars: int,
    ) -> None:
        self.base = base
        self.model = model
        self.snippet_chars = snippet_chars
        self.mteb_model_meta = _wrapper_meta(kind, base)
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
        """Index the base retriever and cache document text."""
        self._text = {row["id"]: row.get("text", "") for row in corpus}
        self.base.index(
            corpus,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
        )

    def _docs_block(self, doc_ids: list[str]) -> str:
        return "\n".join(
            f"[{d}] {self._text.get(d, '')[: self.snippet_chars]}" for d in doc_ids
        )

    def _listwise(self, qid: str, query: str, candidates: list[str]) -> list[str]:
        """LLM listwise ordering of candidates; unranked ids keep their order."""
        out = self.model.generate(
            [
                {
                    "role": "user",
                    "content": _RERANK.format(
                        q=query, docs=self._docs_block(candidates)
                    ),
                }
            ]
        )
        ranked = [d for d in (_parse_ids(out.text) or []) if d in set(candidates)]
        if not ranked:
            logger.warning(
                "listwise rerank parse failed for query %s; keeping base order", qid
            )
        return ranked + [d for d in candidates if d not in set(ranked)]


class RerankRetriever(_TextCachingRetriever):
    """Base retrieves a candidate pool; the LLM listwise reranks it to top_k."""

    def __init__(
        self,
        base: SearchProtocol,
        model: ChatModelProtocol,
        *,
        pool_size: int = 20,
        snippet_chars: int = 500,
    ) -> None:
        super().__init__(base, model, "llm-rerank", snippet_chars)
        self.pool_size = pool_size

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
            ordered = self._listwise(qid, query_text[qid], candidates)
            results[qid] = _to_scores(ordered, top_k)
        return results


class TournamentRerankRetriever(_TextCachingRetriever):
    """Tournament listwise rerank over a large pool (OBLIQ-Bench appendix C).

    The pool is shuffled (deterministically per query id) and partitioned into
    batches of batch_size; one listwise call ranks each batch, the top
    promote_k advance, and the rest form that depth's tail. Rounds repeat
    until one batch remains, which is ranked directly. The final ranking is
    the survivors followed by the tails in reverse order of elimination.
    OBLIQ-Bench uses batch_size=20, promote_k=4.
    """

    def __init__(
        self,
        base: SearchProtocol,
        model: ChatModelProtocol,
        *,
        pool_size: int = 100,
        batch_size: int = 20,
        promote_k: int = 4,
        snippet_chars: int = 500,
    ) -> None:
        if promote_k >= batch_size:
            raise ValueError("promote_k must be smaller than batch_size.")
        super().__init__(base, model, "tournament-rerank", snippet_chars)
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.promote_k = promote_k

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
        """Retrieve a pool, then rank it with a listwise tournament."""
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
            random.Random(qid).shuffle(candidates)
            tails: list[list[str]] = []
            while len(candidates) > self.batch_size:
                survivors: list[str] = []
                tail: list[str] = []
                for i in range(0, len(candidates), self.batch_size):
                    ranked = self._listwise(
                        qid, query_text[qid], candidates[i : i + self.batch_size]
                    )
                    survivors += ranked[: self.promote_k]
                    tail += ranked[self.promote_k :]
                tails.append(tail)
                candidates = survivors
            ordered = self._listwise(qid, query_text[qid], candidates)
            for tail in reversed(tails):
                ordered += tail
            results[qid] = _to_scores(ordered, top_k)
        return results


def _note_after_ids(text: str) -> str:
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    note = text[match.end() :].strip() if match else text.strip()
    return note[:300]


class MultiHopRetriever(_TextCachingRetriever):
    """Iterative search agent producing a ranking (OBLIQ-Bench multi-hop setup).

    Each hop the LLM writes a search query from the question and accumulated
    notes, the base retrieves per_hop candidates, and the LLM reads the batch,
    selecting relevant ids and noting an observation for the next hop.
    Selected documents are promoted to the top in selection order; the
    retrieved-but-unselected ones fill the tail by base score. OBLIQ-Bench
    uses hops=4, per_hop=25.
    """

    def __init__(
        self,
        base: SearchProtocol,
        model: ChatModelProtocol,
        *,
        hops: int = 4,
        per_hop: int = 25,
        snippet_chars: int = 500,
    ) -> None:
        super().__init__(base, model, "multi-hop", snippet_chars)
        self.hops = hops
        self.per_hop = per_hop

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
        """Run the hop loop per query and rank selected docs above the pool."""
        from datasets import Dataset

        results: dict[str, dict[str, float]] = {}
        for row in queries:
            qid, question = row["id"], row["text"]
            selected: list[str] = []
            pool: dict[str, float] = {}
            notes: list[str] = []
            for _ in range(self.hops):
                out = self.model.generate(
                    [
                        {
                            "role": "user",
                            "content": _HOP_QUERY.format(
                                q=question, notes="\n".join(notes) or "(none)"
                            ),
                        }
                    ]
                )
                hop_query = out.text.strip() or question
                ranking = self.base.search(
                    Dataset.from_list([{"id": qid, "text": hop_query}]),
                    task_metadata=task_metadata,
                    hf_split=hf_split,
                    hf_subset=hf_subset,
                    top_k=self.per_hop,
                    encode_kwargs=encode_kwargs,
                    top_ranked=top_ranked,
                    num_proc=num_proc,
                )[qid]
                for d, score in ranking.items():
                    pool[d] = max(pool.get(d, score), score)
                batch = [
                    d
                    for d in sorted(ranking, key=lambda d: -ranking[d])
                    if d not in selected
                ]
                read = self.model.generate(
                    [
                        {
                            "role": "user",
                            "content": _HOP_READ.format(
                                q=question, docs=self._docs_block(batch)
                            ),
                        }
                    ]
                )
                selected += [
                    d
                    for d in (_parse_ids(read.text) or [])
                    if d in ranking and d not in selected
                ]
                note = _note_after_ids(read.text)
                if note:
                    notes.append(note)
            tail = [
                d for d in sorted(pool, key=lambda d: -pool[d]) if d not in selected
            ]
            results[qid] = _to_scores(selected + tail, top_k)
        return results
