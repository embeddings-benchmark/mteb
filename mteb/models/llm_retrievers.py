"""LLM retriever wrappers: query rewrite, HyDE, rerank, and multi-hop search.

Each wraps a base retriever (BM25, dense, late-interaction) and implements
SearchProtocol itself, so the wrappers compose with each other and run on
ordinary retrieval tasks.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import TYPE_CHECKING

from datasets import Dataset

from mteb.models.hybrid_wrappers import fuse_rrf
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

# Rewriter prompt from Ma et al. (arXiv:2305.14283), table 1.
_REWRITE = (
    "Provide a better search query for a web search engine to answer the "
    "given question. Reply with only the query.\n\nQuestion: {q}"
)
_HYDE = (
    "Write a short passage that plausibly answers the question. "
    "Reply with only the passage.\n\nQuestion: {q}"
)
_RERANK = (
    "Rank the documents by relevance to the question. Reply with a JSON array "
    "of document ids, best first.\n\nQuestion: {q}\n\nDocuments:\n{docs}"
)
_MULTI_QUERY = (
    "Write {n} diverse search queries for the question, one per line. "
    "Reply with only the queries.\n\nQuestion: {q}"
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
    base_meta = getattr(base, "mteb_model_meta", None)
    base_name = (getattr(base_meta, "name", None) or "unknown").rsplit("/", 1)[-1]
    return ModelMeta.create_empty(
        overwrites={
            "name": f"{kind}-{base_name}",
            "model_type": ["hybrid"],
        }
    )


def _schema(
    name: str, properties: dict[str, object], required: list[str]
) -> dict[str, object]:
    """A response_format asking the provider for structured JSON."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


_IDS = {"type": "array", "items": {"type": "string"}}
_RANKING_SCHEMA = _schema("ranking", {"doc_ids": _IDS}, ["doc_ids"])
_HOP_SCHEMA = _schema(
    "hop_read", {"doc_ids": _IDS, "note": {"type": "string"}}, ["doc_ids", "note"]
)


def _parse_reply(text: str) -> dict[str, object] | None:
    """Parse a JSON reply, accepting a bare id array from unconstrained models."""
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(value, list):
        return {"doc_ids": value}
    return value if isinstance(value, dict) else None


def _parse_ids(text: str) -> list[str] | None:
    """Document ids from a structured reply or a bare JSON array."""
    reply = _parse_reply(text)
    ids = reply.get("doc_ids") if reply else None
    return [str(x) for x in ids] if isinstance(ids, list) else None


def _to_scores(ordered: list[str], top_k: int) -> dict[str, float]:
    # Descending scores preserve the order for downstream top-k.
    ordered = ordered[:top_k]
    return {d: float(len(ordered) - i) for i, d in enumerate(ordered)}


def _transform_queries(
    queries: QueryDatasetType, model: ChatModelProtocol, template: str
) -> QueryDatasetType:
    rows = []
    for row in queries:
        out = model.generate(
            [{"role": "user", "content": template.format(q=row["text"])}]
        )
        rows.append({"id": row["id"], "text": out.text.strip() or row["text"]})
    return Dataset.from_list(rows)


class _QueryTransformRetriever:
    """Base for retrievers that LLM-transform the query text before searching."""

    def __init__(
        self,
        base: SearchProtocol,
        model: ChatModelProtocol,
        kind: str,
        prompt: str,
    ) -> None:
        self.base = base
        self.model = model
        self.prompt = prompt
        self.mteb_model_meta = _wrapper_meta(kind, base)

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
            _transform_queries(queries, self.model, self.prompt),
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            top_k=top_k,
            encode_kwargs=encode_kwargs,
            top_ranked=top_ranked,
            num_proc=num_proc,
        )


class QueryRewriteRetriever(_QueryTransformRetriever):
    """LLM rewrites each query into a keyword query, then the base searches.

    Reference: Ma et al., Query Rewriting for Retrieval-Augmented LLMs (arXiv:2305.14283).
    """

    def __init__(
        self,
        base: SearchProtocol,
        model: ChatModelProtocol,
        *,
        prompt: str = _REWRITE,
    ) -> None:
        super().__init__(base, model, "query-rewrite", prompt)


class HyDERetriever(_QueryTransformRetriever):
    """LLM writes a hypothetical answer passage; the base retrieves with it.

    Reference: Gao et al., Precise Zero-Shot Dense Retrieval without Relevance Labels (arXiv:2212.10496).
    """

    def __init__(
        self,
        base: SearchProtocol,
        model: ChatModelProtocol,
        *,
        prompt: str = _HYDE,
    ) -> None:
        super().__init__(base, model, "hyde", prompt)


def _docs_block(
    corpus: CorpusDatasetType | None,
    doc_id_to_idx: dict[str, int],
    doc_ids: list[str],
    snippet_chars: int,
) -> tuple[str, list[str]]:
    """Render the given documents as an id-prefixed block, with the ids kept."""
    if corpus is None:
        raise ValueError("Corpus must be indexed before searching.")
    known = [d for d in doc_ids if d in doc_id_to_idx]
    rows = corpus.select([doc_id_to_idx[d] for d in known])
    block = "\n".join(
        f"[{d}] {row.get('text', '')[:snippet_chars]}" for d, row in zip(known, rows)
    )
    return block, known


def _listwise_rank(
    model: ChatModelProtocol,
    prompt: str,
    qid: str,
    query: str,
    candidates: list[str],
    docs: str,
) -> list[str]:
    """LLM listwise ordering of candidates; unranked ids keep their order."""
    out = model.generate(
        [{"role": "user", "content": prompt.format(q=query, docs=docs)}],
        response_format=_RANKING_SCHEMA,
    )
    ranked = [d for d in (_parse_ids(out.text) or []) if d in set(candidates)]
    if not ranked:
        logger.warning(
            "listwise rerank parse failed for query %s; keeping base order", qid
        )
    return ranked + [d for d in candidates if d not in set(ranked)]


class RerankRetriever:
    """Base retrieves a candidate pool; the LLM listwise reranks it to top_k.

    Reference: Sun et al., Is ChatGPT Good at Search? (RankGPT, arXiv:2304.09542).
    """

    def __init__(
        self,
        base: SearchProtocol,
        model: ChatModelProtocol,
        *,
        pool_size: int = 20,
        snippet_chars: int = 500,
        prompt: str = _RERANK,
    ) -> None:
        self.base = base
        self.model = model
        self.prompt = prompt
        self.snippet_chars = snippet_chars
        self.mteb_model_meta = _wrapper_meta("llm-rerank", base)
        self.task_corpus: CorpusDatasetType | None = None
        self._doc_id_to_idx: dict[str, int] = {}

        self.pool_size = pool_size

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
        """Index the base retriever and keep the corpus for document lookups."""
        self.task_corpus = corpus
        self._doc_id_to_idx = {doc: idx for idx, doc in enumerate(corpus["id"])}
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
            docs, _ = _docs_block(
                self.task_corpus, self._doc_id_to_idx, candidates, self.snippet_chars
            )
            ordered = _listwise_rank(
                self.model, self.prompt, qid, query_text[qid], candidates, docs
            )
            results[qid] = _to_scores(ordered, top_k)
        return results


class TournamentRerankRetriever:
    """Tournament listwise rerank over a large candidate pool.

    Reference: OBLIQ-Bench (arXiv:2605.06235), appendix C.

    The pool is shuffled (deterministically per query id) and partitioned into
    batches of batch_size; one listwise call ranks each batch, the top
    promote_k advance, and the rest form that depth's tail. Rounds repeat
    until one batch remains, which is ranked directly. The final ranking is
    the survivors followed by the tails in reverse order of elimination.
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
        prompt: str = _RERANK,
    ) -> None:
        if promote_k >= batch_size:
            raise ValueError("promote_k must be smaller than batch_size.")
        self.base = base
        self.model = model
        self.prompt = prompt
        self.snippet_chars = snippet_chars
        self.mteb_model_meta = _wrapper_meta("tournament-rerank", base)
        self.task_corpus: CorpusDatasetType | None = None
        self._doc_id_to_idx: dict[str, int] = {}

        self.pool_size = pool_size
        self.batch_size = batch_size
        self.promote_k = promote_k

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
        """Index the base retriever and keep the corpus for document lookups."""
        self.task_corpus = corpus
        self._doc_id_to_idx = {doc: idx for idx, doc in enumerate(corpus["id"])}
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
                    batch = candidates[i : i + self.batch_size]
                    docs, _ = _docs_block(
                        self.task_corpus, self._doc_id_to_idx, batch, self.snippet_chars
                    )
                    ranked = _listwise_rank(
                        self.model, self.prompt, qid, query_text[qid], batch, docs
                    )
                    survivors += ranked[: self.promote_k]
                    tail += ranked[self.promote_k :]
                tails.append(tail)
                candidates = survivors
            docs, _ = _docs_block(
                self.task_corpus, self._doc_id_to_idx, candidates, self.snippet_chars
            )
            ordered = _listwise_rank(
                self.model, self.prompt, qid, query_text[qid], candidates, docs
            )
            for tail in reversed(tails):
                ordered += tail
            results[qid] = _to_scores(ordered, top_k)
        return results


def _parse_note(text: str) -> str:
    """The observation from a hop reply, or the text after the id array."""
    reply = _parse_reply(text)
    note = reply.get("note") if reply else None
    if not isinstance(note, str):
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        note = text[match.end() :] if match else text
    return note.strip()[:300]


class MultiHopRetriever:
    """Iterative multi-hop search agent producing a ranking.

    Reference: OBLIQ-Bench (arXiv:2605.06235), section 5.

    Each hop the LLM writes a search query from the question and accumulated
    notes, the base retrieves per_hop candidates, and the LLM reads the batch,
    selecting relevant ids and noting an observation for the next hop.
    Selected documents are promoted to the top in selection order; the
    retrieved-but-unselected ones fill the tail by base score.
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
        self.base = base
        self.model = model
        self.prompt = _RERANK
        self.snippet_chars = snippet_chars
        self.mteb_model_meta = _wrapper_meta("multi-hop", base)
        self.task_corpus: CorpusDatasetType | None = None
        self._doc_id_to_idx: dict[str, int] = {}

        self.hops = hops
        self.per_hop = per_hop

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
        """Index the base retriever and keep the corpus for document lookups."""
        self.task_corpus = corpus
        self._doc_id_to_idx = {doc: idx for idx, doc in enumerate(corpus["id"])}
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
        """Run the hop loop per query and rank selected docs above the pool."""
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
                hop_docs, _ = _docs_block(
                    self.task_corpus, self._doc_id_to_idx, batch, self.snippet_chars
                )
                read = self.model.generate(
                    [
                        {
                            "role": "user",
                            "content": _HOP_READ.format(q=question, docs=hop_docs),
                        }
                    ],
                    response_format=_HOP_SCHEMA,
                )
                selected += [
                    d
                    for d in (_parse_ids(read.text) or [])
                    if d in ranking and d not in selected
                ]
                note = _parse_note(read.text)
                if note:
                    notes.append(note)
            tail = [
                d for d in sorted(pool, key=lambda d: -pool[d]) if d not in selected
            ]
            results[qid] = _to_scores(selected + tail, top_k)
        return results


class MultiQueryRetriever:
    """LLM writes query variants; base rankings fuse via reciprocal rank fusion.

    Reference: Cormack et al., Reciprocal Rank Fusion (SIGIR 2009) for the fusion;
    multi-query expansion follows common RAG practice.

    The original query and num_queries LLM variants are searched together,
    and the per-variant rankings merge with the same reciprocal rank fusion
    used by HybridSearch.
    """

    def __init__(
        self,
        base: SearchProtocol,
        model: ChatModelProtocol,
        *,
        num_queries: int = 3,
        rrf_k: int = 60,
    ) -> None:
        self.base = base
        self.model = model
        self.mteb_model_meta = _wrapper_meta("multi-query", base)
        self.num_queries = num_queries
        self.rrf_k = rrf_k

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
        """Search the original query plus LLM variants, then fuse the rankings."""
        results: dict[str, dict[str, float]] = {}
        for row in queries:
            qid, question = row["id"], row["text"]
            out = self.model.generate(
                [
                    {
                        "role": "user",
                        "content": _MULTI_QUERY.format(n=self.num_queries, q=question),
                    }
                ]
            )
            variants = [v.strip() for v in out.text.splitlines() if v.strip()]
            texts = [question, *variants[: self.num_queries]]
            rankings = self.base.search(
                Dataset.from_list(
                    [{"id": f"v{i}", "text": t} for i, t in enumerate(texts)]
                ),
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                top_k=top_k,
                encode_kwargs=encode_kwargs,
                top_ranked=top_ranked,
                num_proc=num_proc,
            )
            per_variant = [rankings[f"v{i}"] for i in range(len(texts))]
            fused = fuse_rrf(per_variant, [1.0] * len(per_variant), rrf_k=self.rrf_k)
            ordered = sorted(fused, key=lambda d: -fused[d])
            results[qid] = _to_scores(ordered, top_k)
        return results
