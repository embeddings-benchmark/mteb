from __future__ import annotations

import heapq
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mteb._create_dataloaders import (
    create_dataloader,
)
from mteb.types import (
    PromptType,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import (
        Array,
        BatchedInput,
        CorpusDatasetType,
        EncodeKwargs,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )

    from .models_protocols import CrossEncoderProtocol, EncoderProtocol
    from .search_encoder_index.search_backend_protocol import IndexEncoderSearchProtocol

logger = logging.getLogger(__name__)


def chunked_full_corpus_search(  # noqa: PLR0913
    *,
    task_corpus: CorpusDatasetType,
    corpus_chunk_size: int,
    query_idx_to_id: dict[int, str],
    query_embeddings: Array,
    task_metadata: TaskMetadata,
    hf_subset: str,
    hf_split: str,
    top_k: int,
    encode_kwargs: EncodeKwargs,
    encode_fn: Callable[..., Array],
    similarity_fn: Callable[[Array, Array], Array],
    search_k_offset: int = 0,
    num_proc: int | None = None,
) -> dict[str, list[tuple[float, str]]]:
    """Chunk over `task_corpus`, encode + score each chunk against `query_embeddings`, and merge per-query top-k results with a heap.

    Shared by `SearchEncoderWrapper` (for any `EncoderProtocol` model, via
    `encode_fn=model.encode`/`similarity_fn=model.similarity`) and
    `MultiVectorSearchEncoderWrapper` ([mteb.models.sentence_transformer_wrapper][], via
    `encode_fn=self._encode`/`similarity_fn=self.model.similarity`, since `MultiVectorWrapper`
    deliberately has no public `encode()` -- see its docstring). Kept as a plain function taking
    `encode_fn`/`similarity_fn` callables, rather than a shared base class, since the two callers
    can't share one via inheritance: `SearchEncoderWrapper` wraps an *external* `EncoderProtocol`
    object (`self.model.encode(...)`), while `MultiVectorSearchEncoderWrapper` is a self-referential mixin
    (`self._encode(...)`) -- inheriting one from the other would collide on the meaning of
    `self.model` and reintroduce the very `encode()` exposure `MultiVectorWrapper` is built to
    avoid.

    Args:
        task_corpus: The (already-indexed) corpus to search.
        corpus_chunk_size: Number of documents to encode and score at once.
        query_idx_to_id: Maps each query's position in `query_embeddings` to its query ID.
        query_embeddings: Pre-encoded query embeddings.
        task_metadata: Metadata of the task, forwarded to `encode_fn`.
        hf_subset: Subset of the current task, forwarded to `encode_fn`.
        hf_split: Split of the current task, forwarded to `encode_fn`.
        top_k: Number of top documents to keep per query.
        encode_kwargs: Additional arguments to pass to `encode_fn`.
        encode_fn: Encodes a document dataloader into embeddings, matching `EncoderProtocol.encode`.
        similarity_fn: Scores query embeddings against document embeddings, matching
            `EncoderProtocol.similarity`.
        search_k_offset: Added to `top_k` for the per-chunk `torch.topk` call (before the
            cross-chunk heap merge trims back down to `top_k`). `SearchEncoderWrapper` has
            historically used 1 here; `MultiVectorSearchEncoderWrapper` uses 0.
        num_proc: Number of processes to use for dataloading.

    Returns:
        A dictionary mapping query IDs to a list of `(score, corpus_id)` tuples.
    """
    result_heaps: dict[str, list[tuple[float, str]]] = {
        qid: [] for qid in query_idx_to_id.values()
    }
    itr = range(0, len(task_corpus), corpus_chunk_size)
    for batch_num, corpus_start_idx in enumerate(itr):
        logger.info(f"Encoding corpus chunk {batch_num + 1}/{len(itr)}...")
        corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(task_corpus))
        sub_corpus = task_corpus.select(range(corpus_start_idx, corpus_end_idx))
        sub_corpus_ids = list(sub_corpus["id"])

        sub_corpus_embeddings = encode_fn(
            create_dataloader(
                sub_corpus,
                task_metadata=task_metadata,
                prompt_type=PromptType.document,
                batch_size=encode_kwargs.get("batch_size", 32),
                num_proc=num_proc,
            ),
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.document,
            **encode_kwargs,
        )

        scores = torch.as_tensor(similarity_fn(query_embeddings, sub_corpus_embeddings))
        top_k_values, top_k_idx = torch.topk(
            scores,
            min(top_k + search_k_offset, scores.shape[1]),
            dim=1,
            largest=True,
        )
        top_k_idx_list = top_k_idx.cpu().tolist()
        top_k_values_list = top_k_values.cpu().tolist()

        for q_idx, qid in query_idx_to_id.items():
            for idx, score in zip(
                top_k_idx_list[q_idx], top_k_values_list[q_idx], strict=True
            ):
                corpus_id = sub_corpus_ids[idx]
                if len(result_heaps[qid]) < top_k:
                    heapq.heappush(result_heaps[qid], (score, corpus_id))
                else:
                    heapq.heappushpop(result_heaps[qid], (score, corpus_id))
    return result_heaps


def rerank_top_ranked_documents(  # noqa: PLR0913
    *,
    task_corpus: CorpusDatasetType,
    query_idx_to_id: dict[int, str],
    query_embeddings: Array,
    top_ranked: TopRankedDocumentsType,
    top_k: int,
    task_metadata: TaskMetadata,
    hf_subset: str,
    hf_split: str,
    encode_kwargs: EncodeKwargs,
    encode_fn: Callable[..., Array],
    similarity_fn: Callable[[Array, Array], Array],
    num_proc: int | None = None,
) -> dict[str, list[tuple[float, str]]]:
    """Encode the full corpus once, then rerank each query's pre-ranked `top_ranked` candidates against it.

    Shared by `SearchEncoderWrapper` and `MultiVectorSearchEncoderWrapper` -- see
    `chunked_full_corpus_search` for why this is a plain function rather than a shared base class.

    Args:
        task_corpus: The (already-indexed) corpus the `top_ranked` document IDs are drawn from.
        query_idx_to_id: Maps each query's position in `query_embeddings` to its query ID.
        query_embeddings: Pre-encoded query embeddings.
        top_ranked: Maps each query ID to its pre-ranked candidate document IDs.
        top_k: Number of top documents to keep per query.
        task_metadata: Metadata of the task, forwarded to `encode_fn`.
        hf_subset: Subset of the current task, forwarded to `encode_fn`.
        hf_split: Split of the current task, forwarded to `encode_fn`.
        encode_kwargs: Additional arguments to pass to `encode_fn`.
        encode_fn: Encodes a document dataloader into embeddings, matching `EncoderProtocol.encode`.
        similarity_fn: Scores query embeddings against document embeddings, matching
            `EncoderProtocol.similarity`.
        num_proc: Number of processes to use for dataloading.

    Returns:
        A dictionary mapping query IDs to a list of `(score, corpus_id)` tuples.
    """
    result_heaps: dict[str, list[tuple[float, str]]] = {
        qid: [] for qid in query_idx_to_id.values()
    }
    doc_id_to_idx = {doc: idx for idx, doc in enumerate(task_corpus["id"])}

    all_doc_embeddings = encode_fn(
        create_dataloader(
            task_corpus,
            task_metadata=task_metadata,
            prompt_type=PromptType.document,
            batch_size=encode_kwargs.get("batch_size", 32),
            num_proc=num_proc,
        ),
        task_metadata=task_metadata,
        hf_split=hf_split,
        hf_subset=hf_subset,
        prompt_type=PromptType.document,
        **encode_kwargs,
    )

    for q_idx, qid in query_idx_to_id.items():
        if qid not in top_ranked:
            logger.warning(f"No pre-ranked documents found for query {qid}")
            continue
        ranked_ids = top_ranked[qid]
        if not ranked_ids:
            continue

        doc_indices = [doc_id_to_idx[doc_id] for doc_id in ranked_ids]
        candidate_embeddings: Array | list[Any]
        if isinstance(all_doc_embeddings, (torch.Tensor, np.ndarray)):
            candidate_embeddings = all_doc_embeddings[doc_indices]
        else:
            # Ragged (variable-length) multi-vector embeddings: a plain list of per-document
            # tensors, which doesn't support fancy indexing with a list of indices.
            candidate_embeddings = [all_doc_embeddings[idx] for idx in doc_indices]

        # Ensure the query embedding is scored as a batch of one.
        query_embedding = torch.as_tensor(query_embeddings[q_idx]).unsqueeze(0)

        scores = torch.as_tensor(similarity_fn(query_embedding, candidate_embeddings))

        is_nan = torch.isnan(scores)
        if is_nan.sum() > 0:
            raise ValueError(
                f"NaN values detected in the similarity scores: {is_nan.sum()}"
            )

        scores_top_k_values, scores_top_k_idx = torch.topk(
            scores,
            min(top_k, len(ranked_ids)),
            dim=1,
            largest=True,
        )
        scores_top_k_values = scores_top_k_values.cpu()
        scores_top_k_idx = scores_top_k_idx.cpu()

        for doc_idx, score in zip(
            scores_top_k_idx[0].tolist(), scores_top_k_values[0].tolist(), strict=True
        ):
            corpus_id = ranked_ids[doc_idx]
            heapq.heappush(result_heaps[qid], (score, corpus_id))

    return result_heaps


class SearchEncoderWrapper:
    """Wrapper for Encoder models to be used in search tasks."""

    task_corpus: CorpusDatasetType | None

    def __init__(
        self,
        model: EncoderProtocol,
        corpus_chunk_size: int = 50_000,
        index_backend: IndexEncoderSearchProtocol | None = None,
    ) -> None:
        self.model = model
        self.task_corpus = None
        self.mteb_model_meta = model.mteb_model_meta
        self.corpus_chunk_size = corpus_chunk_size
        self.index_backend = index_backend

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None = None,
    ) -> None:
        """Index the corpus for retrieval.

        Args:
            corpus: Corpus dataset to index.
            task_metadata: Metadata of the task, used to determine how to index the corpus.
            hf_split: Split of current task, allows to know some additional information about current split.
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            encode_kwargs: Additional arguments to pass to the encoder during indexing.
            num_proc: Number of processes to use for dataloading.
        """
        # Always retain corpus for potential reranking or fallback flows
        self.task_corpus = corpus
        if self.index_backend is not None:
            all_doc_embeddings = self.model.encode(
                create_dataloader(
                    corpus,
                    task_metadata=task_metadata,
                    prompt_type=PromptType.document,
                    num_proc=num_proc,
                    **encode_kwargs,
                ),
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=PromptType.document,
                **encode_kwargs,
            )

            self.index_backend.add_documents(all_doc_embeddings, corpus["id"])

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
        num_proc: int | None = None,
    ) -> RetrievalOutputType:
        """Search the corpus for the given queries.

        Args:
            queries: Queries to find
            task_metadata: Task metadata
            hf_split: split of the dataset
            hf_subset: subset of the dataset
            top_ranked: Top-ranked documents for each query, mapping query IDs to a list of document IDs.
                Passed only from Reranking tasks.
            top_k: Number of top documents to return for each query.
            encode_kwargs: Additional arguments to pass to the encoder during indexing.
            num_proc: Number of processes to use for dataloading.

        Returns:
            Dictionary with query IDs as keys with dict as values, where each value is a mapping of document IDs to their relevance scores.
        """
        if self.task_corpus is None:
            raise ValueError("Corpus must be indexed before searching.")

        queries_dataloader = create_dataloader(
            queries,
            task_metadata=task_metadata,
            prompt_type=PromptType.query,
            num_proc=num_proc,
            **encode_kwargs,
        )

        query_embeddings = self.model.encode(
            queries_dataloader,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.query,
            **encode_kwargs,
        )
        query_idx_to_id = dict(enumerate(queries["id"]))

        if top_ranked is not None:
            logger.info("Reranking pre-ranked documents...")
            if self.index_backend is None:
                result_heaps = self._rerank_documents(
                    query_idx_to_id=query_idx_to_id,
                    query_embeddings=query_embeddings,
                    top_ranked=top_ranked,
                    top_k=top_k,
                    task_metadata=task_metadata,
                    hf_subset=hf_subset,
                    hf_split=hf_split,
                    encode_kwargs=encode_kwargs,
                )
            else:
                cos_scores_top_k_values, cos_scores_top_k_idx = (
                    self.index_backend.search(
                        query_embeddings,
                        top_k,
                        similarity_fn=self.model.similarity,
                        top_ranked=top_ranked,
                        query_idx_to_id=query_idx_to_id,
                    )
                )
                result_heaps = {qid: [] for qid in query_idx_to_id.values()}
                for query_itr in range(len(query_embeddings)):
                    result_heaps = self._rerank_sort_results(
                        result_heaps=result_heaps,
                        query_id=query_idx_to_id[query_itr],
                        ranked_ids=top_ranked[query_idx_to_id[query_itr]],
                        scores_top_k_idx=torch.tensor(
                            [cos_scores_top_k_idx[query_itr]]
                        ),
                        scores_top_k_values=torch.tensor(
                            [cos_scores_top_k_values[query_itr]]
                        ),
                    )
                self.index_backend.clear()
        else:
            logger.info("Performing full corpus search...")
            if self.index_backend is None:
                result_heaps = self._full_corpus_search(
                    query_idx_to_id=query_idx_to_id,
                    query_embeddings=query_embeddings,
                    task_metadata=task_metadata,
                    hf_subset=hf_subset,
                    hf_split=hf_split,
                    top_k=top_k,
                    encode_kwargs=encode_kwargs,
                )
            else:
                cos_scores_top_k_values, cos_scores_top_k_idx = (
                    self.index_backend.search(
                        query_embeddings,
                        top_k,
                        similarity_fn=self.model.similarity,
                        top_ranked=None,
                        query_idx_to_id=None,
                    )
                )
                result_heaps = {qid: [] for qid in query_idx_to_id.values()}
                result_heaps = self._sort_full_corpus_results(
                    result_heaps=result_heaps,
                    query_idx_to_id=query_idx_to_id,
                    query_embeddings=query_embeddings,
                    cos_scores_top_k_idx=cos_scores_top_k_idx,
                    cos_scores_top_k_values=cos_scores_top_k_values,
                    sub_corpus_ids=self.task_corpus["id"],
                    top_k=top_k,
                )
                self.index_backend.clear()

        # Reset the task corpus dataloader to None to free up memory
        self.task_corpus = None

        results: RetrievalOutputType = {qid: {} for qid in query_idx_to_id.values()}
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                results[qid][corpus_id] = score

        return results

    def _full_corpus_search(
        self,
        *,
        query_idx_to_id: dict[int, str],
        query_embeddings: Array,
        task_metadata: TaskMetadata,
        hf_subset: str,
        hf_split: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
    ) -> dict[str, list[tuple[float, str]]]:
        if self.task_corpus is None:
            raise ValueError("Corpus must be indexed before searching.")

        return chunked_full_corpus_search(
            task_corpus=self.task_corpus,
            corpus_chunk_size=self.corpus_chunk_size,
            query_idx_to_id=query_idx_to_id,
            query_embeddings=query_embeddings,
            task_metadata=task_metadata,
            hf_subset=hf_subset,
            hf_split=hf_split,
            top_k=top_k,
            encode_kwargs=encode_kwargs,
            encode_fn=self.model.encode,
            similarity_fn=self.model.similarity,
            search_k_offset=1,
        )

    def _sort_full_corpus_results(  # noqa: PLR6301
        self,
        *,
        result_heaps: dict[str, list[tuple[float, str]]],
        query_idx_to_id: dict[int, str],
        query_embeddings: Array,
        cos_scores_top_k_idx: list[list[int]],
        cos_scores_top_k_values: list[list[float]],
        sub_corpus_ids: list[str],
        top_k: int,
    ) -> dict[str, list[tuple[float, str]]]:
        """Sort the heaps into descending order lists.

        Returns:
            A dictionary mapping query IDs to a sorted list of tuples, each containing a relevance score and a document ID.
        """
        for query_itr in range(len(query_embeddings)):
            query_id = query_idx_to_id[query_itr]
            for sub_corpus_id, score in zip(
                cos_scores_top_k_idx[query_itr],
                cos_scores_top_k_values[query_itr],
                strict=True,
            ):
                corpus_id = sub_corpus_ids[sub_corpus_id]
                if len(result_heaps[query_id]) < top_k:
                    # push item on the heap
                    heapq.heappush(result_heaps[query_id], (score, corpus_id))
                else:
                    # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                    heapq.heappushpop(result_heaps[query_id], (score, corpus_id))
        return result_heaps

    def _rerank_documents(
        self,
        *,
        query_idx_to_id: dict[int, str],
        query_embeddings: Array,
        top_ranked: TopRankedDocumentsType,
        top_k: int,
        task_metadata: TaskMetadata,
        hf_subset: str,
        hf_split: str,
        encode_kwargs: EncodeKwargs,
    ) -> dict[str, list[tuple[float, str]]]:
        """Rerank documents based on pre-ranked documents.

        Returns:
            A dictionary mapping query IDs to a list of tuples, each containing a relevance score and a document ID.
        """
        if self.task_corpus is None:
            raise ValueError("Corpus must be indexed before searching.")

        return rerank_top_ranked_documents(
            task_corpus=self.task_corpus,
            query_idx_to_id=query_idx_to_id,
            query_embeddings=query_embeddings,
            top_ranked=top_ranked,
            top_k=top_k,
            task_metadata=task_metadata,
            hf_subset=hf_subset,
            hf_split=hf_split,
            encode_kwargs=encode_kwargs,
            encode_fn=self.model.encode,
            similarity_fn=self.model.similarity,
        )

    def _rerank_sort_results(  # noqa: PLR6301
        self,
        result_heaps: dict[str, list[tuple[float, str]]],
        query_id: str,
        ranked_ids: list[str],
        scores_top_k_idx: torch.Tensor,
        scores_top_k_values: torch.Tensor,
    ) -> dict[str, list[tuple[float, str]]]:
        """Sort the heap into descending order list.

        Returns:
            A sorted list of tuples, each containing a relevance score and a document ID.
        """
        for doc_idx, score in zip(
            scores_top_k_idx[0].tolist(),
            scores_top_k_values[0].tolist(),
            strict=True,
        ):
            corpus_id = ranked_ids[doc_idx]
            heapq.heappush(result_heaps[query_id], (score, corpus_id))
        return result_heaps

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        """Encode inputs using the model' s encode."""
        return self.model.encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )

    def similarity(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Compute the similarity between two collections of embeddings."""
        return self.model.similarity(embeddings1, embeddings2)

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Compute the pairwise similarity between two collections of embeddings."""
        return self.model.similarity_pairwise(embeddings1, embeddings2)


class SearchCrossEncoderWrapper:
    """Wrapper for CrossEncoder models to be used in search tasks."""

    task_corpus: CorpusDatasetType | None

    def __init__(self, model: CrossEncoderProtocol):
        self.model = model
        self.task_corpus = None
        self.mteb_model_meta = model.mteb_model_meta

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None = None,
    ) -> None:
        """Index the corpus for retrieval.

        Args:
            corpus: Corpus dataset to index.
            task_metadata: Metadata of the task, used to determine how to index the corpus.
            hf_split: Split of current task, allows to know some additional information about current split.
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            encode_kwargs: Additional arguments to pass to the encoder during indexing.
            num_proc: Number of processes to use.
        """
        self.task_corpus = corpus

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
        num_proc: int | None = None,
    ) -> RetrievalOutputType:
        """Search the corpus using the given queries.

        Args:
            queries: Queries to find
            task_metadata: Task metadata
            hf_split: split of the dataset
            hf_subset: subset of the dataset
            top_ranked: Top-ranked documents for each query, mapping query IDs to a list of document IDs.
                Passed only from Reranking tasks.
            top_k: Number of top documents to return for each query.
            encode_kwargs: Additional arguments to pass to the encoder during indexing.
            num_proc: Number of processes to use.

        Returns:
            Dictionary with query IDs as keys with dict as values, where each value is a mapping of document IDs to their relevance scores.
        """
        if top_ranked is None:
            raise ValueError(
                "CrossEncoder search requires top_ranked documents for reranking."
            )
        if self.task_corpus is None:
            raise ValueError("Corpus must be indexed before searching.")

        query_id_to_idx = {row: i for i, row in enumerate(queries["id"])}
        doc_id_to_idx = {doc: idx for idx, doc in enumerate(self.task_corpus["id"])}

        query_indices: list[int] = []
        doc_indices: list[int] = []
        doc_pairs_ids: list[tuple[str, str]] = []
        for query_id, corpus_ids in top_ranked.items():
            if query_id not in top_ranked:
                msg = f"No pre-ranked documents found for query {query_id}"
                logger.warning(msg)
                continue

            query_idx = query_id_to_idx[query_id]
            for corpus_id in corpus_ids:
                doc_pairs_ids.append((query_id, corpus_id))
                query_indices.append(query_idx)
                doc_indices.append(doc_id_to_idx[corpus_id])

        # select() builds an indices-mapping view over the existing datasets
        # instead of materializing one copied (and, for images, decoded) row per
        # pair, which for image corpora multiplies memory by top_k x decode size.
        queries_loader = create_dataloader(
            queries.select(query_indices),
            task_metadata=task_metadata,
            # PromptType.query, matching SearchEncoderWrapper: on mixed-modality
            # tasks the query rows carry the query modality, and a
            # document prompt type would prepare them with the corpus modality.
            prompt_type=PromptType.query,
            num_proc=num_proc,
            **encode_kwargs,
        )
        corpus_loader = create_dataloader(
            self.task_corpus.select(doc_indices),
            task_metadata=task_metadata,
            prompt_type=PromptType.document,
            num_proc=num_proc,
            **encode_kwargs,
        )
        predictions = self.model.predict(
            inputs1=queries_loader,
            inputs2=corpus_loader,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
        )

        results: RetrievalOutputType = {qid: {} for qid in queries["id"]}
        for (query_id, corpus_id), score in zip(
            doc_pairs_ids, predictions, strict=True
        ):
            results[query_id][corpus_id] = float(score)

        return results
