from __future__ import annotations

import heapq
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

from mteb._create_dataloaders import (
    create_dataloader,
)
from mteb.timing import TimingStack
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

    from .model_meta import ModelMeta
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
    num_proc: int | None = None,
    timer: TimingStack | None = None,
) -> dict[str, list[tuple[float, str]]]:
    """Chunk over `task_corpus`, encode + score each chunk against `query_embeddings`, and merge per-query top-k results with a heap.

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
        num_proc: Number of processes to use for dataloading.
        timer: Records a single "Searching corpus" phase; the per-chunk split between encoding
            and computing similarity is logged.

    Returns:
        A dictionary mapping query IDs to a list of `(score, corpus_id)` tuples.
    """
    import torch

    timer = timer or TimingStack()
    result_heaps: dict[str, list[tuple[float, str]]] = {
        qid: [] for qid in query_idx_to_id.values()
    }
    itr = range(0, len(task_corpus), corpus_chunk_size)
    # Encoding and similarity alternate per chunk, so the timer gets a single phase for the
    # whole search (one row per chunk would bloat the saved results); the per-chunk split
    # is only logged.
    encoding_time = 0.0
    similarity_time = 0.0
    with timer("Searching corpus", split=hf_split, subset=hf_subset):
        for batch_num, corpus_start_idx in enumerate(
            tqdm(
                itr,
                desc="Searching corpus chunks",
                disable=not encode_kwargs.get("show_progress_bar", True),
            )
        ):
            logger.info(f"Encoding corpus chunk {batch_num + 1}/{len(itr)}...")
            corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(task_corpus))
            sub_corpus = task_corpus.select(range(corpus_start_idx, corpus_end_idx))
            sub_corpus_ids = list(sub_corpus["id"])

            chunk_start = time.monotonic()
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
            chunk_encoded = time.monotonic()

            scores = torch.as_tensor(
                similarity_fn(query_embeddings, sub_corpus_embeddings)
            )
            top_k_values, top_k_idx = torch.topk(
                scores,
                min(top_k + 1, scores.shape[1]),
                dim=1,
                largest=True,
            )
            top_k_idx_list = top_k_idx.cpu().tolist()
            top_k_values_list = top_k_values.cpu().tolist()
            # Release this chunk before the next one is encoded; otherwise both chunks' embeddings
            # (possibly on the GPU, e.g. multi-vector models) are alive during the next encode.
            del sub_corpus_embeddings, scores, top_k_values, top_k_idx

            for q_idx, qid in query_idx_to_id.items():
                for idx, score in zip(
                    top_k_idx_list[q_idx], top_k_values_list[q_idx], strict=True
                ):
                    corpus_id = sub_corpus_ids[idx]
                    if len(result_heaps[qid]) < top_k:
                        heapq.heappush(result_heaps[qid], (score, corpus_id))
                    else:
                        heapq.heappushpop(result_heaps[qid], (score, corpus_id))
            chunk_done = time.monotonic()

            encoding_time += chunk_encoded - chunk_start
            similarity_time += chunk_done - chunk_encoded
            logger.info(
                f"Corpus chunk {batch_num + 1}/{len(itr)}: encoding took "
                f"{chunk_encoded - chunk_start:.1f}s, computing similarity took "
                f"{chunk_done - chunk_encoded:.1f}s"
            )
    logger.info(
        f"Searched {len(itr)} corpus chunk(s): encoding took {encoding_time:.1f}s, "
        f"computing similarity took {similarity_time:.1f}s"
    )
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
    timer: TimingStack | None = None,
) -> dict[str, list[tuple[float, str]]]:
    """Encode the full corpus once, then rerank each query's pre-ranked `top_ranked` candidates against it.

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
        timer: Records "Encoding corpus" and "Computing similarity" phases.

    Returns:
        A dictionary mapping query IDs to a list of `(score, corpus_id)` tuples.
    """
    import torch

    timer = timer or TimingStack()
    result_heaps: dict[str, list[tuple[float, str]]] = {
        qid: [] for qid in query_idx_to_id.values()
    }
    doc_id_to_idx = {doc: idx for idx, doc in enumerate(task_corpus["id"])}

    with timer("Encoding corpus", split=hf_split, subset=hf_subset):
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

    with timer("Computing similarity", split=hf_split, subset=hf_subset):
        for q_idx, qid in tqdm(
            query_idx_to_id.items(),
            desc="Reranking queries",
            disable=not encode_kwargs.get("show_progress_bar", True),
        ):
            if qid not in top_ranked:
                logger.warning(f"No pre-ranked documents found for query {qid}")
                continue
            ranked_ids = top_ranked[qid]
            if not ranked_ids:
                continue

            doc_indices = [doc_id_to_idx[doc_id] for doc_id in ranked_ids]
            candidate_embeddings: Array | list[Any]
            if isinstance(all_doc_embeddings, (torch.Tensor, np.ndarray)):
                candidate_embeddings = torch.as_tensor(all_doc_embeddings[doc_indices])
            else:
                # Ragged (variable-length) multi-vector embeddings: a plain list of per-document
                # tensors, which doesn't support fancy indexing with a list of indices.
                candidate_embeddings = [all_doc_embeddings[idx] for idx in doc_indices]

            # Ensure the query embedding is scored as a batch of one.
            query_embedding = torch.as_tensor(query_embeddings[q_idx]).unsqueeze(0)

            scores = torch.as_tensor(
                similarity_fn(query_embedding, candidate_embeddings)
            )

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
                scores_top_k_idx[0].tolist(),
                scores_top_k_values[0].tolist(),
                strict=True,
            ):
                corpus_id = ranked_ids[doc_idx]
                heapq.heappush(result_heaps[qid], (score, corpus_id))
    return result_heaps


def _check_index_backend_model_type(
    meta: ModelMeta, index_backend: IndexEncoderSearchProtocol
) -> None:
    """Raise if `index_backend` can't index embeddings of the model's type.

    Every type of the model must be supported, e.g. a `["dense", "sparse"]` model can produce
    sparse embeddings, which `FaissSearchIndex` can't index.
    """
    unsupported = set(meta.model_type) - set(index_backend.supported_model_types)
    if unsupported:
        raise ValueError(
            f"{type(index_backend).__name__} supports models of type "
            f"{list(index_backend.supported_model_types)}, but {meta.name} has model_type "
            f"{meta.model_type}."
        )


def _add_index_backend_experiment_kwargs(
    meta: ModelMeta, index_backend: IndexEncoderSearchProtocol | None
) -> ModelMeta:
    """Return a copy of `meta` whose `experiment_kwargs` include the backend's `experiment_kwargs`.

    Approximate backends (e.g. `FastPlaidSearchIndex`) expose `experiment_kwargs`, so their results
    are saved as an experiment instead of overwriting exact search results. Exact backends, such
    as `FaissSearchIndex`, don't, and leave `meta` unchanged.
    """
    backend_kwargs = getattr(index_backend, "experiment_kwargs", None)
    if not backend_kwargs:
        return meta
    experiment_kwargs = {**(meta.experiment_kwargs or {}), **backend_kwargs}
    return meta.model_copy(update={"experiment_kwargs": experiment_kwargs}, deep=True)


class EncoderSearchMixin:
    """Shared indexing and search for encoder-based search models."""

    task_corpus: CorpusDatasetType | None = None
    corpus_chunk_size: int = 50_000
    index_backend: IndexEncoderSearchProtocol | None = None

    def _search_encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        """Encode queries or documents for search."""
        raise NotImplementedError

    def _search_similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Score query embeddings against document embeddings for search."""
        raise NotImplementedError

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Index the corpus for retrieval.

        Args:
            corpus: Corpus dataset to index.
            task_metadata: Metadata of the task, used to determine how to index the corpus.
            hf_split: Split of current task, allows to know some additional information about current split.
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            encode_kwargs: Additional arguments to pass to the encoder during indexing.
            num_proc: Number of processes to use for dataloading.
            **kwargs: Additional arguments. Currently unused.
        """
        # Always retain corpus for potential reranking or fallback flows
        self.task_corpus = corpus
        if self.index_backend is not None:
            all_doc_embeddings = self._search_encode(
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
        timer: TimingStack | None = None,
        **kwargs: Any,
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
            timer: Records the search phases: "Encoding queries", then "Searching corpus" for a
                full corpus search, or "Encoding corpus" and "Computing similarity" when reranking.
                With an `index_backend`, the corpus is already encoded in `index`, so only
                "Computing similarity" follows. A new stack is used when not given.
            **kwargs: Additional arguments. Currently unused.

        Returns:
            Dictionary with query IDs as keys with dict as values, where each value is a mapping of document IDs to their relevance scores.
        """
        if self.task_corpus is None:
            raise ValueError("Corpus must be indexed before searching.")
        timer = timer or TimingStack()

        queries_dataloader = create_dataloader(
            queries,
            task_metadata=task_metadata,
            prompt_type=PromptType.query,
            num_proc=num_proc,
            **encode_kwargs,
        )

        with timer("Encoding queries", split=hf_split, subset=hf_subset):
            query_embeddings = self._search_encode(
                queries_dataloader,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=PromptType.query,
                **encode_kwargs,
            )
        query_idx_to_id = dict(enumerate(queries["id"]))

        if self.index_backend is not None:
            logger.info("Searching with the index backend...")
            result_heaps = self._index_backend_search(
                query_idx_to_id=query_idx_to_id,
                query_embeddings=query_embeddings,
                top_ranked=top_ranked,
                top_k=top_k,
                hf_subset=hf_subset,
                hf_split=hf_split,
                timer=timer,
            )
        elif top_ranked is not None:
            logger.info("Reranking pre-ranked documents...")
            result_heaps = self._rerank_documents(
                query_idx_to_id=query_idx_to_id,
                query_embeddings=query_embeddings,
                top_ranked=top_ranked,
                top_k=top_k,
                task_metadata=task_metadata,
                hf_subset=hf_subset,
                hf_split=hf_split,
                encode_kwargs=encode_kwargs,
                timer=timer,
                num_proc=num_proc,
            )
        else:
            logger.info("Performing full corpus search...")
            result_heaps = self._full_corpus_search(
                query_idx_to_id=query_idx_to_id,
                query_embeddings=query_embeddings,
                task_metadata=task_metadata,
                hf_subset=hf_subset,
                hf_split=hf_split,
                top_k=top_k,
                encode_kwargs=encode_kwargs,
                timer=timer,
                num_proc=num_proc,
            )

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
        timer: TimingStack,
        num_proc: int | None = None,
    ) -> dict[str, list[tuple[float, str]]]:

        logger.info("Encoding Corpus in batches (this might take a while)...")
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
            encode_fn=self._search_encode,
            similarity_fn=self._search_similarity,
            num_proc=num_proc,
            timer=timer,
        )

    def _rerank_documents(  # noqa: PLR0913
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
        timer: TimingStack,
        num_proc: int | None = None,
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
            encode_fn=self._search_encode,
            similarity_fn=self._search_similarity,
            num_proc=num_proc,
            timer=timer,
        )

    def _index_backend_search(
        self,
        *,
        query_idx_to_id: dict[int, str],
        query_embeddings: Array,
        top_ranked: TopRankedDocumentsType | None,
        top_k: int,
        hf_subset: str,
        hf_split: str,
        timer: TimingStack,
    ) -> dict[str, list[tuple[float, str]]]:
        """Search (or rerank `top_ranked` candidates) with `index_backend`, which holds the corpus added in `index`, then clear it.

        Returns:
            A dictionary mapping query IDs to a list of `(score, corpus_id)` tuples.
        """
        if self.index_backend is None or self.task_corpus is None:
            raise ValueError("Corpus must be indexed before searching.")

        with timer("Computing similarity", split=hf_split, subset=hf_subset):
            top_k_values, top_k_idx = self.index_backend.search(
                query_embeddings,
                top_k,
                similarity_fn=self._search_similarity,
                top_ranked=top_ranked,
                query_idx_to_id=query_idx_to_id if top_ranked is not None else None,
            )
        self.index_backend.clear()

        corpus_ids = self.task_corpus["id"]
        result_heaps: dict[str, list[tuple[float, str]]] = {
            qid: [] for qid in query_idx_to_id.values()
        }
        for q_idx, qid in query_idx_to_id.items():
            # In reranking mode the backend indexes into the query's own candidate list
            candidate_ids = (
                top_ranked.get(qid, []) if top_ranked is not None else corpus_ids
            )
            for idx, score in zip(top_k_idx[q_idx], top_k_values[q_idx], strict=True):
                corpus_id = candidate_ids[idx]
                if len(result_heaps[qid]) < top_k:
                    heapq.heappush(result_heaps[qid], (score, corpus_id))
                else:
                    heapq.heappushpop(result_heaps[qid], (score, corpus_id))
        return result_heaps


class SearchEncoderWrapper(EncoderSearchMixin):
    """Wrapper for Encoder models to be used in search tasks.

    Works with any embedding type: dense, sparse or multi-vector. An `index_backend` matching the
    embeddings can be passed, e.g. `FaissSearchIndex` for dense or `FastPlaidSearchIndex` for
    multi-vector models.
    """

    def __init__(
        self,
        model: EncoderProtocol,
        corpus_chunk_size: int = 50_000,
        index_backend: IndexEncoderSearchProtocol | None = None,
    ) -> None:
        """Wrap an encoder for search.

        Args:
            model: Encoder used to embed queries and documents. Models without a public `encode`,
                such as `MultiVectorWrapper`, are encoded with their `_encode`.
            corpus_chunk_size: Number of documents to encode and score at once in full corpus search.
            index_backend: Optional search index; when given, the corpus is encoded during `index`.
        """
        self.model = model
        # `MultiVectorWrapper` has no public `encode` (it is a search model itself), only `_encode`
        encode_fn = getattr(model, "encode", None) or getattr(model, "_encode", None)
        if encode_fn is None:
            raise TypeError(
                f"{type(model).__name__} has neither `encode` nor `_encode`, so it can't be wrapped by SearchEncoderWrapper."
            )
        self._encode_fn: Callable[..., Array] = encode_fn
        if index_backend is not None:
            _check_index_backend_model_type(model.mteb_model_meta, index_backend)
        self.task_corpus = None
        self.mteb_model_meta = _add_index_backend_experiment_kwargs(
            model.mteb_model_meta, index_backend
        )
        self.corpus_chunk_size = corpus_chunk_size
        self.index_backend = index_backend

    def _search_encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        return self._encode_fn(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )

    def _search_similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        return self.model.similarity(embeddings1, embeddings2)

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
        return self._encode_fn(
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
        **kwargs: Any,
    ) -> None:
        """Index the corpus for retrieval.

        Args:
            corpus: Corpus dataset to index.
            task_metadata: Metadata of the task, used to determine how to index the corpus.
            hf_split: Split of current task, allows to know some additional information about current split.
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            encode_kwargs: Additional arguments to pass to the encoder during indexing.
            num_proc: Number of processes to use.
            **kwargs: Additional arguments. Currently unused.
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
        **kwargs: Any,
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
            **kwargs: Additional arguments. Currently unused.

        Returns:
            Dictionary with query IDs as keys with dict as values, where each value is a mapping of document IDs to their relevance scores.
        """
        if top_ranked is None:
            raise ValueError(
                "CrossEncoder search requires top_ranked documents for reranking."
            )
        if self.task_corpus is None:
            raise ValueError("Corpus must be indexed before searching.")

        doc_id_to_idx = {doc: idx for idx, doc in enumerate(self.task_corpus["id"])}

        query_indices: list[int] = []
        doc_indices: list[int] = []
        doc_pairs_ids: list[tuple[str, str]] = []
        for query_idx, query_id in enumerate(queries["id"]):
            if query_id not in top_ranked:
                msg = f"No pre-ranked documents found for query {query_id}"
                logger.warning(msg)
                continue

            for corpus_id in top_ranked[query_id]:
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
            **encode_kwargs,
        )

        results: RetrievalOutputType = {qid: {} for qid in queries["id"]}
        for (query_id, corpus_id), score in zip(
            doc_pairs_ids, predictions, strict=True
        ):
            results[query_id][corpus_id] = float(score)

        return results
