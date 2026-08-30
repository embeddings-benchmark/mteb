from __future__ import annotations

import heapq
import logging
from typing import TYPE_CHECKING, Any, cast

import torch

from mteb._create_dataloaders import (
    create_dataloader,
)
from mteb.types import (
    PromptType,
)

if TYPE_CHECKING:
    from sentence_transformers import MultiVectorEncoder
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
    from .sentence_transformer_wrapper import MultiVectorTokenEncoder

logger = logging.getLogger(__name__)


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
        logger.info("Encoding Corpus in batches (this might take a while)...")
        if self.task_corpus is None:
            raise ValueError("Corpus must be indexed before searching.")

        itr = range(0, len(self.task_corpus), self.corpus_chunk_size)

        result_heaps: dict[str, list[tuple[float, str]]] = {
            qid: [] for qid in query_idx_to_id.values()
        }
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info(f"Encoding Batch {batch_num + 1}/{len(itr)}...")
            corpus_end_idx = min(
                corpus_start_idx + self.corpus_chunk_size,
                len(self.task_corpus),
            )
            sub_corpus = self.task_corpus.select(
                range(corpus_start_idx, corpus_end_idx)
            )
            sub_corpus_ids = sub_corpus["id"]
            sub_corpus_embeddings = self.model.encode(
                create_dataloader(
                    sub_corpus,
                    task_metadata=task_metadata,
                    prompt_type=PromptType.document,
                    **encode_kwargs,
                ),
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=PromptType.document,
                **encode_kwargs,
            )

            # Compute similarities using either cosine-similarity or dot product
            logger.info("Computing Similarities...")
            scores = self.model.similarity(query_embeddings, sub_corpus_embeddings)

            # get top-k values
            cos_scores_top_k_values_tensor, cos_scores_top_k_idx_tensor = torch.topk(
                torch.as_tensor(scores),
                min(
                    top_k + 1,
                    len(scores[1]) if len(scores) > 1 else len(scores[-1]),
                ),
                dim=1,
                largest=True,
            )
            cos_scores_top_k_idx = cos_scores_top_k_idx_tensor.cpu().tolist()
            cos_scores_top_k_values = cos_scores_top_k_values_tensor.cpu().tolist()

            sub_corpus_ids = list(sub_corpus_ids)
            result_heaps = self._sort_full_corpus_results(
                result_heaps=result_heaps,
                query_idx_to_id=query_idx_to_id,
                query_embeddings=query_embeddings,
                cos_scores_top_k_idx=cos_scores_top_k_idx,
                cos_scores_top_k_values=cos_scores_top_k_values,
                sub_corpus_ids=sub_corpus_ids,
                top_k=top_k,
            )
        return result_heaps

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
        result_heaps: dict[str, list[tuple[float, str]]] = {
            qid: [] for qid in query_idx_to_id.values()
        }
        doc_id_to_idx = {doc: idx for idx, doc in enumerate(self.task_corpus["id"])}

        all_doc_embeddings = self.model.encode(
            create_dataloader(
                self.task_corpus,
                task_metadata=task_metadata,
                prompt_type=PromptType.document,
                **encode_kwargs,
            ),
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.document,
            **encode_kwargs,
        )

        # Process each query
        for query_idx, query_embedding in enumerate(query_embeddings):
            query_id = query_idx_to_id[query_idx]
            if query_id not in top_ranked:
                msg = f"No pre-ranked documents found for query {query_id}"
                logger.warning(msg)
                continue

            ranked_ids = top_ranked[query_id]
            doc_indices = [doc_id_to_idx[doc_id] for doc_id in ranked_ids]
            query_doc_embeddings: torch.Tensor | list[Any]
            if isinstance(all_doc_embeddings, torch.Tensor):
                query_doc_embeddings = all_doc_embeddings[doc_indices]
            else:
                # Ragged (variable-length) multi-vector embeddings: a plain list of per-document
                # tensors, which doesn't support fancy indexing with a tensor of indices.
                query_doc_embeddings = [all_doc_embeddings[idx] for idx in doc_indices]

            # Ensure query embedding is on the correct device and has correct shape
            query_embedding = torch.as_tensor(query_embedding).unsqueeze(0)  # noqa: PLW2901

            scores = self.model.similarity(
                query_embedding,
                cast("Array", query_doc_embeddings),
            )
            scores = torch.as_tensor(scores)

            # Handle NaN values
            is_nan = torch.isnan(scores)
            if is_nan.sum() > 0:
                raise ValueError(
                    f"NaN values detected in the similarity scores: {is_nan.sum()}"
                )

            # Compute top-k scores
            scores_top_k_values, scores_top_k_idx = torch.topk(
                scores,
                min(top_k, len(ranked_ids)),
                dim=1,
                largest=True,
            )

            # Move results back to CPU for heap operations
            scores_top_k_values = scores_top_k_values.cpu()
            scores_top_k_idx = scores_top_k_idx.cpu()

            result_heaps = self._rerank_sort_results(
                result_heaps=result_heaps,
                query_id=query_id,
                ranked_ids=ranked_ids,
                scores_top_k_idx=scores_top_k_idx,
                scores_top_k_values=scores_top_k_values,
            )
        return result_heaps

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


class _MultiVectorTokenEncoderAsEncoderProtocol:
    """Adapts a `MultiVectorTokenEncoder`'s `_encode` to the `encode()` name `SearchEncoderWrapper` calls internally.

    Doesn't give `MultiVectorTokenEncoder` itself a public `encode()` -- see its docstring for why
    that matters. Purely internal plumbing for `MultiVectorEncoderWrapper`; not a public class.
    """

    def __init__(self, encoder: MultiVectorTokenEncoder) -> None:
        self._encoder = encoder
        self.mteb_model_meta = encoder.mteb_model_meta

    def encode(self, *args: Any, **kwargs: Any) -> Array:
        return self._encoder._encode(*args, **kwargs)

    def similarity(self, *args: Any, **kwargs: Any) -> Array:
        return self._encoder.similarity(*args, **kwargs)

    def similarity_pairwise(self, *args: Any, **kwargs: Any) -> Array:
        return self._encoder.similarity_pairwise(*args, **kwargs)


class MultiVectorEncoderWrapper(SearchEncoderWrapper):
    """`SearchProtocol` wrapper for sentence-transformers' native `MultiVectorEncoder` models (ColBERT-style late interaction, e.g. `lightonai/LateOn` or the "Sentence Transformers" usage of `vidore/colpali-v1.3`).

    Registered as the model `loader` directly (rather than being auto-wrapped in
    `SearchEncoderWrapper`, like a plain `EncoderProtocol` model would be): late-interaction models
    produce a ragged, variable-length sequence of per-token embeddings rather than a single
    fixed-size vector, so -- like `MultiVectorModel` for PyLate models -- they don't reduce to a
    single embedding and are retrieval-only.

    Subclasses `SearchEncoderWrapper` to reuse its brute-force chunked search unchanged: MaxSim
    scoring via the model's own `similarity()` already accepts ragged multi-vector embeddings, and
    `_rerank_documents` indexes per-document embeddings as a plain list rather than with tensor
    fancy-indexing for the same reason, so both the full-corpus and pre-ranked-candidate paths work
    as-is.
    """

    def __init__(
        self,
        model: str | MultiVectorEncoder,
        revision: str | None = None,
        device: str | None = None,
        model_prompts: dict[str, str] | None = None,
        corpus_chunk_size: int = 50_000,
        **kwargs: Any,
    ) -> None:
        """Wrapper for MultiVectorEncoder models.

        Args:
            model: The MultiVectorEncoder model to use. Can be a string (model name) or a
                MultiVectorEncoder model.
            revision: The revision of the model to use.
            device: The device used to load the model.
            model_prompts: A dictionary mapping task names to prompt names. See
                `SentenceTransformerEncoderWrapper` for the order of priority used to select a prompt.
            corpus_chunk_size: Number of corpus documents to encode and score against the queries at
                once, during a full-corpus search.
            **kwargs: Additional arguments to pass to the MultiVectorEncoder model.
        """
        from .sentence_transformer_wrapper import MultiVectorTokenEncoder

        encoder = MultiVectorTokenEncoder(
            model,
            revision=revision,
            device=device,
            model_prompts=model_prompts,
            loader=type(self),
            **kwargs,
        )
        super().__init__(
            _MultiVectorTokenEncoderAsEncoderProtocol(encoder),
            corpus_chunk_size=corpus_chunk_size,
        )


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
