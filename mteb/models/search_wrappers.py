import heapq
import logging
from typing import Any

from datasets import Dataset
from torch.utils.data import DataLoader

from mteb._create_dataloaders import (
    create_dataloader,
)
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import (
    Array,
    BatchedInput,
    CorpusDatasetType,
    PromptType,
    QueryDatasetType,
    RetrievalOutputType,
    TopRankedDocumentsType,
)

from .models_protocols import CrossEncoderProtocol, EncoderProtocol
from .search_encoder_index.default_backend_search import DefaultEncoderSearchBackend
from .search_encoder_index.search_backend_protocol import IndexEncoderSearchProtocol

logger = logging.getLogger(__name__)


class SearchEncoderWrapper:
    """Wrapper for Encoder models to be used in search tasks."""

    task_corpus: CorpusDatasetType | None

    def __init__(
        self,
        model: EncoderProtocol,
        corpus_chunk_size: int = 50_000,
        index_backend: IndexEncoderSearchProtocol = DefaultEncoderSearchBackend(),
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
        encode_kwargs: dict[str, Any],
    ) -> None:
        """Index the corpus for retrieval.

        Args:
            corpus: Corpus dataset to index.
            task_metadata: Metadata of the task, used to determine how to index the corpus.
            hf_split: Split of current task, allows to know some additional information about current split.
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            encode_kwargs: Additional arguments to pass to the encoder during indexing.
        """
        # Always retain corpus for potential reranking or fallback flows
        self.task_corpus = corpus

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: dict[str, Any],
        top_ranked: TopRankedDocumentsType | None = None,
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

        Returns:
            Dictionary with query IDs as keys with dict as values, where each value is a mapping of document IDs to their relevance scores.
        """
        if self.task_corpus is None:
            raise ValueError("Corpus must be indexed before searching.")

        queries_dataloader = create_dataloader(
            queries,
            task_metadata,
            prompt_type=PromptType.query,
            batch_size=encode_kwargs.get("batch_size", 32),
        )

        query_embeddings = self.model.encode(
            queries_dataloader,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.query,
            **encode_kwargs,
        )
        query_idx_to_id = {i: row["id"] for i, row in enumerate(queries)}

        if top_ranked is not None:
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
            )

        # Reset the task corpus dataloader to None to free up memory
        self.task_corpus = None
        self.index_backend.clear()

        results = {qid: {} for qid in query_idx_to_id.values()}
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                results[qid][corpus_id] = score

        return results

    def _full_corpus_search(
        self,
        query_idx_to_id: dict[int, str],
        query_embeddings: Array,
        task_metadata: TaskMetadata,
        hf_subset: str,
        hf_split: str,
        top_k: int,
        encode_kwargs: dict[str, Any],
    ) -> dict[str, list[tuple[float, str]]]:
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        itr = range(0, len(self.task_corpus), self.corpus_chunk_size)

        result_heaps = {qid: [] for qid in query_idx_to_id.values()}
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info(f"Encoding Batch {batch_num + 1}/{len(itr)}...")
            corpus_end_idx = min(
                corpus_start_idx + self.corpus_chunk_size, len(self.task_corpus)
            )
            sub_corpus = self.task_corpus.select(
                range(corpus_start_idx, corpus_end_idx)
            )
            sub_corpus_ids = sub_corpus["id"]
            sub_corpus_embeddings = self.model.encode(
                create_dataloader(
                    sub_corpus,
                    task_metadata,
                    prompt_type=PromptType.document,
                    batch_size=encode_kwargs.get("batch_size", 32),
                ),
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=PromptType.document,
                **encode_kwargs,
            )
            self.index_backend.add_document(sub_corpus_embeddings, sub_corpus_ids)
            # Compute similarities using either cosine-similarity or dot product
            logger.info("Computing Similarities...")

            cos_scores_top_k_values, cos_scores_top_k_idx = self.index_backend.search(
                query_embeddings, top_k, self.model.similarity
            )

            for query_itr in range(len(query_embeddings)):
                query_id = query_idx_to_id[query_itr]
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr],
                    cos_scores_top_k_values[query_itr],
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
        query_idx_to_id: dict[int, str],
        query_embeddings: Array,
        top_ranked: TopRankedDocumentsType,
        top_k: int,
        task_metadata: TaskMetadata,
        hf_subset: str,
        hf_split: str,
        encode_kwargs: dict[str, Any],
    ) -> dict[str, list[tuple[float, str]]]:
        """Rerank documents using backend's search with top_ranked support."""
        result_heaps = {qid: [] for qid in query_idx_to_id.values()}

        # Encode corpus
        all_doc_embeddings = self.model.encode(
            create_dataloader(
                self.task_corpus,
                task_metadata,
                prompt_type=PromptType.document,
                batch_size=encode_kwargs.get("batch_size", 32),
            ),
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.document,
            **encode_kwargs,
        )
        all_doc_ids = [doc["id"] for doc in self.task_corpus]
        self.index_backend.add_document(all_doc_embeddings, all_doc_ids)

        # Unified search call
        cos_scores_top_k_values, cos_scores_top_k_idx = self.index_backend.search(
            query_embeddings,
            top_k,
            similarity_fn=self.model.similarity,
            top_ranked=top_ranked,
            query_idx_to_id=query_idx_to_id,
        )

        # Populate results
        for query_itr, query_id in query_idx_to_id.items():
            ranked_ids = top_ranked.get(query_id, [])
            for score, idx in zip(
                cos_scores_top_k_values[query_itr],
                cos_scores_top_k_idx[query_itr],
            ):
                if idx < len(ranked_ids):
                    corpus_id = ranked_ids[idx]
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
        encode_kwargs: dict[str, Any],
    ) -> None:
        """Index the corpus for retrieval.

        Args:
            corpus: Corpus dataset to index.
            task_metadata: Metadata of the task, used to determine how to index the corpus.
            hf_split: Split of current task, allows to know some additional information about current split.
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            encode_kwargs: Additional arguments to pass to the encoder during indexing.
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
        encode_kwargs: dict[str, Any],
        top_ranked: TopRankedDocumentsType | None = None,
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

        Returns:
            Dictionary with query IDs as keys with dict as values, where each value is a mapping of document IDs to their relevance scores.
        """
        if top_ranked is None:
            raise ValueError(
                "CrossEncoder search requires top_ranked documents for reranking."
            )

        query_id_to_idx = {row["id"]: i for i, row in enumerate(queries)}
        doc_id_to_idx = {doc["id"]: idx for idx, doc in enumerate(self.task_corpus)}

        total_queries = []
        total_docs = []
        doc_pairs_ids: list[tuple[str, str]] = []
        for query_id, corpus_ids in top_ranked.items():
            if query_id not in top_ranked:
                logger.warning(f"No pre-ranked documents found for query {query_id}")
                continue

            query_idx = query_id_to_idx[query_id]
            for corpus_id in corpus_ids:
                doc_pairs_ids.append((query_id, corpus_id))
                total_queries.append(queries[query_idx])
                total_docs.append(self.task_corpus[doc_id_to_idx[corpus_id]])

        queries_loader = create_dataloader(
            Dataset.from_list(total_queries),
            task_metadata,
            prompt_type=PromptType.document,
            batch_size=encode_kwargs.get("batch_size", 32),
        )
        corpus_loader = create_dataloader(
            Dataset.from_list(total_docs),
            task_metadata,
            prompt_type=PromptType.document,
            batch_size=encode_kwargs.get("batch_size", 32),
        )
        predictions = self.model.predict(
            inputs1=queries_loader,
            inputs2=corpus_loader,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
        )

        results = {qid: {} for qid in queries["id"]}
        for (query_id, corpus_id), score in zip(doc_pairs_ids, predictions):
            results[query_id][corpus_id] = float(score)

        return results
