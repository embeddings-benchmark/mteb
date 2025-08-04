from __future__ import annotations

import heapq
import logging
from typing import Any

import torch

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.create_dataloaders import (
    create_dataloader_for_retrieval_corpus,
    create_text_queries_dataloader,
)
from mteb.types import (
    Array,
    CorpusDatasetType,
    PromptType,
    QueryDatasetType,
    RetrievalOutputType,
    TopRankedDocumentsType,
)

from .encoder_interface import Encoder

logger = logging.getLogger(__name__)


class AbsEncoderSearch(Encoder):
    corpus_chunk_size = 50_000
    task_corpus: CorpusDatasetType | None = None

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
        # to make more efficient corpus encoding, they will be encoded in search method
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

        queries_dataloader = create_text_queries_dataloader(
            queries, batch_size=encode_kwargs.get("batch_size", 32)
        )

        query_embeddings = self.encode(
            queries_dataloader,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.query,
            **encode_kwargs,
        )
        query_ids = queries["id"]

        if top_ranked is not None:
            logger.info("Performing reranking on pre-ranked documents...")
            result_heaps = self._rerank_documents(
                query_ids=query_ids,
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
                query_ids=query_ids,
                query_embeddings=query_embeddings,
                task_metadata=task_metadata,
                hf_subset=hf_subset,
                hf_split=hf_split,
                top_k=top_k,
                encode_kwargs=encode_kwargs,
            )

        # Reset the task corpus dataloader to None to free up memory
        self.task_corpus = None

        results = {qid: {} for qid in query_ids}
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                results[qid][corpus_id] = score

        return results

    def _full_corpus_search(
        self,
        query_ids: list[str],
        query_embeddings: Array,
        task_metadata: TaskMetadata,
        hf_subset: str,
        hf_split: str,
        top_k: int,
        encode_kwargs: dict[str, Any],
    ) -> dict[str, list[tuple[float, str]]]:
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        itr = range(0, len(self.task_corpus), self.corpus_chunk_size)

        result_heaps = {qid: [] for qid in query_ids}
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info(f"Encoding Batch {batch_num + 1}/{len(itr)}...")
            corpus_end_idx = min(
                corpus_start_idx + self.corpus_chunk_size, len(self.task_corpus)
            )
            sub_corpus = self.task_corpus.select(
                range(corpus_start_idx, corpus_end_idx)
            )
            sub_corpus_embeddings = self.encode(
                create_dataloader_for_retrieval_corpus(
                    sub_corpus, batch_size=encode_kwargs.get("batch_size", 32)
                ),
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=PromptType.passage,
                **encode_kwargs,
            )

            # Compute similarities using either cosine-similarity or dot product
            logging.info("Computing Similarities...")
            scores = self.similarity(query_embeddings, sub_corpus_embeddings)

            # get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                scores,
                min(
                    top_k + 1,
                    len(scores[1]) if len(scores) > 1 else len(scores[-1]),
                ),
                dim=1,
                largest=True,
            )

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr].cpu().tolist(),
                    cos_scores_top_k_values[query_itr].cpu().tolist(),
                ):
                    corpus_id = sub_corpus[sub_corpus_id]["id"]
                    if len(result_heaps[query_id]) < top_k:
                        # push item on the heap
                        heapq.heappush(result_heaps[query_id], (score, corpus_id))
                    else:
                        # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                        heapq.heappushpop(result_heaps[query_id], (score, corpus_id))
        return result_heaps

    def _rerank_documents(
        self,
        query_ids: list[str],
        query_embeddings: Array,
        top_ranked: TopRankedDocumentsType,
        top_k: int,
        task_metadata: TaskMetadata,
        hf_subset: str,
        hf_split: str,
        encode_kwargs: dict[str, Any],
    ) -> dict[str, list[tuple[float, str]]]:
        """Rerank documents based on pre-ranked documents."""
        result_heaps = {qid: [] for qid in query_ids}
        doc_id_to_idx = {doc["id"]: idx for idx, doc in enumerate(self.task_corpus)}

        all_doc_embeddings = self.encode(
            create_dataloader_for_retrieval_corpus(
                self.task_corpus, batch_size=encode_kwargs.get("batch_size", 32)
            ),
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.passage,
            **encode_kwargs,
        )

        # Process each query
        for query_idx, query_id in enumerate(query_ids):
            if query_id not in top_ranked:
                logger.warning(f"No pre-ranked documents found for query {query_id}")
                continue

            ranked_ids = top_ranked[query_id]
            doc_indices = torch.tensor([doc_id_to_idx[doc_id] for doc_id in ranked_ids])
            query_doc_embeddings = torch.as_tensor(all_doc_embeddings[doc_indices])

            # Ensure query embedding is on the correct device and has correct shape
            query_embedding = torch.as_tensor(query_embeddings[query_idx]).unsqueeze(0)

            scores = self.similarity(
                query_embedding,
                query_doc_embeddings,
            )

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

            # Build result heap
            for doc_idx, score in zip(
                scores_top_k_idx[0].tolist(),
                scores_top_k_values[0].tolist(),
            ):
                corpus_id = ranked_ids[doc_idx]
                heapq.heappush(result_heaps[query_id], (score, corpus_id))

        return result_heaps
