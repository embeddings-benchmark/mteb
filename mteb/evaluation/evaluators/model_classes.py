from __future__ import annotations

import heapq
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.encoder_interface import Encoder

from ...create_dataloaders import (
    create_dataloader_for_queries,
    create_dataloader_for_queries_conversation,
    create_dataloader_for_retrieval_corpus,
)
from ...types import Array, BatchedInput, PromptType
from .utils import download

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/search/dense/exact_search.py#L12
class DenseRetrievalExactSearch:
    def __init__(
        self,
        model: Encoder,
        encode_kwargs: dict[str, Any] = {},
        corpus_chunk_size: int = 50000,
        previous_results: str | Path | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.encode_kwargs = encode_kwargs.copy()

        if "show_progress_bar" not in encode_kwargs:
            self.encode_kwargs["show_progress_bar"] = True

        self.corpus_chunk_size = corpus_chunk_size
        if isinstance(previous_results, Path):
            self.previous_results = str(previous_results)
        else:
            self.previous_results = previous_results
        self.batch_size = self.encode_kwargs.get("batch_size", 32)
        self.show_progress_bar = self.encode_kwargs.get("show_progress_bar")
        self.results = {}

        if self.previous_results is not None:
            self.previous_results = self.load_results_file()

        if hasattr(self.model, "predict"):
            # load the predict instance from the CrossEncoder
            # custom functions can be used by extending the DenseRetrievalExactSearch class
            self.predict = self.model.predict

        self.combine_query_and_instruction = (
            lambda query, instruction: f"{query.strip()} {instruction}".strip()
        )

        if hasattr(self.model, "combine_query_and_instruction"):
            self.combine_query_and_instruction = (
                self.model.combine_query_and_instruction
            )

    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        instructions: dict[str, str] | None = None,
        request_qid: str | None = None,
        return_sorted: bool = False,
        top_ranked: dict[str, list[str]] | None = None,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """Perform semantic search (retrieval or reranking).

        Args:
            corpus: Dictionary mapping corpus IDs to document dictionaries
            queries: Dictionary mapping query IDs to query strings
            top_k: Number of top results to return
            task_metadata: Name of the task
            hf_split: Name of split
            hf_subset: Name of subset
            instructions: Optional instructions to append to queries
            request_qid: Optional request query ID
            return_sorted: Whether to return results sorted
            top_ranked: Optional dict mapping query IDs to lists of pre-ranked corpus IDs
            **kwargs: Additional keyword arguments passed to the underlying model
        """
        logger.info("Encoding Queries.")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        query_ids, query_list = zip(*queries.items())

        # Prepare query-instruction pairs if instructions are provided
        if instructions:
            query_instruction_pairs = [
                (query_list[q_idx], instructions[qid])
                for q_idx, qid in enumerate(query_ids)
            ]
        else:
            query_instruction_pairs = [(query, None) for query in query_list]

        # Create mapping of unique query-instruction pairs to their indices
        unique_pairs = []
        pair_to_idx = {}
        pair_idx_mapping = []

        for pair in query_instruction_pairs:
            query, instruction = pair
            # Create a hashable key for the pair
            query_key = tuple(query) if isinstance(query, list) else query
            pair_key = (query_key, instruction)

            if pair_key not in pair_to_idx:
                pair_to_idx[pair_key] = len(unique_pairs)
                unique_pairs.append(pair)
            pair_idx_mapping.append(pair_to_idx[pair_key])

        # Extract unique queries and their corresponding instructions
        unique_queries = [pair[0] for pair in unique_pairs]
        unique_instructions = (
            [pair[1] for pair in unique_pairs] if instructions else None
        )

        # Encode only unique queries using the dataloader
        if isinstance(query_list[0], list):
            # For conversations, still use the original encode_conversations method
            unique_query_dataloader = create_dataloader_for_queries_conversation(
                queries=unique_queries,
                instructions=unique_instructions,
                combine_query_and_instruction=self.combine_query_and_instruction
                if instructions
                else None,
            )
        else:
            # Create dataloader for text queries with their matched instructions
            unique_query_dataloader = create_dataloader_for_queries(
                queries=unique_queries,
                instructions=unique_instructions,
                combine_query_and_instruction=self.combine_query_and_instruction
                if instructions
                else None,
            )

        # Encode queries using the model with the dataloader
        unique_query_embeddings = self.model.encode(
            unique_query_dataloader,
            task_metadata=task_metadata,
            hf_subset=hf_subset,
            hf_split=hf_split,
            prompt_type=PromptType.query,
            **self.encode_kwargs,
        )
        query_embeddings = unique_query_embeddings[pair_idx_mapping]

        if top_ranked is not None:
            logger.info("Performing reranking on pre-ranked documents...")
            result_heaps = self._rerank_documents(
                query_ids=query_ids,
                query_embeddings=query_embeddings,
                corpus=corpus,
                top_ranked=top_ranked,
                top_k=top_k,
                task_metadata=task_metadata,
                hf_subset=hf_subset,
                hf_split=hf_split,
                request_qid=request_qid,
                return_sorted=return_sorted,
            )
        else:
            logger.info("Performing full corpus search...")
            result_heaps = self._full_corpus_search(
                query_ids=query_ids,
                query_embeddings=query_embeddings,
                corpus=corpus,
                top_k=top_k,
                task_metadata=task_metadata,
                hf_subset=hf_subset,
                hf_split=hf_split,
                request_qid=request_qid,
                return_sorted=return_sorted,
            )

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results

    def _rerank_documents(
        self,
        query_ids: list[str],
        query_embeddings: np.ndarray,
        corpus: dict[str, dict[str, str]],
        top_ranked: dict[str, list[str]],
        top_k: int,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        request_qid: str | None = None,
        return_sorted: bool = False,
    ) -> dict[str, list[tuple[float, str]]]:
        """Rerank documents for each query using top_ranked."""
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Move query embeddings to appropriate device
        query_embeddings = torch.as_tensor(query_embeddings).to(device)

        result_heaps = {qid: [] for qid in query_ids}

        # Get unique document IDs across all queries
        unique_doc_ids = list(
            {
                doc_id
                for qid in query_ids
                if qid in top_ranked
                for doc_id in top_ranked[qid]
            }
        )

        # Create mapping from unique doc IDs to their index in the embedding matrix
        doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(unique_doc_ids)}

        # Encode unique documents only once
        unique_docs = [corpus[doc_id] for doc_id in unique_doc_ids]
        all_doc_embeddings = self.model.encode(
            create_dataloader_for_retrieval_corpus(unique_docs),
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.passage,
            request_qid=request_qid,
            **self.encode_kwargs,
        )

        # Let's make sure we don't get the warnings for the tokenizer here via torch.compile
        if hasattr(torch, "compile"):
            os.environ["TOKENIZERS_PARALLELISM"] = "false"  # we don't need it anymore

        # Process each query
        for query_idx, query_id in enumerate(tqdm.tqdm(query_ids)):
            if query_id not in top_ranked:
                logger.warning(f"No pre-ranked documents found for query {query_id}")
                continue

            ranked_ids = top_ranked[query_id]
            doc_indices = torch.tensor([doc_id_to_idx[doc_id] for doc_id in ranked_ids])
            query_doc_embeddings = torch.as_tensor(all_doc_embeddings[doc_indices]).to(
                device
            )

            # Ensure query embedding is on the correct device and has correct shape
            query_embedding = query_embeddings[query_idx].unsqueeze(0)

            with torch.inference_mode():
                scores = self.model.similarity(
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
                sorted=return_sorted,
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

        # Clear CUDA cache after processing
        if device.type == "cuda":
            del query_doc_embeddings
            torch.cuda.empty_cache()

        return result_heaps

    def _full_corpus_search(
        self,
        query_ids: list[str],
        query_embeddings: np.ndarray,
        corpus: dict[str, dict[str, str]],
        top_k: int,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        request_qid: str | None = None,
        return_sorted: bool = False,
    ) -> dict[str, list[tuple[float, str]]]:
        """Perform full corpus search using batched processing."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        logger.info("Sorting Corpus by document length (Longest first)...")
        corpus_ids = sorted(corpus, reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        itr = range(0, len(corpus), self.corpus_chunk_size)

        result_heaps = {qid: [] for qid in query_ids}
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info(f"Encoding Batch {batch_num + 1}/{len(itr)}...")
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))
            # Encode chunk of corpus
            sub_corpus_embeddings = self.model.encode(
                create_dataloader_for_retrieval_corpus(
                    corpus[corpus_start_idx:corpus_end_idx]
                ),  # type: ignore
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=PromptType.passage,
                request_qid=request_qid,
                **self.encode_kwargs,
            )

            # Compute similarities using either cosine-similarity or dot product
            logging.info("Computing Similarities...")
            query_embeddings = torch.as_tensor(query_embeddings).to(device)
            sub_corpus_embeddings = torch.as_tensor(sub_corpus_embeddings).to(device)

            with torch.inference_mode():
                scores = self.model.similarity(query_embeddings, sub_corpus_embeddings)

            # get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                scores,
                min(
                    top_k + 1,
                    len(scores[1]) if len(scores) > 1 else len(scores[-1]),
                ),
                dim=1,
                largest=True,
                sorted=return_sorted,
            )

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr].cpu().tolist(),
                    cos_scores_top_k_values[query_itr].cpu().tolist(),
                ):
                    corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                    if len(result_heaps[query_id]) < top_k:
                        # push item on the heap
                        heapq.heappush(result_heaps[query_id], (score, corpus_id))
                    else:
                        # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                        heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        return result_heaps

    def load_results_file(self):
        # load the first stage results from file in format {qid: {doc_id: score}}
        if "https://" in self.previous_results:
            # download the file
            if not os.path.exists(self.previous_results):
                url_descriptor = self.previous_results.split("https://")[-1].replace(
                    "/", "--"
                )
                dest_file = os.path.join(
                    "results", f"cached_predictions--{url_descriptor}"
                )
                os.makedirs(os.path.dirname(os.path.abspath(dest_file)), exist_ok=True)
                download(self.previous_results, dest_file)
                logger.info(
                    f"Downloaded the previous results at {self.previous_results} to {dest_file}"
                )
            self.previous_results = dest_file

        with open(self.previous_results) as f:
            previous_results = json.load(f)

        if not isinstance(previous_results, dict) or not isinstance(
            previous_results[list(previous_results.keys())[0]], dict
        ):
            raise ValueError(
                "Previous results file must be in format {qid: {doc_id: score}}. Got "
                + type(previous_results)
            )

        return previous_results

    def search_cross_encoder(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str | list[str]],
        top_k: int,
        hf_split: str,
        hf_subset: str,
        task_metadata: TaskMetadata,
        instructions: dict[str, str] | None = None,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """This function provides support for reranker (or cross-encoder) models that encoder query and document at the same time (typically with attention).
        Some notable examples include MonoBERT, MonoT5, RankLlama, etc.
        Note: you must provide the path to the results to rerank to the __init__ function as `previous_results` or else rerank all documents in the corpus
        """
        pairs = []  # create the pairs for reranking
        for qid in tqdm.tqdm(queries.keys()):
            if self.previous_results is None:
                # try to use all of them
                logging.info(
                    f"previous_results is None. Using all the documents to rerank: {len(corpus)}"
                )
                q_results = dict.fromkeys(corpus.keys(), 0.0)
            else:
                q_results = self.previous_results[qid]
            # take the top-k only
            q_results_sorted = dict(
                sorted(q_results.items(), key=lambda item: item[1], reverse=True)
            )
            top_n = [k for k, v in list(q_results_sorted.items())[:top_k]]
            query = queries[qid]
            query = (
                self.convert_conv_history_to_query(self.model, [query])[0]
                if isinstance(query, list)
                else query
            )
            for doc_id in top_n:
                pairs.append(
                    (
                        query,
                        corpus[doc_id],
                        instructions[qid] if instructions is not None else None,
                        qid,
                        doc_id,
                    )
                )

        logger.info(f"Reranking the top {top_k} in batches... This might take a while!")

        results = {qid: {} for qid in queries.keys()}
        for batch_num, corpus_start_idx in enumerate(
            tqdm.tqdm(
                range(0, len(pairs), self.batch_size),
                leave=False,
                # disable=not self.show_progress_bar,
            )
        ):
            corpus_end_idx = min(corpus_start_idx + self.batch_size, len(pairs))
            cur_batch = pairs[corpus_start_idx:corpus_end_idx]

            (
                queries_in_pair,
                corpus_in_pair,
                instructions_in_pair,
                query_ids,
                corpus_ids,
            ) = zip(*cur_batch)

            queries_dataset = create_dataloader_for_queries(
                queries=queries_in_pair,
                instructions=instructions_in_pair,
                combine_query_and_instruction=self.combine_query_and_instruction
                if hasattr(self, "combine_query_and_instruction")
                else None,
            )

            corpus_dataset = create_dataloader_for_retrieval_corpus(
                corpus_in_pair,
            )

            if not (
                len(queries_in_pair) == len(corpus_in_pair) == len(instructions_in_pair)
            ):
                raise ValueError(
                    "Queries, corpus, and instructions must be the same length"
                )

            # cross-encoders may use the instructions in a unique way
            # due to the many ways of combining query+instruct+doc, so let them decide
            scores = self.model.predict(
                queries_dataset,
                corpus_dataset,
                hf_split=hf_split,
                hf_subset=hf_subset,
                task_metadata=task_metadata,
                **self.encode_kwargs,
            )

            for i, score in enumerate(scores):
                results[query_ids[i]][corpus_ids[i]] = float(score)

        return results

    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        raise NotImplementedError(
            "You must implement a predict method for your reranker model"
        )


def is_cross_encoder_compatible(model) -> bool:
    model_attr = getattr(model, "model", None)
    op = None
    # if we don't have `model_meta` then check if base model has `predict`
    if model_attr is not None:
        op = getattr(model_attr, "predict", None)
    return (
        model.mteb_model_meta.is_cross_encoder
        if hasattr(model, "mteb_model_meta") and model.mteb_model_meta is not None
        else callable(op)
    )
