from __future__ import annotations

import heapq
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, cast, get_args

import torch
from torch.utils.data import DataLoader

import mteb
from mteb.abstasks.task_metadata import TaskMetadata, TaskType
from mteb.create_dataloaders import (
    create_dataloader_for_retrieval_corpus,
    create_text_queries_dataloader,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.similarity_functions import (
    cos_sim,
    dot_score,
    max_sim,
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_max_sim,
)
from mteb.types import (
    Array,
    BatchedInput,
    CorpusDatasetType,
    InstructionDatasetType,
    PromptType,
    QueryDatasetType,
    RetrievalOutput,
    TopRankedDocumentsType,
)

logger = logging.getLogger(__name__)


class AbsEncoder(ABC):
    """Base class to indicate that this is a wrapper for a model.
    Also contains some utility functions for wrappers for working with prompts and instructions.
    """

    model: Any
    mteb_model_meta: ModelMeta | None = None
    model_prompts: dict[str, str] | None = None
    instruction_template: str | Callable[[str, PromptType], str] | None = None
    prompts_dict: dict[str, str] | None = None
    task_corpus: CorpusDatasetType | None = None

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        if self.mteb_model_meta is None or (
            self.mteb_model_meta is not None
            and self.mteb_model_meta.similarity_fn_name is None
        ):
            if (
                hasattr(self, "model")
                and hasattr(self.model, "similarity")
                and callable(self.model.similarity)
            ):
                arr = self.model.similarity(embeddings1, embeddings2)
                # We assume that the model returns an Array-like object:
                arr = cast(Array, arr)
                return arr
            return cos_sim(embeddings1, embeddings2)
        if self.mteb_model_meta.similarity_fn_name is ScoringFunction.COSINE:
            return cos_sim(embeddings1, embeddings2)
        elif self.mteb_model_meta.similarity_fn_name is ScoringFunction.DOT_PRODUCT:
            return dot_score(embeddings1, embeddings2)
        elif self.mteb_model_meta.similarity_fn_name is ScoringFunction.MAX_SIM:
            return max_sim(embeddings1, embeddings2)
        raise ValueError("Similarity function not specified.")

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        if self.mteb_model_meta is None or (
            self.mteb_model_meta is not None
            and self.mteb_model_meta.similarity_fn_name is None
        ):
            if (
                hasattr(self, "model")
                and hasattr(self.model, "similarity_pairwise")
                and callable(self.model.similarity_pairwise)
            ):
                arr = self.model.similarity_pairwise(embeddings1, embeddings2)
                # We assume that the model returns an Array-like object:
                arr = cast(Array, arr)
                return arr
            return pairwise_cos_sim(embeddings1, embeddings2)
        if self.mteb_model_meta.similarity_fn_name is ScoringFunction.COSINE:
            return pairwise_cos_sim(embeddings1, embeddings2)
        elif self.mteb_model_meta.similarity_fn_name is ScoringFunction.DOT_PRODUCT:
            return pairwise_dot_score(embeddings1, embeddings2)
        elif self.mteb_model_meta.similarity_fn_name is ScoringFunction.MAX_SIM:
            return pairwise_max_sim(embeddings1, embeddings2)
        raise ValueError("Similarity function not specified.")

    @abstractmethod
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
        """Encodes the given sentences using the encoder.

        Args:
            inputs: Batch of inputs to encode.
            task_metadata: The metadata of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
                The order of priorities for prompt selection are:
                    1. Composed prompt of task name + prompt type (query or passage)
                    2. Specific task prompt
                    3. Composed prompt of task type + prompt type (query or passage)
                    4. Specific task type prompt
                    5. Specific prompt type (query or passage)
            hf_split: Split of current task
            hf_subset: Subset of current task
            prompt_type: The name type of prompt. (query or passage)
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded input in a numpy array or torch tensor of the shape (Number of sentences) x (Embedding dimension).
        """
        raise NotImplementedError(
            "The encode method must be implemented in the subclass."
        )

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
        """Predicts relevance scores for pairs of inputs.

        Args:
            inputs1: First Dataloader of inputs to encode.
            inputs2: Second Dataloader of inputs to encode.
            task_metadata: Metadata of the current task.
            hf_split: Split of current task, allows to know some additional information about current split.
                E.g. Current language
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            prompt_type: The name type of prompt. (query or passage)
            **kwargs: Additional arguments to pass to the cross-encoder.

        Returns:
            The predicted relevance scores for each inputs pair.
        """
        embeddings1 = self.encode(
            inputs1,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.query,
            **kwargs,
        )
        embeddings2 = self.encode(
            inputs2,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.passage,
            **kwargs,
        )
        return self.similarity_pairwise(embeddings1, embeddings2)

    def get_prompt_name(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str | None:
        """A wrapper function around the model.encode method that handles the prompt_name argument and standardizes the output to a numpy array.
        The order of priorities for prompt selection are:
            1. Composed prompt of task name + prompt type (query or passage)
            2. Specific task prompt
            3. Composed prompt of task type + prompt type (query or passage)
            4. Specific task type prompt
            5. Specific prompt type (query or passage)


        Args:
            task_metadata: The task name to use for building the encoding prompt
            prompt_type: The prompt type (e.g. "query" | "passage") to use for building the encoding prompt
        """
        if self.model_prompts is None:
            return None
        task_type = task_metadata.type
        prompt_type_value = prompt_type.value if prompt_type else None
        task_name = task_metadata.name

        if (
            task_name
            and prompt_type
            and f"{task_name}-{prompt_type_value}" in self.model_prompts
        ):
            return f"{task_name}-{prompt_type_value}"
        if task_name and task_name in self.model_prompts:
            return task_name
        if (
            task_type
            and prompt_type
            and f"{task_type}-{prompt_type_value}" in self.model_prompts
        ):
            return f"{task_type}-{prompt_type_value}"
        if task_type and task_type in self.model_prompts:
            return task_type
        if prompt_type and prompt_type_value in self.model_prompts:
            return prompt_type_value
        logger.info(
            "No combination of task name and prompt type was found in model prompts."
        )
        return None

    def get_prompt(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str | None:
        if not self.model_prompts:
            return None
        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        return self.model_prompts.get(prompt_name)  # type: ignore

    def validate_task_to_prompt_name(self) -> None:
        """Validate the task name and prompt type against the model prompts.

        All keys in model_prompts should be valid task names, prompt types or the combination of both.
        """
        if self.model_prompts is None:
            return
        task_types = get_args(TaskType)
        prompt_types = [e.value for e in PromptType]
        for task_name in self.model_prompts:
            if "-" in task_name:
                task_name, prompt_type = task_name.split("-")
                if prompt_type not in prompt_types:
                    msg = f"Prompt type {prompt_type} is not valid. Valid prompt types are {prompt_types}"
                    logger.warning(msg)
                    raise KeyError(msg)
            if task_name not in task_types and task_name not in prompt_types:
                task = mteb.get_task(task_name=task_name)
                if not task:
                    msg = f"Task name {task_name} is not valid. Valid task names are task types [{task_types}], prompt types [{prompt_types}] and task names"
                    logger.warning(msg)
                    raise KeyError(msg)

    def get_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str:
        """Get the instruction/prompt to be used for encoding sentences."""
        prompt = task_metadata.prompt
        if self.prompts_dict and task_metadata.name in self.prompts_dict:
            prompt = self.prompts_dict[task_metadata.name]

        if isinstance(prompt, dict) and prompt_type:
            if prompt.get(prompt_type.value):
                return prompt[prompt_type.value]
            logger.warning(
                f"Prompt type '{prompt_type}' not found in task metadata for task '{task_metadata.name}'."
            )
            return ""

        if prompt:
            return prompt

        abstask = mteb.get_task(task_name=task_metadata.name)
        return abstask.abstask_prompt

    def format_instruction(
        self, instruction: str, prompt_type: PromptType | None = None
    ) -> str:
        if self.instruction_template is None:
            raise ValueError(
                "Attempting to format an instruction without an instruction template."
            )
        if isinstance(self.instruction_template, str):
            if "{instruction}" not in self.instruction_template:
                raise ValueError(
                    "Instruction template must contain the string '{instruction}'."
                )
            return self.instruction_template.format(instruction=instruction)
        return self.instruction_template(instruction, prompt_type)

    def get_task_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str:
        instruction = self.get_instruction(task_metadata, prompt_type)
        if self.instruction_template and len(instruction) > 0:
            return self.format_instruction(instruction)
        return instruction

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
        self.task_corpus_dataloader = create_dataloader_for_retrieval_corpus(
            corpus, batch_size=encode_kwargs.get("batch_size", 32)
        )

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: dict[str, Any],
        instructions: InstructionDatasetType | None = None,
        top_ranked: TopRankedDocumentsType | None = None,
    ) -> RetrievalOutput:
        """Search the corpus for the given queries.

        Args:
            queries: Queries to find
            task_metadata: Task metadata
            hf_split: split of the dataset
            hf_subset: subset of the dataset
            instructions: Optional instructions to use for the search.
            top_ranked: Top-ranked documents for each query, mapping query IDs to a list of document IDs.
                Passed only from Reranking tasks.
            top_k: Optional number of top documents to return for each query.
            encode_kwargs: Additional arguments to pass to the encoder during indexing.

        Returns:
            Dictionary with query IDs as keys with dict as values, where each value is a mapping of document IDs to their relevance scores.
        """
        if self.task_corpus_dataloader is None:
            raise ValueError("Corpus must be indexed before searching.")
        if not isinstance(self.task_corpus, DataLoader):
            raise TypeError(
                "Corpus must be a DataLoader. Please use the index method to index the corpus."
            )

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
            result = self._rerank_documents(
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
            result = self._full_corpus_search(
                query_ids=query_ids,
                query_embeddings=query_embeddings,
                task_metadata=task_metadata,
                hf_subset=hf_subset,
                hf_split=hf_split,
                top_k=top_k,
                encode_kwargs=encode_kwargs,
            )

        # Reset the task corpus dataloader to None to free up memory
        self.task_corpus_dataloader = None
        return result

    def _full_corpus_search(
        self,
        query_ids: list[str],
        query_embeddings: Array,
        task_metadata: TaskMetadata,
        hf_subset: str,
        hf_split: str,
        top_k: int,
        encode_kwargs: dict[str, Any],
    ) -> RetrievalOutput:
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        result_heaps = {qid: [] for qid in query_ids}
        for batch_num, corpus_batch in enumerate(self.task_corpus_dataloader):
            logger.info(
                f"Encoding Batch {batch_num + 1}/{len(self.task_corpus_dataloader)}..."
            )
            # Encode chunk of corpus
            sub_corpus_embeddings = self.model.encode(
                corpus_batch,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=PromptType.passage,
                **encode_kwargs,
            )

            # Compute similarities using either cosine-similarity or dot product
            logging.info("Computing Similarities...")
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
            )

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr].cpu().tolist(),
                    cos_scores_top_k_values[query_itr].cpu().tolist(),
                ):
                    corpus_id = corpus_batch[sub_corpus_id]["id"]
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
    ) -> RetrievalOutput:
        """Rerank documents based on pre-ranked documents."""
        result_heaps = {qid: [] for qid in query_ids}
        doc_id_to_idx = {
            doc["id"]: idx
            for idx, doc in enumerate(self.task_corpus_dataloader.dataset)
        }

        all_doc_embeddings = self.model.encode(
            self.task_corpus_dataloader,
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
