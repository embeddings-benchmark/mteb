from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from packaging.version import Version
from torch.utils.data import DataLoader

from mteb._log_once import LogOnce
from mteb.models import ModelMeta
from mteb.types import Array, BatchedInput, PromptType

from .abs_encoder import AbsEncoder

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)

SENTENCE_TRANSFORMERS_QUERY_ENCODE_VERSION = "5.0.0"


def sentence_transformers_loader(
    model_name: str, revision: str | None = None, **kwargs
) -> SentenceTransformerEncoderWrapper:
    """Loads a SentenceTransformer model and wraps it in a SentenceTransformerEncoderWrapper.

    Args:
        model_name: The name of the SentenceTransformer model to load.
        revision: The revision of the model to load.
        kwargs: Additional arguments to pass to the SentenceTransformer model.
    """
    return SentenceTransformerEncoderWrapper(
        model=model_name, revision=revision, **kwargs
    )


class SentenceTransformerEncoderWrapper(AbsEncoder):
    """Multimodal wrapper for SentenceTransformer models."""

    mteb_model_meta: ModelMeta

    def __init__(
        self,
        model: str | SentenceTransformer,
        revision: str | None = None,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        """Wrapper for SentenceTransformer models.

        Args:
            model: The SentenceTransformer model to use. Can be a string (model name), a SentenceTransformer model, or a CrossEncoder model.
            revision: The revision of the model to use.
            model_prompts: A dictionary mapping task names to prompt names.
                First priority is given to the composed prompt of task name + prompt type (query or passage), then to the specific task prompt,
                then to the composed prompt of task type + prompt type, then to the specific task type prompt,
                and finally to the specific prompt type.
            **kwargs: Additional arguments to pass to the SentenceTransformer model.
        """
        from sentence_transformers import SentenceTransformer

        if isinstance(model, str):
            self.model = SentenceTransformer(model, revision=revision, **kwargs)
        else:
            self.model = model

        self.mteb_model_meta = ModelMeta.from_sentence_transformer_model(self.model)

        built_in_prompts = getattr(self.model, "prompts", None)
        if built_in_prompts and not model_prompts:
            model_prompts = built_in_prompts
        elif model_prompts and built_in_prompts:
            msg = f"Model prompts specified, these will overwrite the default model prompts. Current prompts will be:\n {model_prompts}"
            logger.warning(msg)
            warnings.warn(msg)
            self.model.prompts = model_prompts

        self.model_prompts, invalid_prompts = self.validate_task_to_prompt_name(
            model_prompts, raise_for_invalid_keys=False
        )

        if invalid_prompts:
            invalid_prompts = "\n".join(invalid_prompts)
            msg = f"Some prompts are not in the expected format and will be ignored. Problems:\n\n{invalid_prompts}"
            logger.warning(msg)
            warnings.warn(msg)

        if (
            self.model_prompts
            and len(self.model_prompts) <= 2
            and (
                PromptType.query.value not in self.model_prompts
                or PromptType.document.value not in self.model_prompts
            )
        ):
            msg = f"SentenceTransformers that use prompts most often need to be configured with at least 'query' and 'document' prompts to ensure optimal performance. Received {self.model_prompts}"
            logger.warning(msg)
            warnings.warn(msg)

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Compute the similarity between two collections of embeddings."""
        if hasattr(self.model, "similarity") and callable(self.model.similarity):
            return self.model.similarity(embeddings1, embeddings2)
        return super().similarity(embeddings1, embeddings2)

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
            inputs: The sentences to encode.
            task_metadata: The metadata of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            prompt_type: The name type of prompt. (query or passage)
            hf_split: Split of current task
            hf_subset: Subset of current task
            **kwargs: Additional arguments to pass to the encoder.

            The order of priorities for prompt selection are:
                1. Composed prompt of task name + prompt type (query or passage)
                2. Specific task prompt
                3. Composed prompt of task type + prompt type (query or passage)
                4. Specific task type prompt
                5. Specific prompt type (query or passage)

        Returns:
            The encoded sentences.
        """
        from sentence_transformers import __version__ as st_version

        has_query_encode = (
            Version(st_version).release
            >= Version(SENTENCE_TRANSFORMERS_QUERY_ENCODE_VERSION).release
        )

        _inputs = [text for batch in inputs for text in batch["text"]]

        prompt = None
        prompt_name = None
        if self.model_prompts is not None:
            prompt_name = self.get_prompt_name(task_metadata, prompt_type)
            prompt = self.model_prompts.get(prompt_name, None)  # type: ignore[arg-type]
        if prompt_name:
            prompt_log = f"Using {prompt_name=} for task={task_metadata.name} {prompt_type=} with {prompt=}"
        else:
            prompt_log = (
                f"No model prompts found for task={task_metadata.name} {prompt_type=}"
            )

        LogOnce(logger).info(prompt_log)
        logger.debug(f"Encoding {len(_inputs)} sentences.")

        if prompt_type and has_query_encode:
            if prompt_type == PromptType.query:
                encode_function = self.model.encode_query
            elif prompt_type == PromptType.document:
                encode_function = self.model.encode_document
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")
        else:
            encode_function = self.model.encode

        embeddings = encode_function(
            _inputs,
            prompt=prompt,
            **kwargs,
        )
        if isinstance(embeddings, torch.Tensor):
            # ensure everything is on CPU and is float
            embeddings = embeddings.cpu().detach().float()
        return embeddings


class SentenceTransformerMultimodalEncoderWrapper(SentenceTransformerEncoderWrapper):
    """Wrapper for multimodal SentenceTransformer models."""

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
            inputs: The sentences to encode.
            task_metadata: The metadata of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            prompt_type: The name type of prompt. (query or passage)
            hf_split: Split of current task
            hf_subset: Subset of current task
            **kwargs: Additional arguments to pass to the encoder.

            The order of priorities for prompt selection are:
                1. Composed prompt of task name + prompt type (query or passage)
                2. Specific task prompt
                3. Composed prompt of task type + prompt type (query or passage)
                4. Specific task type prompt
                5. Specific prompt type (query or passage)


        Returns:
            The encoded sentences.
        """
        prompt = None
        prompt_name = None
        if self.model_prompts is not None:
            prompt_name = self.get_prompt_name(task_metadata, prompt_type)
            prompt = self.model_prompts.get(prompt_name, None)  # type: ignore[arg-type]
        if prompt_name:
            logger.info(
                f"Using {prompt_name=} for task={task_metadata.name} {prompt_type=} with {prompt=}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_metadata.name} {prompt_type=}"
            )

        all_embeddings = []
        for batch in inputs:
            batch_column = next(iter(batch.keys()))
            batched_input: list[dict[str, Any]] = [
                dict() for _ in range(len(batch[batch_column]))
            ]

            # transform from {"text": [text1, text2], "image": [image1, image2]} to
            # [{"text": text1, "image": image1}, {"text": text2, "image": image2}]
            for key, values in batch.items():
                for i, value in enumerate(values):
                    batched_input[i][key] = value

            embeddings = self.model.encode(
                batched_input,
                prompt=prompt,
                **kwargs,
            )
            if isinstance(embeddings, torch.Tensor):
                # ensure everything is on CPU and is float
                embeddings = embeddings.cpu().detach().float()
            all_embeddings.append(embeddings)
        return np.stack(all_embeddings)


class CrossEncoderWrapper:
    """Wrapper for CrossEncoder models."""

    def __init__(
        self,
        model: CrossEncoder | str,
        revision: str | None = None,
        **kwargs,
    ) -> None:
        from sentence_transformers import CrossEncoder

        if isinstance(model, CrossEncoder):
            self.model = model
        elif isinstance(model, str):
            self.model = CrossEncoder(model, revision=revision, **kwargs)

        self.mteb_model_meta = ModelMeta.from_cross_encoder(self.model)

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
        """Predicts relevance scores for pairs of inputs. Note that, unlike the encoder, the cross-encoder can compare across inputs.

        Args:
            inputs1: First Dataloader of inputs to encode. For reranking tasks, these are queries (for text only tasks `QueryDatasetType`).
            inputs2: Second Dataloader of inputs to encode. For reranking, these are documents (for text only tasks `RetrievalOutputType`).
            task_metadata: Metadata of the current task.
            hf_split: Split of current task, allows to know some additional information about current split.
                E.g. Current language
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            prompt_type: The name type of prompt. (query or passage)
            **kwargs: Additional arguments to pass to the cross-encoder.

        Returns:
            The predicted relevance scores for each inputs pair.
        """
        all_queries_with_instructions = [
            text for batch in inputs1 for text in batch["text"]
        ]
        all_corpus_with_instructions = [
            text for batch in inputs2 for text in batch["text"]
        ]

        return self.model.predict(
            list(zip(all_queries_with_instructions, all_corpus_with_instructions)),
            **kwargs,
        )
