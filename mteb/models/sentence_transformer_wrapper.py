from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from packaging.version import Version
from torch.utils.data import DataLoader

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
    return SentenceTransformerEncoderWrapper(
        model=model_name, revision=revision, **kwargs
    )


class SentenceTransformerEncoderWrapper(AbsEncoder):
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
        from mteb.models.get_model_meta import (
            _model_meta_from_sentence_transformers,
        )

        self.mteb_model_meta = _model_meta_from_sentence_transformers(self.model)

        built_in_prompts = getattr(self.model, "prompts", None)
        if built_in_prompts and not model_prompts:
            model_prompts = built_in_prompts
        elif model_prompts and built_in_prompts:
            logger.warning(f"Model prompts will be overwritten with {model_prompts}")
            self.model.prompts = model_prompts

        self.model_prompts, invalid_prompts = self.validate_task_to_prompt_name(
            model_prompts, raise_for_invalid_keys=False
        )

        if invalid_prompts:
            invalid_prompts = "\n".join(invalid_prompts)
            logger.warning(
                f"Some prompts are not in the expected format and will be ignored. Problems:\n\n{invalid_prompts}"
            )

        if (
            self.model_prompts
            and len(self.model_prompts) <= 2
            and (
                PromptType.query.value not in self.model_prompts
                or PromptType.document.value not in self.model_prompts
            )
        ):
            logger.warning(
                "SentenceTransformers that use prompts most often need to be configured with at least 'query' and"
                f" 'document' prompts to ensure optimal performance. Received {self.model_prompts}"
            )

        if hasattr(self.model, "similarity") and callable(self.model.similarity):
            self.similarity = self.model.similarity

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

        HAS_QUERY_ENCODE = (
            Version(st_version).release
            >= Version(SENTENCE_TRANSFORMERS_QUERY_ENCODE_VERSION).release
        )

        _inputs = [text for batch in inputs for text in batch["text"]]

        prompt = None
        prompt_name = None
        if self.model_prompts is not None:
            prompt_name = self.get_prompt_name(task_metadata, prompt_type)
            prompt = self.model_prompts.get(prompt_name, None)
        if prompt_name:
            logger.info(
                f"Using {prompt_name=} for task={task_metadata.name} {prompt_type=} with {prompt=}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_metadata.name} {prompt_type=}"
            )
        logger.info(f"Encoding {len(_inputs)} sentences.")

        if prompt_type and HAS_QUERY_ENCODE:
            if prompt_type == PromptType.query:
                encode_function = self.model.encode_query
            elif prompt_type == PromptType.document:
                encode_function = self.model.encode_corpus
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
            prompt = self.model_prompts.get(prompt_name, None)
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
            batch_column = list(batch.keys())[0]
            batched_input = [dict() for _ in range(len(batch[batch_column]))]

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
    def __init__(
        self,
        model: CrossEncoder | str,
        revision: str | None = None,
        **kwargs,
    ) -> None:
        from sentence_transformers import CrossEncoder

        from mteb.models.get_model_meta import _model_meta_from_cross_encoder

        if isinstance(model, CrossEncoder):
            self.model = model
        elif isinstance(model, str):
            self.model = CrossEncoder(model, revision=revision, **kwargs)

        self.mteb_model_meta = _model_meta_from_cross_encoder(self.model)

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
