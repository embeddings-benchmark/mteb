from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from torch.utils.data import DataLoader

from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, BatchedInput, PromptType

if TYPE_CHECKING:
    from mteb import Encoder, TaskMetadata

logger = logging.getLogger(__name__)


def sentence_transformers_loader(
    model_name: str, revision: str | None = None, **kwargs
) -> Encoder:
    return SentenceTransformerWrapper(model=model_name, revision=revision, **kwargs)


class SentenceTransformerWrapper(AbsEncoder):
    def __init__(
        self,
        model: str | SentenceTransformer | CrossEncoder,
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
        if isinstance(model, str):
            self.model = SentenceTransformer(model, revision=revision, **kwargs)
        else:
            self.model = model

        if (
            model_prompts is None
            and hasattr(self.model, "prompts")
            and len(self.model.prompts) > 0
        ):
            try:
                self.model_prompts = self.model.prompts
                self.validate_task_to_prompt_name()
            except KeyError:
                logger.warning(
                    "Model prompts are not in the expected format. Ignoring them."
                )
        elif model_prompts is not None and hasattr(self.model, "prompts"):
            logger.info(f"Model prompts will be overwritten with {model_prompts}")
            self.model_prompts = model_prompts
            self.validate_task_to_prompt_name()

        if isinstance(self.model, CrossEncoder):
            self.predict = self.handle_instructions_predict

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
        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        if prompt_name:
            logger.info(
                f"Using prompt_name={prompt_name} for task={task_metadata.name} prompt_type={prompt_type}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_metadata.name} prompt_type={prompt_type}"
            )
        logger.info(f"Encoding {len(inputs)} inputs.")

        inputs = [text for batch in inputs for text in batch["text"]]

        embeddings = self.model.encode(
            inputs,
            prompt_name=prompt_name,
            **kwargs,
        )
        if isinstance(embeddings, torch.Tensor):
            # ensure everything is on CPU and is float
            embeddings = embeddings.cpu().detach().float()
        return embeddings

    def _predict(
        self,
        sentences: Sequence[tuple[str, str]],
        **kwargs: Any,
    ) -> np.ndarray:
        return self.model.predict(
            sentences,
            convert_to_numpy=True,
            **kwargs,
        )

    def handle_instructions_predict(
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
        """Handles the prediction for cross-encoders with instructions.

        Args:
            inputs1: Queries with optional instructions to encode.
            inputs2: Passages to encode.
            task_metadata: Task metadata.
            hf_split: Split of the task
            hf_subset: Split of the subset
            prompt_type: The type of prompt to use (query or passage).
            **kwargs: Kwargs to pass to the model.

        Returns:
            Similarity scores for the query-passage pairs.
        """
        all_queries_with_instructions = [
            text for batch in inputs1 for text in batch["text"]
        ]
        all_corpus_with_instructions = [
            text for batch in inputs2 for text in batch["text"]
        ]

        return self._predict(
            list(zip(all_queries_with_instructions, all_corpus_with_instructions)),
            **kwargs,
        )
