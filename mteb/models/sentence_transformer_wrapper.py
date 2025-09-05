from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from mteb.encoder_interface import PromptType
from mteb.models.wrapper import Wrapper

logger = logging.getLogger(__name__)


class SentenceTransformerWrapper(Wrapper):
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
            built_in_prompts := getattr(self.model, "prompts", None)
        ) and not model_prompts:
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

        if isinstance(self.model, CrossEncoder):
            self.predict = self._predict

        if hasattr(self.model, "similarity") and callable(self.model.similarity):
            self.similarity = self.model.similarity

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            prompt_type: The name type of prompt. (query or passage)
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
            prompt_name = self.get_prompt_name(
                self.model_prompts, task_name, prompt_type
            )
            prompt = self.model_prompts.get(prompt_name, None)
        if prompt_name:
            logger.info(
                f"Using {prompt_name=} for task={task_name} {prompt_type=} with {prompt=}"
            )
        else:
            logger.info(f"No model prompts found for task={task_name} {prompt_type=}")
        logger.info(f"Encoding {len(sentences)} sentences.")

        embeddings = self.model.encode(
            sentences,
            prompt=prompt,
            **kwargs,
        )
        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings

    def _predict(
        self,
        sentences: Sequence[str],
        **kwargs: Any,
    ) -> np.ndarray:
        return self.model.predict(
            sentences,
            convert_to_numpy=True,
            **kwargs,
        )
