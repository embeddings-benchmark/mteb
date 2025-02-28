from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from torch.utils.data import DataLoader

from mteb.encoder_interface import BatchedInput, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper

logger = logging.getLogger(__name__)


class SentenceTransformerWrapper(Wrapper):
    mteb_model_meta: ModelMeta | None = None

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
                model_prompts = self.validate_task_to_prompt_name(self.model.prompts)
            except KeyError:
                model_prompts = None
                logger.warning(
                    "Model prompts are not in the expected format. Ignoring them."
                )
        elif model_prompts is not None and hasattr(self.model, "prompts"):
            logger.info(f"Model prompts will be overwritten with {model_prompts}")
            self.model.prompts = model_prompts
        self.model_prompts = self.validate_task_to_prompt_name(model_prompts)

        if isinstance(self.model, CrossEncoder):
            self.predict = self.handle_instructions_predict

        if hasattr(self.model, "similarity") and callable(self.model.similarity):
            self.similarity = self.model.similarity

    def encode(
        self,
        inputs: Iterable[BatchedInput] | DataLoader[BatchedInput],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            inputs: The sentences to encode.
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
        prompt_name = None
        if self.model_prompts is not None:
            prompt_name = self.get_prompt_name(
                self.model_prompts, task_name, prompt_type
            )
        if prompt_name:
            logger.info(
                f"Using prompt_name={prompt_name} for task={task_name} prompt_type={prompt_type}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_name} prompt_type={prompt_type}"
            )
        logger.info(f"Encoding {len(inputs)} inputs.")

        if self.mteb_model_meta is not None and self.mteb_model_meta.modalities == [
            "text"
        ]:
            # compatability for multi-modal models, e.g. mmE5
            inputs = [text for batch in inputs for text in batch["text"]]

        embeddings = self.model.encode(
            inputs,
            prompt_name=prompt_name,
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

    def handle_instructions_predict(self, sentences, **kwargs):
        # unzip the queries, corpus, and instruction so we can add the instructions to the queries
        # as ST models can't take an arg for instructions
        queries, corpus, instructions = list(zip(*sentences))
        # combine the queries and instructions
        queries_with_instructions = [
            f"{query.strip()} {instruction}".strip() if instruction else query
            for query, instruction in zip(queries, instructions)
        ]
        return self._predict(list(zip(queries_with_instructions, corpus)), **kwargs)
