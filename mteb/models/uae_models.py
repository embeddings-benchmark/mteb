from __future__ import annotations

from functools import partial
from typing import Any, Sequence

import numpy as np
import torch

from .sentence_transformer_wrapper import SentenceTransformerWrapper, get_prompt_name
from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
import logging

logger = logging.getLogger(__name__)


class UAEWrapper(SentenceTransformerWrapper):
    """following the hf model card documentation."""

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
        prompt_name = get_prompt_name(self.model_prompts, task_name, prompt_type)
        if prompt_name:
            logger.info(
                f"Using prompt_nane={prompt_name} for task={task_name} prompt_type={prompt_type}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_name} prompt_type={prompt_type}"
            )
        logger.info(f"Encoding {len(sentences)} sentences.")
        if prompt_name and prompt_name in self.model.prompts:
            prompt = self.model.prompts[prompt_name]
            sentences = [prompt.format(text=sentence) for sentence in sentences]

        embeddings = self.model.encode(
            sentences,
            **kwargs,
        )
        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings


uae_large_v1 = ModelMeta(
    loader=partial(
        UAEWrapper,
        model_name="WhereIsAI/UAE-Large-V1",
        revision="369c368f70f16a613f19f5598d4f12d9f44235d4",
        trust_remote_code=True,
        # https://github.com/SeanLee97/AnglE/blob/b04eae166d8596b47293c75b4664d3ad820d7331/angle_emb/angle.py#L291-L314
        model_prompts={
            "query": 'Represent this sentence for searching relevant passages: {text}',
            "Summarization": 'Summarize sentence "{text}" in one word:"',
        },
    ),
    name="WhereIsAI/UAE-Large-V1",
    languages=["eng_Latn"],
    open_source=True,
    revision="369c368f70f16a613f19f5598d4f12d9f44235d4",
    release_date="2023-12-04",  # initial commit of hf model.
)