from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import torch

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper

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
        prompt_name = self.get_prompt_name(self.model_prompts, task_name, prompt_type)
        if prompt_name:
            logger.info(
                f"Using prompt_name={prompt_name} for task={task_name} prompt_type={prompt_type}"
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
    loader=partial(  # type: ignore
        UAEWrapper,
        model="WhereIsAI/UAE-Large-V1",
        revision="369c368f70f16a613f19f5598d4f12d9f44235d4",
        # https://github.com/SeanLee97/AnglE/blob/b04eae166d8596b47293c75b4664d3ad820d7331/angle_emb/angle.py#L291-L314
        model_prompts={
            "query": "Represent this sentence for searching relevant passages: {text}",
            "Summarization": 'Summarize sentence "{text}" in one word:"',
        },
    ),
    name="WhereIsAI/UAE-Large-V1",
    languages=["eng_Latn"],
    open_weights=True,
    revision="369c368f70f16a613f19f5598d4f12d9f44235d4",
    release_date="2023-12-04",  # initial commit of hf model.
    n_parameters=335 * 1e6,
    memory_usage_mb=1278,
    max_tokens=512,
    embed_dim=1024,
    license="mit",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/WhereIsAI/UAE-Large-V1",
    use_instructions=True,
    training_datasets={
        # source: https://arxiv.org/pdf/2309.12871
        # not in MTEB
        "MNLI": [],
        "NLI": [],
        "SNLI": [],
    },
    public_training_code=None,
    public_training_data=None,
)
