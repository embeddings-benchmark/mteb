from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper

logger = logging.getLogger(__name__)


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return f"Instruct: {instruction}\nQuery: " if instruction else ""


class NvEmbedWrapper(SentenceTransformerWrapper):
    def __init__(
        self,
        model: str | SentenceTransformer | CrossEncoder,
        revision: str | None = None,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model, revision, model_prompts, **kwargs)
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"
        logger.warning(
            "Instructions are used in both query and docs, which may cause performance discrepancies from the original implementation."
        )

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        # Add eos token to each input example
        sentences = [example + self.model.tokenizer.eos_token for example in sentences]

        instruction = ""
        if prompt_type == PromptType.query:
            instruction = self.get_instruction(task_name, prompt_type)

        prompt = instruction_template(instruction)

        if prompt:
            logger.info(f"Using {prompt=} for task={task_name} {prompt_type=}")
        else:
            logger.info(f"No model prompts found for task={task_name} {prompt_type=}")

        logger.info(f"Encoding {len(sentences)} sentences.")

        embeddings = self.model.encode(
            sentences,
            prompt=prompt,
            normalize_embeddings=True,
            **kwargs,
        )
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings


NV_embed_v2 = ModelMeta(
    loader=partial(  # type: ignore
        NvEmbedWrapper,
        model="nvidia/NV-Embed-v2",
        trust_remote_code=True,
    ),
    name="nvidia/NV-Embed-v2",
    languages=["eng_Latn"],
    open_weights=True,
    revision="7604d305b621f14095a1aa23d351674c2859553a",
    release_date="2024-09-09",  # initial commit of hf model.
    n_parameters=7_850_000_000,
    memory_usage=None,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/nvidia/NV-Embed-v2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
)

NV_embed_v1 = ModelMeta(
    loader=partial(  # type: ignore
        NvEmbedWrapper,
        model="nvidia/NV-Embed-v1",
        trust_remote_code=True,
    ),
    name="nvidia/NV-Embed-v1",
    languages=["eng_Latn"],
    open_weights=True,
    revision="570834afd5fef5bf3a3c2311a2b6e0a66f6f4f2c",
    release_date="2024-09-13",  # initial commit of hf model.
    n_parameters=7_850_000_000,
    memory_usage=None,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/nvidia/NV-Embed-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
)
