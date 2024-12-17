from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import mteb
from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

from .wrapper import Wrapper

logger = logging.getLogger(__name__)


class JasperWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        revision: str,
        instruction_template: str | Callable[[str], str] | None = None,
        max_seq_length: int = 2048,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, revision=revision, **kwargs)
        self.instruction_template = instruction_template
        self.model.max_seq_length = max_seq_length

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        task = mteb.get_task(task_name=task_name)
        instruction = self.get_task_instruction(task_name, prompt_type)

        # to passage prompts won't be applied to passages
        if prompt_type == PromptType.passage and task.metadata.type == "s2p":
            instruction = None

        embeddings = self.model.encode(
            sentences,
            normalize_embeddings=True,
            prompt=instruction,
            **kwargs,
        )

        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings


jasper_en_v1 = ModelMeta(
    loader=partial(  # type: ignore
        JasperWrapper,
        model_name="infgrad/jasper_en_vision_language_v1",
        revision="d6330ce98f8a0d741e781df845904c9484f00efa",
        config_kwargs={"is_text_encoder": True, "vector_dim": 12288},
        model_kwargs={
            "attn_implementation": "sdpa",
            "torch_dtype": torch.float16,
        },
        trust_remote_code=True,
        max_seq_length=2048,
        instruction_template="Instruct: {instruction}\nQuery: ",
    ),
    name="infgrad/jasper_en_vision_language_v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="d6330ce98f8a0d741e781df845904c9484f00efa",
    release_date="2024-12-11",  # first commit
    n_parameters=1_999_000_000,
    memory_usage=None,
    max_tokens=131072,
    embed_dim=8960,
    license="apache-2.0",
    reference="https://huggingface.co/infgrad/jasper_en_vision_language_v1/tree/main",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    training_datasets={
        "non_mteb": ["BAAI/Infinity-MM", "HuggingFaceFW/fineweb-edu"],
    },
)
