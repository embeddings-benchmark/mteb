from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

from .wrapper import Wrapper

logger = logging.getLogger(__name__)


def jasper_vl_forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
    trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
    if "pixel_values" in features:
        trans_features["pixel_values"] = features["pixel_values"]
    sentence_embedding = self.auto_model(**trans_features, **kwargs)["sentence_embedding"]
    features.update({"sentence_embedding": sentence_embedding})
    return features


class JasperWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        revision: str,
        model_prompts: dict[str, str] | None = None,
        instruction_template: str | Callable[[str], str] | None = None,
        max_length: int = 400,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, revision=revision, **kwargs)
        self.model._first_module().forward = partial(jasper_vl_forward, self.model._first_module())
        self.model_prompts = (
            self.validate_task_to_prompt_name(model_prompts) if model_prompts else None
        )
        self.model.max_seq_length = max_length
        self.pool = self.model.start_multi_process_pool()
        self.instruction_template = instruction_template

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        instruction = self.get_task_instruction(task_name, prompt_type)

        # to passage prompts won't be applied
        if prompt_type == PromptType.passage:
            instruction = None

        # process white space data
        sentences = [i if i.strip() else "<|endoftext|>" for i in sentences]

        vectors = self.model.encode_multi_process(
            sentences=sentences,
            pool=self.pool,
            normalize_embeddings=True,
            prompt=instruction,
            **kwargs,
        )
        return vectors.astype(dtype=np.float32)


jasper_en_v1 = ModelMeta(
    loader=partial(  # type: ignore
        JasperWrapper,
        trust_remote_code=True,
        model_name="infgrad/jasper_en_vision_language_v1",
        revision="d6330ce98f8a0d741e781df845904c9484f00efa",
        config_kwargs={"is_text_encoder": True, "vector_dim": 12288},
        tokenizer_kwargs={"padding_side": "right"},
        model_kwargs={"attn_implementation": "sdpa"},
        # https://huggingface.co/infgrad/jasper_en_vision_language_v1/blob/d6330ce98f8a0d741e781df845904c9484f00efa/scripts/evaluate_en_mteb/run_evaluate_mteb.py#L14
        max_length=400,
        instruction_template="Instruct: {instruction}\nQuery: "
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
    }
)
