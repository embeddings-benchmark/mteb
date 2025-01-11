from __future__ import annotations

import logging
from functools import partial

import torch

from mteb.model_meta import ModelMeta

from .instruct_wrapper import InstructSentenceTransformerWrapper

logger = logging.getLogger(__name__)


jasper_en_v1 = ModelMeta(
    loader=partial(  # type: ignore
        InstructSentenceTransformerWrapper,
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
