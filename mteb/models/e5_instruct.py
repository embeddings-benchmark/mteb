from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import torch

from mteb.model_meta import ModelMeta

from ..encoder_interface import PromptType
from .e5_models import E5_PAPER_RELEASE_DATE, XLMR_LANGUAGES
from .instructions import task_to_instruction
from .wrapper import Wrapper

MISTRAL_LANGUAGES = ["eng_Latn", "fra_Latn", "deu_Latn", "ita_Latn", "spa_Latn"]


def e5_instruction(instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery: "


def e5_loader(**kwargs):
    try:
        from gritlm import GritLM
    except ImportError:
        raise ImportError(
            "Please install `pip install gritlm` to use E5 Instruct models."
        )

    class E5InstructWrapper(GritLM, Wrapper):
        def encode(
            self,
            sentences: Sequence[str],
            *args,
            task_name: str,
            prompt_type: PromptType | None = None,
            **kwargs: Any,
        ) -> np.ndarray:
            if "instruction" in kwargs:
                instruction = kwargs.pop("instruction", "")
            else:
                instruction = task_to_instruction(
                    task_name, prompt_type == PromptType.query
                )
            if instruction:
                kwargs["instruction"] = e5_instruction(instruction)
            return super().encode(sentences, *args, **kwargs)

    return E5InstructWrapper(**kwargs)


e5_instruct = ModelMeta(
    loader=partial(
        e5_loader,
        model_name_or_path="intfloat/multilingual-e5-large-instruct",
        attn="cccc",
        pooling_method="mean",
        mode="embedding",
        torch_dtype=torch.float16,
        normalized=True,
    ),
    name="intfloat/multilingual-e5-large-instruct",
    languages=XLMR_LANGUAGES,
    open_weights=True,
    revision="baa7be480a7de1539afce709c8f13f833a510e0a",
    release_date=E5_PAPER_RELEASE_DATE,
    framework=["GritLM", "PyTorch"],
    similarity_fn_name="cosine",
    use_instuctions=True,
    reference="https://huggingface.co/intfloat/multilingual-e5-large-instruct",
    n_parameters=560_000_000,
    memory_usage=None,
    embed_dim=1024,
    license="mit",
    max_tokens=514,
)

e5_mistral = ModelMeta(
    loader=partial(
        e5_loader,
        model_name_or_path="intfloat/e5-mistral-7b-instruct",
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.float16,
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/intfloat/e5-mistral-7b-instruct#transformers
        normalized=True,
    ),
    name="intfloat/e5-mistral-7b-instruct",
    languages=MISTRAL_LANGUAGES,
    open_weights=True,
    revision="07163b72af1488142a360786df853f237b1a3ca1",
    release_date=E5_PAPER_RELEASE_DATE,
    framework=["GritLM", "PyTorch"],
    similarity_fn_name="cosine",
    use_instuctions=True,
    reference="https://huggingface.co/intfloat/e5-mistral-7b-instruct",
    n_parameters=7_111_000_000,
    memory_usage=None,
    embed_dim=4096,
    license="mit",
    max_tokens=32768,
)
