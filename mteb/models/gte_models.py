from __future__ import annotations

from functools import partial
from typing import Any, Sequence

import numpy as np

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

from .instructions import task_to_instruction


def gte_instruction(instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery: "


def gte_loader(**kwargs):
    try:
        from gritlm import GritLM
    except ImportError:
        raise ImportError(
            "Please install `pip install gritlm` to use gte-Qwen2-7B-instruct."
        )

    class GTEWrapper(GritLM):
        def encode(
            self,
            sentences: Sequence[str],
            *args,
            task_name: str,
            prompt_type: PromptType | None = None,
            **kwargs: Any,
        ) -> np.ndarray:
            # TODO check sentences and API (what it returns)
            if "instruction" in kwargs:
                instruction = kwargs.pop("instruction", "")
            else:
                instruction = task_to_instruction(
                    task_name, prompt_type == PromptType.query
                )
            if instruction:
                kwargs["instruction"] = gte_instruction(instruction)
            return super().encode(*args, **kwargs)

    return GTEWrapper(**kwargs)


gte_Qwen2_7B_instruct = ModelMeta(
    loader=partial(
        gte_loader,
        model_name_or_path="Alibaba-NLP/gte-Qwen2-7B-instruct",
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct#sentence-transformers
        normalized=True,
    ),
    name="Alibaba-NLP/gte-Qwen2-7B-instruct",
    languages=None,
    open_source=True,
    revision="e26182b2122f4435e8b3ebecbf363990f409b45b",
    release_date="2024-06-15",  # initial commit of hf model.
)
