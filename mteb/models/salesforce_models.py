from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import torch

from mteb.model_meta import ModelMeta

from .wrapper import Wrapper
from ..encoder_interface import PromptType
from .instructions import task_to_instruction


def sfr_instruction(instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery: "


def sfr_loader(**kwargs):
    try:
        from gritlm import GritLM
    except ImportError:
        raise ImportError(
            "Please install `pip install gritlm` to use SFR_Embedding_2_R."
        )

    class SFRWrapper(GritLM, Wrapper):
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
                kwargs["instruction"] = sfr_instruction(instruction)
            return super().encode(*args, **kwargs)

    return SFRWrapper(**kwargs)


SFR_Embedding_2_R = ModelMeta(
    loader=partial(
        sfr_loader,
        model_name_or_path="Salesforce/SFR-Embedding-2_R",
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.bfloat16,
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/Salesforce/SFR-Embedding-2_R
        normalized=True,
    ),
    name="Salesforce/SFR-Embedding-2_R",
    languages=["eng_Latn"],
    open_source=True,
    revision="91762139d94ed4371a9fa31db5551272e0b83818",
    release_date="2024-06-14",  # initial commit of hf model.
)
