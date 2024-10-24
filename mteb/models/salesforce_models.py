from __future__ import annotations

from functools import partial

import torch

from mteb.model_meta import ModelMeta

from .instruct_wrapper import instruct_wrapper


def sfr_instruction(instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery: "


SFR_Embedding_2_R = ModelMeta(
    loader=partial(
        instruct_wrapper,
        model_name_or_path="Salesforce/SFR-Embedding-2_R",
        instruction_template=sfr_instruction,
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
