from __future__ import annotations

from functools import partial

import torch

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.e5_instruct import E5_MISTRAL_TRAINING_DATA
from mteb.models.instruct_wrapper import instruct_wrapper


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return f"Instruct: {instruction}\nQuery: " if instruction else ""


Linq_Embed_Mistral = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="Linq-AI-Research/Linq-Embed-Mistral",
        instruction_template=instruction_template,
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.bfloat16,
        normalized=True,
    ),
    name="Linq-AI-Research/Linq-Embed-Mistral",
    languages=["eng_Latn"],
    open_weights=True,
    revision="0c1a0b0589177079acc552433cad51d7c9132379",
    release_date="2024-05-29",  # initial commit of hf model.
    n_parameters=7_110_000_000,
    memory_usage_mb=13563,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    adapted_from="intfloat/e5-mistral-7b-instruct",
    training_datasets=E5_MISTRAL_TRAINING_DATA,
)
