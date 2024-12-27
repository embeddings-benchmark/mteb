from __future__ import annotations

from functools import partial

import torch

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import instruct_wrapper


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return (
        f"Instruct: {instruction}\nQuery: "
        if (prompt_type is None or prompt_type == PromptType.query) and instruction
        else ""
    )


gte_Qwen2_7B_instruct = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="Alibaba-NLP/gte-Qwen2-7B-instruct",
        instruction_template=instruction_template,
        attn="bbcc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.float16,
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct#sentence-transformers
        normalized=True,
        embed_eos="<|endoftext|>",
    ),
    name="Alibaba-NLP/gte-Qwen2-7B-instruct",
    languages=None,
    open_weights=True,
    revision="e26182b2122f4435e8b3ebecbf363990f409b45b",
    release_date="2024-06-15",  # initial commit of hf model.
    n_parameters=7_613_000_000,
    memory_usage=None,
    embed_dim=3584,
    license="apache-2.0",
    reference="https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
)


gte_Qwen1_5_7B_instruct = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="Alibaba-NLP/gte-Qwen1.5-7B-instruct",
        instruction_template=instruction_template,
        attn="bbcc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.float16,
        normalized=True,
        embed_eos="<|endoftext|>",
    ),
    name="Alibaba-NLP/gte-Qwen1.5-7B-instruct",
    languages=["eng_Latn"],
    open_weights=True,
    revision="07d27e5226328010336563bc1b564a5e3436a298",
    release_date="2024-04-20",  # initial commit of hf model.
    n_parameters=7_720_000_000,
    memory_usage=None,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Alibaba-NLP/gte-Qwen1.5-7B-instruct",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
)


gte_Qwen2_1_5B_instruct = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        instruction_template=instruction_template,
        attn="bbcc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.float16,
        normalized=True,
        embed_eos="<|endoftext|>",
    ),
    name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    languages=["eng_Latn"],
    open_weights=True,
    revision="c6c1b92f4a3e1b92b326ad29dd3c8433457df8dd",
    release_date="2024-07-29",  # initial commit of hf model.
    n_parameters=1_780_000_000,
    memory_usage=None,
    embed_dim=8960,
    license="apache-2.0",
    max_tokens=131072,
    reference="https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
)
