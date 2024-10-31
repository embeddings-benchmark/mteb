from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta

from .instruct_wrapper import instruct_wrapper

gte_Qwen2_7B_instruct = ModelMeta(
    loader=partial(
        instruct_wrapper,
        model_name_or_path="Alibaba-NLP/gte-Qwen2-7B-instruct",
        instruction_template="Instruct: {instruction}\nQuery: ",
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
