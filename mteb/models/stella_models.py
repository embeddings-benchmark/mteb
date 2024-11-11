from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import instruct_wrapper

stella_en_400M = ModelMeta(
    # https://huggingface.co/dunzhang/stella_en_400M_v5/discussions/21#671a6205ac1e2416090f2bf4
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="dunzhang/stella_en_400M_v5",
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
    ),
    name="dunzhang/stella_en_400M_v5",
    languages=["eng_Latn"],
    open_weights=True,
    use_instructions=True,
    revision="1bb50bc7bb726810eac2140e62155b88b0df198f",
    release_date="2024-07-12",
    n_parameters=435_000,
    max_tokens=8192,
    embed_dim=4096,
    license="mit",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "GritLM"],
    reference="https://huggingface.co/dunzhang/stella_en_400M_v5",
)

stella_en_1_5b = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="dunzhang/stella_en_1.5B_v5",
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
    ),
    name="dunzhang/stella_en_1.5B_v5",
    languages=["eng_Latn"],
    open_weights=True,
    use_instructions=True,
    revision="d03be74b361d4eb24f42a2fe5bd2e29917df4604",
    release_date="2024-07-12",
    n_parameters=1_540_000,
    max_tokens=131072,
    embed_dim=8960,
    license="mit",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "GritLM"],
    reference="https://huggingface.co/dunzhang/stella_en_1.5B_v5",
)
