from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

mme5_model = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="intfloat/mmE5-mllama-11b-instruct",
        revision="cbb328b9bf9ff5362c852c3166931903226d46f1",
    ),
    name="intfloat/mmE5-mllama-11b-instruct",
    languages=["eng_Latn"],
    revision="cbb328b9bf9ff5362c852c3166931903226d46f1",
    release_date="2025-02-16",
    modalities=["image", "text"],
    n_parameters=10_600_000_000,
    memory_usage_mb=20300,
    max_tokens=8192,
    embed_dim=7680,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/intfloat/mmE5-synthetic",
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/intfloat/mmE5-mllama-11b-instruct",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets=None,
)
