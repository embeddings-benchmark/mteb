from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

chain19_en = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="bchoiced/CHAIN19",
        revision="5ba01fcb4e90ede5e2772b8a9ca68c12515dc6af",
    ),
    name="bchoiced/CHAIN19",
    languages=[
        "eng-Latn",
    ],
    open_weights=True,
    revision="5ba01fcb4e90ede5e2772b8a9ca68c12515dc6af",
    release_date="2025-05-07",
    n_parameters=7_110_000_000,
    memory_usage_mb=27125,
    embed_dim=4096,
    license="cc-by-sa-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/bchoiced/CHAIN19",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)
