from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

relle_en = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="bchoiced/RELLE",
        revision="093417798a062ed8ddd8424df3be62145d4ace91",
    ),
    name="bchoiced/RELLE",
    languages=[
        "eng-Latn",
    ],
    open_weights=True,
    revision="093417798a062ed8ddd8424df3be62145d4ace91",
    release_date="2025-04-17",
    n_parameters=7_110_000_000,
    memory_usage_mb=27125,
    embed_dim=4096,
    license="cc-by-sa-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/bchoiced/RELLE",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)
