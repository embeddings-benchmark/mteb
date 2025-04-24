from __future__ import annotations

from mteb.model_meta import ModelMeta


seed_embedding = ModelMeta(
    name="ByteDance-Seed/Seed-Embedding",
    revision="1",
    release_date="2025-04-25",
    languages=[
        "eng-Latn",
        "zho-Hans",
    ],
    loader=None,
    max_tokens=32768,
    embed_dim=2048,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license="other",
    reference="https://huggingface.co/ByteDance-Seed/Seed-Embedding",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    training_datasets=None,
    public_training_code=None,
    public_training_data=None,
)
