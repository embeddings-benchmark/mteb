from __future__ import annotations

from mteb.model_meta import ModelMeta


xyz_embedding = ModelMeta(
    name="fangxq/XYZ-embedding",
    languages=["zh"], # follows ISO 639-3 and BCP-47
    open_weights=False,
    revision="4004120220b99baea764a1d3508427248ac3bccf",
    release_date="2024-09-13",
    n_parameters=326000000,
    memory_usage_mb=1242,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/fangxq/XYZ-embedding",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None, 
)
