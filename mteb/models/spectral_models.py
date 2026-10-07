from functools import partial

from .model_meta import ModelMeta
from .sentence_transformer_wrapper import sentence_transformers_loader

spectral_embed_v1_140m = ModelMeta(
    name="JMullings/spectral-embed-v1-140m",
    revision="39169ba7aa8d1de29da32d97a6129bca790e9193",
    release_date="2026-10-06",
    languages=["eng-Latn"],
    n_parameters=140_000_000,
    memory_usage_mb=560,
    max_tokens=512,
    embed_dim=2048,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["NumPy", "Sentence Transformers"],
    reference="https://huggingface.co/JMullings/spectral-embed-v1-140m",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=None,
    loader=partial(
        sentence_transformers_loader,
        model_name="JMullings/spectral-embed-v1-140m",
        revision="39169ba7aa8d1de29da32d97a6129bca790e9193",
    ),
    adapted_from=None,
    superseded_by=None,
)
