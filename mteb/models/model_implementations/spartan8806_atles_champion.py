"""ATLES Champion Embedding Model for MTEB."""

from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

spartan8806_atles_champion_embedding = ModelMeta(
    loader=sentence_transformers_loader,
    name="spartan8806/atles-champion-embedding",
    languages=["eng-Latn"],
    open_weights=True,
    revision="d4c74d7000bbd25f3597fc0f2dcde59ef1386e8f",
    release_date="2025-11-16",
    n_parameters=110_000_000,
    memory_usage_mb=420,
    max_tokens=512,
    embed_dim=768,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers"],
    reference="https://huggingface.co/spartan8806/atles-champion-embedding",
    use_instructions=False,
    training_datasets={"STSBenchmark"},
    public_training_code=True,
    public_training_data=True,
)
