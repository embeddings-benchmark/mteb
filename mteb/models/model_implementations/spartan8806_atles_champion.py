"""ATLES Champion Embedding Model for MTEB."

from mteb.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

spartan8806_atles_champion_embedding = ModelMeta(
    loader=sentence_transformers_loader,
    name="spartan8806/atles-champion-embedding",
    languages=["eng-Latn"],
    open_source=True,
    revision="main",
    release_date="2025-01-16",
    n_parameters=110_000_000,
    max_tokens=512,
    embed_dim=768,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers"],
    reference="https://huggingface.co/spartan8806/atles-champion-embedding",
    use_instructions=False,
)