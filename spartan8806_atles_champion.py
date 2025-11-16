"""ATLES Champion Embedding Model wrapper for MTEB."""

from __future__ import annotations

from mteb.model_meta import ModelMeta

from .wrapper import Wrapper


class SpartanATLESChampionEmbedding(Wrapper):
    """ATLES Champion Embedding model wrapper for MTEB benchmarking."""

    def __init__(self, **kwargs):
        """Initialize the ATLES Champion model."""
        super().__init__(
            model_name_or_path="spartan8806/atles-champion-embedding",
            **kwargs
        )


spartan8806_atles_champion_embedding = ModelMeta(
    loader=lambda **kwargs: SpartanATLESChampionEmbedding(**kwargs),
    name="spartan8806/atles-champion-embedding",
    languages=["eng_Latn"],
    open_source=True,
    revision="main",
    release_date="2025-01-16",
    n_parameters=110_000_000,
    memory_usage=None,
    max_tokens=512,
    embed_dim=768,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers"],
    reference="https://huggingface.co/spartan8806/atles-champion-embedding",
    use_instructions=False,
)

