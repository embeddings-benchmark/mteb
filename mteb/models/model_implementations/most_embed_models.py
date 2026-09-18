from __future__ import annotations

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import OutputDType

most_embed_de = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    # Mirrors the `nvidia/Nemotron-3-Embed-1B-BF16` base model entry so that results stay
    # directly comparable to the baseline on the leaderboard.
    loader_kwargs=dict(processor_kwargs={"model_max_length": 4096}),
    name="malteos/most-embed-de",
    revision="30be696e29c09763afc3588ee9e04d6d0cbe8e43",
    release_date="2026-08-11",
    languages=["deu-Latn", "eng-Latn"],
    n_parameters=1_140_918_272,
    n_active_parameters_override=None,
    n_embedding_parameters=268_435_456,
    memory_usage_mb=2176,
    max_tokens=32768,
    embed_dim=2048,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch", "Transformers"],
    reference="https://huggingface.co/malteos/most-embed-de",
    similarity_fn_name=ScoringFunction.COSINE,
    # The model ships `query: ` / `passage: ` prompts in `config_sentence_transformers.json`,
    # which `SentenceTransformerEncoderWrapper` picks up automatically.
    use_instructions=True,
    training_datasets=set(),  # proprietary German support corpora, no mteb task training splits
    adapted_from="nvidia/Nemotron-3-Embed-1B-BF16",
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation=None,
    contacts=None,
    output_dtypes=OutputDType.BF16,
    extra_requirements_groups=["nemotron-3-embed"],
)
