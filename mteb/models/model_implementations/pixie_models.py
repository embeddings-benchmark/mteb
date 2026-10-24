from mteb.models.model_implementations.arctic_models import (
    ARCTIC_V2_CITATION,
    LANGUAGES_V2_0,
    arctic_v2_training_datasets,
)
from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerEncoderWrapper,
    SparseEncoderWrapper,
)

PIXIE_RUNE_V1_PROMPTS = {
    "query": "query: ",
    "document": "",
}

# it is further fine-tuned on TelePIX proprietary IR data (not public).
pixie_rune_v1_training_datasets = set(arctic_v2_training_datasets) | {
    "TelePIX-Proprietary-IR-Triplets",
}

pixie_rune_v1_0 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs={
        "model_prompts": PIXIE_RUNE_V1_PROMPTS,
    },
    name="telepix/PIXIE-Rune-v1.0",
    model_type=["dense"],
    revision="b2486496da71191626666a88f9bfec844933a134",
    release_date="2026-01-15",
    languages=LANGUAGES_V2_0,
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    n_parameters=567754752,
    n_embedding_parameters=256002048,
    memory_usage_mb=2166,
    max_tokens=6144,
    embed_dim=1024,
    license="apache-2.0",
    reference="https://huggingface.co/telepix/PIXIE-Rune-v1.0",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    adapted_from="Snowflake/snowflake-arctic-embed-l-v2.0",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=pixie_rune_v1_training_datasets,
    citation=ARCTIC_V2_CITATION,
)

pixie_rune_v1_5 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs={
        "model_prompts": PIXIE_RUNE_V1_PROMPTS,
    },
    name="telepix/PIXIE-Rune-v1.5",
    model_type=["dense"],
    revision="29dd334196af53e6cfc16674379e743334f5fa66",
    release_date="2026-04-22",
    languages=["kor-Hang", "eng-Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=567754752,
    n_embedding_parameters=256002048,
    memory_usage_mb=2166,
    max_tokens=8192,
    embed_dim=1024,
    license="apache-2.0",
    reference="https://huggingface.co/telepix/PIXIE-Rune-v1.5",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    adapted_from="Snowflake/snowflake-arctic-embed-l-v2.0",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    # Model card only says "trained on Korean and English information
    # retrieval dataset" without naming it, so the fine-tuning data is unknown.
    training_datasets=None,
)

pixie_splade_v1_5 = ModelMeta(
    loader=SparseEncoderWrapper,
    name="telepix/PIXIE-Splade-v1.5",
    model_type=["sparse"],
    revision="bb818c826aa9abc251e7a94c1ea64b58ae3a9ac7",
    release_date="2026-04-22",
    languages=["kor-Hang", "eng-Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    n_parameters=149372240,
    n_embedding_parameters=38_400_000,
    memory_usage_mb=570,
    max_tokens=5632,
    embed_dim=50000,
    license="apache-2.0",
    reference="https://huggingface.co/telepix/PIXIE-Splade-v1.5",
    similarity_fn_name=ScoringFunction.DOT_PRODUCT,
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    # Trained on TelePIX proprietary data (not public), so no training datasets are declared.
    training_datasets=None,
    extra_requirements_groups=["sparse-encoder"],
)
