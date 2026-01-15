from mteb.models.model_implementations.arctic_models import (
    ARCTIC_V2_CITATION,
    LANGUAGES_V2_0,
    arctic_v2_training_datasets,
)
from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

PIXIE_RUNE_V1_CITATION = """@misc{TelePIX-PIXIE-Rune-v1.0,
  title        = {PIXIE-Rune-v1.0},
  author       = {TelePIX AI Research Team and Bongmin Kim},
  year         = {2026},
  howpublished = {Hugging Face model card},
  url          = {https://huggingface.co/telepix/PIXIE-Rune-v1.0}
}"""

PIXIE_RUNE_V1_PROMPTS = {
    "query": "query: ",
    "document": "",
}

# it is further fine-tuned on TelePIX proprietary IR data (not public).
pixie_rune_v1_training_datasets = set(arctic_v2_training_datasets) | {
    "TelePIX-Proprietary-IR-Triplets",
}

pixie_rune_v1_0 = ModelMeta(
    loader=sentence_transformers_loader,
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
    citation=PIXIE_RUNE_V1_CITATION + "\n\n" + ARCTIC_V2_CITATION,
)
