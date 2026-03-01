from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

auto_g_embed_st = ModelMeta(
    loader=sentence_transformers_loader,
    name="geoffsee/auto-g-embed-st",
    model_type=["dense"],
    revision="3e0bf6004ec386dea06d55dda4efe38fd96b5f7b",
    release_date="2026-02-08",
    languages=["eng-Latn"],
    open_weights=True,
    n_parameters=22_713_216,
    n_embedding_parameters=11_720_448,
    memory_usage_mb=87,
    embed_dim=384,
    license="mit",
    max_tokens=256,
    reference="https://huggingface.co/geoffsee/auto-g-embed-st",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=[
        "Sentence Transformers",
        "PyTorch",
        "safetensors",
        "Transformers",
    ],
    use_instructions=False,
    superseded_by=None,
    adapted_from="sentence-transformers/all-MiniLM-L6-v2",
    training_datasets=set(),  # no known overlap with MTEB datasets
    public_training_code="https://github.com/geoffsee/auto-g-embed",
    public_training_data=None,
    citation=None,
)
