from mteb.models.model_meta import ModelMeta, ScoringFunction

amazon_titan_text_embeddings_v2 = ModelMeta(
    loader=None,
    name="amazon/Titan-text-embeddings-v2",
    revision="1",
    release_date="2024-04-30",
    languages=["eng-Latn"],
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license="https://aws.amazon.com/service-terms/",
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    framework=[],
    reference="https://huggingface.co/amazon/Titan-text-embeddings-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    superseded_by=None,
)
