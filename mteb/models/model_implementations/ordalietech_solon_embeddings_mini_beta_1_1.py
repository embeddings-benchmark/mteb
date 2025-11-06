from mteb.models import ModelMeta, sentence_transformers_loader

solon_embeddings_1_1 = ModelMeta(
    name="OrdalieTech/Solon-embeddings-mini-beta-1.1",
    languages=["fra-Latn"],
    n_parameters=210_000_000,
    public_training_code=None,
    memory_usage_mb=808.0,
    open_weights=True,
    revision="8e4ea66eb7eb6109b47b7d97d7556f154d9aec4a",
    release_date="2025-01-01",
    embed_dim=768,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/OrdalieTech/Solon-embeddings-mini-beta-1.1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_data=(
        "https://huggingface.co/datasets/PleIAs/common_corpus; "
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb; "
        "https://huggingface.co/datasets/OrdalieTech/wiki_fr; "
        "private LLM-synthetic (train)"
    ),
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
    training_datasets=set(),  # No mteb dataset
)
