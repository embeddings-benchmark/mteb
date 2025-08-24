from mteb.model_meta import ModelMeta

m2_32k = ModelMeta(
    name="togethercomputer/m2-bert-80M-32k-retrieval",
    languages=["eng-Latn"],
    open_weights=True,
    revision="a2ccdc5b5661a282c77545e586a019f387ab7a48",
    release_date="2023-11-04",
    n_parameters=80_000_000,
    memory_usage_mb=305,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/togethercomputer/m2-bert-80M-32k-retrieval",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/HazyResearch/m2",
    public_training_data="https://huggingface.co/datasets/allenai/c4",
    training_datasets={}, # C4
)
