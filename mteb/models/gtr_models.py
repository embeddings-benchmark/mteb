from mteb.model_meta import ModelMeta

gtr_t5_large = ModelMeta(
    name="sentence-transformers/gtr-t5-large",
    languages=["eng-Latn"],  # in format eng-Latn
    open_weights=True,
    revision="a2c8ac47f998531948d4cbe32a0b577a7037a5e3",
    release_date="2022-02-09",
    n_parameters=335_000_000,
    memory_usage_mb=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/gtr-t5-large",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

gtr_t5_xl = ModelMeta(
    name="sentence-transformers/gtr-t5-xl",
    languages=["eng-Latn"],  # in format eng-Latn
    open_weights=True,
    revision="23a8d667a1ad2578af181ce762867003c498d1bf",
    release_date="2022-02-09",
    n_parameters=1_240_000_000,
    memory_usage_mb=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/gtr-t5-xl",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

gtr_t5_xxl = ModelMeta(
    name="sentence-transformers/gtr-t5-xxl",
    languages=["eng-Latn"],  # in format eng-Latn
    open_weights=True,
    revision="73f2a9156a3dcc2194dfdb2bf201cd7d17e17884",
    release_date="2022-02-09",
    n_parameters=4_860_000_000,
    memory_usage_mb=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/gtr-t5-xxl",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

gtr_t5_base = ModelMeta(
    name="sentence-transformers/gtr-t5-base",
    languages=["eng-Latn"],  # in format eng-Latn
    open_weights=True,
    revision="7027e9594267928589816394bdd295273ddc0739",
    release_date="2022-02-09",
    n_parameters=110_000_000,
    memory_usage_mb=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/gtr-t5-base",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)