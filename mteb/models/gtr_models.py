from mteb.model_meta import ModelMeta

gtr_t5_xxl = ModelMeta(
    name="gtr-t5-xxl",
    languages=["eng-Latn"],  # in format eng-Latn
    open_weights=True,
    revision="73f2a9156a3dcc2194dfdb2bf201cd7d17e17884",
    release_date="2022-02-09",
    n_parameters=4_860_000_000,
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
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/gtr-t5-bases",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)