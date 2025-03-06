from mteb.model_meta import ModelMeta

arabic_triplet_matryoshka = ModelMeta(
    name="Arabic-Triplet-Matryoshka-V2",
    languages=["ara-Arab"],  # Arabic in correct format
    open_weights=True,
    revision="ed357f222f0b6ea6670d2c9b5a1cb93950d34200",  # Your actual revision hash
    release_date="2024-07-28",
    n_parameters=135_000_000,  # Update if different
    memory_usage_mb=0.5,  # Update if different
    embed_dim=768,  # Update if different
    license="apache-2.0",  # Update if different
    max_tokens=768,  # Update if different
    reference="https://huggingface.co/Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",  # Add correct model link
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,  # Update dataset link
    training_datasets=None,
)

