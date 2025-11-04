from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

embedding_gemma_300m = ModelMeta(
    loader=sentence_transformers_loader,  # type: ignore
    name="emillykkejensen/EmbeddingGemma-Scandi-300m",
    languages=["dan-Latn", "swe-Latn", "nor-Latn", "nob-Latn", "nno-Latn"],
    open_weights=True,
    revision="64614b0b8b64f0c6c1e52b07e4e9a4e8fe4d2da2",
    release_date="2025-10-17",
    n_parameters=307_581_696,
    embed_dim=768,
    max_tokens=2048,
    license="apache-2.0",
    reference="https://huggingface.co/emillykkejensen/EmbeddingGemma-Scandi-300m",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/DDSC/nordic-embedding-training-data",
    training_datasets=set(),
    similarity_fn_name="cosine",  # type: ignore[arg-type]
    memory_usage_mb=578,
)
