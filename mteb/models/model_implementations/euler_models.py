from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

Euler_Legal_Embedding_V1 = ModelMeta(
    loader=sentence_transformers_loader,
    name="LawRank/Euler-Legal-Embedding-V1",
    revision="main",
    release_date="2025-11-06",
    languages=["eng-Latn", "multilingual"],
    n_parameters=8000000000,
    memory_usage_mb=None,
    max_tokens=1536,
    embed_dim=4096,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/LawRank/Euler-Legal-Embedding-V1",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=["final-data-new-anonymized-grok4-filtered.jsonl"],
    adapted_from="Qwen/Qwen3-Embedding-8B",
    superseded_by=None,
)
