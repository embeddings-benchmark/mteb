from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

Euler_Legal_Embedding_V1 = ModelMeta(
    loader=sentence_transformers_loader,
    name="Mira190/Euler-Legal-Embedding-V1",
    model_type=["dense"],
    revision="df607ed9e25e569514a99c27cdaaab16e76b6dd4",
    release_date="2025-11-06",
    languages=["eng-Latn"],
    n_parameters=8000000000,
    memory_usage_mb=15618,
    max_tokens=1536,
    embed_dim=4096,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/Mira190/Euler-Legal-Embedding-V1",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=set(),  # final-data-new-anonymized-grok4-filtered
    adapted_from="Qwen/Qwen3-Embedding-8B",
    superseded_by=None,
    citation="""@misc{euler2025legal,
      title={Euler-Legal-Embedding: Advanced Legal Representation Learning}, 
      author={LawRank Team},
      year={2025},
      publisher={Hugging Face}
}""",
)
