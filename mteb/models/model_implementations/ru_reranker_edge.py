from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import CrossEncoderWrapper

ru_reranker_edge_150m = ModelMeta(
    loader=CrossEncoderWrapper,
    loader_kwargs=dict(max_length=512),
    name="sshalimov04/ru-reranker-edge-150m",
    revision="d3f4bbf84c4f180696a06a1565981ea37c65fc06",
    release_date="2026-09-06",
    languages=["rus-Cyrl"],
    n_parameters=149_000_000,
    n_embedding_parameters=38_682_624,
    memory_usage_mb=570,
    max_tokens=8192,
    embed_dim=None,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/sshalimov04/ru-reranker-teacher-scores",
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/sshalimov04/ru-reranker-edge-150m",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={"MIRACLRetrieval", "MIRACLReranking", "MMarcoRetrieval"},
    modalities=["text"],
    adapted_from="deepvk/RuModernBERT-base",
    model_type=["cross-encoder"],
)
