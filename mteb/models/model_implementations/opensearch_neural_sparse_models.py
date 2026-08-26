from __future__ import annotations

from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import SparseEncoderWrapper

v2_training_data = {
    "MSMARCO",
    # not in MTEB. see https://huggingface.co/datasets/sentence-transformers/embedding-training-data
    # "eli5_question_answer",
    # "gooaq_pairs",
    # "searchQA_top5_snippets",
    # "squad_pairs",
    # "stackexchange_duplicate_questions_body_body",
    # "stackexchange_duplicate_questions_title_title",
    # "stackexchange_duplicate_questions_title-body_title-body",
    # "WikiAnswers",
    # "wikihow",
    # "yahoo_answers_question_answer",
    # "yahoo_answers_title_answer",
    # "yahoo_answers_title_question",
}


v3_training_data = v2_training_data | {
    "HotpotQA",
    "FEVER",
    "FIQA",
    "NFCORPUS",
    "SCIFACT",
    # not in MTEB. see https://huggingface.co/datasets/sentence-transformers/embedding-training-data
    # "NQ-train_pairs",
    # "quora_duplicates",
}


opensearch_neural_sparse_encoding_doc_v3_gte = ModelMeta(
    name="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
    extra_requirements_groups=["sparse-encoder"],
    model_type=["sparse"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="a8abaa916125ee512a7a8f4d706d07eb0128a8e6",
    release_date="2025-06-18",
    n_parameters=136771584,
    n_embedding_parameters=23_440_896,
    memory_usage_mb=549,
    embed_dim=30522,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
    similarity_fn_name="dot",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    public_training_code="https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample",
    public_training_data=True,
    use_instructions=True,
    training_datasets=v3_training_data,
    loader=SparseEncoderWrapper,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
)


opensearch_neural_sparse_encoding_doc_v3_distill = ModelMeta(
    name="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
    extra_requirements_groups=["sparse-encoder"],
    model_type=["sparse"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="babf71f3c48695e2e53a978208e8aba48335e3c0",
    release_date="2025-03-28",
    n_parameters=66362880,
    n_embedding_parameters=23_440_896,
    memory_usage_mb=267,
    embed_dim=30522,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
    similarity_fn_name="dot",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    public_training_code="https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample",
    public_training_data=True,
    use_instructions=True,
    training_datasets=v3_training_data,
    loader=SparseEncoderWrapper,
)

opensearch_neural_sparse_encoding_doc_v2_distill = ModelMeta(
    name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    extra_requirements_groups=["sparse-encoder"],
    model_type=["sparse"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="8921a26c78b8559d6604eb1f5c0b74c079bee38f",
    release_date="2024-07-17",
    n_parameters=66362880,
    n_embedding_parameters=23_440_896,
    memory_usage_mb=267,
    embed_dim=30522,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    similarity_fn_name="dot",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    public_training_code="https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample",
    public_training_data=True,
    use_instructions=True,
    training_datasets=v2_training_data,
    loader=SparseEncoderWrapper,
)


opensearch_neural_sparse_encoding_doc_v2_mini = ModelMeta(
    name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini",
    extra_requirements_groups=["sparse-encoder"],
    model_type=["sparse"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="4af867a426867dfdd744097531046f4289a32fdd",
    release_date="2024-07-18",
    n_parameters=22713216,
    n_embedding_parameters=11_720_448,
    memory_usage_mb=86,
    embed_dim=30522,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini",
    similarity_fn_name="dot",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    public_training_code="https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample",
    public_training_data=True,
    use_instructions=True,
    training_datasets=v2_training_data,
    loader=SparseEncoderWrapper,
)

opensearch_neural_sparse_encoding_doc_v1 = ModelMeta(
    name="opensearch-project/opensearch-neural-sparse-encoding-doc-v1",
    extra_requirements_groups=["sparse-encoder"],
    model_type=["sparse"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="98cdcbd72867c547f72f2b7b7bed9cdf9f09922d",
    release_date="2024-03-07",
    n_parameters=132955194,
    n_embedding_parameters=23_440_896,
    memory_usage_mb=507,
    embed_dim=30522,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v1",
    similarity_fn_name="dot",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    public_training_code="https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample",
    public_training_data=True,
    use_instructions=True,
    training_datasets={
        "MSMARCO",
    },
    loader=SparseEncoderWrapper,
)
