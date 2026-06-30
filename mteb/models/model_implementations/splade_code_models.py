from __future__ import annotations

from mteb.models.model_implementations.opensearch_neural_sparse_models import (
    SparseEncoderWrapper,
)
from mteb.models.model_meta import ModelMeta

splade_code_languages = [
    "python-Code",
    "javascript-Code",
    "go-Code",
    "ruby-Code",
    "java-Code",
    "php-Code",
]

splade_code_06b = ModelMeta(
    name="naver/splade-code-06B",
    model_type=["sparse"],
    languages=splade_code_languages,
    open_weights=True,
    revision="e53a0b8bd312d83955598a392dc826b3fc4028f7",
    release_date="2026-02-24",
    n_parameters=596049920,
    n_embedding_parameters=155582464,
    memory_usage_mb=1137,
    embed_dim=151936,
    license="cc-by-nc-sa-4.0",
    max_tokens=40960,
    reference="https://huggingface.co/naver/splade-code-06B",
    similarity_fn_name="dot",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=None,
    loader=SparseEncoderWrapper,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
)

splade_code_8b = ModelMeta(
    name="naver/splade-code-8B",
    model_type=["sparse"],
    languages=splade_code_languages,
    open_weights=True,
    revision="fe9fb2fc9fd930187ede95085cd189c7dc5d55a4",
    release_date="2026-02-24",
    n_parameters=8365357488,
    n_embedding_parameters=622329856,
    memory_usage_mb=15956,
    embed_dim=151936,
    license="cc-by-nc-sa-4.0",
    max_tokens=40960,
    reference="https://huggingface.co/naver/splade-code-8B",
    similarity_fn_name="dot",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=None,
    loader=SparseEncoderWrapper,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
)
