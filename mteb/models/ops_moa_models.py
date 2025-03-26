from __future__ import annotations

from mteb.model_meta import ModelMeta

ops_moa_conan_embedding = ModelMeta(
    name="OpenSearch-AI/Ops-MoA-Conan-embedding-v1",
    revision="cd42de6d61c103047b7bcd780ef0dbaa9a9d0472",
    release_date="2025-03-26",
    languages=["zho_Hans"],
    loader=None,
    n_parameters=343 * 1e6,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=1536,
    license=None
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/OpenSearch-AI/Ops-MoA-Conan-embedding-v1",
    similarity_fn_name="cosine",
    use_instructions=None,
    training_datasets=None,
    superseded_by=None,
)

ops_moa_yuan_embedding = ModelMeta(
    name="OpenSearch-AI/Ops-MoA-Yuan-embedding-1.0",
    revision="09b8857bbd74f189d9bc45bf59adaf34f9378e17",
    release_date="2025-03-26",
    languages=["zho_Hans"],
    loader=None,
    n_parameters=343 * 1e6,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=1536,
    license=None
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/OpenSearch-AI/Ops-MoA-Yuan-embedding-1.0",
    similarity_fn_name="cosine",
    use_instructions=None,
    training_datasets=None,
    superseded_by=None,
)
