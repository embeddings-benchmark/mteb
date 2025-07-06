from __future__ import annotations

from mteb.model_meta import ModelMeta

listconranker_training_datasets = {
    "CMedQAv1-reranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "MMarcoReranking": ["train"],
    "T2Reranking": ["train"],
    # 'Huatuo26M-Lite': ['train'],
    # 'MARC': ['train'],
    # 'XL-sum-chinese_simplified': ['train'],
    # 'CSL': ['train'],
}

listconranker = ModelMeta(
    loader=None,  # implemented in v2.0.0 branch
    name="ByteDance/ListConRanker",
    languages=["zho-Hans"],
    open_weights=True,
    revision="95ae6a5f422a916bc36520f0f3e198e7d91520a0",
    release_date="2024-12-11",
    n_parameters=401_000_000,
    memory_usage_mb=1242,
    similarity_fn_name="cosine",
    training_datasets=listconranker_training_datasets,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/ByteDance/ListConRanker",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    is_cross_encoder=True,
)
