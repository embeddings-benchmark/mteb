from __future__ import annotations

import mteb
from mteb.models.hybrid_wrappers import HybridSearch
from mteb.models.model_meta import ModelMeta


def hybrid_bm25s_e5_loader(model_name: str, **kwargs) -> HybridSearch:
    bm25 = mteb.get_model("mteb/baseline-bm25s")
    dense = mteb.get_model("intfloat/multilingual-e5-small")
    return HybridSearch(
        models=[bm25, dense],
        fusion_strategy="rrf",
    )


hybrid_bm25s_e5_rrf = ModelMeta(
    loader=hybrid_bm25s_e5_loader,
    name="mteb/hybrid-rrf-baseline-bm25s-multilingual-e5-small",
    model_type=["hybrid"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="0.1.0",
    release_date="2024-02-08",
    n_parameters=118_000_000,
    n_embedding_parameters=96_014_208,
    memory_usage_mb=None,
    embed_dim=None,
    license=None,
    max_tokens=None,
    reference=None,
    similarity_fn_name=None,
    framework=[],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=None,
)


def hybrid_bm25s_e5_dbsf_loader(model_name: str, **kwargs) -> HybridSearch:
    bm25 = mteb.get_model("mteb/baseline-bm25s")
    dense = mteb.get_model("intfloat/multilingual-e5-small")
    return HybridSearch(
        models=[bm25, dense],
        fusion_strategy="dbsf",
    )


hybrid_bm25s_e5_dbsf = ModelMeta(
    loader=hybrid_bm25s_e5_dbsf_loader,
    name="mteb/hybrid-dbsf-baseline-bm25s-multilingual-e5-small",
    model_type=["hybrid"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="0.1.0",
    release_date="2024-02-08",
    n_parameters=118_000_000,
    n_embedding_parameters=96_014_208,
    memory_usage_mb=None,
    embed_dim=None,
    license=None,
    max_tokens=None,
    reference=None,
    similarity_fn_name=None,
    framework=[],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=None,
)


def hybrid_bm25s_e5_relative_score_fusion_loader(
    model_name: str, **kwargs
) -> HybridSearch:
    bm25 = mteb.get_model("mteb/baseline-bm25s")
    dense = mteb.get_model("intfloat/multilingual-e5-small")
    return HybridSearch(
        models=[bm25, dense],
        fusion_strategy="relative-score-fusion",
    )


hybrid_bm25s_e5_relative_score_fusion = ModelMeta(
    loader=hybrid_bm25s_e5_relative_score_fusion_loader,
    name="mteb/hybrid-relative-score-fusion-baseline-bm25s-multilingual-e5-small",
    model_type=["hybrid"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="0.1.0",
    release_date="2024-02-08",
    n_parameters=118_000_000,
    n_embedding_parameters=96_014_208,
    memory_usage_mb=None,
    embed_dim=None,
    license=None,
    max_tokens=None,
    reference=None,
    similarity_fn_name=None,
    framework=[],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=None,
)


def hybrid_baseline_encoder_e5_loader(model_name: str, **kwargs) -> HybridSearch:
    bm25 = mteb.get_model("mteb/baseline-random-encoder")
    dense = mteb.get_model("intfloat/multilingual-e5-small")
    return HybridSearch(
        models=[bm25, dense],
        fusion_strategy="rrf",
    )


hybrid_baseline_encoder_e5_rrf = ModelMeta(
    loader=hybrid_baseline_encoder_e5_loader,
    name="mteb/hybrid-rrf-baseline-random-encoder-multilingual-e5-small",
    model_type=["hybrid"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="0.1.0",
    release_date="2024-02-08",
    n_parameters=118_000_000,
    n_embedding_parameters=96_014_208,
    memory_usage_mb=None,
    embed_dim=None,
    license=None,
    max_tokens=None,
    reference=None,
    similarity_fn_name=None,
    framework=[],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=None,
)
