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


hybrid_bm25s_e5 = ModelMeta(
    loader=hybrid_bm25s_e5_loader,
    name="mteb/hybrid-bm25s-e5-small",
    model_type=["hybrid"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="0_1_0",
    release_date="2026-05-27",
    n_parameters=0,
    n_embedding_parameters=0,
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
    citation="",
)


def hybrid_baseline_encoder_e5_loader(model_name: str, **kwargs) -> HybridSearch:
    bm25 = mteb.get_model("mteb/baseline-random-encoder")
    dense = mteb.get_model("intfloat/multilingual-e5-small")
    return HybridSearch(
        models=[bm25, dense],
        fusion_strategy="rrf",
    )


hybrid_baseline_encoder_e5 = ModelMeta(
    loader=hybrid_baseline_encoder_e5_loader,
    name="mteb/hybrid-baseline_encoder-e5-small",
    model_type=["hybrid"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="0_1_0",
    release_date="2026-05-27",
    n_parameters=0,
    n_embedding_parameters=0,
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
    citation="",
)
