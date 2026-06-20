from __future__ import annotations

import mteb
from mteb.models.hybrid_wrappers import HybridSearch
from mteb.models.model_meta import ModelMeta, ScoringFunction

from .e5_models import (
    E5_PAPER_RELEASE_DATE,
    ME5_TRAINING_DATA,
    MULTILINGUAL_E5_CITATION,
)
from .facebookai import XLMR_LANGUAGES


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
    languages=XLMR_LANGUAGES,
    open_weights=True,
    revision="0.1.0",
    release_date=E5_PAPER_RELEASE_DATE,
    n_parameters=118_000_000,
    n_embedding_parameters=96_014_208,
    memory_usage_mb=449,
    embed_dim=384,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/intfloat/multilingual-e5-small",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "ONNX", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=ME5_TRAINING_DATA,
    adapted_from="microsoft/Multilingual-MiniLM-L12-H384",
    citation=MULTILINGUAL_E5_CITATION,
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
    languages=XLMR_LANGUAGES,
    open_weights=True,
    revision="0.1.0",
    release_date=E5_PAPER_RELEASE_DATE,
    n_parameters=118_000_000,
    n_embedding_parameters=96_014_208,
    memory_usage_mb=449,
    embed_dim=384,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/intfloat/multilingual-e5-small",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "ONNX", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=ME5_TRAINING_DATA,
    adapted_from="microsoft/Multilingual-MiniLM-L12-H384",
    citation=MULTILINGUAL_E5_CITATION,
)


def hybrid_bm25s_e5_rsf_loader(model_name: str, **kwargs) -> HybridSearch:
    bm25 = mteb.get_model("mteb/baseline-bm25s")
    dense = mteb.get_model("intfloat/multilingual-e5-small")
    return HybridSearch(
        models=[bm25, dense],
        fusion_strategy="rsf",
    )


hybrid_bm25s_e5_rsf = ModelMeta(
    loader=hybrid_bm25s_e5_rsf_loader,
    name="mteb/hybrid-rsf-baseline-bm25s-multilingual-e5-small",
    model_type=["hybrid"],
    languages=XLMR_LANGUAGES,
    open_weights=True,
    revision="0.1.0",
    release_date=E5_PAPER_RELEASE_DATE,
    n_parameters=118_000_000,
    n_embedding_parameters=96_014_208,
    memory_usage_mb=449,
    embed_dim=384,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/intfloat/multilingual-e5-small",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "ONNX", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=ME5_TRAINING_DATA,
    adapted_from="microsoft/Multilingual-MiniLM-L12-H384",
    citation=MULTILINGUAL_E5_CITATION,
)
