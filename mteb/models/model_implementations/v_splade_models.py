from __future__ import annotations

from mteb.models.model_implementations.colpali_models import COLPALI_TRAINING_DATA
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SparseEncoderWrapper

V_SPLADE_CITATION = """@misc{cho2026vsplade,
  title         = {Inference-Free Multimodal Learned Sparse Retrieval for Production-Scale Visual Document Search},
  author        = {Cho, Gyu-Hwung and Lee, Youngjune and Jeong, Kiyoon and Lee, Siyoung and Han, Sanggyu and Dejean, Herv{\\'e} and Clinchant, St{\\'e}phane and Hwang, Seung-won},
  year          = {2026},
  eprint        = {2605.30917},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR}
}"""

RLHN_680K_TRAINING_DATA = {
    # from https://huggingface.co/datasets/rlhn/rlhn-680K
    "ArguAna",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NQ",
    "SciDocsRR",
}

V_SPLADE_TRAINING_DATA = COLPALI_TRAINING_DATA | RLHN_680K_TRAINING_DATA

v_splade_quality = ModelMeta(
    name="naver/v-splade-quality",
    model_type=["sparse"],
    languages=["eng-Latn"],
    modalities=["text", "image"],
    open_weights=True,
    revision="99bdc93f42460e595b2fb1e78b96edd44e898441",
    release_date="2026-07-01",
    n_parameters=330010817,
    n_embedding_parameters=38713344,
    memory_usage_mb=629,
    embed_dim=50368,
    license="apache-2.0",
    max_tokens=7999,
    reference="https://huggingface.co/naver/v-splade-quality",
    similarity_fn_name=ScoringFunction.DOT_PRODUCT,
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    public_training_code="https://github.com/naver/v-splade",
    public_training_data=True,
    use_instructions=False,
    training_datasets=V_SPLADE_TRAINING_DATA,
    citation=V_SPLADE_CITATION,
    loader=SparseEncoderWrapper,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
    extra_requirements_groups=["sparse-encoder"],
)

v_splade_efficient = ModelMeta(
    name="naver/v-splade-efficient",
    model_type=["sparse"],
    languages=["eng-Latn"],
    modalities=["text", "image"],
    open_weights=True,
    revision="ab0c2260c6d78bcb7d05076a9407a71f55d57eb1",
    release_date="2026-07-01",
    n_parameters=330010817,
    n_embedding_parameters=38713344,
    memory_usage_mb=629,
    embed_dim=50368,
    license="apache-2.0",
    max_tokens=7999,
    reference="https://huggingface.co/naver/v-splade-efficient",
    similarity_fn_name=ScoringFunction.DOT_PRODUCT,
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    public_training_code="https://github.com/naver/v-splade",
    public_training_data=True,
    use_instructions=False,
    training_datasets=V_SPLADE_TRAINING_DATA,
    citation=V_SPLADE_CITATION,
    loader=SparseEncoderWrapper,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
    extra_requirements_groups=["sparse-encoder"],
)
