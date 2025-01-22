from __future__ import annotations

import logging

from mteb.model_meta import ModelMeta

from .bge_models import bge_full_data

logger = logging.getLogger(__name__)


cde_small_v1 = ModelMeta(
    loader=None,  # I will leave this at None for now,
    name="jxm/cde-small-v1",
    languages=["eng_Latn"],
    open_weights=True,
    revision="8d5736163718a8b65cd787b75ed61020d18bad3c",
    release_date="2024-09-24",
    n_parameters=int(281 * 1e6),  # Though the second-stage model is only 140M
    max_tokens=512,
    embed_dim=768,
    license="mit",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers"],
    reference="https://huggingface.co/jxm/cde-small-v1",
    use_instructions=True,
    adapted_from="nomic-ai/nomic-bert-2048",
    superseded_by="jxm/cde-small-v2",
    training_datasets=bge_full_data,
    public_training_code="https://github.com/jxmorris12/cde",
    public_training_data="https://huggingface.co/datasets/cfli/bge-full-data",
)

cde_small_v2 = ModelMeta(
    loader=None,  # I will leave this at None for now,
    name="jxm/cde-small-v2",
    languages=["eng_Latn"],
    open_weights=True,
    revision="a7e5882ad52c27ea2831fc8258f24379c25cb459",
    release_date="2025-01-13",
    n_parameters=int(306 * 1e6),  # Though the second-stage model is only 140M
    max_tokens=512,
    embed_dim=768,
    license="mit",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers"],
    reference="https://huggingface.co/jxm/cde-small-v1",
    use_instructions=True,
    adapted_from="answerdotai/ModernBERT-base",
    superseded_by="jxm/cde-small-v2",
    training_datasets=bge_full_data,
    public_training_code="https://github.com/jxmorris12/cde",
    public_training_data="https://huggingface.co/datasets/cfli/bge-full-data",
)
