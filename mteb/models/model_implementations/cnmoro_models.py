"""Model definitions for cnmoro's static embedding models."""

from __future__ import annotations

import numpy as np

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

static_nomic_384_pten_v2_st = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="cnmoro/static-nomic-384-pten-v2-st",
    model_type=["dense"],
    languages=["eng-Latn", "por-Latn"],
    open_weights=True,
    revision="50fed70340b370b707f0f220010eac250bee7c18",
    release_date="2026-08-04",
    # one row per tokenizer token
    n_parameters=276214 * 384,
    n_embedding_parameters=276214 * 384,
    memory_usage_mb=405,
    max_tokens=np.inf,
    embed_dim=384,
    license="apache-2.0",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["NumPy", "Sentence Transformers", "safetensors"],
    reference="https://huggingface.co/cnmoro/static-nomic-384-pten-v2-st",
    use_instructions=False,
    adapted_from="nomic-ai/nomic-embed-text-v2-moe",
    superseded_by=None,
    # Distilled with Tokenlearn; finetuned on a pt-BR translation of MS MARCO triplets.
    training_datasets={"MSMARCO"},
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/cnmoro/AllTripletsMsMarco-PTBR",
    citation=None,
)
