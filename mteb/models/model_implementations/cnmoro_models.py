"""Model definitions for cnmoro's static embedding models."""

from __future__ import annotations

import numpy as np

from mteb.models.model_meta import ModelMeta, ScoringFunction

from .model2vec_models import Model2VecModel

static_nomic_384_pten_v2 = ModelMeta(
    loader=Model2VecModel,
    name="cnmoro/static-nomic-384-pten-v2",
    model_type=["dense"],
    languages=["eng-Latn", "por-Latn"],
    open_weights=True,
    revision="08981248ac665f19e71df149a2f2b5ef9b514bd7",
    release_date="2026-05-29",
    # 32000 x 384 embedding matrix + 276214 per-token scalar weights
    n_parameters=32000 * 384 + 276214,
    n_embedding_parameters=32000 * 384 + 276214,
    memory_usage_mb=50,
    max_tokens=np.inf,
    embed_dim=384,
    license="apache-2.0",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["NumPy", "safetensors"],
    reference="https://huggingface.co/cnmoro/static-nomic-384-pten-v2",
    use_instructions=False,
    adapted_from="nomic-ai/nomic-embed-text-v2-moe",
    superseded_by=None,
    # Distilled with Tokenlearn; finetuned on a pt-BR translation of MS MARCO triplets.
    training_datasets={"MSMARCO"},
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/cnmoro/AllTripletsMsMarco-PTBR",
    citation=None,
    extra_requirements_groups=["model2vec"],
)
