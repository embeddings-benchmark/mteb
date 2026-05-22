"""Ingot-8B-R3 model registration for MTEB.

Ingot-8B-R3 is a gated embedding model by Voxell (JCorners/Ingot-8B-R3).
It is accessible via the Forge API (voxell.ai/forge) and not via direct
weight download. The SentenceTransformerEncoderWrapper loader is listed
for registry identification only; evaluation requires API access.
"""
from __future__ import annotations

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

ingot_8b_r3 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="JCorners/Ingot-8B-R3",
    revision="c44ba2a73964602dfa287f0b4d55981561387e86",
    release_date="2026-05-21",
    languages=["eng-Latn"],
    n_parameters=7_567_295_488,
    memory_usage_mb=14433,
    max_tokens=32768,
    embed_dim=4096,
    license="apache-2.0",
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/JCorners/Ingot-8B-R3",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=set(),
    adapted_from="Qwen/Qwen3-Embedding-8B",
    superseded_by=None,
    model_type=["dense"],
    citation=None,
)
