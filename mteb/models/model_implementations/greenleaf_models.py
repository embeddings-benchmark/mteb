"""GreenLeaf Law Embed models by JudicialMind.

Custom embedding models purpose-built for legal text retrieval.
Uses bidirectional attention on Qwen3 architecture with custom code
for flexible quantization (bf16, int8, binary).
"""

from __future__ import annotations

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

greenleaf_law_embed_tiny = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs={"trust_remote_code": True},
    name="judicialmind/greenleaf-law-embed-tiny",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="bff06c9474bda26e7b5883640821d6910689eb82",
    release_date="2026-08-22",
    n_parameters=595_776_512,
    n_embedding_parameters=155_189_248,
    memory_usage_mb=1136,
    max_tokens=32768,
    embed_dim=1024,
    license="apache-2.0",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=False,
    reference="https://huggingface.co/judicialmind/greenleaf-law-embed-tiny",
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/judicialmind/legal-training-dataset",
    training_datasets=None,
)
