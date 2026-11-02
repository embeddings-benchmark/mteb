from __future__ import annotations

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

etsu_embedding_08b_memory = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="Rebine/Etsu-Embedding-0.8B-Memory",
    revision="deaac7b1d4e62c12ecff3d3dbb2caa79b966c10b",
    release_date="2026-10-03",
    languages=["cmn-Hans", "eng-Latn"],
    n_parameters=752_393_024,
    n_embedding_parameters=254_279_680,
    memory_usage_mb=1505,
    max_tokens=2048,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors", "Sentence Transformers"],
    reference="https://huggingface.co/Rebine/Etsu-Embedding-0.8B-Memory",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=None,
    model_type=["dense"],
)
