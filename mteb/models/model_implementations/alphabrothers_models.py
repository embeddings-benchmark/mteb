from __future__ import annotations

from mteb.models import ModelMeta, SentenceTransformerEncoderWrapper

alpha_sts_v0 = ModelMeta(
    name="alphabrothers/alpha-sts-v0",
    loader=SentenceTransformerEncoderWrapper,
    revision="e737a8ac80fcd6c33f1945f2596c853a18f4cd06",
    release_date="2026-10-07",
    languages=["kor-Hang"],
    open_weights=True,
    n_parameters=567754752,
    n_embedding_parameters=256002048,
    memory_usage_mb=2166,
    embed_dim=1024,
    max_tokens=128,
    license="apache-2.0",
    reference="https://huggingface.co/alphabrothers/alpha-sts-v0",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"KLUE-STS", "KorSTS"},
    adapted_from="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
    modalities=["text"],
    model_type=["dense"],
)
