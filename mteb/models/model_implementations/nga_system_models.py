from __future__ import annotations

from mteb.models import ModelMeta, SentenceTransformerEncoderWrapper
from mteb.models.model_meta import ScoringFunction

ko_embed_v0 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(
        model_prompts={"query": "query: ", "document": "", "passage": "passage: "}
    ),
    name="NGA-KR/ko-embed-v0",
    revision="f321741f682310b04b8d96afb3a36d177f1ab900",
    release_date="2026-09-10",
    languages=["eng-Latn", "kor-Hang"],
    n_parameters=148731648,
    n_embedding_parameters=38400000,
    memory_usage_mb=284,
    max_tokens=512,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "safetensors"],
    reference="https://huggingface.co/NGA-KR/ko-embed-v0",
    similarity_fn_name=ScoringFunction.COSINE,
    training_datasets={"MSMARCO", "GooAQ", "NQ", "HotpotQA"},
    adapted_from="skt/A.X-Encoder-base",
    modalities=["text"],
    model_type=["dense"],
)

ko_embed_cls = ModelMeta(
    name="NGA-KR/ko-embed-cls",
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(),
    revision="64bf48823ed89c0e14fe8f00030c6f1e68053d2c",
    release_date="2026-09-10",
    languages=["kor-Hang", "eng-Latn"],
    open_weights=True,
    n_parameters=567754752,
    memory_usage_mb=1083,
    embed_dim=1024,
    max_tokens=128,
    license="mit",
    reference="https://huggingface.co/NGA-KR/ko-embed-cls",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"KLUE-TC"},
    adapted_from="BAAI/bge-m3",
    modalities=["text"],
    model_type=["dense"],
)
