from __future__ import annotations

from mteb.models import ModelMeta, SentenceTransformerEncoderWrapper

ko_embed_v0 = ModelMeta(
    name="NGA-KR/ko-embed-v0",
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(model_prompts={"query": "query: ", "document": "", "passage": "passage: "}),
    revision="f321741f682310b04b8d96afb3a36d177f1ab900",
    release_date="2026-09-10",
    languages=["kor-Hang", "eng-Latn"],
    open_weights=True,
    n_parameters=148731648,
    memory_usage_mb=284,
    embed_dim=768,
    max_tokens=512,
    license="apache-2.0",
    reference="https://huggingface.co/NGA-KR/ko-embed-v0",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets={'MSMARCO', 'GooAQ', 'NQ', 'HotpotQA'},
    adapted_from="skt/A.X-Encoder-base",
)
