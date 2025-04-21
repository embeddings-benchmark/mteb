from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

frony_training_datasets = {
    "AI허브_금융법률": ["train"],
    "AI허브_기계독해": ["train"],
    "AI허브_뉴스기사": ["train"],
    "AI허브_도서자료": ["train"],
    "AI허브_도서자료_AUG": ["train"], # Augmented (Multi-Query & Multi-Passage)
    "AI허브_도서자료_EN_PASSAGE": ["train"], # Augmented (Language transfer to English on passage)
    "AI허브_도서자료_EN_QUERY": ["train"], # Augmented (Language transfer to English on query)
    "AI허브_도서자료_MD": ["train"], # Augmented (Style transfer to markdown on passage)
    "AI허브_숫자연산": ["train"],
    "AI허브_이벤트": ["train"],
    "AI허브_일반상식": ["train"],
    "AI허브_추상요약": ["train"],
    "AI허브_표정보": ["train"], 
    "AI허브_행정문서": ["train"],
    "AI허브_헬스케어": ["train"],
    "HF_miracl": ["train"],
    "HF_mr-tydi": ["train"],
    "KoStrategyQA": ["train"],
    "LGNLP": ["train"],
    "kakao-nli": ["train"],
    "kakao-sts": ["train"],
    "klue-mrc": ["train"],
    "klue-nli": ["train"],
    "klue-sts": ["train"],
    "msmarco": ["train"],
    "공공데이터포털-deepqa": ["train"],
}

FronyEmbed_tiny_ko_v1 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="FronyAI/frony-embed-tiny-ko-v1",
        revision="frony-embed-tiny-ko-v1-20250421",
        model_prompts={
           "query": "<Q>",
           "passage": "<P>",
        },
    ),
    name="FronyAI/frony-embed-tiny-ko-v1",
    revision="frony-embed-tiny-ko-v1-20250421",
    languages=["kor-Hang", "eng-Latn"],
    open_weights=True,
    release_date="2025-04-21",
    n_parameters=111_000_000,
    memory_usage_mb=2048,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/FronyAI/frony-embed-tiny-ko-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    training_datasets=frony_training_datasets,
    public_training_code=None,
    public_training_data=None,
)

FronyEmbed_small_ko_v1 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="FronyAI/frony-embed-small-ko-v1",
        revision="frony-embed-small-ko-v1-20250421",
        model_prompts={
           "query": "<Q>",
           "passage": "<P>",
        },
    ),
    name="FronyAI/frony-embed-small-ko-v1",
    languages=["kor-Hang", "eng-Latn"],
    open_weights=True,
    release_date="2025-04-21",
    n_parameters=111_000_000,
    memory_usage_mb=2048,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/FronyAI/frony-embed-small-ko-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    training_datasets=frony_training_datasets,
    public_training_code=None,
    public_training_data=None,
)

FronyEmbed_medium_ko_v1 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="FronyAI/frony-embed-medium-ko-v1",
        revision="frony-embed-medium-ko-v1-20250421",
        model_prompts={
           "query": "<Q>",
           "passage": "<P>",
        },
    ),
    name="FronyAI/frony-embed-medium-ko-v1",
    languages=["kor-Hang", "eng-Latn"],
    open_weights=True,
    release_date="2025-04-21",
    n_parameters=337_000_000,
    memory_usage_mb=2048,
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/FronyAI/frony-embed-medium-ko-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    training_datasets=frony_training_datasets,
    public_training_code=None,
    public_training_data=None,
)
