from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

GRANITE_LANGUAGES = [
    "ara_Latn",
    "ces_Latn",
    "deu_Latn",
    "eng_Latn",
    "spa_Latn",
    "fra_Latn",
    "ita_Latn",
    "jpn_Latn",
    "kor_Latn",
    "nld_Latn",
    "por_Latn",
    "zho_Hant",
    "zho_Hans",
]


granite_107m_multilingual = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="ibm-granite/granite-embedding-107m-multilingual",
        revision="47db56afe692f731540413c67dd818ff492277e7",
    ),
    name="ibm-granite/granite-embedding-107m-multilingual",
    languages=GRANITE_LANGUAGES,
    open_weights=True,
    revision="47db56afe692f731540413c67dd818ff492277e7",
    release_date="2024-12-18",
    n_parameters=107_000_000,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=None,
)

granite_278m_multilingual = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="ibm-granite/granite-embedding-278m-multilingual",
        revision="84e3546b88b0cb69f8078608a1df558020bcbf1f",
    ),
    name="ibm-granite/granite-embedding-278m-multilingual",
    languages=GRANITE_LANGUAGES,
    open_weights=True,
    revision="84e3546b88b0cb69f8078608a1df558020bcbf1f",
    release_date="2024-12-18",
    n_parameters=278_000_000,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=None,
)

granite_30m_english = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="ibm-granite/granite-embedding-30m-english",
        revision="eddbb57470f896b5f8e2bfcb823d8f0e2d2024a5",
    ),
    name="ibm-granite/granite-embedding-30m-english",
    languages=["eng_Latn"],
    open_weights=True,
    revision="eddbb57470f896b5f8e2bfcb823d8f0e2d2024a5",
    release_date="2024-12-18",
    n_parameters=30_000_000,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/ibm-granite/granite-embedding-30m-english",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=None,
)

granite_125m_english = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="ibm-granite/granite-embedding-125m-english",
        revision="e48d3a5b47eaa18e3fe07d4676e187fd80f32730",
    ),
    name="ibm-granite/granite-embedding-125m-english",
    languages=["eng_Latn"],
    open_weights=True,
    revision="e48d3a5b47eaa18e3fe07d4676e187fd80f32730",
    release_date="2024-12-18",
    n_parameters=125_000_000,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/ibm-granite/granite-embedding-125m-english",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=None,
)
