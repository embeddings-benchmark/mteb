from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

LANGUAGES_V2_0 = [
    "afr-Latn",
    "ara-Arab",
    "aze-Latn",
    "bel-Cyrl",
    "bul-Cyrl",
    "ben-Beng",
    "cat-Latn",
    "ceb-Latn",
    "ces-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "spa-Latn",
    "est-Latn",
    "eus-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "glg-Latn",
    "guj-Gujr",
    "heb-Hebr",
    "hin-Deva",
    "hrv-Latn",
    "hat-Latn",
    "hun-Latn",
    "hye-Armn",
    "ind-Latn",
    "isl-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "jav-Latn",
    "kat-Geor",
    "kaz-Cyrl",
    "khm-Khmr",
    "kan-Knda",
    "kor-Hang",
    "kir-Cyrl",
    "lao-Laoo",
    "lit-Latn",
    "lav-Latn",
    "mkd-Cyrl",
    "mal-Mlym",
    "mon-Cyrl",
    "mar-Deva",
    "msa-Latn",
    "mya-Mymr",
    "nep-Deva",
    "nld-Latn",
    "pan-Guru",
    "pol-Latn",
    "por-Latn",
    "que-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "sin-Sinh",
    "slk-Latn",
    "slv-Latn",
    "som-Latn",
    "sqi-Latn",
    "srp-Cyrl",
    "swe-Latn",
    "swa-Latn",
    "tam-Taml",
    "tel-Telu",
    "tha-Thai",
    "tgl-Latn",
    "tur-Latn",
    "ukr-Cyrl",
    "urd-Arab",
    "vie-Latn",
    "yor-Latn",
    "zho-Hans",
]

arctic_v1_training_datasets = {
    # source: https://arxiv.org/pdf/2405.05374
    # splits not specified to assuming everything
    # in MTEB
    "NQ": ["test"],
    "NQ-NL": ["test"],  # translated from NQ (not trained on)
    "NQHardNegatives": ["test"],
    "NQ-PL": ["test"],
    "HotPotQA": ["test"],  # translated, not trained on
    "HotPotQAHardNegatives": ["test"],
    "HotPotQA-PL": ["test"],  # translated from hotpotQA (not trained on)
    "HotpotQA-NL": ["test"],  # translated from hotpotQA (not trained on)
    "FEVER": ["test"],
    "FEVER-NL": ["test"],  # translated from FEVER (not trained on)
    "FEVERHardNegatives": ["test"],
    # not in MTEB
    # trained on stack exchange (title-body)
    # "stackexchange": [],
    # potentially means that:
    # "StackExchangeClusteringP2P": ["test"],
    # "StackExchangeClusteringP2P.v2": ["test"],
    # "StackExchangeClustering": ["test"],
    # "StackExchangeClustering.v2": ["test"],
    # not in MTEB
    # "paq": [],
    # "s2orc": [],
    # "other": [],  # undisclosed including webdata
}  # also use synthetic

arctic_v2_training_datasets = {
    **arctic_v1_training_datasets,
    "MIRACLRetrieval": ["train"],
    "MIRACLRetrievalHardNegatives": ["train"],
    "MIRACLReranking": ["train"],
}

arctic_embed_xs = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="Snowflake/snowflake-arctic-embed-xs",
        revision="742da4f66e1823b5b4dbe6c320a1375a1fd85f9e",
    ),
    name="Snowflake/snowflake-arctic-embed-xs",
    revision="742da4f66e1823b5b4dbe6c320a1375a1fd85f9e",
    release_date="2024-07-08",  # initial commit of hf model.
    languages=["eng-Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=22_600_000,
    memory_usage_mb=86,
    max_tokens=512,
    embed_dim=384,
    license="apache-2.0",
    reference="https://huggingface.co/Snowflake/snowflake-arctic-embed-xs",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from="sentence-transformers/all-MiniLM-L6-v2",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=arctic_v1_training_datasets,
)


arctic_embed_s = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="Snowflake/snowflake-arctic-embed-s",
        revision="d3c1d2d433dd0fdc8e9ca01331a5f225639e798f",
    ),
    name="Snowflake/snowflake-arctic-embed-s",
    revision="d3c1d2d433dd0fdc8e9ca01331a5f225639e798f",
    release_date="2024-04-12",  # initial commit of hf model.
    languages=["eng-Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=32_200_000,
    memory_usage_mb=127,
    max_tokens=512,
    embed_dim=384,
    license="apache-2.0",
    reference="https://huggingface.co/Snowflake/snowflake-arctic-embed-s",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from="intfloat/e5-small-unsupervised",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,  # couldn't find
    training_datasets=arctic_v1_training_datasets,
)


arctic_embed_m = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="Snowflake/snowflake-arctic-embed-m",
        revision="cc17beacbac32366782584c8752220405a0f3f40",
    ),
    name="Snowflake/snowflake-arctic-embed-m",
    revision="cc17beacbac32366782584c8752220405a0f3f40",
    release_date="2024-04-12",  # initial commit of hf model.
    languages=["eng-Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=109_000_000,
    memory_usage_mb=415,
    max_tokens=512,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/Snowflake/snowflake-arctic-embed-m",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from="intfloat/e5-base-unsupervised",
    superseded_by="Snowflake/snowflake-arctic-embed-m-v1.5",
    public_training_code=None,
    public_training_data=None,  # couldn't find
    training_datasets=arctic_v1_training_datasets,
)

arctic_embed_m_long = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="Snowflake/snowflake-arctic-embed-m-long",
        revision="89d0f6ab196eead40b90cb6f9fefec01a908d2d1",
        trust_remote_code=True,
    ),
    name="Snowflake/snowflake-arctic-embed-m-long",
    revision="89d0f6ab196eead40b90cb6f9fefec01a908d2d1",
    release_date="2024-04-12",  # initial commit of hf model.
    languages=["eng-Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=137_000_000,
    memory_usage_mb=522,
    max_tokens=2048,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from="nomic-ai/nomic-embed-text-v1-unsupervised",
    superseded_by="Snowflake/snowflake-arctic-embed-m-v2.0",
    public_training_code=None,
    public_training_data=None,  # couldn't find
    training_datasets=arctic_v1_training_datasets,
)

arctic_embed_l = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="Snowflake/snowflake-arctic-embed-l",
        revision="9a9e5834d2e89cdd8bb72b64111dde496e4fe78c",
    ),
    name="Snowflake/snowflake-arctic-embed-l",
    revision="9a9e5834d2e89cdd8bb72b64111dde496e4fe78c",
    release_date="2024-04-12",  # initial commit of hf model.
    languages=["eng-Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=335_000_000,
    memory_usage_mb=1274,
    max_tokens=512,
    embed_dim=1024,
    license="apache-2.0",
    reference="https://huggingface.co/Snowflake/snowflake-arctic-embed-l",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from="intfloat/e5-base-unsupervised",
    superseded_by="Snowflake/snowflake-arctic-embed-l-v2.0",
    public_training_code=None,
    public_training_data=None,  # couldn't find
    training_datasets=arctic_v1_training_datasets,
)

arctic_embed_m_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="Snowflake/snowflake-arctic-embed-m-v1.5",
        revision="97eab2e17fcb7ccb8bb94d6e547898fa1a6a0f47",
        model_prompts={
            "query": "Represent this sentence for searching relevant passages: "
        },
    ),
    name="Snowflake/snowflake-arctic-embed-m-v1.5",
    revision="97eab2e17fcb7ccb8bb94d6e547898fa1a6a0f47",
    release_date="2024-07-08",  # initial commit of hf model.
    languages=["eng-Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=109_000_000,
    memory_usage_mb=415,
    max_tokens=512,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from=None,
    superseded_by="Snowflake/snowflake-arctic-embed-m-v2.0",
    public_training_code=None,
    public_training_data=None,
    training_datasets=arctic_v1_training_datasets,
)

arctic_embed_m_v2_0 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="Snowflake/snowflake-arctic-embed-m-v2.0",
        revision="f2a7d59d80dfda5b1d14f096f3ce88bb6bf9ebdc",
        trust_remote_code=True,
    ),
    name="Snowflake/snowflake-arctic-embed-m-v2.0",
    revision="f2a7d59d80dfda5b1d14f096f3ce88bb6bf9ebdc",
    release_date="2024-12-04",  # initial commit of hf model.
    languages=LANGUAGES_V2_0,
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=305_000_000,
    memory_usage_mb=1165,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from="Alibaba-NLP/gte-multilingual-base",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,  # couldn't find
    training_datasets=arctic_v2_training_datasets,
)

arctic_embed_l_v2_0 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
        revision="edc2df7b6c25794b340229ca082e7c78782e6374",
    ),
    name="Snowflake/snowflake-arctic-embed-l-v2.0",
    revision="edc2df7b6c25794b340229ca082e7c78782e6374",
    release_date="2024-12-04",  # initial commit of hf model.
    languages=LANGUAGES_V2_0,
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=568_000_000,
    memory_usage_mb=2166,
    max_tokens=8192,
    embed_dim=1024,
    license="apache-2.0",
    reference="https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from="BAAI/bge-m3-retromae",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,  # couldn't find
    training_datasets=arctic_v2_training_datasets,
)
