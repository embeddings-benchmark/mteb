"""Sentence models for evaluation on the Russian part of MTEB"""

from mteb.model_meta import ModelMeta

rubert_tiny2 = ModelMeta(
    name="cointegrated/rubert-tiny2",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="dad72b8f77c5eef6995dd3e4691b758ba56b90c3",
    release_date="2021-10-28",
)

rubert_tiny = ModelMeta(
    name="cointegrated/rubert-tiny",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="5441c5ea8026d4f6d7505ec004845409f1259fb1",
    release_date="2021-05-24",
)

sbert_large_nlu_ru = ModelMeta(
    name="ai-forever/sbert_large_nlu_ru",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="af977d5dfa46a3635e29bf0ef383f2df2a08d47a",
    release_date="2020-11-20",
)

sbert_large_mt_nlu_ru = ModelMeta(
    name="ai-forever/sbert_large_mt_nlu_ru",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="05300876c2b83f46d3ddd422a7f17e45cf633bb0",
    release_date="2021-05-18",
)
