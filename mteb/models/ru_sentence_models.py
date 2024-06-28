"""Sentence models for evaluation on the Russian part of MTEB"""

from functools import partial

from mteb.model_meta import ModelMeta

from .e5_models import E5Wrapper

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

user_base_ru = ModelMeta(
    loader=partial(E5Wrapper, model_name="deepvk/USER-base"),  # type: ignore
    name="deepvk/USER-base",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="436a489a2087d61aa670b3496a9915f84e46c861",
    release_date="2024-06-10",
)

deberta_v1_ru = ModelMeta(
    name="deepvk/deberta-v1-base",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="bdd30b0e19757e6940c92c7aff19e8fc0a60dff4",
    release_date="2023-02-07",
)

rubert_base_cased = ModelMeta(
    name="DeepPavlov/rubert-base-cased",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="4036cab694767a299f2b9e6492909664d9414229",
    release_date="2020-03-04",
)

distilrubert_small_cased_conversational = ModelMeta(
    name="DeepPavlov/distilrubert-small-cased-conversational",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="e348066b4a7279b97138038299bddc6580a9169a",
    release_date="2022-06-28",
)

rubert_base_cased_sentence = ModelMeta(
    name="DeepPavlov/rubert-base-cased-sentence",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="78b5122d6365337dd4114281b0d08cd1edbb3bc8",
    release_date="2020-03-04",
)

labse_en_ru = ModelMeta(
    name="cointegrated/LaBSE-en-ru",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="cf0714e606d4af551e14ad69a7929cd6b0da7f7e",
    release_date="2021-06-10",
)

rubert_tiny_turbo = ModelMeta(
    name="sergeyzh/rubert-tiny-turbo",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="8ce0cf757446ce9bb2d5f5a4ac8103c7a1049054",
    release_date="2024-06-21",
)

labse_ru_turbo = ModelMeta(
    name="sergeyzh/LaBSE-ru-turbo",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="1940b046c6b5e125df11722b899130329d0a46da",
    release_date="2024-06-27",
)
