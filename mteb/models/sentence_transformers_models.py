"""Implementation of Sentence Transformers model validated in MTEB."""

from mteb.model_meta import ModelMeta

paraphrase_langs = [
    "ara_Arab",
    "bul_Cyrl",
    "cat_Latn",
    "ces_Latn",
    "dan_Latn",
    "deu_Latn",
    "ell_Grek",
    "eng_Latn",
    "spa_Latn",
    "est_Latn",
    "fas_Arab",
    "fin_Latn",
    "fra_Latn",
    "fra_Latn",
    "glg_Latn",
    "guj_Gujr",
    "heb_Hebr",
    "hin_Deva",
    "hrv_Latn",
    "hun_Latn",
    "hye_Armn",
    "ind_Latn",
    "ita_Latn",
    "jpn_Jpan",
    "kat_Geor",
    "kor_Hang",
    "kur_Arab",
    "lit_Latn",
    "lav_Latn",
    "mkd_Cyrl",
    "mon_Cyrl",
    "mar_Deva",
    "msa_Latn",
    "mya_Mymr",
    "nob_Latn",
    "nld_Latn",
    "pol_Latn",
    "por_Latn",
    "por_Latn",
    "ron_Latn",
    "rus_Cyrl",
    "slk_Latn",
    "slv_Latn",
    "sqi_Latn",
    "srp_Cyrl",
    "swe_Latn",
    "tha_Thai",
    "tur_Latn",
    "ukr_Cyrl",
    "urd_Arab",
    "vie_Latn",
    "zho_Hans",
    "zho_Hant",
]

all_MiniLM_L6_v2 = ModelMeta(
    name="sentence-transformers/all-MiniLM-L6-v2",
    languages=["eng-Latn"],
    open_source=True,
    revision="8b3219a92973c328a8e22fadcfa821b5dc75636a",  # can be any
    release_date="2021-08-30",
)

paraphrase_multilingual_MiniLM_L12_v2 = ModelMeta(
    name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    languages=paraphrase_langs,
    open_source=True,
    revision="bf3bf13ab40c3157080a7ab344c831b9ad18b5eb",  # can be any
    release_date="2019-11-01",  # release date of paper
)

paraphrase_multilingual_mpnet_base_v2 = ModelMeta(
    name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    languages=paraphrase_langs,
    open_source=True,
    revision="79f2382ceacceacdf38563d7c5d16b9ff8d725d6",  # can be any
    release_date="2019-11-01",  # release date of paper
)

labse = ModelMeta(
    name="sentence-transformers/LaBSE",
    languages=paraphrase_langs,
    open_source=True,
    revision="e34fab64a3011d2176c99545a93d5cbddc9a91b7",  # can be any
    release_date="2019-11-01",  # release date of paper
)

rubert_tiny2 = ModelMeta(
    name="cointegrated/rubert-tiny2",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="dad72b8f77c5eef6995dd3e4691b758ba56b90c3",  # can be any
    release_date="2021-10-28",
)

rubert_tiny = ModelMeta(
    name="cointegrated/rubert-tiny",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="5441c5ea8026d4f6d7505ec004845409f1259fb1",  # can be any
    release_date="2021-05-24",
)


sbert_large_nlu_ru = ModelMeta(
    name="ai-forever/sbert_large_nlu_ru",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="af977d5dfa46a3635e29bf0ef383f2df2a08d47a",  # can be any
    release_date="2020-11-20",
)

sbert_large_mt_nlu_ru = ModelMeta(
    name="ai-forever/sbert_large_mt_nlu_ru",
    languages=["rus_Cyrl"],
    open_source=True,
    revision="05300876c2b83f46d3ddd422a7f17e45cf633bb0",  # can be any
    release_date="2021-05-18",
)
