"""Implementation of Sentence Transformers model validated in MTEB."""

from __future__ import annotations

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
    open_weights=True,
    revision="8b3219a92973c328a8e22fadcfa821b5dc75636a",  # can be any
    release_date="2021-08-30",
    n_parameters=22_700_000,
    memory_usage=None,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instuctions=False,
)

paraphrase_multilingual_MiniLM_L12_v2 = ModelMeta(
    name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    languages=paraphrase_langs,
    open_weights=True,
    revision="bf3bf13ab40c3157080a7ab344c831b9ad18b5eb",  # can be any
    release_date="2019-11-01",  # release date of paper
    n_parameters=118_000_000,
    memory_usage=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instuctions=False,
)

paraphrase_multilingual_mpnet_base_v2 = ModelMeta(
    name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    languages=paraphrase_langs,
    open_weights=True,
    revision="79f2382ceacceacdf38563d7c5d16b9ff8d725d6",  # can be any
    release_date="2019-11-01",  # release date of paper
    n_parameters=278_000_000,
    memory_usage=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instuctions=False,
)

labse = ModelMeta(
    name="sentence-transformers/LaBSE",
    languages=paraphrase_langs,
    open_weights=True,
    revision="e34fab64a3011d2176c99545a93d5cbddc9a91b7",  # can be any
    release_date="2019-11-01",  # release date of paper
    n_parameters=471_000_000,
    memory_usage=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/LaBSE",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instuctions=False,
)
