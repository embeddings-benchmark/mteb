from __future__ import annotations

from functools import partial
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from mteb.model_meta import ModelMeta

e5_paper_release_date = "2024-02-08"
xlmr_languages = [
    "afr_Latn",
    "amh_Latn",
    "ara_Latn",
    "asm_Latn",
    "aze_Latn",
    "bel_Latn",
    "bul_Latn",
    "ben_Latn",
    "ben_Beng",
    "bre_Latn",
    "bos_Latn",
    "cat_Latn",
    "ces_Latn",
    "cym_Latn",
    "dan_Latn",
    "deu_Latn",
    "ell_Latn",
    "eng_Latn",
    "epo_Latn",
    "spa_Latn",
    "est_Latn",
    "eus_Latn",
    "fas_Latn",
    "fin_Latn",
    "fra_Latn",
    "fry_Latn",
    "gle_Latn",
    "gla_Latn",
    "glg_Latn",
    "guj_Latn",
    "hau_Latn",
    "heb_Latn",
    "hin_Latn",
    "hin_Deva",
    "hrv_Latn",
    "hun_Latn",
    "hye_Latn",
    "ind_Latn",
    "isl_Latn",
    "ita_Latn",
    "jpn_Latn",
    "jav_Latn",
    "kat_Latn",
    "kaz_Latn",
    "khm_Latn",
    "kan_Latn",
    "kor_Latn",
    "kur_Latn",
    "kir_Latn",
    "lat_Latn",
    "lao_Latn",
    "lit_Latn",
    "lav_Latn",
    "mlg_Latn",
    "mkd_Latn",
    "mal_Latn",
    "mon_Latn",
    "mar_Latn",
    "msa_Latn",
    "mya_Latn",
    "nep_Latn",
    "nld_Latn",
    "nob_Latn",
    "orm_Latn",
    "ori_Latn",
    "pan_Latn",
    "pol_Latn",
    "pus_Latn",
    "por_Latn",
    "ron_Latn",
    "rus_Latn",
    "san_Latn",
    "snd_Latn",
    "sin_Latn",
    "slk_Latn",
    "slv_Latn",
    "som_Latn",
    "sqi_Latn",
    "srp_Latn",
    "sun_Latn",
    "swe_Latn",
    "swa_Latn",
    "tam_Latn",
    "tam_Taml",
    "tel_Latn",
    "tel_Telu",
    "tha_Latn",
    "tgl_Latn",
    "tur_Latn",
    "uig_Latn",
    "ukr_Latn",
    "urd_Latn",
    "urd_Arab",
    "uzb_Latn",
    "vie_Latn",
    "xho_Latn",
    "yid_Latn",
    "zho_Hant",
    "zho_Hans",
]


class E5Wrapper:
    """following the implementation within the Scandinavian Embedding Benchmark and the intfloat/multilingual-e5-small documentation."""

    def __init__(self, model_name: str, sep: str = " ", **kwargs: Any):
        self.model_name = model_name
        self.mdl = SentenceTransformer(model_name)
        self.sep = sep

    def to(self, device: torch.device) -> None:
        self.mdl.to(device)

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        return self.encode_queries(sentences, batch_size=batch_size, **kwargs)

    def encode_queries(self, queries: list[str], batch_size: int = 32, **kwargs: Any):
        sentences = ["query: " + sentence for sentence in queries]
        emb = self.mdl.encode(sentences, batch_size=batch_size, **kwargs)
        return emb

    def encode_corpus(
        self, corpus: list[dict[str, str]], batch_size: int = 32, **kwargs: Any
    ):
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()  # type: ignore
                if "title" in corpus
                else corpus["text"][i].strip()  # type: ignore
                for i in range(len(corpus["text"]))  # type: ignore
            ]
        else:
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip()
                if "title" in doc
                else doc["text"].strip()
                for doc in corpus
            ]
        sentences = ["passage: " + sentence for sentence in sentences]
        emb = self.mdl.encode(sentences, batch_size=batch_size, **kwargs)
        return emb


e5_mult_small = ModelMeta(
    loader=partial(E5Wrapper, model_name="intfloat/multilingual-e5-small"),  # type: ignore
    name="intfloat/multilingual-e5-small",
    languages=xlmr_languages,
    open_source=True,
    revision="e4ce9877abf3edfe10b0d82785e83bdcb973e22e",
    release_date=e5_paper_release_date,
)

e5_mult_base = ModelMeta(
    loader=partial(E5Wrapper, model_name="intfloat/multilingual-e5-base"),  # type: ignore
    name="intfloat/multilingual-e5-small",
    languages=xlmr_languages,
    open_source=True,
    revision="d13f1b27baf31030b7fd040960d60d909913633f",
    release_date=e5_paper_release_date,
)


e5_mult_large = ModelMeta(
    loader=partial(E5Wrapper, model_name="intfloat/multilingual-e5-large"),  # type: ignore
    name="intfloat/multilingual-e5-small",
    languages=xlmr_languages,
    open_source=True,
    revision="4dc6d853a804b9c8886ede6dda8a073b7dc08a81",
    release_date=e5_paper_release_date,
)
