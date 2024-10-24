from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta

MODEL_NAME = "jinaai/jina-embeddings-v3"
REVISION = "fa78e35d523dcda8d3b5212c7487cf70a4b277da"
XLMR_LANGUAGES = [
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


model_prompts = {
    "retrieval.query": "Represent the query for retrieving evidence documents: ",
    "retrieval.passage": "Represent the document for retrieval: ",
    "separation": "",
    "classification": "",
    "text-matching": "",
}

# Lora adaptation for specific downstream tasks.
# Empty represents for no-lora weights (or checkpoints after pair tuning).
supported_tasks = list(model_prompts.keys()) + [""]


def jina_embeddings_v3_loader(**kwargs):
    class JinaV3Wrapper(Wrapper):
        def encode(
            self,
            sentences: Sequence[str],
            task: str,
            *args,
            **kwargs: Any,
        ) -> np.ndarray:
            return super().encode(
                sentences, task=task, prompt_name=task, *args, **kwargs
            )

    return JinaV3Wrapper(**kwargs)


jina_embeddings_v3 = ModelMeta(
    loader=partial(
        jina_embeddings_v3_loader,
        model_name=MODEL_NAME,
        revision=REVISION,
        model_prompts=model_prompts,
        trust_remote_code=True,
    ),
    name=MODEL_NAME,
    languages=XLMR_LANGUAGES,
    open_source=True,  # CC-BY-NC-4.0
    revision=REVISION,
    release_date="2024-09-18",
)
