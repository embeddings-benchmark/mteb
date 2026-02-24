from collections import defaultdict

import datasets
from datasets import DatasetDict
from tqdm.auto import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_FLEURS_EVAL_LANGS = {
    "af_za": ["afr-Latn"],  # Afrikaans
    "am_et": ["amh-Ethi"],  # Amharic
    "ar_eg": ["ara-Arab"],  # Arabic
    "as_in": ["asm-Beng"],  # Assamese
    "ast_es": ["ast-Latn"],  # Asturian
    "az_az": ["aze-Latn"],  # Azerbaijani
    "be_by": ["bel-Cyrl"],  # Belarusian
    "bn_in": ["ben-Beng"],  # Bengali
    "bs_ba": ["bos-Latn"],  # Bosnian
    "ca_es": ["cat-Latn"],  # Catalan
    "ceb_ph": ["ceb-Latn"],  # Cebuano
    "cmn_hans_cn": ["cmn-Hans"],  # Mandarin Chinese (Simplified)
    "yue_hant_hk": ["yue-Hant"],  # Cantonese Chinese (Traditional)
    "cs_cz": ["ces-Latn"],  # Czech
    "cy_gb": ["cym-Latn"],  # Welsh
    "da_dk": ["dan-Latn"],  # Danish
    "de_de": ["deu-Latn"],  # German
    "el_gr": ["ell-Grek"],  # Greek
    "en_us": ["eng-Latn"],  # English
    "es_419": ["spa-Latn"],  # Spanish (Latin America)
    "et_ee": ["est-Latn"],  # Estonian
    "fa_ir": ["fas-Arab"],  # Persian
    "ff_sn": ["ful-Latn"],  # Fula
    "fi_fi": ["fin-Latn"],  # Finnish
    "fil_ph": ["fil-Latn"],  # Filipino
    "fr_fr": ["fra-Latn"],  # French
    "ga_ie": ["gle-Latn"],  # Irish
    "gl_es": ["glg-Latn"],  # Galician
    "gu_in": ["guj-Gujr"],  # Gujarati
    "ha_ng": ["hau-Latn"],  # Hausa
    "he_il": ["heb-Hebr"],  # Hebrew
    "hi_in": ["hin-Deva"],  # Hindi
    "hr_hr": ["hrv-Latn"],  # Croatian
    "hu_hu": ["hun-Latn"],  # Hungarian
    "hy_am": ["hye-Armn"],  # Armenian
    "id_id": ["ind-Latn"],  # Indonesian
    "ig_ng": ["ibo-Latn"],  # Igbo
    "is_is": ["isl-Latn"],  # Icelandic
    "it_it": ["ita-Latn"],  # Italian
    "ja_jp": ["jpn-Jpan"],  # Japanese
    "jv_id": ["jav-Latn"],  # Javanese
    "ka_ge": ["kat-Geor"],  # Georgian
    "kam_ke": ["kam-Latn"],  # Kamba
    "kea_cv": ["kea-Latn"],  # Kabuverdianu
    "kk_kz": ["kaz-Cyrl"],  # Kazakh
    "km_kh": ["khm-Khmr"],  # Khmer
    "kn_in": ["kan-Knda"],  # Kannada
    "ko_kr": ["kor-Hang"],  # Korean
    "ckb_iq": ["ckb-Arab"],  # Sorani Kurdish
    "ky_kg": ["kir-Cyrl"],  # Kyrgyz
    "lb_lu": ["ltz-Latn"],  # Luxembourgish
    "lg_ug": ["lug-Latn"],  # Ganda
    "ln_cd": ["lin-Latn"],  # Lingala
    "lo_la": ["lao-Laoo"],  # Lao
    "lt_lt": ["lit-Latn"],  # Lithuanian
    "luo_ke": ["luo-Latn"],  # Luo
    "lv_lv": ["lvs-Latn"],  # Latvian (Standard)
    "mi_nz": ["mri-Latn"],  # Maori
    "mk_mk": ["mkd-Cyrl"],  # Macedonian
    "ml_in": ["mal-Mlym"],  # Malayalam
    "mn_mn": ["mon-Cyrl"],  # Mongolian
    "mr_in": ["mar-Deva"],  # Marathi
    "ms_my": ["msa-Latn"],  # Malay
    "mt_mt": ["mlt-Latn"],  # Maltese
    "my_mm": ["mya-Mymr"],  # Burmese
    "nb_no": ["nob-Latn"],  # Norwegian Bokmål
    "ne_np": ["npi-Deva"],  # Nepali
    "nl_nl": ["nld-Latn"],  # Dutch
    "nso_za": ["nso-Latn"],  # Northern Sotho
    "ny_mw": ["nya-Latn"],  # Nyanja
    "oc_fr": ["oci-Latn"],  # Occitan
    "om_et": ["orm-Latn"],  # Oromo
    "or_in": ["ori-Orya"],  # Odia (Oriya)
    "pa_in": ["pan-Guru"],  # Punjabi
    "pl_pl": ["pol-Latn"],  # Polish
    "ps_af": ["pus-Arab"],  # Pashto
    "pt_br": ["por-Latn"],  # Portuguese (Brazil)
    "ro_ro": ["ron-Latn"],  # Romanian
    "ru_ru": ["rus-Cyrl"],  # Russian
    "bg_bg": ["bul-Cyrl"],  # Bulgarian
    "sd_in": ["snd-Arab"],  # Sindhi
    "sk_sk": ["slk-Latn"],  # Slovak
    "sl_si": ["slv-Latn"],  # Slovenian
    "sn_zw": ["sna-Latn"],  # Shona
    "so_so": ["som-Latn"],  # Somali
    "sr_rs": ["srp-Cyrl"],  # Serbian
    "sv_se": ["swe-Latn"],  # Swedish
    "sw_ke": ["swh-Latn"],  # Swahili
    "ta_in": ["tam-Taml"],  # Tamil
    "te_in": ["tel-Telu"],  # Telugu
    "tg_tj": ["tgk-Cyrl"],  # Tajik
    "th_th": ["tha-Thai"],  # Thai
    "tr_tr": ["tur-Latn"],  # Turkish
    "uk_ua": ["ukr-Cyrl"],  # Ukrainian
    "umb_ao": ["umb-Latn"],  # Umbundu
    "ur_pk": ["urd-Arab"],  # Urdu
    "uz_uz": ["uzn-Latn"],  # Uzbek (Northern)
    "vi_vn": ["vie-Latn"],  # Vietnamese
    "wo_sn": ["wol-Latn"],  # Wolof
    "xh_za": ["xho-Latn"],  # Xhosa
    "yo_ng": ["yor-Latn"],  # Yoruba
    "zu_za": ["zul-Latn"],  # Zulu
}


class FleursA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FleursA2TRetrieval",
        description="Speech recordings with corresponding text transcriptions from the FLEURS dataset.",
        reference="https://github.com/google-research-datasets/fleurs",
        dataset={
            "path": "mteb/fleurs",
            "revision": "720cfffb024c8a4691f26a2b79d471b1f99a49a0",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=_FLEURS_EVAL_LANGS,
        main_score="hit_rate_at_5",
        date=("2021-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{conneau2023fleurs,
  author = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  booktitle = {2022 IEEE Spoken Language Technology Workshop (SLT)},
  organization = {IEEE},
  pages = {798--805},
  title = {Fleurs: Few-shot learning evaluation of universal representations of speech},
  year = {2023},
}
""",
    )

    def load_data(self, **kwargs):
        if getattr(self, "data_loaded", False):
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(
        self, id_col="path", text_col="transcription", audio_col="audio"
    ):
        """A2T: Query = audio, Corpus = text."""
        for lang in tqdm(self.hf_subsets, desc="Loading FleursA2TRetrieval subsets"):
            lang_dataset = datasets.load_dataset(
                self.metadata.dataset["path"],
                lang,
                revision=self.metadata.dataset["revision"],
            )

            for split in self.metadata.eval_splits:
                split_dataset = lang_dataset[split]

                queries_ds = split_dataset.select_columns(
                    [id_col, audio_col]
                ).rename_column(id_col, "id")

                corpus_ds = (
                    split_dataset.select_columns([id_col, text_col])
                    .rename_column(id_col, "id")
                    .rename_column(text_col, "text")
                )

                relevant_docs_ = {
                    str(row[id_col]): {str(row[id_col]): 1} for row in split_dataset
                }

                self.corpus[lang][split] = corpus_ds
                self.queries[lang][split] = queries_ds
                self.relevant_docs[lang][split] = relevant_docs_


class FleursT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FleursT2ARetrieval",
        description="Speech recordings with corresponding text transcriptions from the FLEURS dataset.",
        reference="https://github.com/google-research-datasets/fleurs",
        dataset={
            "path": "mteb/fleurs",
            "revision": "720cfffb024c8a4691f26a2b79d471b1f99a49a0",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=_FLEURS_EVAL_LANGS,
        main_score="hit_rate_at_5",
        date=("2021-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Retrieval"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{conneau2023fleurs,
  author = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  booktitle = {2022 IEEE Spoken Language Technology Workshop (SLT)},
  organization = {IEEE},
  pages = {798--805},
  title = {Fleurs: Few-shot learning evaluation of universal representations of speech},
  year = {2023},
}
""",
    )

    def load_data(self, **kwargs):
        if getattr(self, "data_loaded", False):
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(
        self, id_col="path", text_col="transcription", audio_col="audio"
    ):
        """T2A: Query = text, Corpus = audio."""
        for lang in tqdm(self.hf_subsets, desc="Loading FleursT2ARetrieval subsets"):
            lang_dataset = datasets.load_dataset(
                self.metadata.dataset["path"],
                lang,
                revision=self.metadata.dataset["revision"],
            )

            for split in self.metadata.eval_splits:
                split_dataset = lang_dataset[split]

                # Create datasets directly without intermediate lists
                queries_ds = (
                    split_dataset.select_columns([id_col, text_col])
                    .rename_column(id_col, "id")
                    .rename_column(text_col, "text")
                )

                corpus_ds = split_dataset.select_columns(
                    [id_col, audio_col]
                ).rename_column(id_col, "id")

                # Create relevant_docs mapping
                relevant_docs_ = {
                    str(row[id_col]): {str(row[id_col]): 1} for row in split_dataset
                }

                self.corpus[lang][split] = corpus_ds
                self.queries[lang][split] = queries_ds
                self.relevant_docs[lang][split] = relevant_docs_
