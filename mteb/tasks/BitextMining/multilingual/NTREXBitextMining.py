from __future__ import annotations

from typing import Any

import datasets

from mteb.abstasks import AbsTaskBitextMining, CrosslingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

# Mapping from ISO 639-3 + script to ISO 639-1 used in NTREX
_LANGUAGES = [
    # "afr_Latn",
    # "amh_Ethi",
    "arb_Arab",
    # "aze_Latn",
    # "bak_Cyrl",
    # "bel_Cyrl",
    # "bem_Latn",
    # "ben_Beng",
    # "bod_Tibt",
    # "bos_Latn",
    # "bul_Cyrl",
    # "cat_Latn",
    # "ces_Latn",
    # "ckb_Arab",
    # "cym_Latn",
    # "dan_Latn",
    "deu_Latn",
    "div_Thaa",  # new BitextMining language
    "dzo_Tibt",  # new BitextMining language
    # "ell_Grek",
    "eng_Latn",
    # "est_Latn",
    # "eus_Latn",
    # "ewe_Latn",
    # "fao_Latn",
    "fas_Arab",  # new BitextMining language
    # "fij_Latn",
    "fil_Latn",  # new BitextMining language
    # "fin_Latn",
    "fra_Latn",
    "fuc_Latn",  # new BitextMining language
    # "gle_Latn",
    # "glg_Latn",
    # "guj_Gujr",
    # "hau_Latn",
    # "heb_Hebr",
    "hin_Deva",
    "hmn_Latn",  # new BitextMining language
    # "hrv_Latn",
    # "hun_Latn",
    # "hye_Armn",
    # "ibo_Latn",
    "ind_Latn",
    # "isl_Latn",
    "ita_Latn",
    "jpn_Jpan",
    # "kan_Knda",
    # "kat_Geor",
    # "kaz_Cyrl",
    # "khm_Khmr",
    # "kin_Latn",
    # "kir_Cyrl",
    "kmr_Latn",  # new BitextMining language
    "kor_Hang",
    # "lao_Laoo",
    "lav_Latn",  # new BitextMining language
    # "lit_Latn",
    # "ltz_Latn",
    # "mal_Mlym",
    # "mar_Deva",
    "mey_Arab",  # new BitextMining language
    # "mkd_Cyrl",
    "mlg_Latn",  # new BitextMining language
    # "mlt_Latn",
    # "mon_Mong",
    # "mri_Latn",
    "msa_Latn",  # new BitextMining language
    # "mya_Mymr",
    "nde_Latn",  # new BitextMining language
    "nep_Deva",  # new BitextMining language
    # "nld_Latn",
    # "nno_Latn",
    # "nob_Latn",
    # "nso_Latn",
    # "nya_Latn",
    "orm_Ethi",  # new BitextMining language
    # "pan_Guru",
    # "pol_Latn",
    "por_Latn",
    # "prs_Arab",
    "pus_Arab",  # new BitextMining language
    # "ron_Latn",
    "rus_Cyrl",
    "shi_Arab",  # new BitextMining language
    # "sin_Sinh",
    # "slk_Latn",
    # "slv_Latn",
    # "smo_Latn",
    # "sna_Latn",
    # "snd_Arab",
    # "som_Latn",
    "spa_Latn",
    # "sqi_Latn",
    "srp_Cyrl",  # new BitextMining language (only srp_Latn available)
    # "srp_Latn",
    # "ssw_Latn",
    "swa_Latn",  # new BitextMining language
    # "swe_Latn",
    "tah_Latn",  # new BitextMining language
    # "tam_Taml",
    # "tat_Cyrl",
    # "tel_Telu",
    # "tgk_Cyrl",
    "tha_Thai",
    # "tir_Ethi",
    "ton_Latn",  # new BitextMining language
    # "tsn_Latn",
    # "tuk_Latn",
    "tur_Latn",
    # "uig_Arab",
    # "ukr_Cyrl",
    # "urd_Arab",
    # "uzb_Latn",
    "ven_Latn",  # new BitextMining language
    "vie_Latn",
    # "wol_Latn",
    # "xho_Latn",
    # "yor_Latn",
    # "yue_Hant",
    "zho_Hans",
    # "zho_Hant",
    # "zul_Latn",
]

_SPLIT = ["test"]

# number of sentences to use for evaluation
_N = 256


def extend_lang_pairs() -> dict[str, list[str]]:
    # add all possible language pairs
    hf_lang_subset2isolang = {}
    for x in _LANGUAGES:
        if "-" not in x:
            for y in _LANGUAGES:
                if x != y:
                    pair = f"{x}-{y}"
                    hf_lang_subset2isolang[pair] = [
                        x.replace("_", "-"),
                        y.replace("_", "-"),
                    ]
    return hf_lang_subset2isolang


_EVAL_LANGS = extend_lang_pairs()


class NTREXBitextMining(AbsTaskBitextMining, CrosslingualTask):
    metadata = TaskMetadata(
        name="NTREXBitextMining",
        dataset={
            "path": "davidstap/NTREX",
            "revision": "fd20d54141b6da952d5c68a2989472892489da0f",
            "trust_remote_code": True,
        },
        description="NTREX is a News Test References for MT Evaluation from English into a total of 128 target languages.",
        reference="https://huggingface.co/datasets/davidstap/NTREX",
        type="BitextMining",
        category="s2s",
        eval_splits=_SPLIT,
        eval_langs=_EVAL_LANGS,
        main_score="f1",
        date=("2022-11-01", "2022-11-01"),
        form=["written"],
        domains=["News"],
        task_subtypes=[],
        license="CC-BY-SA-4.0",
        socioeconomic_status="medium",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="created",
        n_samples={"test": _N},
        avg_character_length={"test": 120},
        bibtex_citation="""
@inproceedings{federmann-etal-2022-ntrex,
    title = "{NTREX}-128 {--} News Test References for {MT} Evaluation of 128 Languages",
    author = "Federmann, Christian and Kocmi, Tom and Xin, Ying",
    booktitle = "Proceedings of the First Workshop on Scaling Up Multilingual Evaluation",
    month = "nov",
    year = "2022",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.sumeval-1.4",
    pages = "21--24",
}
""",
    )

    def load_data(self, **kwargs: Any) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = {}

        all_data = {
            l: datasets.load_dataset(
                name=l,
                split=f"test[:{_N}]",
                **self.metadata_dict["dataset"],
            )
            for l in _LANGUAGES
        }

        for lang in self.langs:
            l1, l2 = lang.split("-")
            l1_data = all_data[l1].rename_column("text", "sentence1")
            l2_data = all_data[l2].rename_column("text", "sentence2")
            assert l1_data.num_rows == l2_data.num_rows
            # Combine languages
            data = l1_data.add_column("sentence2", l2_data["sentence2"])
            self.dataset[lang] = datasets.DatasetDict({"test": data})

        self.data_loaded = True
