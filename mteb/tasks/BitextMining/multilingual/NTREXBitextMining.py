from __future__ import annotations

from typing import Any

import datasets

from mteb.abstasks import AbsTaskBitextMining, CrosslingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_BRIDGE_LANGUAGES = (
    "arb_Arab",
    "ben_Beng",
    "deu_Latn",
    "ell_Grek",
    "eng_Latn",
    "fas_Arab",
    "fin_Latn",
    "fra_Latn",
    "heb_Hebr",
    "hin_Deva",
    "hun_Latn",
    "ind_Latn",
    "jpn_Jpan",
    "kor_Hang",
    "lit_Latn",
    "nld_Latn",
    "pol_Latn",
    "por_Latn",
    "rus_Cyrl",
    "spa_Latn",
    "swa_Latn",
    "swe_Latn",
    "tam_Taml",
    "tur_Latn",
    "vie_Latn",
    "zho_Hant",
    "zul_Latn",
)

# Mapping from ISO 639-3 + script to ISO 639-1 used in NTREX
_LANGUAGES = {
    "afr_Latn": {"group": "Germanic"},
    "amh_Ethi": {"group": "Ethiopian/Niger-Congo/Afro-Asiatic/Cushitic"},
    "arb_Arab": {"group": "Arabic/Semitic/Iranian"},
    "aze_Latn": {"group": "Turkic"},
    "bak_Cyrl": {"group": "Turkic"},
    "bel_Cyrl": {"group": "Slavic"},
    # Manually added to group (not in M2M100 languages)
    "bem_Latn": {"group": "Niger-Congo/Afro-Asiatic/Cushitic"},
    "ben_Beng": {"group": "Indo-Aryan/Tamil/Dravidian"},
    # Manually added to group (not in M2M100 languages)
    "bod_Tibt": {"group": "Sino-Tibetan/Khmer/Kra-Dai/Mongolic"},
    "bos_Latn": {"group": "Slavic"},
    "bul_Cyrl": {"group": "Slavic"},
    "cat_Latn": {"group": "Romance"},
    # Manually added to group (not in M2M100 languages)
    "ces_Latn": {"group": "Slavic"},
    # Manually added to group (not in M2M100 languages)
    "ckb_Arab": {"group": "Arabic/Semitic/Iranian"},
    "cym_Latn": {"group": "Celtic/Irish"},
    "dan_Latn": {"group": "Germanic"},
    "deu_Latn": {"group": "Germanic"},
    # new BitextMining language. Manually added to group (not in M2M100 languages)
    "div_Thaa": {"group": "Indo-Aryan/Tamil/Dravidian"},
    # new BitextMining language. Manually added to group (not in M2M100 languages)
    "dzo_Tibt": {"group": "Sino-Tibetan/Khmer/Kra-Dai/Mongolic"},
    "ell_Grek": {"group": "Albanian/Armenian/Kartvelian/Hellenic"},
    "eng_Latn": {"group": "Germanic"},
    # Manually added to group (not in M2M100 languages), Basque closest to Indo-European
    "eus_Latn": {"group": "Indo-Aryan/Tamil/Dravidian"},
    # Manually added to group (not in M2M100 languages)
    "ewe_Latn": {"group": "Niger-Congo/Afro-Asiatic/Cushitic"},
    # Manually added to group (not in M2M100 languages)
    "fao_Latn": {"group": "Germanic"},
    # new BitextMining language
    "fas_Arab": {"group": "Arabic/Semitic/Iranian"},
    # Manually added to group (not in M2M100 languages)
    "fij_Latn": {"group": "Malayo-Polyn/Philippine/Dravidian"},
    # new BitextMining language. Manually added to group (not in M2M100 languages)
    "fil_Latn": {"group": "Malayo-Polyn/Philippine/Dravidian"},
    "fin_Latn": {"group": "Uralic/Baltic"},
    "fra_Latn": {"group": "Romance"},
    # new BitextMining language. Manually added to group (not in M2M100 languages)
    "fuc_Latn": {"group": "Niger-Congo/Afro-Asiatic/Cushitic"},
    # Manually added to group (not in M2M100 languages)
    "gle_Latn": {"group": "Celtic/Irish"},
    "glg_Latn": {"group": "Romance"},
    "guj_Gujr": {"group": "Indo-Aryan/Tamil/Dravidian"},
    "hau_Latn": {"group": "Ethiopian/Niger-Congo/Afro-Asiatic/Cushitic"},
    "heb_Hebr": {"group": "Arabic/Semitic/Iranian"},
    "hin_Deva": {"group": "Indo-Aryan/Tamil/Dravidian"},
    # New BitextMining language. Manually added to group (not in M2M100 languages)
    "hmn_Latn": {"group": "Hmong-Mien"},
    "hrv_Latn": {"group": "Slavic"},
    "hun_Latn": {"group": "Uralic/Baltic"},
    "hye_Armn": {"group": "Albanian/Armenian/Kartvelian/Hellenic"},
    "ibo_Latn": {"group": "Ethiopian/Niger-Congo/Afro-Asiatic/Cushitic"},
    "ind_Latn": {"group": "Malayo-Polyn/Philippine/Dravidian"},
    "isl_Latn": {"group": "Germanic"},
    "ita_Latn": {"group": "Romance"},
    "jpn_Jpan": {"group": "Japonic/Koreanic/Vietic/Chinese"},
    "kan_Knda": {"group": "Indo-Aryan/Tamil/Dravidian"},
    "kat_Geor": {"group": "Albanian/Armenian/Kartvelian/Hellenic"},
    "kaz_Cyrl": {"group": "Turkic"},
    "khm_Khmr": {"group": "Sino-Tibetan/Khmer/Kra-Dai/Mongolic"},
    # Manually added to group (not in M2M100 languages)
    "kin_Latn": {"group": "Niger-Congo/Afro-Asiatic/Cushitic"},
    # Manually added to group (not in M2M100 languages)
    "kir_Cyrl": {"group": "Turkic"},
    # new BitextMining language. Manually added to group (not in M2M100 languages)
    "kmr_Latn": {"group": "Arabic/Semitic/Iranian"},
    "kor_Hang": {"group": "Japonic/Koreanic/Vietic/Chinese"},
    "lao_Laoo": {"group": "Sino-Tibetan/Khmer/Kra-Dai/Mongolic"},
    # new BitextMining language
    "lav_Latn": {"group": "Uralic/Baltic"},
    "lit_Latn": {"group": "Uralic/Baltic"},
    # Manually added to group (not in M2M100 languages)
    "ltz_Latn": {"group": "Germanic"},
    "mal_Mlym": {"group": "Malayo-Polyn/Philippine/Dravidian"},
    "mar_Deva": {"group": "Indo-Aryan/Tamil/Dravidian"},
    # New BitextMining language. Manually added to group (not in M2M100 languages)
    "mey_Arab": {"group": "Arabic/Semitic/Iranian"},
    "mkd_Cyrl": {"group": "Slavic"},
    # new BitextMining language
    "mlg_Latn": {"group": "Malayo-Polyn/Philippine/Dravidian"},
    # Manually added to group (not in M2M100 languages)
    "mlt_Latn": {"group": "Romance"},
    "mon_Mong": {"group": "Sino-Tibetan/Khmer/Kra-Dai/Mongolic"},
    # Manually added to group (not in M2M100 languages)
    "mri_Latn": {"group": "Malayo-Polyn/Philippine/Dravidian"},
    # new BitextMining language
    "msa_Latn": {"group": "Malayo-Polyn/Philippine/Dravidian"},
    "mya_Mymr": {"group": "Sino-Tibetan/Khmer/Kra-Dai/Mongolic"},
    # New BitextMining language. Manually added to group (not in M2M100 languages)
    "nde_Latn": {"group": "Niger-Congo/Afro-Asiatic/Cushitic"},
    # new BitextMining language
    "nep_Deva": {"group": "Indo-Aryan/Tamil/Dravidian"},
    "nld_Latn": {"group": "Germanic"},
    # Manually added to group (not in M2M100 languages)
    "nno_Latn": {"group": "Germanic"},
    "nob_Latn": {"group": "Germanic"},
    "nso_Latn": {"group": "Ethiopian/Niger-Congo/Afro-Asiatic/Cushitic"},
    # Manually added to group (not in M2M100 languages)
    "nya_Latn": {"group": "Niger-Congo/Afro-Asiatic/Cushitic"},
    # new BitextMining language. Manually added to group (not in M2M100 languages)
    "orm_Ethi": {"group": "Ethiopian/Niger-Congo/Afro-Asiatic/Cushitic"},
    "pan_Guru": {"group": "Indo-Aryan/Tamil/Dravidian"},
    "pol_Latn": {"group": "Slavic"},
    "por_Latn": {"group": "Romance"},
    # Manually added to group (not in M2M100 languages)
    "prs_Arab": {"group": "Arabic/Semitic/Iranian"},
    # new BitextMining language
    "pus_Arab": {"group": "Arabic/Semitic/Iranian"},
    "ron_Latn": {"group": "Romance"},
    "rus_Cyrl": {"group": "Slavic"},
    # new BitextMining language. Manually added to group (not in M2M100 languages)
    "shi_Arab": {"group": "Arabic/Semitic/Iranian"},
    "sin_Sinh": {"group": "Indo-Aryan/Tamil/Dravidian"},
    "slk_Latn": {"group": "Slavic"},
    "slv_Latn": {"group": "Slavic"},
    # Manually added to group (not in M2M100 languages)
    "smo_Latn": {"group": "Malayo-Polyn/Philippine/Dravidian"},
    # Manually added to group (not in M2M100 languages)
    "sna_Latn": {"group": "Niger-Congo/Afro-Asiatic/Cushitic"},
    # Manually added to group (not in M2M100 languages)
    "snd_Arab": {"group": "Indo-Aryan/Tamil/Dravidian"},
    "som_Latn": {"group": "Ethiopian/Niger-Congo/Afro-Asiatic/Cushitic"},
    "spa_Latn": {"group": "Romance"},
    "sqi_Latn": {"group": "Albanian/Armenian/Kartvelian/Hellenic"},
    # new BitextMining language (only srp_Latn available)
    "srp_Cyrl": {"group": "Slavic"},
    # Manually added to group (not in M2M100 languages)
    "srp_Latn": {"group": "Slavic"},
    "ssw_Latn": {"group": "Ethiopian/Niger-Congo/Afro-Asiatic/Cushitic"},
    # new BitextMining language
    "swa_Latn": {"group": "Ethiopian/Niger-Congo/Afro-Asiatic/Cushitic"},
    "swe_Latn": {"group": "Germanic"},
    # new BitextMining language. Manually added to group (not in M2M100 languages)
    "tah_Latn": {"group": "Malayo-Polyn/Philippine/Dravidian"},
    "tam_Taml": {"group": "Indo-Aryan/Tamil/Dravidian"},
    # Manually added to group (not in M2M100 languages)
    "tat_Cyrl": {"group": "Turkic"},
    # Manually added to group (not in M2M100 languages)
    "tel_Telu": {"group": "Indo-Aryan/Tamil/Dravidian"},
    # Manually added to group (not in M2M100 languages)
    "tgk_Cyrl": {"group": "Arabic/Semitic/Iranian"},
    "tha_Thai": {"group": "Sino-Tibetan/Khmer/Kra-Dai/Mongolic"},
    # Manually added to group (not in M2M100 languages)
    "tir_Ethi": {"group": "Ethiopian/Niger-Congo/Afro-Asiatic/Cushitic"},
    # new BitextMining language. Manually added to group (not in M2M100 languages)
    "ton_Latn": {"group": "Malayo-Polyn/Philippine/Dravidian"},
    "tsn_Latn": {"group": "Ethiopian/Niger-Congo/Afro-Asiatic/Cushitic"},
    # Manually added to group (not in M2M100 languages)
    "tuk_Latn": {"group": "Turkic"},
    "tur_Latn": {"group": "Turkic"},
    # Manually added to group (not in M2M100 languages)
    "uig_Arab": {"group": "Turkic"},
    "ukr_Cyrl": {"group": "Slavic"},
    "urd_Arab": {"group": "Indo-Aryan/Tamil/Dravidian"},
    "uzb_Latn": {"group": "Turkic"},
    # new BitextMining language. Manually added to group (not in M2M100 languages)
    "ven_Latn": {"group": "Niger-Congo/Afro-Asiatic/Cushitic"},
    "vie_Latn": {"group": "Japonic/Koreanic/Vietic/Chinese"},
    "wol_Latn": {"group": "Ethiopian/Niger-Congo/Afro-Asiatic/Cushitic"},
    "xho_Latn": {"group": "Ethiopian/Niger-Congo/Afro-Asiatic/Cushitic"},
    "yor_Latn": {"group": "Ethiopian/Niger-Congo/Afro-Asiatic/Cushitic"},
    # Manually added to group (not in M2M100 languages)
    "yue_Hant": {"group": "Japonic/Koreanic/Vietic/Chinese"},
    "zho_Hans": {"group": "Japonic/Koreanic/Vietic/Chinese"},
    "zho_Hant": {"group": "Japonic/Koreanic/Vietic/Chinese"},
    "zul_Latn": {"group": "Ethiopian/Niger-Congo/Afro-Asiatic/Cushitic"},
}

_SPLIT = ["test"]

# number of sentences to use for evaluation
_N = 256


def extend_lang_pairs() -> dict[str, list[str]]:
    """Add language pairs according to M2M-100 language grouping strategy. A pair is only included if:
    - source or target is English; or
    - source or target is a bridge language; or
    - source and target are from same language grouping
    """
    hf_lang_subset2isolang = {}
    for x in _LANGUAGES.keys():
        for y in _LANGUAGES.keys():
            if x != y:
                if (
                    ("eng_Latn" in (x, y))
                    or (all(var in _BRIDGE_LANGUAGES for var in (x, y)))
                    or (_LANGUAGES[x]["group"] == _LANGUAGES[y]["group"])
                ):
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
        description="NTREX is a News Test References dataset for Machine Translation Evaluation, covering translation from English into 128 languages. We select language pairs according to the M2M-100 language grouping strategy, resulting in 1916 directions.",
        reference="https://huggingface.co/datasets/davidstap/NTREX",
        type="BitextMining",
        category="s2s",
        eval_splits=_SPLIT,
        eval_langs=_EVAL_LANGS,
        main_score="f1",
        date=("2019-08-01", "2022-11-01"), # publication date newstest19 until publication date NTREX paper
        form=["written"],
        domains=["News"],
        task_subtypes=[],
        license="CC-BY-SA-4.0",
        socioeconomic_status="medium",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="human-translated and localized",
        n_samples={"test": _N*len(_EVAL_LANGS)},
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
            for l in _LANGUAGES.keys()
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
