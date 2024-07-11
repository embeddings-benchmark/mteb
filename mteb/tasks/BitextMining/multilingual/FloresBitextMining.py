from __future__ import annotations

from typing import Any

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, MultilingualTask

_LANGUAGES = [
    "ace_Arab",
    "bam_Latn",
    "dzo_Tibt",
    "hin_Deva",
    "khm_Khmr",
    "mag_Deva",
    "pap_Latn",
    "sot_Latn",
    "tur_Latn",
    "ace_Latn",
    "ban_Latn",
    "ell_Grek",
    "hne_Deva",
    "kik_Latn",
    "mai_Deva",
    "pbt_Arab",
    "spa_Latn",
    "twi_Latn",
    "acm_Arab",
    "bel_Cyrl",
    "eng_Latn",
    "hrv_Latn",
    "kin_Latn",
    "mal_Mlym",
    "pes_Arab",
    "srd_Latn",
    "tzm_Tfng",
    "acq_Arab",
    "bem_Latn",
    "epo_Latn",
    "hun_Latn",
    "kir_Cyrl",
    "mar_Deva",
    "plt_Latn",
    "srp_Cyrl",
    "uig_Arab",
    "aeb_Arab",
    "ben_Beng",
    "est_Latn",
    "hye_Armn",
    "kmb_Latn",
    "min_Arab",
    "pol_Latn",
    "ssw_Latn",
    "ukr_Cyrl",
    "afr_Latn",
    "bho_Deva",
    "eus_Latn",
    "ibo_Latn",
    "kmr_Latn",
    "min_Latn",
    "por_Latn",
    "sun_Latn",
    "umb_Latn",
    "ajp_Arab",
    "bjn_Arab",
    "ewe_Latn",
    "ilo_Latn",
    "knc_Arab",
    "mkd_Cyrl",
    "prs_Arab",
    "swe_Latn",
    "urd_Arab",
    "aka_Latn",
    "bjn_Latn",
    "fao_Latn",
    "ind_Latn",
    "knc_Latn",
    "mlt_Latn",
    "quy_Latn",
    "swh_Latn",
    "uzn_Latn",
    "als_Latn",
    "bod_Tibt",
    "fij_Latn",
    "isl_Latn",
    "kon_Latn",
    "mni_Beng",
    "ron_Latn",
    "szl_Latn",
    "vec_Latn",
    "amh_Ethi",
    "bos_Latn",
    "fin_Latn",
    "ita_Latn",
    "kor_Hang",
    "mos_Latn",
    "run_Latn",
    "tam_Taml",
    "vie_Latn",
    "apc_Arab",
    "bug_Latn",
    "fon_Latn",
    "jav_Latn",
    "lao_Laoo",
    "mri_Latn",
    "rus_Cyrl",
    "taq_Latn",
    "war_Latn",
    "arb_Arab",
    "bul_Cyrl",
    "fra_Latn",
    "jpn_Jpan",
    "lij_Latn",
    "mya_Mymr",
    "sag_Latn",
    "taq_Tfng",
    "wol_Latn",
    "arb_Latn",
    "cat_Latn",
    "fur_Latn",
    "kab_Latn",
    "lim_Latn",
    "nld_Latn",
    "san_Deva",
    "tat_Cyrl",
    "xho_Latn",
    "ars_Arab",
    "ceb_Latn",
    "fuv_Latn",
    "kac_Latn",
    "lin_Latn",
    "nno_Latn",
    "sat_Olck",
    "tel_Telu",
    "ydd_Hebr",
    "ary_Arab",
    "ces_Latn",
    "gaz_Latn",
    "kam_Latn",
    "lit_Latn",
    "nob_Latn",
    "scn_Latn",
    "tgk_Cyrl",
    "yor_Latn",
    "arz_Arab",
    "cjk_Latn",
    "gla_Latn",
    "kan_Knda",
    "lmo_Latn",
    "npi_Deva",
    "shn_Mymr",
    "tgl_Latn",
    "yue_Hant",
    "asm_Beng",
    "ckb_Arab",
    "gle_Latn",
    "kas_Arab",
    "ltg_Latn",
    "nso_Latn",
    "sin_Sinh",
    "tha_Thai",
    "zho_Hans",
    "ast_Latn",
    "crh_Latn",
    "glg_Latn",
    "kas_Deva",
    "ltz_Latn",
    "nus_Latn",
    "slk_Latn",
    "tir_Ethi",
    "zho_Hant",
    "awa_Deva",
    "cym_Latn",
    "grn_Latn",
    "kat_Geor",
    "lua_Latn",
    "nya_Latn",
    "slv_Latn",
    "tpi_Latn",
    "zsm_Latn",
    "ayr_Latn",
    "dan_Latn",
    "guj_Gujr",
    "kaz_Cyrl",
    "lug_Latn",
    "oci_Latn",
    "smo_Latn",
    "tsn_Latn",
    "zul_Latn",
    "azb_Arab",
    "deu_Latn",
    "hat_Latn",
    "kbp_Latn",
    "luo_Latn",
    "ory_Orya",
    "sna_Latn",
    "tso_Latn",
    "azj_Latn",
    "dik_Latn",
    "hau_Latn",
    "kea_Latn",
    "lus_Latn",
    "pag_Latn",
    "snd_Arab",
    "tuk_Latn",
    "bak_Cyrl",
    "dyu_Latn",
    "heb_Hebr",
    "khk_Cyrl",
    "lvs_Latn",
    "pan_Guru",
    "som_Latn",
    "tum_Latn",
]
_SPLIT = ["devtest"]


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


_LANGUAGES_MAPPING = extend_lang_pairs()


class FloresBitextMining(AbsTaskBitextMining, MultilingualTask):
    parallel_subsets = True
    metadata = TaskMetadata(
        name="FloresBitextMining",
        dataset={
            "path": "mteb/flores",
            "revision": "e6b647fcb6299a2f686f742f4d4c023e553ea67e",
            "trust_remote_code": True,
        },
        description="FLORES is a benchmark dataset for machine translation between English and low-resource languages.",
        reference="https://huggingface.co/datasets/facebook/flores",
        type="BitextMining",
        category="s2s",
        eval_splits=_SPLIT,
        eval_langs=_LANGUAGES_MAPPING,
        main_score="f1",
        date=("2022-01-01", "2022-12-31"),
        form=["written"],
        domains=["Non-fiction", "Encyclopaedic"],
        task_subtypes=[],
        license="CC BY-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""
        @inproceedings{goyal2022flores,
        title={The FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation},
        author={Goyal, Naman and Gao, Cynthia and Chaudhary, Vishrav and Chen, Peng-Jen and Wenzek, Guillaume and Ju, Da and Krishnan, Sanjana and Ranzato, Marc'Aurelio and Guzm{\'a}n, Francisco},
        booktitle={Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
        pages={19--35},
        year={2022}
        }
        """,
        n_samples={"dev": 997, "devtest": 1012},
        avg_character_length={},
    )

    def load_data(self, **kwargs: Any) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])
        self.data_loaded = True
