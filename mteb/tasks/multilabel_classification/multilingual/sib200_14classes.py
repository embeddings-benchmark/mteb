from __future__ import annotations

"""MTEB task definition for the SIB‑200 (14‑class) multilingual *multi‑label*
news‑topic classification benchmark.

Each sentence is assigned **one** of the 14 topic labels listed in
``_TOPICS`` below, but the task is framed as *multi‑label* classification so
that evaluation remains compatible with MTEB’s multilabel pipeline (each
example simply has a one‑element label list).

The dataset contains 205 Flores‑200 languages with the original splits
(train / validation / test).  The train split contains five examples per
language to facilitate few‑shot evaluation; the test split has 1 225
examples.
"""

from typing import Iterable, List

from datasets import ClassLabel
from mteb.abstasks.AbsTaskMultilabelClassification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

# ---------------------------------------------------------------------------
# Languages (Flores‑200 ISO‑639‑3 + script ⇒ BCP‑47 tag)
# Copied verbatim from the single‑label SIB‑200 task to guarantee identical
# coverage.
_LANGS = {
    "ace_Latn": ["ace-Latn"],
    "acm_Arab": ["acm-Arab"],
    "acq_Arab": ["acq-Arab"],
    "aeb_Arab": ["aeb-Arab"],
    "afr_Latn": ["afr-Latn"],
    "ajp_Arab": ["ajp-Arab"],
    "aka_Latn": ["aka-Latn"],
    "als_Latn": ["als-Latn"],
    "amh_Ethi": ["amh-Ethi"],
    "apc_Arab": ["apc-Arab"],
    "arb_Latn": ["arb-Latn"],
    "ars_Arab": ["ars-Arab"],
    "ary_Arab": ["ary-Arab"],
    "arz_Arab": ["arz-Arab"],
    "asm_Beng": ["asm-Beng"],
    "ast_Latn": ["ast-Latn"],
    "awa_Deva": ["awa-Deva"],
    "ayr_Latn": ["ayr-Latn"],
    "azb_Arab": ["azb-Arab"],
    "azj_Latn": ["azj-Latn"],
    "bak_Cyrl": ["bak-Cyrl"],
    "bam_Latn": ["bam-Latn"],
    "ban_Latn": ["ban-Latn"],
    "bel_Cyrl": ["bel-Cyrl"],
    "bem_Latn": ["bem-Latn"],
    "ben_Beng": ["ben-Beng"],
    "bho_Deva": ["bho-Deva"],
    "bjn_Latn": ["bjn-Latn"],
    "bod_Tibt": ["bod-Tibt"],
    "bos_Latn": ["bos-Latn"],
    "bug_Latn": ["bug-Latn"],
    "bul_Cyrl": ["bul-Cyrl"],
    "cat_Latn": ["cat-Latn"],
    "ceb_Latn": ["ceb-Latn"],
    "ces_Latn": ["ces-Latn"],
    "cjk_Latn": ["cjk-Latn"],
    "ckb_Arab": ["ckb-Arab"],
    "crh_Latn": ["crh-Latn"],
    "cym_Latn": ["cym-Latn"],
    "dan_Latn": ["dan-Latn"],
    "deu_Latn": ["deu-Latn"],
    "dik_Latn": ["dik-Latn"],
    "dyu_Latn": ["dyu-Latn"],
    "dzo_Tibt": ["dzo-Tibt"],
    "ell_Grek": ["ell-Grek"],
    "eng_Latn": ["eng-Latn"],
    "epo_Latn": ["epo-Latn"],
    "est_Latn": ["est-Latn"],
    "eus_Latn": ["eus-Latn"],
    "ewe_Latn": ["ewe-Latn"],
    "fao_Latn": ["fao-Latn"],
    "fij_Latn": ["fij-Latn"],
    "fin_Latn": ["fin-Latn"],
    "fon_Latn": ["fon-Latn"],
    "fra_Latn": ["fra-Latn"],
    "fur_Latn": ["fur-Latn"],
    "fuv_Latn": ["fuv-Latn"],
    "gaz_Latn": ["gaz-Latn"],
    "gla_Latn": ["gla-Latn"],
    "gle_Latn": ["gle-Latn"],
    "glg_Latn": ["glg-Latn"],
    "grn_Latn": ["grn-Latn"],
    "guj_Gujr": ["guj-Gujr"],
    "hat_Latn": ["hat-Latn"],
    "hau_Latn": ["hau-Latn"],
    "heb_Hebr": ["heb-Hebr"],
    "hin_Deva": ["hin-Deva"],
    "hne_Deva": ["hne-Deva"],
    "hrv_Latn": ["hrv-Latn"],
    "hun_Latn": ["hun-Latn"],
    "hye_Armn": ["hye-Armn"],
    "ibo_Latn": ["ibo-Latn"],
    "ilo_Latn": ["ilo-Latn"],
    "ind_Latn": ["ind-Latn"],
    "isl_Latn": ["isl-Latn"],
    "ita_Latn": ["ita-Latn"],
    "jav_Latn": ["jav-Latn"],
    "jpn_Jpan": ["jpn-Jpan"],
    "kab_Latn": ["kab-Latn"],
    "kac_Latn": ["kac-Latn"],
    "kam_Latn": ["kam-Latn"],
    "kan_Knda": ["kan-Knda"],
    "kas_Deva": ["kas-Deva"],
    "kat_Geor": ["kat-Geor"],
    "kaz_Cyrl": ["kaz-Cyrl"],
    "kbp_Latn": ["kbp-Latn"],
    "kea_Latn": ["kea-Latn"],
    "khk_Cyrl": ["khk-Cyrl"],
    "khm_Khmr": ["khm-Khmr"],
    "kik_Latn": ["kik-Latn"],
    "kin_Latn": ["kin-Latn"],
    "kir_Cyrl": ["kir-Cyrl"],
    "kmb_Latn": ["kmb-Latn"],
    "kmr_Latn": ["kmr-Latn"],
    "knc_Latn": ["knc-Latn"],
    "kon_Latn": ["kon-Latn"],
    "kor_Hang": ["kor-Hang"],
    "lao_Laoo": ["lao-Laoo"],
    "lij_Latn": ["lij-Latn"],
    "lim_Latn": ["lim-Latn"],
    "lin_Latn": ["lin-Latn"],
    "lit_Latn": ["lit-Latn"],
    "lmo_Latn": ["lmo-Latn"],
    "ltg_Latn": ["ltg-Latn"],
    "ltz_Latn": ["ltz-Latn"],
    "lua_Latn": ["lua-Latn"],
    "lug_Latn": ["lug-Latn"],
    "luo_Latn": ["luo-Latn"],
    "lus_Latn": ["lus-Latn"],
    "lvs_Latn": ["lvs-Latn"],
    "mag_Deva": ["mag-Deva"],
    "mai_Deva": ["mai-Deva"],
    "mal_Mlym": ["mal-Mlym"],
    "mar_Deva": ["mar-Deva"],
    "min_Latn": ["min-Latn"],
    "mkd_Cyrl": ["mkd-Cyrl"],
    "mlt_Latn": ["mlt-Latn"],
    "mni_Beng": ["mni-Beng"],
    "mos_Latn": ["mos-Latn"],
    "mri_Latn": ["mri-Latn"],
    "mya_Mymr": ["mya-Mymr"],
    "nld_Latn": ["nld-Latn"],
    "nno_Latn": ["nno-Latn"],
    "nob_Latn": ["nob-Latn"],
    "npi_Deva": ["npi-Deva"],
    "nqo_Nkoo": ["nqo-Nkoo"],
    "nso_Latn": ["nso-Latn"],
    "nus_Latn": ["nus-Latn"],
    "nya_Latn": ["nya-Latn"],
    "oci_Latn": ["oci-Latn"],
    "ory_Orya": ["ory-Orya"],
    "pag_Latn": ["pag-Latn"],
    "pan_Guru": ["pan-Guru"],
    "pap_Latn": ["pap-Latn"],
    "pbt_Arab": ["pbt-Arab"],
    "pes_Arab": ["pes-Arab"],
    "plt_Latn": ["plt-Latn"],
    "pol_Latn": ["pol-Latn"],
    "por_Latn": ["por-Latn"],
    "prs_Arab": ["prs-Arab"],
    "quy_Latn": ["quy-Latn"],
    "ron_Latn": ["ron-Latn"],
    "run_Latn": ["run-Latn"],
    "rus_Cyrl": ["rus-Cyrl"],
    "sag_Latn": ["sag-Latn"],
    "san_Deva": ["san-Deva"],
    "sat_Olck": ["sat-Olck"],
    "scn_Latn": ["scn-Latn"],
    "shn_Mymr": ["shn-Mymr"],
    "sin_Sinh": ["sin-Sinh"],
    "slk_Latn": ["slk-Latn"],
    "slv_Latn": ["slv-Latn"],
    "smo_Latn": ["smo-Latn"],
    "sna_Latn": ["sna-Latn"],
    "snd_Arab": ["snd-Arab"],
    "som_Latn": ["som-Latn"],
    "sot_Latn": ["sot-Latn"],
    "spa_Latn": ["spa-Latn"],
    "srd_Latn": ["srd-Latn"],
    "srp_Cyrl": ["srp-Cyrl"],
    "ssw_Latn": ["ssw-Latn"],
    "sun_Latn": ["sun-Latn"],
    "swe_Latn": ["swe-Latn"],
    "swh_Latn": ["swh-Latn"],
    "szl_Latn": ["szl-Latn"],
    "tam_Taml": ["tam-Taml"],
    "taq_Tfng": ["taq-Tfng"],
    "tat_Cyrl": ["tat-Cyrl"],
    "tel_Telu": ["tel-Telu"],
    "tgk_Cyrl": ["tgk-Cyrl"],
    "tgl_Latn": ["tgl-Latn"],
    "tha_Thai": ["tha-Thai"],
    "tir_Ethi": ["tir-Ethi"],
    "tpi_Latn": ["tpi-Latn"],
    "tsn_Latn": ["tsn-Latn"],
    "tso_Latn": ["tso-Latn"],
    "tuk_Latn": ["tuk-Latn"],
    "tum_Latn": ["tum-Latn"],
    "tur_Latn": ["tur-Latn"],
    "twi_Latn": ["twi-Latn"],
    "tzm_Tfng": ["tzm-Tfng"],
    "uig_Arab": ["uig-Arab"],
    "ukr_Cyrl": ["ukr-Cyrl"],
    "umb_Latn": ["umb-Latn"],
    "urd_Arab": ["urd-Arab"],
    "uzn_Latn": ["uzn-Latn"],
    "vec_Latn": ["vec-Latn"],
    "vie_Latn": ["vie-Latn"],
    "war_Latn": ["war-Latn"],
    "wol_Latn": ["wol-Latn"],
    "xho_Latn": ["xho-Latn"],
    "ydd_Hebr": ["ydd-Hebr"],
    "yor_Latn": ["yor-Latn"],
    "yue_Hant": ["yue-Hant"],
    "zho_Hant": ["zho-Hant"],
    "zsm_Latn": ["zsm-Latn"],
    "zul_Latn": ["zul-Latn"],
}

# ---------------------------------------------------------------------------
# Topics / label set (order matches numeric IDs in the raw dataset)
_TOPICS = [
    "travel",
    "politics",
    "science",
    "sports",
    "technology",
    "health",
    "nature",
    "entertainment",
    "geography",
    "business",
    "disasters",
    "crime",
    "education",
    "religion",
]


# ---------------------------------------------------------------- labels
_TOPICS = [
    "travel", "politics", "science", "sports", "technology", "health",
    "nature", "entertainment", "geography", "business", "disasters",
    "crime", "education", "religion",
]
_TOPIC2ID = {name: idx for idx, name in enumerate(_TOPICS)}

# ---------------------------------------------------------------------------
# Task implementation
class SIB200_14Classes(MultilingualTask, AbsTaskMultilabelClassification):
    """SIB‑200 (14 topics) multilingual multilabel classification benchmark."""

    metadata = TaskMetadata(
        name="SIB200-14Classes",
        description=(
            "SIB‑200 14‑class edition for multilingual *topic* classification. "
            "It covers the full 205 Flores‑200 languages and provides only 5 "
            "training examples per language to enable few‑shot evaluation. "
            "The original paper describing the SIB‑200 family of datasets is "
            "Adelani *et al.* (2023)."
        ),
        reference="https://arxiv.org/abs/2309.07445",
        dataset={
            "path": "Davlan/sib200_14classes",
            "revision": "main",
        },
        type="MultilabelClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train", "validation", "test"],
        eval_langs=_LANGS,
        main_score="accuracy",
        date=("2023-09-14", "2024-01-27"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="human-translated and localized",
        bibtex_citation=r"""
@misc{adelani2023sib200,
      title={SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects},
      author={Adelani, David Ifeoluwa and Liu, Hannah and Shen, Xiaoyu and Vassilyev, Nikita and Alabi, Jesujoba O. and Mao, Yanke and Gao, Haonan and Lee, Annie En-Shiun},
      year={2023},
      eprint={2309.07445},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
""",
    )



    # ---------------------------------------------------------------- transform
    from datasets import ClassLabel

    # ---------------------------------------------------------------- transform
    def dataset_transform(self) -> None:
        """
        Convert each split to the MTEB multilabel format:

        * text  : str  (sentence)
        * label : list[int]  (always a list; each item is the topic ID)
        """
        for lang in self.dataset:
            for split in self.dataset[lang]:
                ds = self.dataset[lang][split]

                cols = ds.column_names
                text_col = next(c for c in ("text", "sentence", "utterance") if c in cols)
                lab_col  = next(c for c in ("label", "labels", "category")   if c in cols)

                # normalise every example -------------------------------------------------
                def to_mteb(example):
                    raw = example[lab_col]

                    # ensure list
                    if not isinstance(raw, list):
                        raw = [raw]

                    # NOTICE THE CHANGE ➡  self._TOPIC2ID
                    norm = [
                        _TOPIC2ID[x] if isinstance(x, str) else int(x)
                        for x in raw
                    ]
                    return {"text": example[text_col], "label": norm}

                ds = ds.map(
                    to_mteb,
                    remove_columns=cols,          # drop original columns
                    desc=f"{lang}/{split}",
                )

                # cast if the feature was a ClassLabel
                if isinstance(ds.features["label"].feature, ClassLabel):
                    ds = ds.cast_column("label", list[int])

                self.dataset[lang][split] = ds

        # fall back to another split if “train” is missing
        for lang, splits in self.dataset.items():
            if "train" not in splits:
                self.dataset[lang]["train"] = splits.get(
                    "validation", splits.get("dev", splits["test"])
                )
