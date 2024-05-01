from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

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


class XSimPlusPlusRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="XSimPlusPlusRetrieval",
        dataset={
            "path": "jaygala24/xsimplusplus",
            "revision": "67d739657845d42aeed8169189254a788ab60f17",
        },
        description="xsim++ is a machine translation testset that seeks to capture more subtle improvements in bitext mining by adding challenging negative examples",
        reference="https://arxiv.org/abs/2306.12907",
        type="Retrieval",
        category="s2s",
        eval_splits=_SPLIT,
        eval_langs={x: ["eng-Latn", x.replace("_", "-")] for x in _LANGUAGES},
        main_score="ndcg_at_10",
        date=("2022-08-01", "2022-12-20"),
        form=["written"],
        domains=["Web", "Non-fiction", "News"],
        task_subtypes=[],
        license="CC0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@article{chen2023xsim,
  title     = {xSIM++: An Improved Proxy to Bitext Mining Performance for Low-Resource Languages},
  author    = {Mingda Chen and Kevin Heffernan and Onur cCelebi and Alexandre Mourachko and Holger Schwenk},
  journal   = {Annual Meeting of the Association for Computational Linguistics},
  year      = {2023},
  doi       = {10.48550/arXiv.2306.12907}
}""",
        n_samples={"devtest": 1012},
        avg_character_length={"devtest": 134},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        split = self.metadata.eval_splits[0]
        queries = {lang: {split: {}} for lang in self.langs}
        corpus = {lang: {split: {}} for lang in self.langs}
        relevant_docs = {lang: {split: {}} for lang in self.langs}

        for lang in self.langs:
            data = datasets.load_dataset(name=lang, **self.metadata_dict["dataset"])[
                split
            ]
            # Generate unique IDs for queries and documents
            query_id_counter = 1
            document_id_counter = 1

            for row in data:
                query_text = row["query"]
                positive_texts = row["positive"]
                negative_texts = row["negative"]

                # Assign unique ID to the query
                query_id = f"Q{query_id_counter}"
                queries[lang][split][query_id] = query_text
                query_id_counter += 1

                # Add positive and negative texts to corpus with unique IDs
                for text in positive_texts + negative_texts:
                    doc_id = f"D{document_id_counter}"
                    corpus[lang][split][doc_id] = {"text": text}
                    document_id_counter += 1

                    # Add relevant document information to relevant_docs for positive texts only
                    if text in positive_texts:
                        if query_id not in relevant_docs[lang][split]:
                            relevant_docs[lang][split][query_id] = {}
                        relevant_docs[lang][split][query_id][doc_id] = 1

        self.corpus = datasets.DatasetDict(corpus)
        self.queries = datasets.DatasetDict(queries)
        self.relevant_docs = datasets.DatasetDict(relevant_docs)

        self.data_loaded = True
