from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"

_LANGUAGES = [
    "acm_Arab",
    "afr_Latn",
    "als_Latn",
    "amh_Ethi",
    "apc_Arab",
    "arb_Arab",
    "arb_Latn",
    "ars_Arab",
    "ary_Arab",
    "arz_Arab",
    "asm_Beng",
    "azj_Latn",
    "bam_Latn",
    "ben_Beng",
    "ben_Latn",
    "bod_Tibt",
    "bul_Cyrl",
    "cat_Latn",
    "ceb_Latn",
    "ces_Latn",
    "ckb_Arab",
    "dan_Latn",
    "deu_Latn",
    "ell_Grek",
    "eng_Latn",
    "est_Latn",
    "eus_Latn",
    "fin_Latn",
    "fra_Latn",
    "fuv_Latn",
    "gaz_Latn",
    "grn_Latn",
    "guj_Gujr",
    "hat_Latn",
    "hau_Latn",
    "heb_Hebr",
    "hin_Deva",
    "hin_Latn",
    "hrv_Latn",
    "hun_Latn",
    "hye_Armn",
    "ibo_Latn",
    "ilo_Latn",
    "ind_Latn",
    "isl_Latn",
    "ita_Latn",
    "jav_Latn",
    "jpn_Jpan",
    "kac_Latn",
    "kan_Knda",
    "kat_Geor",
    "kaz_Cyrl",
    "kea_Latn",
    "khk_Cyrl",
    "khm_Khmr",
    "kin_Latn",
    "kir_Cyrl",
    "kor_Hang",
    "lao_Laoo",
    "lin_Latn",
    "lit_Latn",
    "lug_Latn",
    "luo_Latn",
    "lvs_Latn",
    "mal_Mlym",
    "mar_Deva",
    "mkd_Cyrl",
    "mlt_Latn",
    "mri_Latn",
    "mya_Mymr",
    "nld_Latn",
    "nob_Latn",
    "npi_Deva",
    "npi_Latn",
    "nso_Latn",
    "nya_Latn",
    "ory_Orya",
    "pan_Guru",
    "pbt_Arab",
    "pes_Arab",
    "plt_Latn",
    "pol_Latn",
    "por_Latn",
    "ron_Latn",
    "rus_Cyrl",
    "shn_Mymr",
    "sin_Latn",
    "sin_Sinh",
    "slk_Latn",
    "slv_Latn",
    "sna_Latn",
    "snd_Arab",
    "som_Latn",
    "sot_Latn",
    "spa_Latn",
    "srp_Cyrl",
    "ssw_Latn",
    "sun_Latn",
    "swe_Latn",
    "swh_Latn",
    "tam_Taml",
    "tel_Telu",
    "tgk_Cyrl",
    "tgl_Latn",
    "tha_Thai",
    "tir_Ethi",
    "tsn_Latn",
    "tso_Latn",
    "tur_Latn",
    "ukr_Cyrl",
    "urd_Arab",
    "urd_Latn",
    "uzn_Latn",
    "vie_Latn",
    "war_Latn",
    "wol_Latn",
    "xho_Latn",
    "yor_Latn",
    "zho_Hans",
    "zho_Hant",
    "zsm_Latn",
    "zul_Latn",
]


def get_lang_pairs() -> dict[str, list[str]]:
    # add pairs with same langauge as the source and target
    # add pairs with english as source or target
    lang_pairs = {}
    for x in _LANGUAGES:
        pair = f"{x}-{x}"
        lang_pairs[pair] = [x.replace("_", "-"), x.replace("_", "-")]

        if x != "eng_Latn":
            pair = f"{x}-eng_Latn"
            lang_pairs[pair] = [x.replace("_", "-"), "eng-Latn"]
            pair = f"eng_Latn-{x}"
            lang_pairs[pair] = ["eng-Latn", x.replace("_", "-")]

    # add pairs for languages with a base script and a Latn script
    lang_base_scripts = [
        "arb_Arab",
        "ben_Beng",
        "hin_Deva",
        "npi_Deva",
        "sin_Sinh",
        "urd_Arab",
    ]
    for lang_base_script in lang_base_scripts:
        lang = lang_base_script.split("_")[0]
        lang_latn_script = f"{lang}_Latn"
        pair = f"{lang_base_script}-{lang_latn_script}"
        lang_pairs[pair] = [
            lang_base_script.replace("_", "-"),
            lang_latn_script.replace("_", "-"),
        ]
        pair = f"{lang_latn_script}-{lang_base_script}"
        lang_pairs[pair] = [
            lang_latn_script.replace("_", "-"),
            lang_base_script.replace("_", "-"),
        ]

    return lang_pairs


_LANGUAGES_MAPPING = get_lang_pairs()


class BelebeleRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BelebeleRetrieval",
        dataset={
            "path": "facebook/belebele",
            "revision": "75b399394a9803252cfec289d103de462763db7c",
        },
        description=(
            "Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants "
            + "(including 115 distinct languages and their scripts)"
        ),
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGUAGES_MAPPING,
        reference="https://arxiv.org/abs/2308.16884",
        main_score="ndcg_at_10",
        license="cc-by-sa-4.0",
        domains=["Web", "News", "Written"],
        sample_creation="created",  # number of languages * 900
        date=("2023-08-31", "2023-08-31"),
        task_subtypes=["Question answering"],
        annotations_creators="expert-annotated",
        dialect=[],
        bibtex_citation=r"""
@article{bandarkar2023belebele,
  author = {Lucas Bandarkar and Davis Liang and Benjamin Muller and Mikel Artetxe and Satya Narayan Shukla and Donald Husa and Naman Goyal and Abhinandan Krishnan and Luke Zettlemoyer and Madian Khabsa},
  journal = {arXiv preprint arXiv:2308.16884},
  title = {The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants},
  year = {2023},
}
""",
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return

        self.dataset = load_dataset(**self.metadata.dataset)

        self.queries = {lang_pair: {_EVAL_SPLIT: {}} for lang_pair in self.hf_subsets}
        self.corpus = {lang_pair: {_EVAL_SPLIT: {}} for lang_pair in self.hf_subsets}
        self.relevant_docs = {
            lang_pair: {_EVAL_SPLIT: {}} for lang_pair in self.hf_subsets
        }

        for lang_pair in self.hf_subsets:
            languages = self.metadata.eval_langs[lang_pair]
            lang_corpus, lang_question = (
                languages[0].replace("-", "_"),
                languages[1].replace("-", "_"),
            )
            ds_corpus = self.dataset[lang_corpus]
            ds_question = self.dataset[lang_question]

            question_ids = {
                question: _id
                for _id, question in enumerate(set(ds_question["question"]))
            }

            link_to_context_id = {}
            context_idx = 0
            for row in ds_corpus:
                if row["link"] not in link_to_context_id:
                    context_id = f"C{context_idx}"
                    link_to_context_id[row["link"]] = context_id
                    self.corpus[lang_pair][_EVAL_SPLIT][context_id] = {
                        "title": "",
                        "text": row["flores_passage"],
                    }
                    context_idx = context_idx + 1

            for row in ds_question:
                query = row["question"]
                query_id = f"Q{question_ids[query]}"
                self.queries[lang_pair][_EVAL_SPLIT][query_id] = query

                context_link = row["link"]
                context_id = link_to_context_id[context_link]
                if query_id not in self.relevant_docs[lang_pair][_EVAL_SPLIT]:
                    self.relevant_docs[lang_pair][_EVAL_SPLIT][query_id] = {}
                self.relevant_docs[lang_pair][_EVAL_SPLIT][query_id][context_id] = 1

        self.data_loaded = True
