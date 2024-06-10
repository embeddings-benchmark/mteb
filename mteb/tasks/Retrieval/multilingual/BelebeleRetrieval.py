from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_EVAL_SPLIT = "test"

_LANGS = {
    "acm-Arab": ["acm-Arab"],
    "afr-Latn": ["afr-Latn"],
    "als-Latn": ["als-Latn"],
    "amh-Ethi": ["amh-Ethi"],
    "apc-Arab": ["apc-Arab"],
    "arb-Arab": ["arb-Arab"],
    "arb-Latn": ["arb-Latn"],
    "ars-Arab": ["ars-Arab"],
    "ary-Arab": ["ary-Arab"],
    "arz-Arab": ["arz-Arab"],
    "asm-Beng": ["asm-Beng"],
    "azj-Latn": ["azj-Latn"],
    "bam-Latn": ["bam-Latn"],
    "ben-Beng": ["ben-Beng"],
    "ben-Latn": ["ben-Latn"],
    "bod-Tibt": ["bod-Tibt"],
    "bul-Cyrl": ["bul-Cyrl"],
    "cat-Latn": ["cat-Latn"],
    "ceb-Latn": ["ceb-Latn"],
    "ces-Latn": ["ces-Latn"],
    "ckb-Arab": ["ckb-Arab"],
    "dan-Latn": ["dan-Latn"],
    "deu-Latn": ["deu-Latn"],
    "ell-Grek": ["ell-Grek"],
    "eng-Latn": ["eng-Latn"],
    "est-Latn": ["est-Latn"],
    "eus-Latn": ["eus-Latn"],
    "fin-Latn": ["fin-Latn"],
    "fra-Latn": ["fra-Latn"],
    "fuv-Latn": ["fuv-Latn"],
    "gaz-Latn": ["gaz-Latn"],
    "grn-Latn": ["grn-Latn"],
    "guj-Gujr": ["guj-Gujr"],
    "hat-Latn": ["hat-Latn"],
    "hau-Latn": ["hau-Latn"],
    "heb-Hebr": ["heb-Hebr"],
    "hin-Deva": ["hin-Deva"],
    "hin-Latn": ["hin-Latn"],
    "hrv-Latn": ["hrv-Latn"],
    "hun-Latn": ["hun-Latn"],
    "hye-Armn": ["hye-Armn"],
    "ibo-Latn": ["ibo-Latn"],
    "ilo-Latn": ["ilo-Latn"],
    "ind-Latn": ["ind-Latn"],
    "isl-Latn": ["isl-Latn"],
    "ita-Latn": ["ita-Latn"],
    "jav-Latn": ["jav-Latn"],
    "jpn-Jpan": ["jpn-Jpan"],
    "kac-Latn": ["kac-Latn"],
    "kan-Knda": ["kan-Knda"],
    "kat-Geor": ["kat-Geor"],
    "kaz-Cyrl": ["kaz-Cyrl"],
    "kea-Latn": ["kea-Latn"],
    "khk-Cyrl": ["khk-Cyrl"],
    "khm-Khmr": ["khm-Khmr"],
    "kin-Latn": ["kin-Latn"],
    "kir-Cyrl": ["kir-Cyrl"],
    "kor-Hang": ["kor-Hang"],
    "lao-Laoo": ["lao-Laoo"],
    "lin-Latn": ["lin-Latn"],
    "lit-Latn": ["lit-Latn"],
    "lug-Latn": ["lug-Latn"],
    "luo-Latn": ["luo-Latn"],
    "lvs-Latn": ["lvs-Latn"],
    "mal-Mlym": ["mal-Mlym"],
    "mar-Deva": ["mar-Deva"],
    "mkd-Cyrl": ["mkd-Cyrl"],
    "mlt-Latn": ["mlt-Latn"],
    "mri-Latn": ["mri-Latn"],
    "mya-Mymr": ["mya-Mymr"],
    "nld-Latn": ["nld-Latn"],
    "nob-Latn": ["nob-Latn"],
    "npi-Deva": ["npi-Deva"],
    "npi-Latn": ["npi-Latn"],
    "nso-Latn": ["nso-Latn"],
    "nya-Latn": ["nya-Latn"],
    "ory-Orya": ["ory-Orya"],
    "pan-Guru": ["pan-Guru"],
    "pbt-Arab": ["pbt-Arab"],
    "pes-Arab": ["pes-Arab"],
    "plt-Latn": ["plt-Latn"],
    "pol-Latn": ["pol-Latn"],
    "por-Latn": ["por-Latn"],
    "ron-Latn": ["ron-Latn"],
    "rus-Cyrl": ["rus-Cyrl"],
    "shn-Mymr": ["shn-Mymr"],
    "sin-Latn": ["sin-Latn"],
    "sin-Sinh": ["sin-Sinh"],
    "slk-Latn": ["slk-Latn"],
    "slv-Latn": ["slv-Latn"],
    "sna-Latn": ["sna-Latn"],
    "snd-Arab": ["snd-Arab"],
    "som-Latn": ["som-Latn"],
    "sot-Latn": ["sot-Latn"],
    "spa-Latn": ["spa-Latn"],
    "srp-Cyrl": ["srp-Cyrl"],
    "ssw-Latn": ["ssw-Latn"],
    "sun-Latn": ["sun-Latn"],
    "swe-Latn": ["swe-Latn"],
    "swh-Latn": ["swh-Latn"],
    "tam-Taml": ["tam-Taml"],
    "tel-Telu": ["tel-Telu"],
    "tgk-Cyrl": ["tgk-Cyrl"],
    "tgl-Latn": ["tgl-Latn"],
    "tha-Thai": ["tha-Thai"],
    "tir-Ethi": ["tir-Ethi"],
    "tsn-Latn": ["tsn-Latn"],
    "tso-Latn": ["tso-Latn"],
    "tur-Latn": ["tur-Latn"],
    "ukr-Cyrl": ["ukr-Cyrl"],
    "urd-Arab": ["urd-Arab"],
    "urd-Latn": ["urd-Latn"],
    "uzn-Latn": ["uzn-Latn"],
    "vie-Latn": ["vie-Latn"],
    "war-Latn": ["war-Latn"],
    "wol-Latn": ["wol-Latn"],
    "xho-Latn": ["xho-Latn"],
    "yor-Latn": ["yor-Latn"],
    "zho-Hans": ["zho-Hans"],
    "zho-Hant": ["zho-Hant"],
    "zsm-Latn": ["zsm-Latn"],
    "zul-Latn": ["zul-Latn"]
}


class BelebeleRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BelebeleRetrieval",
        dataset={
            "path": "facebook/belebele",
            "revision": "75b399394a9803252cfec289d103de462763db7c",
        },
        description=(
            "Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning 115 distinct languages."
        ),
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGS,
        reference="https://arxiv.org/abs/2308.16884",
        main_score="ndcg_at_10",
        license="CC-BY-SA-4.0",
        domains=["Web", "News"],
        text_creation="created",
        n_samples={_EVAL_SPLIT: 103500},  # number of languages * 900
        date=("2023-08-31", "2023-08-31"),
        form=["written"],
        task_subtypes=["Question answering"],
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        avg_character_length={_EVAL_SPLIT: 568},  # avg length of query-passage pairs
        bibtex_citation="""@article{bandarkar2023belebele,
      title={The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants}, 
      author={Lucas Bandarkar and Davis Liang and Benjamin Muller and Mikel Artetxe and Satya Narayan Shukla and Donald Husa and Naman Goyal and Abhinandan Krishnan and Luke Zettlemoyer and Madian Khabsa},
      year={2023},
      journal={arXiv preprint arXiv:2308.16884}
}""",
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return

        self.dataset = load_dataset(**self.metadata_dict["dataset"])

        self.queries = {lang: {_EVAL_SPLIT: {}} for lang in self.hf_subsets}
        self.corpus = {lang: {_EVAL_SPLIT: {}} for lang in self.hf_subsets}
        self.relevant_docs = {lang: {_EVAL_SPLIT: {}} for lang in self.hf_subsets}

        for lang in self.hf_subsets:
            belebele_lang = _LANGS[lang][0].replace("-", "_")
            ds = self.dataset[belebele_lang]

            question_ids = {
                question: _id for _id, question in enumerate(set(ds["question"]))
            }
            context_ids = {
                passage: _id for _id, passage in enumerate(set(ds["flores_passage"]))
            }

            for row in ds:
                query = row["question"]
                query_id = f"Q{question_ids[query]}"
                self.queries[lang][_EVAL_SPLIT][query_id] = query
                context = row["flores_passage"]
                context_id = f"C{context_ids[context]}"
                self.corpus[lang][_EVAL_SPLIT][context_id] = {
                    "title": "",
                    "text": context,
                }
                if query_id not in self.relevant_docs[lang][_EVAL_SPLIT]:
                    self.relevant_docs[lang][_EVAL_SPLIT][query_id] = {}
                self.relevant_docs[lang][_EVAL_SPLIT][query_id][context_id] = 1

        self.data_loaded = True
