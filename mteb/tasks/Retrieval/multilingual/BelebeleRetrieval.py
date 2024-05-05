from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_EVAL_SPLIT = "test"

_LANGS = {
    "acm": ["acm-Arab"],
    "afr": ["afr-Latn"],
    "als": ["als-Latn"],
    "amh": ["amh-Ethi"],
    "apc": ["apc-Arab"],
    "arb": ["arb-Arab"],
    "ars": ["ars-Arab"],
    "ary": ["ary-Arab"],
    "arz": ["arz-Arab"],
    "asm": ["asm-Beng"],
    "azj": ["azj-Latn"],
    "bam": ["bam-Latn"],
    "ben": ["ben-Beng"],
    "bod": ["bod-Tibt"],
    "bul": ["bul-Cyrl"],
    "cat": ["cat-Latn"],
    "ceb": ["ceb-Latn"],
    "ces": ["ces-Latn"],
    "ckb": ["ckb-Arab"],
    "dan": ["dan-Latn"],
    "deu": ["deu-Latn"],
    "ell": ["ell-Grek"],
    "eng": ["eng-Latn"],
    "est": ["est-Latn"],
    "eus": ["eus-Latn"],
    "fin": ["fin-Latn"],
    "fra": ["fra-Latn"],
    "fuv": ["fuv-Latn"],
    "gaz": ["gaz-Latn"],
    "grn": ["grn-Latn"],
    "guj": ["guj-Gujr"],
    "hat": ["hat-Latn"],
    "hau": ["hau-Latn"],
    "heb": ["heb-Hebr"],
    "hin": ["hin-Deva"],
    "hrv": ["hrv-Latn"],
    "hun": ["hun-Latn"],
    "hye": ["hye-Armn"],
    "ibo": ["ibo-Latn"],
    "ilo": ["ilo-Latn"],
    "ind": ["ind-Latn"],
    "isl": ["isl-Latn"],
    "ita": ["ita-Latn"],
    "jav": ["jav-Latn"],
    "jpn": ["jpn-Jpan"],
    "kac": ["kac-Latn"],
    "kan": ["kan-Knda"],
    "kat": ["kat-Geor"],
    "kaz": ["kaz-Cyrl"],
    "kea": ["kea-Latn"],
    "khk": ["khk-Cyrl"],
    "khm": ["khm-Khmr"],
    "kin": ["kin-Latn"],
    "kir": ["kir-Cyrl"],
    "kor": ["kor-Hang"],
    "lao": ["lao-Laoo"],
    "lin": ["lin-Latn"],
    "lit": ["lit-Latn"],
    "lug": ["lug-Latn"],
    "luo": ["luo-Latn"],
    "lvs": ["lvs-Latn"],
    "mal": ["mal-Mlym"],
    "mar": ["mar-Deva"],
    "mkd": ["mkd-Cyrl"],
    "mlt": ["mlt-Latn"],
    "mri": ["mri-Latn"],
    "mya": ["mya-Mymr"],
    "nld": ["nld-Latn"],
    "nob": ["nob-Latn"],
    "npi": ["npi-Deva"],
    "nso": ["nso-Latn"],
    "nya": ["nya-Latn"],
    "ory": ["ory-Orya"],
    "pan": ["pan-Guru"],
    "pbt": ["pbt-Arab"],
    "pes": ["pes-Arab"],
    "plt": ["plt-Latn"],
    "pol": ["pol-Latn"],
    "por": ["por-Latn"],
    "ron": ["ron-Latn"],
    "rus": ["rus-Cyrl"],
    "shn": ["shn-Mymr"],
    "sin": ["sin-Latn"],
    "slk": ["slk-Latn"],
    "slv": ["slv-Latn"],
    "sna": ["sna-Latn"],
    "snd": ["snd-Arab"],
    "som": ["som-Latn"],
    "sot": ["sot-Latn"],
    "spa": ["spa-Latn"],
    "srp": ["srp-Cyrl"],
    "ssw": ["ssw-Latn"],
    "sun": ["sun-Latn"],
    "swe": ["swe-Latn"],
    "swh": ["swh-Latn"],
    "tam": ["tam-Taml"],
    "tel": ["tel-Telu"],
    "tgk": ["tgk-Cyrl"],
    "tgl": ["tgl-Latn"],
    "tha": ["tha-Thai"],
    "tir": ["tir-Ethi"],
    "tsn": ["tsn-Latn"],
    "tso": ["tso-Latn"],
    "tur": ["tur-Latn"],
    "ukr": ["ukr-Cyrl"],
    "urd": ["urd-Arab"],
    "uzn": ["uzn-Latn"],
    "vie": ["vie-Latn"],
    "war": ["war-Latn"],
    "wol": ["wol-Latn"],
    "xho": ["xho-Latn"],
    "yor": ["yor-Latn"],
    "zho": ["zho-Hans"],
    "zsm": ["zsm-Latn"],
    "zul": ["zul-Latn"],
}


class BelebeleRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BelebeleRetrieval",
        dataset={
            "path": "facebook/belebele",
            "revision": "75b399394a9803252cfec289d103de462763db7c",
        },
        description=(
            "Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants."
        ),
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGS,
        reference="https://arxiv.org/abs/2308.16884",
        main_score="ndcg_at_20",
        license="CC-BY-SA-4.0",
        domains=["Web", "News"],
        text_creation="created",
        n_samples={_EVAL_SPLIT: 900},
        date=("2023-08-31", "2023-08-31"),
        form=["written"],
        task_subtypes=["Question answering"],
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        avg_character_length={_EVAL_SPLIT: 329},
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

        self.queries = {lang: {_EVAL_SPLIT: {}} for lang in self.langs}
        self.corpus = {lang: {_EVAL_SPLIT: {}} for lang in self.langs}
        self.relevant_docs = {lang: {_EVAL_SPLIT: {}} for lang in self.langs}

        for lang in self.langs:
            belebele_lang = _LANGS[lang][0].replace("-", "_")
            ds = self.dataset[belebele_lang]

            question_ids = {
                question: _id for _id, question in enumerate(set(ds["question"]))
            }
            context_ids = {
                passage: _id for _id, passage in enumerate(set(ds["flores_passage"]))
            }
            answers = [
                row[f"mc_answer{num}"] for row, num in zip(ds, ds["correct_answer_num"])
            ]
            answer_ids = {answer: _id for _id, answer in enumerate(set(answers))}

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
                answer = row[f"mc_answer{row['correct_answer_num']}"]
                answer_id = f"A{answer_ids[answer]}"
                self.corpus[lang][_EVAL_SPLIT][answer_id] = {
                    "title": "",
                    "text": answer,
                }

                self.relevant_docs[lang][_EVAL_SPLIT][query_id] = {
                    context_id: 1,
                    answer_id: 1,
                }

        self.data_loaded = True
