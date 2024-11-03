from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_LANGS = {
    "ara-ara": ["ara-Arab", "ara-Arab"],
    "eng-ara": ["eng-Latn", "ara-Arab"],
    "ara-eng": ["ara-Arab", "eng-Latn"],
    "deu-deu": ["deu-Latn", "deu-Latn"],
    "eng-deu": ["eng-Latn", "deu-Latn"],
    "deu-eng": ["deu-Latn", "eng-Latn"],
    "spa-spa": ["spa-Latn", "spa-Latn"],
    "eng-spa": ["eng-Latn", "spa-Latn"],
    "spa-eng": ["spa-Latn", "eng-Latn"],
    "fra-fra": ["fra-Latn", "fra-Latn"],
    "eng-fra": ["eng-Latn", "fra-Latn"],
    "fra-eng": ["fra-Latn", "eng-Latn"],
    "hin-hin": ["hin-Deva", "hin-Deva"],
    "eng-hin": ["eng-Latn", "hin-Deva"],
    "hin-eng": ["hin-Deva", "eng-Latn"],
    "ita-ita": ["ita-Latn", "ita-Latn"],
    "eng-ita": ["eng-Latn", "ita-Latn"],
    "ita-eng": ["ita-Latn", "eng-Latn"],
    "jpn-jpn": ["jpn-Hira", "jpn-Hira"],
    "eng-jpn": ["eng-Latn", "jpn-Hira"],
    "jpn-eng": ["jpn-Hira", "eng-Latn"],
    "kor-kor": ["kor-Hang", "kor-Hang"],
    "eng-kor": ["eng-Latn", "kor-Hang"],
    "kor-eng": ["kor-Hang", "eng-Latn"],
    "pol-pol": ["pol-Latn", "pol-Latn"],
    "eng-pol": ["eng-Latn", "pol-Latn"],
    "pol-eng": ["pol-Latn", "eng-Latn"],
    "por-por": ["por-Latn", "por-Latn"],
    "eng-por": ["eng-Latn", "por-Latn"],
    "por-eng": ["por-Latn", "eng-Latn"],
    "tam-tam": ["tam-Taml", "tam-Taml"],
    "eng-tam": ["eng-Latn", "tam-Taml"],
    "tam-eng": ["tam-Taml", "eng-Latn"],
    "cmn-cmn": ["cmn-Hans", "cmn-Hans"],
    "eng-cmn": ["eng-Latn", "cmn-Hans"],
    "cmn-eng": ["cmn-Hans", "eng-Latn"],
}

_LANG_CONVERSION = {
    "ara": "ar",
    "deu": "de",
    "spa": "es",
    "fra": "fr",
    "hin": "hi",
    "ita": "it",
    "jpn": "ja",
    "kor": "ko",
    "pol": "pl",
    "por": "pt",
    "tam": "ta",
    "cmn": "zh",
    "eng": "en",
}


class XPQARetrieval(AbsTaskRetrieval, MultilingualTask):
    metadata = TaskMetadata(
        name="XPQARetrieval",
        description="XPQARetrieval",
        reference="https://arxiv.org/abs/2305.09249",
        dataset={
            "path": "jinaai/xpqa",
            "revision": "c99d599f0a6ab9b85b065da6f9d94f9cf731679f",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2022-01-01", "2023-07-31"),  # best guess
        domains=["Reviews", "Written"],
        task_subtypes=["Question answering"],
        license="cdla-sharing-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{shen2023xpqa,
        title={xPQA: Cross-Lingual Product Question Answering in 12 Languages},
        author={Shen, Xiaoyu and Asai, Akari and Byrne, Bill and De Gispert, Adria},
        booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 5: Industry Track)},
        pages={103--115},
        year={2023}
        }""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        path = self.metadata_dict["dataset"]["path"]
        revision = self.metadata_dict["dataset"]["revision"]
        eval_splits = self.metadata_dict["eval_splits"]
        dataset = _load_dataset_csv(path, revision, eval_splits)

        self.queries, self.corpus, self.relevant_docs = {}, {}, {}
        for lang_pair, _ in self.metadata.eval_langs.items():
            lang_corpus, lang_question = (
                lang_pair.split("-")[0],
                lang_pair.split("-")[1],
            )
            lang_not_english = lang_corpus if lang_corpus != "eng" else lang_question
            dataset_language = dataset.filter(
                lambda x: x["lang"] == _LANG_CONVERSION.get(lang_not_english)
            )
            question_key = "question_en" if lang_question == "eng" else "question"
            corpus_key = "candidate" if lang_corpus == "eng" else "answer"

            queries_to_ids = {
                eval_split: {
                    q: f"Q{str(_id)}"
                    for _id, q in enumerate(
                        set(dataset_language[eval_split][question_key])
                    )
                }
                for eval_split in eval_splits
            }

            self.queries[lang_pair] = {
                eval_split: {v: k for k, v in queries_to_ids[eval_split].items()}
                for eval_split in eval_splits
            }

            corpus_to_ids = {
                eval_split: {
                    document: f"C{str(_id)}"
                    for _id, document in enumerate(
                        set(dataset_language[eval_split][corpus_key])
                    )
                }
                for eval_split in eval_splits
            }

            self.corpus[lang_pair] = {
                eval_split: {
                    v: {"text": k} for k, v in corpus_to_ids[eval_split].items()
                }
                for eval_split in eval_splits
            }

            self.relevant_docs[lang_pair] = {}
            for eval_split in eval_splits:
                self.relevant_docs[lang_pair][eval_split] = {}
                for example in dataset_language[eval_split]:
                    query_id = queries_to_ids[eval_split].get(example[question_key])
                    document_id = corpus_to_ids[eval_split].get(example[corpus_key])
                    if query_id in self.relevant_docs[lang_pair][eval_split]:
                        self.relevant_docs[lang_pair][eval_split][query_id][
                            document_id
                        ] = 1
                    else:
                        self.relevant_docs[lang_pair][eval_split][query_id] = {
                            document_id: 1
                        }

        self.data_loaded = True


def _load_dataset_csv(path: str, revision: str, eval_splits: list[str]):
    data_files = {
        eval_split: f"https://huggingface.co/datasets/{path}/resolve/{revision}/{eval_split}.csv"
        for eval_split in eval_splits
    }
    dataset = datasets.load_dataset("csv", data_files=data_files)
    dataset = dataset.filter(lambda x: x["answer"] is not None)

    return dataset
