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
        descriptive_stats={
            "n_samples": {"test": 19801},
            "avg_character_length": {
                "test": {
                    "ara-ara": {
                        "average_document_length": 61.88361204013378,
                        "average_query_length": 29.688,
                        "num_documents": 1495,
                        "num_queries": 750,
                        "average_relevant_docs_per_query": 2.004,
                    },
                    "eng-ara": {
                        "average_document_length": 125.26940639269407,
                        "average_query_length": 29.688,
                        "num_documents": 1533,
                        "num_queries": 750,
                        "average_relevant_docs_per_query": 2.058666666666667,
                    },
                    "ara-eng": {
                        "average_document_length": 61.88361204013378,
                        "average_query_length": 39.5188679245283,
                        "num_documents": 1495,
                        "num_queries": 742,
                        "average_relevant_docs_per_query": 2.024258760107817,
                    },
                    "deu-deu": {
                        "average_document_length": 69.54807692307692,
                        "average_query_length": 55.51827676240209,
                        "num_documents": 1248,
                        "num_queries": 766,
                        "average_relevant_docs_per_query": 1.6318537859007833,
                    },
                    "eng-deu": {
                        "average_document_length": 115.77118078719145,
                        "average_query_length": 55.51827676240209,
                        "num_documents": 1499,
                        "num_queries": 766,
                        "average_relevant_docs_per_query": 1.9634464751958225,
                    },
                    "deu-eng": {
                        "average_document_length": 69.54807692307692,
                        "average_query_length": 51.88903394255875,
                        "num_documents": 1248,
                        "num_queries": 766,
                        "average_relevant_docs_per_query": 1.6318537859007833,
                    },
                    "spa-spa": {
                        "average_document_length": 68.27511591962906,
                        "average_query_length": 46.711223203026485,
                        "num_documents": 1941,
                        "num_queries": 793,
                        "average_relevant_docs_per_query": 2.4489281210592684,
                    },
                    "eng-spa": {
                        "average_document_length": 123.43698347107438,
                        "average_query_length": 46.711223203026485,
                        "num_documents": 1936,
                        "num_queries": 793,
                        "average_relevant_docs_per_query": 2.472887767969735,
                    },
                    "spa-eng": {
                        "average_document_length": 68.27511591962906,
                        "average_query_length": 47.21059268600252,
                        "num_documents": 1941,
                        "num_queries": 793,
                        "average_relevant_docs_per_query": 2.4489281210592684,
                    },
                    "fra-fra": {
                        "average_document_length": 76.99354005167959,
                        "average_query_length": 56.0520694259012,
                        "num_documents": 1548,
                        "num_queries": 749,
                        "average_relevant_docs_per_query": 2.069425901201602,
                    },
                    "eng-fra": {
                        "average_document_length": 137.31242532855435,
                        "average_query_length": 56.0520694259012,
                        "num_documents": 1674,
                        "num_queries": 749,
                        "average_relevant_docs_per_query": 2.248331108144192,
                    },
                    "fra-eng": {
                        "average_document_length": 76.99354005167959,
                        "average_query_length": 49.58744993324433,
                        "num_documents": 1548,
                        "num_queries": 749,
                        "average_relevant_docs_per_query": 2.069425901201602,
                    },
                    "hin-hin": {
                        "average_document_length": 47.20783373301359,
                        "average_query_length": 33.47783783783784,
                        "num_documents": 1251,
                        "num_queries": 925,
                        "average_relevant_docs_per_query": 1.3902702702702703,
                    },
                    "eng-hin": {
                        "average_document_length": 106.67662682602922,
                        "average_query_length": 33.47783783783784,
                        "num_documents": 1506,
                        "num_queries": 925,
                        "average_relevant_docs_per_query": 1.8054054054054054,
                    },
                    "hin-eng": {
                        "average_document_length": 47.20783373301359,
                        "average_query_length": 34.98574561403509,
                        "num_documents": 1251,
                        "num_queries": 912,
                        "average_relevant_docs_per_query": 1.4100877192982457,
                    },
                    "ita-ita": {
                        "average_document_length": 59.778301886792455,
                        "average_query_length": 49.14932126696833,
                        "num_documents": 1272,
                        "num_queries": 663,
                        "average_relevant_docs_per_query": 1.9245852187028658,
                    },
                    "eng-ita": {
                        "average_document_length": 123.07302075326672,
                        "average_query_length": 49.14932126696833,
                        "num_documents": 1301,
                        "num_queries": 663,
                        "average_relevant_docs_per_query": 1.9849170437405732,
                    },
                    "ita-eng": {
                        "average_document_length": 59.778301886792455,
                        "average_query_length": 49.040723981900456,
                        "num_documents": 1272,
                        "num_queries": 663,
                        "average_relevant_docs_per_query": 1.9245852187028658,
                    },
                    "jpn-jpn": {
                        "average_document_length": 41.030605871330415,
                        "average_query_length": 23.296969696969697,
                        "num_documents": 1601,
                        "num_queries": 825,
                        "average_relevant_docs_per_query": 1.9406060606060607,
                    },
                    "eng-jpn": {
                        "average_document_length": 126.2647564469914,
                        "average_query_length": 23.296969696969697,
                        "num_documents": 1745,
                        "num_queries": 825,
                        "average_relevant_docs_per_query": 2.1187878787878787,
                    },
                    "jpn-eng": {
                        "average_document_length": 41.030605871330415,
                        "average_query_length": 51.416058394160586,
                        "num_documents": 1601,
                        "num_queries": 822,
                        "average_relevant_docs_per_query": 1.9476885644768855,
                    },
                    "kor-kor": {
                        "average_document_length": 31.22722159730034,
                        "average_query_length": 21.81804281345566,
                        "num_documents": 889,
                        "num_queries": 654,
                        "average_relevant_docs_per_query": 1.5642201834862386,
                    },
                    "eng-kor": {
                        "average_document_length": 112.41231822070145,
                        "average_query_length": 21.81804281345566,
                        "num_documents": 1169,
                        "num_queries": 654,
                        "average_relevant_docs_per_query": 1.952599388379205,
                    },
                    "kor-eng": {
                        "average_document_length": 31.22722159730034,
                        "average_query_length": 43.9527687296417,
                        "num_documents": 889,
                        "num_queries": 614,
                        "average_relevant_docs_per_query": 1.6661237785016287,
                    },
                    "pol-pol": {
                        "average_document_length": 50.66814439518683,
                        "average_query_length": 53.72101910828025,
                        "num_documents": 1579,
                        "num_queries": 785,
                        "average_relevant_docs_per_query": 2.080254777070064,
                    },
                    "eng-pol": {
                        "average_document_length": 112.96919566457501,
                        "average_query_length": 53.72101910828025,
                        "num_documents": 1753,
                        "num_queries": 785,
                        "average_relevant_docs_per_query": 2.385987261146497,
                    },
                    "pol-eng": {
                        "average_document_length": 50.66814439518683,
                        "average_query_length": 54.1994851994852,
                        "num_documents": 1579,
                        "num_queries": 777,
                        "average_relevant_docs_per_query": 2.101673101673102,
                    },
                    "por-por": {
                        "average_document_length": 75.9845869297164,
                        "average_query_length": 42.58875,
                        "num_documents": 1622,
                        "num_queries": 800,
                        "average_relevant_docs_per_query": 2.14,
                    },
                    "eng-por": {
                        "average_document_length": 111.42525930445393,
                        "average_query_length": 42.58875,
                        "num_documents": 1639,
                        "num_queries": 800,
                        "average_relevant_docs_per_query": 2.21875,
                    },
                    "por-eng": {
                        "average_document_length": 75.9845869297164,
                        "average_query_length": 46.57967377666248,
                        "num_documents": 1622,
                        "num_queries": 797,
                        "average_relevant_docs_per_query": 2.148055207026349,
                    },
                    "tam-tam": {
                        "average_document_length": 64.89019607843137,
                        "average_query_length": 33.267263427109974,
                        "num_documents": 1275,
                        "num_queries": 782,
                        "average_relevant_docs_per_query": 1.6994884910485935,
                    },
                    "eng-tam": {
                        "average_document_length": 96.96361185983828,
                        "average_query_length": 33.267263427109974,
                        "num_documents": 1484,
                        "num_queries": 782,
                        "average_relevant_docs_per_query": 2.0255754475703327,
                    },
                    "tam-eng": {
                        "average_document_length": 64.89019607843137,
                        "average_query_length": 34.777633289986994,
                        "num_documents": 1275,
                        "num_queries": 769,
                        "average_relevant_docs_per_query": 1.728218465539662,
                    },
                    "cmn-cmn": {
                        "average_document_length": 20.958944281524925,
                        "average_query_length": 12.21116504854369,
                        "num_documents": 1705,
                        "num_queries": 824,
                        "average_relevant_docs_per_query": 2.0716019417475726,
                    },
                    "eng-cmn": {
                        "average_document_length": 108.31593874078276,
                        "average_query_length": 12.21116504854369,
                        "num_documents": 1763,
                        "num_queries": 824,
                        "average_relevant_docs_per_query": 2.2633495145631066,
                    },
                    "cmn-eng": {
                        "average_document_length": 20.958944281524925,
                        "average_query_length": 41.24390243902439,
                        "num_documents": 1705,
                        "num_queries": 820,
                        "average_relevant_docs_per_query": 2.0817073170731706,
                    },
                }
            },
        },
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
