from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LEMBQMSumRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBQMSumRetrieval",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
            "name": "qmsum",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("qmsum subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("1950-01-01", "2021-12-31"),
        domains=["Spoken", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
            @inproceedings{zhong-etal-2021-qmsum,
            title = "{QMS}um: A New Benchmark for Query-based Multi-domain Meeting Summarization",
            author = "Zhong, Ming  and
            Yin, Da  and
            Yu, Tao  and
            Zaidi, Ahmad  and
            Mutuma, Mutethia  and
            Jha, Rahul  and
            Awadallah, Ahmed Hassan  and
            Celikyilmaz, Asli  and
            Liu, Yang  and
            Qiu, Xipeng  and
            Radev, Dragomir",
            editor = "Toutanova, Kristina  and
            Rumshisky, Anna  and
            Zettlemoyer, Luke  and
            Hakkani-Tur, Dilek  and
            Beltagy, Iz  and
            Bethard, Steven  and
            Cotterell, Ryan  and
            Chakraborty, Tanmoy  and
            Zhou, Yichao",
            booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
            month = jun,
            year = "2021",
            address = "Online",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2021.naacl-main.472",
            doi = "10.18653/v1/2021.naacl-main.472",
            pages = "5905--5921",
            abstract = "",
            }
        """,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
            "queries"
        ]  # dict_keys(['qid', 'text'])
        queries = {row["qid"]: row["text"] for row in query_list}

        corpus_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
            "corpus"
        ]  # dict_keys(['doc_id', 'text'])
        corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
            "qrels"
        ]  # dict_keys(['qid', 'doc_id'])
        qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True
