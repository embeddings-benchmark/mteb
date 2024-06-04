import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LEMBPasskeyRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = [
        "test_256",
        "test_512",
        "test_1024",
        "test_2048",
        "test_4096",
        "test_8192",
        "test_16384",
        "test_32768",
    ]

    metadata = TaskMetadata(
        name="LEMBPasskeyRetrieval",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
            "name": "passkey",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("passkey subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        eval_splits=_EVAL_SPLIT,
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_1",
        date=("2000-01-01", "2023-12-31"),
        form=["written"],
        domains=["Fiction"],
        task_subtypes=["Article retrieval"],
        license="Not specified",
        socioeconomic_status="low",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @article{zhu2024longembed,
            title={LongEmbed: Extending Embedding Models for Long Context Retrieval},
            author={Zhu, Dawei and Wang, Liang and Yang, Nan and Song, Yifan and Wu, Wenhao and Wei, Furu and Li, Sujian},
            journal={arXiv preprint arXiv:2404.12096},
            year={2024}
            }
        """,
        n_samples={
            "test_256": 150,
            "test_512": 150,
            "test_1024": 150,
            "test_2048": 150,
            "test_4096": 150,
            "test_8192": 150,
            "test_16384": 150,
            "test_32768": 150,
        },
        avg_character_length={
            "test_256": 914.9,
            "test_512": 1823.0,
            "test_1024": 3644.7,
            "test_2048": 7280.0,
            "test_4096": 14555.5,
            "test_8192": 29108.1,
            "test_16384": 58213.9,
            "test_32768": 116417.9,
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = {}
        self.queries = {}
        self.relevant_docs = {}

        for split in self._EVAL_SPLIT:
            context_length = int(split.split("_")[1])
            query_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
                "queries"
            ]  # dict_keys(['qid', 'text'])
            query_list = query_list.filter(
                lambda x: x["context_length"] == context_length
            )
            queries = {row["qid"]: row["text"] for row in query_list}

            corpus_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
                "corpus"
            ]  # dict_keys(['doc_id', 'text'])
            corpus_list = corpus_list.filter(
                lambda x: x["context_length"] == context_length
            )
            corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

            qrels_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
                "qrels"
            ]  # dict_keys(['qid', 'doc_id'])
            qrels_list = qrels_list.filter(
                lambda x: x["context_length"] == context_length
            )
            qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

            self.corpus[split] = corpus
            self.queries[split] = queries
            self.relevant_docs[split] = qrels

        self.data_loaded = True
