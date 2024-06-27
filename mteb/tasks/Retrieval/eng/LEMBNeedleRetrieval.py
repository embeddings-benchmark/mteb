import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LEMBNeedleRetrieval(AbsTaskRetrieval):
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
        name="LEMBNeedleRetrieval",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
            "name": "needle",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("needle subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        eval_splits=_EVAL_SPLIT,
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_1",
        date=("2000-01-01", "2023-12-31"),
        form=["written"],
        domains=["Academic", "Blog"],
        task_subtypes=["Article retrieval"],
        license="Not specified",
        socioeconomic_status="high",
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
            "test_256": {
                "average_document_length": 1013.22,
                "average_query_length": 60.48,
                "num_documents": 100,
                "num_queries": 50,
                "average_relevant_docs_per_query": 1.0,
            },
            "test_512": {
                "average_document_length": 2009.96,
                "average_query_length": 57.3,
                "num_documents": 100,
                "num_queries": 50,
                "average_relevant_docs_per_query": 1.0,
            },
            "test_1024": {
                "average_document_length": 4069.9,
                "average_query_length": 58.28,
                "num_documents": 100,
                "num_queries": 50,
                "average_relevant_docs_per_query": 1.0,
            },
            "test_2048": {
                "average_document_length": 8453.82,
                "average_query_length": 59.92,
                "num_documents": 100,
                "num_queries": 50,
                "average_relevant_docs_per_query": 1.0,
            },
            "test_4096": {
                "average_document_length": 17395.8,
                "average_query_length": 55.86,
                "num_documents": 100,
                "num_queries": 50,
                "average_relevant_docs_per_query": 1.0,
            },
            "test_8192": {
                "average_document_length": 35203.82,
                "average_query_length": 59.6,
                "num_documents": 100,
                "num_queries": 50,
                "average_relevant_docs_per_query": 1.0,
            },
            "test_16384": {
                "average_document_length": 72054.8,
                "average_query_length": 59.12,
                "num_documents": 100,
                "num_queries": 50,
                "average_relevant_docs_per_query": 1.0,
            },
            "test_32768": {
                "average_document_length": 141769.8,
                "average_query_length": 58.34,
                "num_documents": 100,
                "num_queries": 50,
                "average_relevant_docs_per_query": 1.0,
            },
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
