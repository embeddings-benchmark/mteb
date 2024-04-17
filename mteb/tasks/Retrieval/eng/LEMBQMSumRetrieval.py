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
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=["written"],
        domains=["Spoken"],
        task_subtypes=["Article retrieval"],
        license=None,
        socioeconomic_status=None,
        annotations_creators="derived",
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={_EVAL_SPLIT: 1527},
        avg_character_length=None,
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

        print("Example Query")
        print(list(queries.values())[5])
        print("Example Passage (truncate at 200 characters)")
        print(list(corpus.values())[5]["text"][:200])

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True
