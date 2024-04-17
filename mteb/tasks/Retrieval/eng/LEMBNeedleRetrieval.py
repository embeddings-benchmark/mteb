import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LEMBNeedleRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

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
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=["written"],
        domains=None,
        task_subtypes=["Article retrieval"],
        license=None,
        socioeconomic_status=None,
        annotations_creators="derived",
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={_EVAL_SPLIT: 400},
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        if "context_length" not in kwargs:
            raise ValueError("Need to specify context_length")
        context_length = kwargs["context_length"]

        query_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
            "queries"
        ]  # dict_keys(['qid', 'text'])
        query_list = query_list.filter(lambda x: x["context_length"] == context_length)
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
        qrels_list = qrels_list.filter(lambda x: x["context_length"] == context_length)
        qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

        print("Example Query")
        print(list(queries.values())[5])
        print("Example Passage (truncate at 200 characters)")
        print(list(corpus.values())[5]["text"][:200])

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True
