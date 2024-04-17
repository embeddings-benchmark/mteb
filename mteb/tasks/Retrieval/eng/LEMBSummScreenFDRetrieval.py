import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LEMBSummScreenFDRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "validation"

    metadata = TaskMetadata(
        name="LEMBSummScreenFDRetrieval",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
            "name": "summ_screen_fd",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("summ_screen_fd subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    # @property
    # def description(self):
    #     return {
    #         'name': 'LEMBSummScreenFDRetrieval',
    #         'dataset_name': 'dwzhu/LongEmbed',
    #         'subset': 'summ_screen_fd',
    #         'reference': 'https://huggingface.co/datasets/dwzhu/LongEmbed',
    #         "description": (
    #             "summ_screen_fd subset of dwzhu/LongEmbed dataset. "
    #         ),
    #         "type": "Retrieval",
    #         "category": "s2p",
    #         "eval_splits": ["validation"],
    #         "eval_langs": ["en"],
    #         "main_score": "ndcg_at_10",
    #     }

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
