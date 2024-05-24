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
        date=("2000-01-01", "2021-12-31"),
        form=["written"],
        domains=["Spoken"],
        task_subtypes=["Article retrieval"],
        license="Not specified",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @inproceedings{chen-etal-2022-summscreen,
                title = "{S}umm{S}creen: A Dataset for Abstractive Screenplay Summarization",
                author = "Chen, Mingda  and
                Chu, Zewei  and
                Wiseman, Sam  and
                Gimpel, Kevin",
                editor = "Muresan, Smaranda  and
                Nakov, Preslav  and
                Villavicencio, Aline",
                booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
                month = may,
                year = "2022",
                address = "Dublin, Ireland",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/2022.acl-long.589",
                doi = "10.18653/v1/2022.acl-long.589",
                pages = "8602--8615",
                abstract = "",
            }
        """,
        n_samples={_EVAL_SPLIT: 672},
        avg_character_length={_EVAL_SPLIT: 31445.8},
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
