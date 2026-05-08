import datasets
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2000-01-01", "2021-12-31"),
        domains=["Spoken", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{chen-etal-2022-summscreen,
  address = {Dublin, Ireland},
  author = {Chen, Mingda  and
Chu, Zewei  and
Wiseman, Sam  and
Gimpel, Kevin},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2022.acl-long.589},
  editor = {Muresan, Smaranda  and
Nakov, Preslav  and
Villavicencio, Aline},
  month = may,
  pages = {8602--8615},
  publisher = {Association for Computational Linguistics},
  title = {{S}umm{S}creen: A Dataset for Abstractive Screenplay Summarization},
  url = {https://aclanthology.org/2022.acl-long.589},
  year = {2022},
}
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        query_list = datasets.load_dataset(**self.metadata.dataset)[
            "queries"
        ]  # dict_keys(['qid', 'text'])
        queries_dict = {row["qid"]: row["text"] for row in query_list}

        corpus_list = datasets.load_dataset(**self.metadata.dataset)[
            "corpus"
        ]  # dict_keys(['doc_id', 'text'])
        corpus_dict = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = datasets.load_dataset(**self.metadata.dataset)[
            "qrels"
        ]  # dict_keys(['qid', 'doc_id'])
        qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

        corpus_dataset = Dataset.from_list(
            [
                {
                    "id": k,
                    "text": v.get("text", "") if isinstance(v, dict) else v,
                    "title": v.get("title", "") if isinstance(v, dict) else "",
                }
                for k, v in corpus_dict.items()
            ]
        )
        queries_dataset = Dataset.from_list(
            [{"id": k, "text": v} for k, v in queries_dict.items()]
        )

        self.dataset = {
            "default": {
                self._EVAL_SPLIT: {
                    "corpus": corpus_dataset,
                    "queries": queries_dataset,
                    "relevant_docs": qrels,
                    "top_ranked": None,
                }
            }
        }

        self.data_loaded = True
