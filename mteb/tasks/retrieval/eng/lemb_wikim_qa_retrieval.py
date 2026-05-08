import datasets
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LEMBWikimQARetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBWikimQARetrieval",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
            "name": "2wikimqa",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("2wikimqa subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("1950-01-01", "2019-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{ho2020constructing,
  author = {Ho, Xanh and Nguyen, Anh-Khoa Duong and Sugawara, Saku and Aizawa, Akiko},
  booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
  pages = {6609--6625},
  title = {Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps},
  year = {2020},
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
