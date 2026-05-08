import datasets
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SadeemQuestionRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="SadeemQuestionRetrieval",
        dataset={
            "path": "sadeem-ai/sadeem-ar-eval-retrieval-questions",
            "revision": "3cb0752b182e5d5d740df547748b06663c8e0bd9",
            "name": "test",
        },
        reference="https://huggingface.co/datasets/sadeem-ai/sadeem-ar-eval-retrieval-questions",
        description="SadeemQuestion: A Benchmark Data Set for Community Question-Retrieval Research",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["ara-Arab"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-04-01"),
        domains=["Written", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{sadeem-2024-ar-retrieval-questions,
  author = {abubakr.soliman@sadeem.app},
  title = {SadeemQuestionRetrieval: A New Benchmark for Arabic questions-based Articles Searching.},
}
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        query_list = datasets.load_dataset(**self.metadata.dataset)["queries"]
        queries_dict = {row["query-id"]: row["text"] for row in query_list}

        corpus_list = datasets.load_dataset(**self.metadata.dataset)["corpus"]
        corpus_dict = {
            row["corpus-id"]: {"text": row["text"], "title": ""} for row in corpus_list
        }

        qrels_list = datasets.load_dataset(**self.metadata.dataset)["qrels"]
        relevant_docs = {row["query-id"]: {row["corpus-id"]: 1} for row in qrels_list}

        corpus_dataset = Dataset.from_list(
            [
                {"id": k, "text": v["text"], "title": v["title"]}
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
                    "relevant_docs": relevant_docs,
                    "top_ranked": None,
                }
            }
        }

        self.data_loaded = True
