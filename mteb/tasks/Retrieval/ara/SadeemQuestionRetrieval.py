from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


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
        bibtex_citation="""
            @inproceedings{sadeem-2024-ar-retrieval-questions,
            title = "SadeemQuestionRetrieval: A New Benchmark for Arabic questions-based Articles Searching.",
            author = "abubakr.soliman@sadeem.app"
            }
        """,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_list = datasets.load_dataset(**self.metadata.dataset, split="queries")
        queries = {row["query-id"]: row["text"] for row in query_list}

        corpus_list = datasets.load_dataset(**self.metadata.dataset, split="corpus")
        corpus = {row["corpus-id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = datasets.load_dataset(**self.metadata.dataset, split="qrels")
        qrels = {row["query-id"]: {row["corpus-id"]: 1} for row in qrels_list}

        self.dataset = {
            "default": {
                self._EVAL_SPLIT: {
                    "corpus": corpus,
                    "queries": queries,
                    "relevant_docs": qrels,
                }
            }
        }

        self.data_loaded = True
