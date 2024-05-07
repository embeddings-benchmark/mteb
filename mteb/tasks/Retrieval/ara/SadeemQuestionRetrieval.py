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
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["ara-Arab"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-04-01"),
        form=["written"],
        domains=["Spoken"],
        task_subtypes=["Article retrieval"],
        license="Not specified",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @inproceedings{sadeem-2024-ar-retrieval-questions,
            title = "{QMS}um: A New Benchmark for Arabic questions-based Articles Searching.",
            author = "Abu Bakr Soliman | Saddem-ai"
            }
        """,
        n_samples={_EVAL_SPLIT: 22979},
        avg_character_length={_EVAL_SPLIT: 500.0},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
            "queries"
        ]
        queries = {row["query-id"]: row["text"] for row in query_list}

        corpus_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
            "corpus"
        ]
        corpus = {row["corpus-id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
            "qrels"
        ]
        qrels = {row["query-id"]: {row["corpus-id"]: 1} for row in qrels_list}

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True
