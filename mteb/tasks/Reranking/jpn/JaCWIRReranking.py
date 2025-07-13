from __future__ import annotations

import datasets
from datasets import Dataset

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class JaCWIRReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="JaCWIRReranking",
        description=(
            "JaCWIR is a small-scale Japanese information retrieval evaluation dataset consisting of "
            "5000 question texts and approximately 500k web page titles and web page introductions or summaries "
            "(meta descriptions, etc.). The question texts are created based on one of the 500k web pages, "
            "and that data is used as a positive example for the question text."
        ),
        reference="https://huggingface.co/datasets/hotchpotch/JaCWIR",
        dataset={
            "path": "sbintuitions/JMTEB",
            "revision": "b194332dfb8476c7bdd0aaf80e2c4f2a0b4274c2",
            "trust_remote_code": True,
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["jpn-Jpan"],
        main_score="map",
        date=("2020-01-01", "2024-12-31"),
        domains=["Web", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{yuichi-tateno-2024-jacwir,
  author = {Yuichi Tateno},
  title = {JaCWIR: Japanese Casual Web IR - 日本語情報検索評価のための小規模でカジュアルなWebタイトルと概要のデータセット},
  url = {https://huggingface.co/datasets/hotchpotch/JaCWIR},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        # Load queries
        query_list = datasets.load_dataset(
            name="jacwir-reranking-query",
            split=_EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )

        # Load corpus
        corpus_list = datasets.load_dataset(
            name="jacwir-reranking-corpus",
            split="corpus",
            **self.metadata_dict["dataset"],
        )

        # Create corpus mapping
        corpus_map = {}
        for row in corpus_list:
            corpus_map[str(row["docid"])] = row["text"]

        # Transform data to RerankingEvaluator format
        transformed_data = []
        for row in query_list:
            query = row["query"]
            retrieved_docs = row["retrieved_docs"]
            relevance_scores = row["relevance_scores"]

            positive_docs = []
            negative_docs = []

            for doc_id, score in zip(retrieved_docs, relevance_scores):
                doc_text = corpus_map.get(str(doc_id), "")
                if doc_text:  # Only include documents that exist in corpus
                    if score == 1:
                        positive_docs.append(doc_text)
                    else:
                        negative_docs.append(doc_text)

            # Only include samples with both positive and negative documents
            if positive_docs and negative_docs:
                transformed_data.append(
                    {
                        "query": query,
                        "positive": positive_docs,
                        "negative": negative_docs,
                    }
                )

        # Convert to Dataset
        self.dataset = {_EVAL_SPLIT: Dataset.from_list(transformed_data)}
        self.dataset_transform()  # do nothing
        self.data_loaded = True
