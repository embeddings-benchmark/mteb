from __future__ import annotations

import datasets
from datasets import Dataset

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class JQaRAReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="JQaRAReranking",
        description=(
            "JQaRA: Japanese Question Answering with Retrieval Augmentation "
            " - 検索拡張(RAG)評価のための日本語 Q&A データセット. JQaRA is an information retrieval task "
            "for questions against 100 candidate data (including one or more correct answers)."
        ),
        reference="https://huggingface.co/datasets/hotchpotch/JQaRA",
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
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=["jpn-Jpan"],
        sample_creation="found",
        prompt="Given a Japanese question, rerank passages based on their relevance for answering the question",
        bibtex_citation=r"""
@misc{yuichi-tateno-2024-jqara,
  author = {Yuichi Tateno},
  title = {JQaRA: Japanese Question Answering with Retrieval Augmentation - 検索拡張(RAG)評価のための日本語Q&Aデータセット},
  url = {https://huggingface.co/datasets/hotchpotch/JQaRA},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        # Load queries
        query_list = datasets.load_dataset(
            name="jqara-query",
            split=_EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )

        # Load corpus
        corpus_list = datasets.load_dataset(
            name="jqara-corpus",
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
