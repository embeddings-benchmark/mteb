from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class SweFaqRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SweFaqRetrieval",
        dataset={
            "path": "AI-Sweden/SuperLim",
            "revision": "7ebf0b4caa7b2ae39698a889de782c09e6f5ee56",
            "name": "swefaq",
            "trust_remote_code": True,
        },
        description="A Swedish QA dataset derived from FAQ",
        reference="https://spraakbanken.gu.se/en/resources/superlim",
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["swe-Latn"],
        main_score="ndcg_at_10",
        date=("2000-01-01", "2024-12-31"),  # best guess
        task_subtypes=["Question answering"],
        domains=["Government", "Non-fiction", "Written"],
        license="CC-BY-SA-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=None,
        descriptive_stats={
            "n_samples": {"test": 1024},
            "avg_character_length": {
                "test": {
                    "average_document_length": 319.8473581213307,
                    "average_query_length": 70.51461988304094,
                    "num_documents": 511,
                    "num_queries": 513,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata.dataset)  # type: ignore
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        """And transform to a retrieval datset, which have the following attributes

        self.corpus = Dict[doc_id, Dict[str, str]] #id => dict with document datas like title and text
        self.queries = Dict[query_id, str] #id => query
        self.relevant_docs = Dict[query_id, Dict[[doc_id, score]]
        """
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}
        text2id = {}

        for split in self.dataset:
            ds: datasets.Dataset = self.dataset[split]  # type: ignore
            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}

            questions = ds["question"]
            ca_answers = ds["candidate_answer"]
            co_answers = ds["correct_answer"]

            n = 0
            for q, ca, co in zip(questions, ca_answers, co_answers):
                self.queries[split][str(n)] = q
                q_n = n
                n += 1
                if ca not in text2id:
                    text2id[ca] = n
                    self.corpus[split][str(n)] = {"title": "", "text": ca}
                    n += 1
                if co not in text2id:
                    text2id[co] = n
                    self.corpus[split][str(n)] = {"title": "", "text": co}
                    n += 1
                cor_n = text2id[co]

                self.relevant_docs[split][str(q_n)] = {
                    str(cor_n): 1,
                }  # only one correct match
