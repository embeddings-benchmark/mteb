from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"
_MAX_EVAL_SIZE = 2048


class JaGovFaqsRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="JaGovFaqsRetrieval",
        description="JaGovFaqs is a dataset consisting of FAQs manully extracted from the website of Japanese bureaus. The dataset consists of 22k FAQs, where the queries (questions) and corpus (answers) have been shuffled, and the goal is to match the answer with the question.",
        reference="https://github.com/sbintuitions/JMTEB",
        dataset={
            "path": "sbintuitions/JMTEB",
            "revision": "e4af6c73182bebb41d94cb336846e5a452454ea7",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2000-01-01", "2023-12-31"),
        domains=["Web", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {_EVAL_SPLIT: _MAX_EVAL_SIZE},
            "avg_character_length": {
                "test": {
                    "average_document_length": 210.02601561814512,
                    "average_query_length": 59.48193359375,
                    "num_documents": 22794,
                    "num_queries": 2048,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_list = datasets.load_dataset(
            name="jagovfaqs_22k-query",
            split=_EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )

        # Limit the dataset size to make sure the task does not take too long to run, sample the dataset to 2048 queries
        query_list = query_list.shuffle(seed=self.seed)
        max_samples = min(_MAX_EVAL_SIZE, len(query_list))
        query_list = query_list.select(range(max_samples))

        queries = {}
        qrels = {}
        for row_id, row in enumerate(query_list):
            queries[str(row_id)] = row["query"]
            qrels[str(row_id)] = {str(row["relevant_docs"][0]): 1}

        corpus_list = datasets.load_dataset(
            name="jagovfaqs_22k-corpus", split="corpus", **self.metadata_dict["dataset"]
        )

        corpus = {str(row["docid"]): {"text": row["text"]} for row in corpus_list}

        self.corpus = {_EVAL_SPLIT: corpus}
        self.queries = {_EVAL_SPLIT: queries}
        self.relevant_docs = {_EVAL_SPLIT: qrels}

        self.data_loaded = True
