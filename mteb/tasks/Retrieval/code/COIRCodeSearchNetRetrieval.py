from __future__ import annotations

import logging

from mteb.abstasks import MultilingualTask
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.tasks.Retrieval.code.CodeSearchNetCCRetrieval import (
    _load_code_search_code_retrieval,
)

_LANGS = ["python", "javascript", "go", "ruby", "java", "php"]
_EVAL_SPLIT = "test"

logger = logging.getLogger(__name__)


class COIRCodeSearchNetRetrieval(MultilingualTask, AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    metadata = TaskMetadata(
        name="COIRCodeSearchNetRetrieval",
        description="The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code summary given a code snippet.",
        reference="https://huggingface.co/datasets/code_search_net/",
        dataset={
            "path": "CoIR-Retrieval/CodeSearchNet",
            "revision": "4adc7bc41202b5c13543c9c886a25f340634dab3",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs={lang: [lang + "-Code"] for lang in _LANGS},
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="MIT",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="@article{husain2019codesearchnet, title={{CodeSearchNet} challenge: Evaluating the state of semantic code search}, author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc}, journal={arXiv preprint arXiv:1909.09436}, year={2019} }",
        descriptive_stats={
            "n_samples": {
                _EVAL_SPLIT: 1000,
            },
            "avg_character_length": {
                "test": {
                    "python": {
                        "average_document_length": 862.842,
                        "average_query_length": 466.546,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "javascript": {
                        "average_document_length": 1415.632,
                        "average_query_length": 186.018,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "go": {
                        "average_document_length": 563.729,
                        "average_query_length": 125.213,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ruby": {
                        "average_document_length": 577.634,
                        "average_query_length": 313.818,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "java": {
                        "average_document_length": 420.287,
                        "average_query_length": 690.36,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "php": {
                        "average_document_length": 712.129,
                        "average_query_length": 162.119,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                },
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = (
            _load_code_search_code_retrieval(
                path=self.metadata_dict["dataset"]["path"],
                langs=self.hf_subsets,
                splits=self.metadata_dict["eval_splits"],
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata_dict["dataset"]["revision"],
            )
        )

        self.data_loaded = True
