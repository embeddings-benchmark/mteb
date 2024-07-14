from __future__ import annotations

import datasets

from mteb.abstasks import MultilingualTask
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = [
    "python",
    "javascript",
    "typescript",
    "go",
    "ruby",
    "java",
    "php",
    "c",
    "c++",
    "rust",
    "swift",
    "scala",
    "shell",
]


class CodeEditSearchRetrieval(MultilingualTask, AbsTaskRetrieval):
    _EVAL_SPLIT = "train"
    metadata = TaskMetadata(
        name="CodeEditSearchRetrieval",
        description="The dataset is a collection of unified diffs of code changes, paired with a short instruction that describes the change. The dataset is derived from the CommitPackFT dataset.",
        reference="https://huggingface.co/datasets/cassanof/CodeEditSearch/viewer",
        dataset={
            "path": "cassanof/CodeEditSearch",
            "revision": "4e51c66e0939303f6928472f13ad0848b2a3f4c0",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs={lang: [lang + "-Code"] for lang in _LANGS},
        main_score="ndcg_at_10",
        date=("2011-02-12", "2016-01-01"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="@article{muennighoff2023octopack, title={OctoPack: Instruction Tuning Code Large Language Models}, author={Niklas Muennighoff and Qian Liu and Armel Zebaze and Qinkai Zheng and Binyuan Hui and Terry Yue Zhuo and Swayam Singh and Xiangru Tang and Leandro von Werra and Shayne Longpre}, journal={arXiv preprint arXiv:2308.07124}, year={2023} }",
        descriptive_stats={
            "n_samples": {
                _EVAL_SPLIT: 1000 * len(_LANGS),
            },
            "avg_character_length": {
                "train": {
                    "python": {
                        "average_document_length": 597.592,
                        "average_query_length": 69.519,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "javascript": {
                        "average_document_length": 582.554,
                        "average_query_length": 56.88,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "typescript": {
                        "average_document_length": 580.877,
                        "average_query_length": 60.092,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "go": {
                        "average_document_length": 548.498,
                        "average_query_length": 70.797,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ruby": {
                        "average_document_length": 518.895,
                        "average_query_length": 66.9,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "java": {
                        "average_document_length": 620.332,
                        "average_query_length": 62.984,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "php": {
                        "average_document_length": 545.452,
                        "average_query_length": 61.927,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "c": {
                        "average_document_length": 475.868,
                        "average_query_length": 97.588,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "c++": {
                        "average_document_length": 544.446,
                        "average_query_length": 114.48,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "rust": {
                        "average_document_length": 609.548,
                        "average_query_length": 67.503,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "swift": {
                        "average_document_length": 574.62,
                        "average_query_length": 57.279,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "scala": {
                        "average_document_length": 495.485,
                        "average_query_length": 64.833,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "shell": {
                        "average_document_length": 486.519,
                        "average_query_length": 72.059,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        lang_subs = {lang: [] for lang in _LANGS}
        for lang in _LANGS:
            data = datasets.load_dataset(
                split=self._EVAL_SPLIT,
                data_dir=lang,
                **self.metadata_dict["dataset"],
            )
            for row in data:
                lang_subs[lang].append(row)

        self.queries = {}
        self.corpus = {}
        self.relevant_docs = {}

        for lang, sub in lang_subs.items():
            sub = sub[:1000]

            self.queries[lang] = {
                self._EVAL_SPLIT: {
                    str(i): row["instruction"] for i, row in enumerate(sub)
                }
            }
            self.corpus[lang] = {
                self._EVAL_SPLIT: {
                    str(row["commit"]): {"text": row["diff"]} for row in sub
                }
            }
            self.relevant_docs[lang] = {
                self._EVAL_SPLIT: {
                    str(i): {row["commit"]: 1} for i, row in enumerate(sub)
                }
            }

        self.data_loaded = True
