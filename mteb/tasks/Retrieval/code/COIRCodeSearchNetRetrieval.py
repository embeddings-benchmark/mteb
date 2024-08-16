from __future__ import annotations

import logging

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = ["python", "javascript", "go", "ruby", "java", "php"]
_EVAL_SPLIT = "test"

logger = logging.getLogger(__name__)


def _load_code_search_code_retrieval(
    path: str, langs: list, splits: str, cache_dir: str = None, revision: str = None
):
    corpus = {lang: {split: {} for split in splits} for lang in langs}
    queries = {lang: {split: {} for split in splits} for lang in langs}
    relevant_docs = {lang: {split: {} for split in splits} for lang in langs}

    split = _EVAL_SPLIT

    for lang in langs:
        qrels_data = datasets.load_dataset(
            path,
            name=f"{lang}-qrels",
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )[split]

        for row in qrels_data:
            query_id = row["query-id"]
            doc_id = row["corpus-id"]
            score = row["score"]
            if query_id not in relevant_docs[lang][split]:
                relevant_docs[lang][split][query_id] = {}
            relevant_docs[lang][split][query_id][doc_id] = score

        corpus_data = datasets.load_dataset(
            path,
            name=f"{lang}-corpus",
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )["corpus"]

        for row in corpus_data:
            doc_id = row["_id"]
            doc_title = row["title"]
            doc_text = row["text"]
            corpus[lang][split][doc_id] = {"title": doc_title, "text": doc_text}

        queries_data = datasets.load_dataset(
            path,
            name=f"{lang}-queries",
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )["queries"].filter(lambda x: x["partition"] == "test")

        for row in queries_data:
            query_id = row["_id"]
            query_text = row["text"]
            queries[lang][split][query_id] = query_text

        queries = queries
        logger.info("Loaded %d %s Queries.", len(queries), split.upper())

    return corpus, queries, relevant_docs


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
                        "average_document_length": 466.546,
                        "average_query_length": 862.842,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "javascript": {
                        "average_document_length": 186.018,
                        "average_query_length": 1415.632,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "go": {
                        "average_document_length": 125.213,
                        "average_query_length": 563.729,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ruby": {
                        "average_document_length": 313.818,
                        "average_query_length": 577.634,
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
                        "average_document_length": 162.119,
                        "average_query_length": 712.129,
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
