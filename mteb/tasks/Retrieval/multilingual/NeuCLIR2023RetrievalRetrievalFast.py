from __future__ import annotations

from collections import defaultdict

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import *

_LANGUAGES = {
    "fas": ["fas-Arab"],
    "rus": ["rus-Cyrl"],
    "zho": ["zho-Hans"],
}


def load_neuclir_data(
    path: str,
    langs: list,
    eval_splits: list,
    cache_dir: str | None = None,
    revision: str | None = None,
):
    corpus = {lang: {split: None for split in eval_splits} for lang in langs}
    queries = {lang: {split: None for split in eval_splits} for lang in langs}
    relevant_docs = {lang: {split: None for split in eval_splits} for lang in langs}

    for lang in langs:
        lang_corpus = datasets.load_dataset(
            path, f"corpus-{lang}", cache_dir=cache_dir, revision=revision
        )["corpus"]
        lang_queries = datasets.load_dataset(
            path, f"queries-{lang}", cache_dir=cache_dir, revision=revision
        )["queries"]
        lang_qrels = datasets.load_dataset(
            path, f"{lang}", cache_dir=cache_dir, revision=revision
        )["test"]
        corpus[lang] = {
            "test": {
                str(e["_id"]): {"text": e["text"], "title": e["title"]}
                for e in lang_corpus
            }
        }
        queries[lang] = {"test": {str(e["_id"]): e["text"] for e in lang_queries}}
        relevant_docs[lang]["test"] = defaultdict(dict)
        for item in lang_qrels:
            relevant_docs[lang]["test"][str(item["query-id"])].update(
                {str(item["corpus-id"]): item["score"]}
            )
    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)
    return corpus, queries, relevant_docs


class NeuCLIR2023RetrievalFast(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIR2023Retrieval-Fast",
        description="The task involves identifying and retrieving the documents that are relevant to the queries.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/neuclir-2023-fast",
            "revision": "5ae366d350bb08d660053aca30f9e64a9c0856ca",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_20",
        date=("2022-08-01", "2023-06-30"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="odc-by",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{lawrie2024overview,
      title={Overview of the TREC 2023 NeuCLIR Track}, 
      author={Dawn Lawrie and Sean MacAvaney and James Mayfield and Paul McNamee and Douglas W. Oard and Luca Soldaini and Eugene Yang},
      year={2024},
      eprint={2404.08071},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 2181.3189590322254,
                    "average_query_length": 54.04385964912281,
                    "num_documents": 49185,
                    "num_queries": 228,
                    "average_relevant_docs_per_query": 30.030701754385966,
                    "hf_subset_descriptive_stats": {
                        "fas": {
                            "average_document_length": 2833.6343748035456,
                            "average_query_length": 65.48684210526316,
                            "num_documents": 15907,
                            "num_queries": 76,
                            "average_relevant_docs_per_query": 34.421052631578945,
                        },
                        "rus": {
                            "average_document_length": 2619.8847800459143,
                            "average_query_length": 74.4342105263158,
                            "num_documents": 16117,
                            "num_queries": 76,
                            "average_relevant_docs_per_query": 31.05263157894737,
                        },
                        "zho": {
                            "average_document_length": 1164.7845696637726,
                            "average_query_length": 22.210526315789473,
                            "num_documents": 17161,
                            "num_queries": 76,
                            "average_relevant_docs_per_query": 24.61842105263158,
                        },
                    },
                },
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_neuclir_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.metadata.eval_langs,
            eval_splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )
        self.data_loaded = True
