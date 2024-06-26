from __future__ import annotations

from collections import defaultdict

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskRetrieval, MultilingualTask
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


class NeuCLIR2023Retrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIR2023Retrieval",
        description="The task involves identifying and retrieving the documents that are relevant to the queries.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/neuclir-2023",
            "revision": "dfad7cc7fe4064d6568d6b7d43b99e3a0246d29b",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_20",
        date=("2022-08-01", "2023-06-30"),
        form=["written"],
        domains=["News"],
        task_subtypes=[],
        license="odc-by",
        socioeconomic_status="medium",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@misc{lawrie2024overview,
      title={Overview of the TREC 2023 NeuCLIR Track}, 
      author={Dawn Lawrie and Sean MacAvaney and James Mayfield and Paul McNamee and Douglas W. Oard and Luca Soldaini and Eugene Yang},
      year={2024},
      eprint={2404.08071},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        n_samples={"fas": 2232092, "zho": 3179285, "rus": 4627619},
        avg_character_length={
            "test": {
                "fas": {
                    "average_document_length": 2032.093148525817,
                    "average_query_length": 65.48684210526316,
                    "num_documents": 2232016,
                    "num_queries": 76,
                    "average_relevant_docs_per_query": 66.28947368421052,
                },
                "rus": {
                    "average_document_length": 1757.9129983233004,
                    "average_query_length": 74.4342105263158,
                    "num_documents": 4627543,
                    "num_queries": 76,
                    "average_relevant_docs_per_query": 62.223684210526315,
                },
                "zho": {
                    "average_document_length": 743.1426659901881,
                    "average_query_length": 22.210526315789473,
                    "num_documents": 3179209,
                    "num_queries": 76,
                    "average_relevant_docs_per_query": 53.68421052631579,
                },
            }
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
