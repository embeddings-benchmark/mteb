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
            "n_samples": {"fas": 2232092, "zho": 3179285, "rus": 4627619},
            "avg_character_length": {
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


def load_neuclir_data_hard_negatives(
    path: str,
    langs: list,
    eval_splits: list,
    cache_dir: str | None = None,
    revision: str | None = None,
):
    split = "test"
    corpus = {lang: {split: None for split in eval_splits} for lang in langs}
    queries = {lang: {split: None for split in eval_splits} for lang in langs}
    relevant_docs = {lang: {split: None for split in eval_splits} for lang in langs}

    for lang in langs:
        corpus_identifier = f"corpus-{lang}"
        corpus_data = datasets.load_dataset(
            path,
            corpus_identifier,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )
        corpus[lang][split] = {}
        for row in corpus_data["corpus"]:
            docid = row["_id"]
            doc_title = row["title"]
            doc_text = row["text"]
            corpus[lang][split][docid] = {"title": doc_title, "text": doc_text}

        # Load queries data
        queries_identifier = f"queries-{lang}"
        queries_data = datasets.load_dataset(
            path,
            queries_identifier,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )
        queries[lang][split] = {}
        for row in queries_data["queries"]:
            query_id = row["_id"]
            query_text = row["text"]
            queries[lang][split][query_id] = query_text

        # Load relevant documents data
        qrels_identifier = f"{lang}"
        qrels_data = datasets.load_dataset(
            path,
            qrels_identifier,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )
        relevant_docs[lang][split] = {}
        for row in qrels_data[split]:
            query_id = row["query-id"]
            doc_id = row["corpus-id"]
            score = row["score"]
            if query_id not in relevant_docs[lang][split]:
                relevant_docs[lang][split][query_id] = {}
            relevant_docs[lang][split][query_id][doc_id] = score

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class NeuCLIR2023RetrievalHardNegatives(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIR2023RetrievalHardNegatives",
        description="The task involves identifying and retrieving the documents that are relevant to the queries. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/neuclir-2023-hard-negatives",
            "revision": "5d47e924e632c333d3f087d945642af93b008d2b",
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
                    "average_document_length": 2236.175955333482,
                    "average_query_length": 54.10267857142857,
                    "num_documents": 49433,
                    "num_queries": 224,
                    "average_relevant_docs_per_query": 61.816964285714285,
                    "hf_subset_descriptive_stats": {
                        "fas": {
                            "average_document_length": 2895.869857421016,
                            "average_query_length": 65.89189189189189,
                            "num_documents": 15921,
                            "num_queries": 74,
                            "average_relevant_docs_per_query": 68.08108108108108,
                        },
                        "rus": {
                            "average_document_length": 2724.294762109928,
                            "average_query_length": 74.41333333333333,
                            "num_documents": 16247,
                            "num_queries": 75,
                            "average_relevant_docs_per_query": 63.053333333333335,
                        },
                        "zho": {
                            "average_document_length": 1168.4984071821605,
                            "average_query_length": 22.16,
                            "num_documents": 17265,
                            "num_queries": 75,
                            "average_relevant_docs_per_query": 54.4,
                        },
                    },
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = (
            load_neuclir_data_hard_negatives(
                path=self.metadata_dict["dataset"]["path"],
                langs=self.metadata.eval_langs,
                eval_splits=self.metadata_dict["eval_splits"],
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata_dict["dataset"]["revision"],
            )
        )
        self.data_loaded = True
