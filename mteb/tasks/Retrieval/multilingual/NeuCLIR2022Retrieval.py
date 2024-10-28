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


class NeuCLIR2022Retrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIR2022Retrieval",
        description="The task involves identifying and retrieving the documents that are relevant to the queries.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/neuclir-2022",
            "revision": "920fc15b81e2324e52163904be663f340235cdea",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_20",
        date=("2021-08-01", "2022-06-30"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="odc-by",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{lawrie2023overview,
  title={Overview of the TREC 2022 NeuCLIR track},
  author={Lawrie, Dawn and MacAvaney, Sean and Mayfield, James and McNamee, Paul and Oard, Douglas W and Soldaini, Luca and Yang, Eugene},
  journal={arXiv preprint arXiv:2304.12367},
  year={2023}
}""",
        descriptive_stats={
            "n_samples": {"fas": 2232130, "zho": 3179323, "rus": 4627657},
            "avg_character_length": {
                "test": {
                    "fas": {
                        "average_document_length": 2032.093148525817,
                        "average_query_length": 85.4298245614035,
                        "num_documents": 2232016,
                        "num_queries": 114,
                        "average_relevant_docs_per_query": 12.912280701754385,
                    },
                    "rus": {
                        "average_document_length": 1757.9129983233004,
                        "average_query_length": 85.58771929824562,
                        "num_documents": 4627543,
                        "num_queries": 114,
                        "average_relevant_docs_per_query": 16.57017543859649,
                    },
                    "zho": {
                        "average_document_length": 743.1426659901881,
                        "average_query_length": 24.17543859649123,
                        "num_documents": 3179209,
                        "num_queries": 114,
                        "average_relevant_docs_per_query": 18.710526315789473,
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


class NeuCLIR2022RetrievalHardNegatives(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIR2022RetrievalHardNegatives",
        description="The task involves identifying and retrieving the documents that are relevant to the queries. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/neuclir-2022-hard-negatives",
            "revision": "35dd709a0d846ae987541cf8ca978562636260f0",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_20",
        date=("2021-08-01", "2022-06-30"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="odc-by",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{lawrie2023overview,
  title={Overview of the TREC 2022 NeuCLIR track},
  author={Lawrie, Dawn and MacAvaney, Sean and Mayfield, James and McNamee, Paul and Oard, Douglas W and Soldaini, Luca and Yang, Eugene},
  journal={arXiv preprint arXiv:2304.12367},
  year={2023}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 2066.9453653646488,
                    "average_query_length": 63.529411764705884,
                    "num_documents": 27931,
                    "num_queries": 136,
                    "average_relevant_docs_per_query": 40.39705882352941,
                    "hf_subset_descriptive_stats": {
                        "fas": {
                            "average_document_length": 2816.847782031074,
                            "average_query_length": 83.26666666666667,
                            "num_documents": 8882,
                            "num_queries": 45,
                            "average_relevant_docs_per_query": 32.71111111111111,
                        },
                        "rus": {
                            "average_document_length": 2446.5574277854193,
                            "average_query_length": 85.56818181818181,
                            "num_documents": 8724,
                            "num_queries": 44,
                            "average_relevant_docs_per_query": 42.93181818181818,
                        },
                        "zho": {
                            "average_document_length": 1101.0984987893462,
                            "average_query_length": 24.0,
                            "num_documents": 10325,
                            "num_queries": 47,
                            "average_relevant_docs_per_query": 45.38297872340426,
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
