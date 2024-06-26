from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskRetrieval, MultilingualTask
from ....abstasks.AbsTaskRetrieval import *

_LANGUAGES = {
    "ar": ["ara-Arab"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Jpan"],
    "ko": ["kor-Hang"],
    "pt": ["por-Latn"],
    "ru": ["rus-Cyrl"],
    "th": ["tha-Thai"],
    "zh": ["cmn-Hans"],
}


def load_mldr_data(
    path: str,
    langs: list,
    eval_splits: list,
    cache_dir: str = None,
    revision: str = None,
):
    corpus = {lang: {split: None for split in eval_splits} for lang in langs}
    queries = {lang: {split: None for split in eval_splits} for lang in langs}
    relevant_docs = {lang: {split: None for split in eval_splits} for lang in langs}

    for lang in langs:
        lang_corpus = datasets.load_dataset(
            path, f"corpus-{lang}", cache_dir=cache_dir, revision=revision
        )["corpus"]
        lang_corpus = {e["docid"]: {"text": e["text"]} for e in lang_corpus}
        lang_data = datasets.load_dataset(path, lang, cache_dir=cache_dir)
        for split in eval_splits:
            corpus[lang][split] = lang_corpus
            queries[lang][split] = {e["query_id"]: e["query"] for e in lang_data[split]}
            relevant_docs[lang][split] = {
                e["query_id"]: {e["positive_passages"][0]["docid"]: 1}
                for e in lang_data[split]
            }

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)
    return corpus, queries, relevant_docs


class MultiLongDocRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultiLongDocRetrieval",
        description="MultiLongDocRetrieval",
        reference="https://arxiv.org/abs/2402.03216",
        dataset={
            "path": "Shitao/MLDR",
            "revision": "d67138e705d963e346253a80e59676ddb418810a",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["dev", "test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@misc{bge-m3,
      title={BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
      author={Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
      year={2024},
      eprint={2402.03216},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
""",
        n_samples=None,
        avg_character_length={
            "dev": {
                "ar": {
                    "average_document_length": 29234.48153016958,
                    "average_query_length": 69.27,
                    "num_documents": 7607,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "de": {
                    "average_document_length": 33771.2111,
                    "average_query_length": 153.63,
                    "num_documents": 10000,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "en": {
                    "average_document_length": 13332.76764,
                    "average_query_length": 81.22,
                    "num_documents": 200000,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "es": {
                    "average_document_length": 36567.1736990891,
                    "average_query_length": 123.11,
                    "num_documents": 9551,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "fr": {
                    "average_document_length": 36009.4934,
                    "average_query_length": 142.165,
                    "num_documents": 10000,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "hi": {
                    "average_document_length": 18688.50788229112,
                    "average_query_length": 77.995,
                    "num_documents": 3806,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "it": {
                    "average_document_length": 36633.9969,
                    "average_query_length": 99.615,
                    "num_documents": 10000,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "ja": {
                    "average_document_length": 14480.7508,
                    "average_query_length": 61.625,
                    "num_documents": 10000,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "ko": {
                    "average_document_length": 13813.441224093263,
                    "average_query_length": 58.845,
                    "num_documents": 6176,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "pt": {
                    "average_document_length": 32127.576952351956,
                    "average_query_length": 122.275,
                    "num_documents": 6569,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "ru": {
                    "average_document_length": 35934.8756,
                    "average_query_length": 87.875,
                    "num_documents": 10000,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "th": {
                    "average_document_length": 25993.2696,
                    "average_query_length": 107.81,
                    "num_documents": 10000,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "zh": {
                    "average_document_length": 6039.059725,
                    "average_query_length": 26.79,
                    "num_documents": 200000,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
            },
            "test": {
                "ar": {
                    "average_document_length": 29234.48153016958,
                    "average_query_length": 75.77,
                    "num_documents": 7607,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "de": {
                    "average_document_length": 33771.2111,
                    "average_query_length": 123.65,
                    "num_documents": 10000,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "en": {
                    "average_document_length": 13332.76764,
                    "average_query_length": 81.33,
                    "num_documents": 200000,
                    "num_queries": 800,
                    "average_relevant_docs_per_query": 1.0,
                },
                "es": {
                    "average_document_length": 36567.1736990891,
                    "average_query_length": 131.985,
                    "num_documents": 9551,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "fr": {
                    "average_document_length": 36009.4934,
                    "average_query_length": 149.795,
                    "num_documents": 10000,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "hi": {
                    "average_document_length": 18688.50788229112,
                    "average_query_length": 103.76,
                    "num_documents": 3806,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "it": {
                    "average_document_length": 36633.9969,
                    "average_query_length": 114.595,
                    "num_documents": 10000,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "ja": {
                    "average_document_length": 14480.7508,
                    "average_query_length": 55.73,
                    "num_documents": 10000,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "ko": {
                    "average_document_length": 13813.441224093263,
                    "average_query_length": 58.72,
                    "num_documents": 6176,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "pt": {
                    "average_document_length": 32127.576952351956,
                    "average_query_length": 113.455,
                    "num_documents": 6569,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "ru": {
                    "average_document_length": 35934.8756,
                    "average_query_length": 94.87,
                    "num_documents": 10000,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "th": {
                    "average_document_length": 25993.2696,
                    "average_query_length": 97.99,
                    "num_documents": 10000,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                },
                "zh": {
                    "average_document_length": 6039.059725,
                    "average_query_length": 24.70875,
                    "num_documents": 200000,
                    "num_queries": 800,
                    "average_relevant_docs_per_query": 1.0,
                },
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_mldr_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.metadata.eval_langs,
            eval_splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )
        self.data_loaded = True
