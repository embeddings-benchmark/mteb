from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_EVAL_LANGS = {
    "bg": ["bul-Cyrl"],
    "bn": ["ben-Beng"],
    "cs": ["ces-Latn"],
    "da": ["dan-Latn"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "fa": ["fas-Arab"],
    "fi": ["fin-Latn"],
    "hi": ["hin-Deva"],
    "it": ["ita-Latn"],
    "nl": ["nld-Latn"],
    "pt": ["por-Latn"],
    "ro": ["ron-Latn"],
    "sr": ["srp-Cyrl"],
    "no": ["nor-Latn"],
    "sv": ["swe-Latn"],
}


# adapted from MIRACLRetrieval
def _load_data(
    path: str,
    langs: list,
    split: str,
    cache_dir: str = None,
    revision_queries: str = None,
    revision_corpus: str = None,
    revision_qrels: str = None,
):
    queries = {lang: {split: {}} for lang in langs}
    corpus = {lang: {split: {}} for lang in langs}
    qrels = {lang: {split: {}} for lang in langs}

    for lang in langs:
        queries_path = path
        corpus_path = path.replace("queries", "corpus")
        qrels_path = path.replace("queries", "qrels")
        queries_lang = load_dataset(
            queries_path,
            lang,
            split=split,
            cache_dir=cache_dir,
            revision=revision_queries,
        )
        corpus_lang = load_dataset(
            corpus_path,
            lang,
            split=split,
            cache_dir=cache_dir,
            revision=revision_corpus,
        )
        qrels_lang = load_dataset(
            qrels_path,
            lang,
            split=split,
            cache_dir=cache_dir,
            revision=revision_qrels,
        )
        # don't pass on titles to make task harder
        corpus_lang_dict = {doc["_id"]: {"text": doc["text"]} for doc in corpus_lang}
        queries_lang_dict = {query["_id"]: query["text"] for query in queries_lang}
        # qrels_lang_dict = {qrel["query-id"]: {qrel["corpus-id"]: qrel["score"]} for qrel in qrels_lang}

        qrels_lang_dict = {}
        for qrel in qrels_lang:
            if qrel["score"] == 0.5:
                continue
            # score = 0 if qrel["score"] == 0.5 else qrel["score"]
            # score = int(score)
            score = int(qrel["score"])
            qrels_lang_dict[qrel["query-id"]] = {qrel["corpus-id"]: score}

        corpus[lang][split] = corpus_lang_dict
        queries[lang][split] = queries_lang_dict
        qrels[lang][split] = qrels_lang_dict

    return corpus, queries, qrels


class WikipediaRetrievalMultilingual(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WikipediaRetrievalMultilingual",
        description="The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.",
        reference="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-queries",
        dataset={
            "path": "ellamind/wikipedia-2023-11-retrieval-multilingual-queries",
            "revision": "3b6ea595c94bac3448a2ad167ca2e06abd340d6e",  # avoid validation error
            "revision_corpus": "f20ac0c449c85358d3d5c72a95f92f1eddc98aa5",
            "revision_qrels": "ec88a7bb2da034d538e98e3122d2c98530ca1c8d",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2023-11-01", "2024-05-15"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Question answering", "Article retrieval"],
        license="cc-by-sa-3.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {
                "en": 1500,
                "de": 1500,
                "it": 1500,
                "pt": 1500,
                "nl": 1500,
                "cs": 1500,
                "ro": 1500,
                "bg": 1500,
                "sr": 1500,
                "fi": 1500,
                "da": 1500,
                "fa": 1500,
                "hi": 1500,
                "bn": 1500,
                "no": 1500,
                "sv": 1500,
            },
            "avg_character_length": {
                "test": {
                    "bg": {
                        "average_document_length": 374.376,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "bn": {
                        "average_document_length": 394.05044444444445,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "cs": {
                        "average_document_length": 369.9831111111111,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "da": {
                        "average_document_length": 345.2597037037037,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "de": {
                        "average_document_length": 398.4137777777778,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "en": {
                        "average_document_length": 452.9871111111111,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "fa": {
                        "average_document_length": 345.1568888888889,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "fi": {
                        "average_document_length": 379.71237037037037,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "hi": {
                        "average_document_length": 410.72540740740743,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "it": {
                        "average_document_length": 393.73437037037036,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "nl": {
                        "average_document_length": 375.6695555555556,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "pt": {
                        "average_document_length": 398.27237037037037,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ro": {
                        "average_document_length": 348.3817037037037,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "sr": {
                        "average_document_length": 384.3131851851852,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "no": {
                        "average_document_length": 366.93733333333336,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "sv": {
                        "average_document_length": 369.340962962963,
                        "average_query_length": 1.0,
                        "num_documents": 13500,
                        "num_queries": 1500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.hf_subsets,
            split=self.metadata_dict["eval_splits"][0],
            cache_dir=kwargs.get("cache_dir", None),
            revision_queries=self.metadata_dict["dataset"]["revision"],
            revision_corpus=self.metadata_dict["dataset"]["revision_corpus"],
            revision_qrels=self.metadata_dict["dataset"]["revision_qrels"],
        )

        self.data_loaded = True
