from __future__ import annotations

from datasets import DatasetDict, load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "id": ["ind-Latn"],
    "ja": ["jpn-Jpan"],
    "ru": ["rus-Cyrl"],
    "tr": ["tur-Latn"],
    "zh": ["zho-Hans"],
}


def _load_xflickrco_data(
    path: str, langs: list, splits: str, cache_dir: str = None, revision: str = None
):
    corpus = {lang: {split: None for split in splits} for lang in langs}
    queries = {lang: {split: None for split in splits} for lang in langs}
    relevant_docs = {lang: {split: None for split in splits} for lang in langs}

    split = "test"

    for lang in langs:
        lang_data = load_dataset(
            path,
            cache_dir=cache_dir,
            revision=revision,
            # trust_remote_code=True,
        )[lang]
        lang_corpus = lang_data.map(
            lambda x: {
                "id": "corpus-" + x["id"],
                "text": None,
                "modality": "image",
                "image": x["image"]["bytes"],
            },
            remove_columns=["sentences"],
        )

        lang_queries = lang_data.map(
            lambda x: {
                "id": "query-" + x["id"],
                "text": x["sentences"],
                "modality": "text",
                "image": None,
            },
            remove_columns=["sentences"],
        )

        relevant_docs[lang][split] = {}
        for row in lang_data:
            query_id = "query-" + row["id"]
            corpus_id = "corpus-" + row["id"]
            score = 1
            if query_id not in relevant_docs[lang][split]:
                relevant_docs[lang][split][query_id] = {}
            relevant_docs[lang][split][query_id][corpus_id] = score

        corpus[lang][split] = lang_corpus
        queries[lang][split] = lang_queries

    corpus = DatasetDict({lang: DatasetDict(splits) for lang, splits in corpus.items()})
    queries = DatasetDict(
        {lang: DatasetDict(splits) for lang, splits in queries.items()}
    )
    relevant_docs = DatasetDict(relevant_docs)
    return corpus, queries, relevant_docs


class XFlickr30kCoT2IRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="XFlickr30kCoT2IRetrieval",
        description="Retrieve images based on multilingual descriptions.",
        reference="https://proceedings.mlr.press/v162/bugliarello22a/bugliarello22a.pdf",
        dataset={
            "path": "floschne/xflickrco",
            "revision": "0af2c2eba58b27a71898787e286be04befdd7a20",
            # "trust_remote_code": True,
        },
        type="Any2AnyMultilingualRetrieval",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{bugliarello2022iglue,
  author = {Bugliarello, Emanuele and Liu, Fangyu and Pfeiffer, Jonas and Reddy, Siva and Elliott, Desmond and Ponti, Edoardo Maria and Vuli{\'c}, Ivan},
  booktitle = {International Conference on Machine Learning},
  organization = {PMLR},
  pages = {2370--2392},
  title = {IGLUE: A benchmark for transfer learning across modalities, tasks, and languages},
  year = {2022},
}
""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "de": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 2000,
                        "num_queries": 2000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "en": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 2000,
                        "num_queries": 2000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "es": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 2000,
                        "num_queries": 2000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "id": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 2000,
                        "num_queries": 2000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ja": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 2000,
                        "num_queries": 2000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ru": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 2000,
                        "num_queries": 2000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "tr": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 2000,
                        "num_queries": 2000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "zh": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 2000,
                        "num_queries": 2000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_xflickrco_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.hf_subsets,
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
