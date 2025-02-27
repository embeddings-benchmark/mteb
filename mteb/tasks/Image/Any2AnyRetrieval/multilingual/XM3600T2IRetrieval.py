from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "ar": ["ara-Arab"],
    "bn": ["ben-Beng"],
    "cs": ["ces-Latn"],
    "da": ["dan-Latn"],
    "de": ["deu-Latn"],
    "el": ["ell-Grek"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fa": ["fas-Arab"],
    "fi": ["fin-Latn"],
    "fil": ["fil-Latn"],
    "fr": ["fra-Latn"],
    "he": ["heb-Hebr"],
    "hi": ["hin-Deva"],
    "hr": ["hrv-Latn"],
    "hu": ["hun-Latn"],
    "id": ["ind-Latn"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Jpan"],
    "ko": ["kor-Hang"],
    "mi": ["mri-Latn"],
    "nl": ["nld-Latn"],
    "no": ["nor-Latn"],
    "pl": ["pol-Latn"],
    "pt": ["por-Latn"],
    "quz": ["quz-Latn"],
    "ro": ["ron-Latn"],
    "ru": ["rus-Cyrl"],
    "sv": ["swe-Latn"],
    "sw": ["swa-Latn"],
    "te": ["tel-Telu"],
    "th": ["tha-Thai"],
    "tr": ["tur-Latn"],
    "uk": ["ukr-Cyrl"],
    "vi": ["vie-Latn"],
    "zh": ["zho-Hans"],
}


def _load_xm3600_data(
    path: str, langs: list, splits: str, cache_dir: str = None, revision: str = None
):
    corpus = {lang: {split: None for split in splits} for lang in langs}
    queries = {lang: {split: None for split in splits} for lang in langs}
    relevant_docs = {lang: {split: None for split in splits} for lang in langs}

    split = "test"

    for lang in langs:
        lang_data = load_dataset(
            path,
            split=lang,
            cache_dir=cache_dir,
            revision=revision,
            # trust_remote_code=True,
        )
        lang_corpus = lang_data.map(
            lambda x: {
                "id": "corpus-" + x["image_id"],
                "text": None,
                "modality": "image",
                "image": x["image"]["bytes"],
            },
            remove_columns=[
                "captions",
                "captions_tokenized",
                "captions_tokenized_lowercase",
                "image_locale",
                "image_id",
            ],
        )

        corpus[lang][split] = lang_corpus

        lang_data = lang_data.remove_columns(["image"])

        queries[lang][split] = []
        relevant_docs[lang][split] = {}

        for row in lang_data:
            image_id = "corpus-" + row["image_id"]
            for idx, caption in enumerate(row["captions"]):
                query_id = f"query-{row['image_id']}-{idx}"
                queries[lang][split].append(
                    {
                        "id": query_id,
                        "text": caption,
                        "modality": "text",
                        "image": None,
                    }
                )
                if query_id not in relevant_docs[lang][split]:
                    relevant_docs[lang][split][query_id] = {}
                relevant_docs[lang][split][query_id][image_id] = 1

        queries[lang][split] = Dataset.from_dict(
            {
                "id": [query["id"] for query in queries[lang][split]],
                "text": [query["text"] for query in queries[lang][split]],
                "modality": [query["modality"] for query in queries[lang][split]],
                "image": [None for _ in queries[lang][split]],
            }
        )
    corpus = DatasetDict({lang: DatasetDict(splits) for lang, splits in corpus.items()})
    queries = DatasetDict(
        {lang: DatasetDict(splits) for lang, splits in queries.items()}
    )
    relevant_docs = DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class XM3600T2IRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="XM3600T2IRetrieval",
        description="Retrieve images based on multilingual descriptions.",
        reference="https://aclanthology.org/2022.emnlp-main.45/",
        dataset={
            "path": "floschne/xm3600",
            "revision": "8d3e5665526c55a5855cd6ddfbaba2032bc7cee4",
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
        bibtex_citation="""@inproceedings{thapliyal2022crossmodal,
  title={Crossmodal-3600: A Massively Multilingual Multimodal Evaluation Dataset},
  author={Thapliyal, Ashish V and Tuset, Jordi Pont and Chen, Xi and Soricut, Radu},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  pages={715--729},
  year={2022}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "ar": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "bn": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "cs": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "da": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "de": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "el": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "en": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "es": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "fa": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "fi": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "fil": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "fr": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "he": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "hi": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "hr": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "hu": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "id": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "it": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "ja": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "ko": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "mi": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "nl": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "pl": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "pt": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "quz": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "ro": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "ru": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "sv": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "sw": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "te": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "th": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "tr": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "uk": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "vi": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "zh": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_xm3600_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.hf_subsets,
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
