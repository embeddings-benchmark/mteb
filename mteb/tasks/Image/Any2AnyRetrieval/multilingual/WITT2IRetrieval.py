from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from .....abstasks import AbsTaskAny2AnyRetrieval, MultilingualTask

_LANGUAGES = {
    "ar": ["ara-Arab"],
    "bg": ["bul-Cyrl"],
    "da": ["dan-Latn"],
    "el": ["ell-Grek"],
    "et": ["est-Latn"],
    "id": ["ind-Latn"],
    "ko": ["kor-Hang"],
    "ja": ["jpn-Jpan"],
    "tr": ["tur-Latn"],
    "vi": ["vie-Latn"],
    "en": ["eng-Latn"],
}


def _load_wit_data(
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
                "image": x["image"],
            },
            remove_columns=[
                "captions",
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


class WITT2IRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="WITT2IRetrieval",
        description="Retrieve images based on multilingual descriptions.",
        reference="https://aclanthology.org/2022.emnlp-main.45/",
        dataset={
            "path": "mteb/wit",
            "revision": "91ac153f1371a98b209ed763205e25e115ecd06e",
            # "trust_remote_code": True,
        },
        type="Retrieval",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="CC BY-SA 4.0",
        socioeconomic_status="medium",
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
                "dev": {
                    "ar": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "bg": {
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
                    "el": {
                        "average_document_length": 0.0,
                        "average_query_length": 0.0,
                        "num_documents": 7200,
                        "num_queries": 3600,
                        "average_relevant_docs_per_query": 2.0,
                    },
                    "et": {
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
                    "tr": {
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
                    "en": {
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

        self.corpus, self.queries, self.relevant_docs = _load_wit_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.hf_subsets,
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
