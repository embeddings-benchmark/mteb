from datasets import Dataset, DatasetDict, Image, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

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
    "no": ["nor-Latn", "nno-Latn", "nob-Latn"],
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
    path: str, langs: list, splits: list[str], revision: str | None = None
):
    corpus = {lang: dict.fromkeys(splits) for lang in langs}
    queries = {lang: dict.fromkeys(splits) for lang in langs}
    relevant_docs = {lang: dict.fromkeys(splits) for lang in langs}

    split = "test"

    for lang in langs:
        lang_data = load_dataset(
            path,
            split=lang,
            revision=revision,
        )
        lang_corpus = lang_data.map(
            lambda x: {
                "id": "corpus-" + x["image_id"],
                "modality": "image",
                "image": x["image"],
            },
            remove_columns=[
                "captions",
                "captions_tokenized",
                "captions_tokenized_lowercase",
                "image_locale",
                "image_id",
            ],
        )
        lang_corpus = lang_corpus.cast_column("image", Image())

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


def _load_xm3600_i2t_data(
    path: str, langs: list[str], splits: list[str], revision: str | None = None
) -> dict[str, dict[str, RetrievalSplitData]]:
    split = splits[0]
    dataset: dict[str, dict[str, RetrievalSplitData]] = {}

    for lang in langs:
        lang_data = load_dataset(path, split=lang, revision=revision)
        source_ids = list(lang_data["image_id"])
        query_ids = [f"query-{source_id}" for source_id in source_ids]

        # The pinned dataset stores images as raw {bytes, path} structs.
        queries = (
            lang_data.select_columns(["image"])
            .cast_column("image", Image())
            .add_column("id", query_ids)
        )

        corpus_ids = []
        corpus_texts = []
        relevant_docs = {}
        for source_id, captions in zip(source_ids, lang_data["captions"], strict=True):
            query_id = f"query-{source_id}"
            relevant_docs[query_id] = {}
            for index, caption in enumerate(captions):
                corpus_id = f"corpus-{source_id}-{index}"
                corpus_ids.append(corpus_id)
                corpus_texts.append(caption)
                relevant_docs[query_id][corpus_id] = 1

        corpus = Dataset.from_dict({"id": corpus_ids, "text": corpus_texts})
        dataset[lang] = {
            split: RetrievalSplitData(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                top_ranked=None,
            )
        }

    return dataset


class XM3600T2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="XM3600T2IRetrieval",
        description="Retrieve images based on multilingual descriptions.",
        reference="https://aclanthology.org/2022.emnlp-main.45/",
        dataset={
            "path": "mteb/xm3600",
            "revision": "536cd45bbfe53de9b08c0483bb4a76a4bd3673fa",
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
@inproceedings{thapliyal2022crossmodal,
  author = {Thapliyal, Ashish V and Tuset, Jordi Pont and Chen, Xi and Soricut, Radu},
  booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  pages = {715--729},
  title = {Crossmodal-3600: A Massively Multilingual Multimodal Evaluation Dataset},
  year = {2022},
}
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_xm3600_data(
            path=self.metadata.dataset["path"],
            langs=self.hf_subsets,
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class XM3600I2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="XM3600I2TRetrieval",
        description="Retrieve multilingual captions based on images.",
        reference="https://aclanthology.org/2022.emnlp-main.45/",
        dataset={
            "path": "mteb/xm3600",
            "revision": "536cd45bbfe53de9b08c0483bb4a76a4bd3673fa",
        },
        type="Any2AnyMultilingualRetrieval",
        category="i2t",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="found",
        adapted_from=["XM3600T2IRetrieval"],
        bibtex_citation=r"""
@inproceedings{thapliyal2022crossmodal,
  author = {Thapliyal, Ashish V and Tuset, Jordi Pont and Chen, Xi and Soricut, Radu},
  booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  pages = {715--729},
  title = {Crossmodal-3600: A Massively Multilingual Multimodal Evaluation Dataset},
  year = {2022},
}
""",
        prompt={"query": "Find a caption describing the following image."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        self.dataset = _load_xm3600_i2t_data(
            path=self.metadata.dataset["path"],
            langs=self.hf_subsets,
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True
