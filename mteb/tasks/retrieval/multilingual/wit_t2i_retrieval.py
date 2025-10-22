from datasets import Dataset, DatasetDict, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

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


def _load_wit_data(path: str, langs: list, splits: str, revision: str | None = None):
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


class WITT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WITT2IRetrieval",
        description="Retrieve images based on multilingual descriptions.",
        reference="https://proceedings.mlr.press/v162/bugliarello22a/bugliarello22a.pdf",
        dataset={
            "path": "mteb/wit",
            "revision": "91ac153f1371a98b209ed763205e25e115ecd06e",
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
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_wit_data(
            path=self.metadata.dataset["path"],
            langs=self.hf_subsets,
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True
