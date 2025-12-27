from datasets import DatasetDict, Image, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

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
    path: str, langs: list, splits: list[str], revision: str | None = None
):
    corpus = {lang: dict.fromkeys(splits) for lang in langs}
    queries = {lang: dict.fromkeys(splits) for lang in langs}
    relevant_docs = {lang: dict.fromkeys(splits) for lang in langs}

    split = "test"

    for lang in langs:
        lang_data = load_dataset(
            path,
            revision=revision,
        )[lang]
        lang_corpus = lang_data.map(
            lambda x: {
                "id": "corpus-" + x["id"],
                "modality": "image",
                "image": x["image"],
            },
            remove_columns=["sentences"],
        )
        lang_corpus = lang_corpus.cast_column("image", Image())

        lang_queries = lang_data.map(
            lambda x: {
                "id": "query-" + x["id"],
                "text": x["sentences"],
                "modality": "text",
            },
            remove_columns=["sentences"],
        )
        # None values
        lang_queries = lang_queries.remove_columns(["image"])

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


class XFlickr30kCoT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="XFlickr30kCoT2IRetrieval",
        description="Retrieve images based on multilingual descriptions.",
        reference="https://proceedings.mlr.press/v162/bugliarello22a/bugliarello22a.pdf",
        dataset={
            "path": "floschne/xflickrco",
            "revision": "0af2c2eba58b27a71898787e286be04befdd7a20",
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

        self.corpus, self.queries, self.relevant_docs = _load_xflickrco_data(
            path=self.metadata.dataset["path"],
            langs=self.hf_subsets,
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True
