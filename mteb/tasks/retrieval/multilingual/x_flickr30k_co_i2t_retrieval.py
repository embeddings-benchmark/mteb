from datasets import DatasetDict, Image, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.tasks.retrieval.multilingual.x_flickr30k_co_t2i_retrieval import (
    _LANGUAGES,
)


def _load_xflickrco_i2t_data(
    path: str, langs: list, splits: list[str], revision: str | None = None
):
    corpus = {lang: dict.fromkeys(splits) for lang in langs}
    queries = {lang: dict.fromkeys(splits) for lang in langs}
    relevant_docs = {lang: dict.fromkeys(splits) for lang in langs}
    split = "test"

    for lang in langs:
        lang_data = load_dataset(path, revision=revision)[lang]
        lang_queries = lang_data.map(
            lambda x: {
                "id": "query-" + x["id"],
                "modality": "image",
                "image": x["image"],
            },
            remove_columns=["sentences"],
        ).cast_column("image", Image())
        lang_corpus = lang_data.map(
            lambda x: {
                "id": "corpus-" + x["id"],
                "text": x["sentences"],
                "modality": "text",
            },
            remove_columns=["sentences", "image"],
        )

        lang_relevant_docs = {}
        for row in lang_data:
            query_id = "query-" + row["id"]
            corpus_id = "corpus-" + row["id"]
            lang_relevant_docs[query_id] = {corpus_id: 1}

        queries[lang][split] = lang_queries
        corpus[lang][split] = lang_corpus
        relevant_docs[lang][split] = lang_relevant_docs

    return (
        DatasetDict({lang: DatasetDict(data) for lang, data in corpus.items()}),
        DatasetDict({lang: DatasetDict(data) for lang, data in queries.items()}),
        DatasetDict(relevant_docs),
    )


class XFlickr30kCoI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="XFlickr30kCoI2TRetrieval",
        description="Retrieve multilingual captions based on images.",
        reference="https://proceedings.mlr.press/v162/bugliarello22a/bugliarello22a.pdf",
        dataset={
            "path": "mteb/xflickrco",
            "revision": "4da629e05455d757306174cdc72f2edfe00b9027",
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
        adapted_from=["XFlickr30kCoT2IRetrieval"],
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
        prompt={"query": "Find a caption describing the following image."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_xflickrco_i2t_data(
            path=self.metadata.dataset["path"],
            langs=self.hf_subsets,
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True
