from datasets import Dataset, DatasetDict, Image, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.tasks.retrieval.multilingual.wit_t2i_retrieval import _LANGUAGES


def _load_wit_i2t_data(
    path: str, langs: list, splits: list[str], revision: str | None = None
):
    corpus = {lang: dict.fromkeys(splits) for lang in langs}
    queries = {lang: dict.fromkeys(splits) for lang in langs}
    relevant_docs = {lang: dict.fromkeys(splits) for lang in langs}
    split = "test"

    for lang in langs:
        lang_data = load_dataset(path, split=lang, revision=revision)
        query_rows = []
        corpus_rows = []
        lang_relevant_docs = {}

        for row in lang_data:
            captions = [caption for caption in row["captions"] if caption.strip()]
            if not captions:
                continue

            query_id = f"query-{row['image_id']}"
            query_rows.append(
                {
                    "id": query_id,
                    "modality": "image",
                    "image": row["image"],
                }
            )
            lang_relevant_docs[query_id] = {}
            for idx, caption in enumerate(captions):
                corpus_id = f"corpus-{row['image_id']}-{idx}"
                corpus_rows.append(
                    {
                        "id": corpus_id,
                        "text": caption,
                        "modality": "text",
                    }
                )
                lang_relevant_docs[query_id][corpus_id] = 1

        queries[lang][split] = Dataset.from_list(query_rows).cast_column(
            "image", Image()
        )
        corpus[lang][split] = Dataset.from_list(corpus_rows)
        relevant_docs[lang][split] = lang_relevant_docs

    return (
        DatasetDict({lang: DatasetDict(data) for lang, data in corpus.items()}),
        DatasetDict({lang: DatasetDict(data) for lang, data in queries.items()}),
        DatasetDict(relevant_docs),
    )


class WITI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WITI2TRetrieval",
        description="Retrieve multilingual captions based on images.",
        reference="https://proceedings.mlr.press/v162/bugliarello22a/bugliarello22a.pdf",
        dataset={
            "path": "mteb/wit",
            "revision": "91ac153f1371a98b209ed763205e25e115ecd06e",
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
        adapted_from=["WITT2IRetrieval"],
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

        self.corpus, self.queries, self.relevant_docs = _load_wit_i2t_data(
            path=self.metadata.dataset["path"],
            langs=self.hf_subsets,
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True
