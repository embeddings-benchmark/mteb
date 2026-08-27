from datasets import Dataset, DatasetDict, Image, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.tasks.retrieval.multilingual.xm3600_t2i_retrieval import _LANGUAGES


def _load_xm3600_i2t_data(
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
            query_id = f"query-{row['image_id']}"
            query_rows.append(
                {
                    "id": query_id,
                    "modality": "image",
                    "image": row["image"],
                }
            )
            lang_relevant_docs[query_id] = {}
            for idx, caption in enumerate(row["captions"]):
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

        self.corpus, self.queries, self.relevant_docs = _load_xm3600_i2t_data(
            path=self.metadata.dataset["path"],
            langs=self.hf_subsets,
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True
