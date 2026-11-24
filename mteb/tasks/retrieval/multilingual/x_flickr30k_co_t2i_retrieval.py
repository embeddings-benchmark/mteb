from __future__ import annotations

from typing import Any

from datasets import Image, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
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


def _load_xflickrco_i2t_data(
    path: str, langs: list[str], splits: list[str], revision: str | None = None
) -> dict[str, dict[str, RetrievalSplitData]]:
    """Load XFlickr30k-Co without rebuilding rows with ``Dataset.map``."""
    split = splits[0]
    dataset: dict[str, dict[str, RetrievalSplitData]] = {}

    for lang in langs:
        lang_data = load_dataset(path, split=lang, revision=revision)
        source_ids = list(lang_data["id"])
        query_ids = [f"query-{source_id}" for source_id in source_ids]
        corpus_ids = [f"corpus-{source_id}" for source_id in source_ids]

        # The pinned dataset stores images as raw {bytes, path} structs, so mark the
        # selected column as Image to make evaluators receive decoded PIL images.
        queries = (
            lang_data.select_columns(["image"])
            .cast_column("image", Image())
            .add_column("id", query_ids)
        )
        corpus = (
            lang_data.select_columns(["sentences"])
            .rename_column("sentences", "text")
            .add_column("id", corpus_ids)
        )
        relevant_docs = {
            query_id: {corpus_id: 1}
            for query_id, corpus_id in zip(query_ids, corpus_ids, strict=True)
        }
        dataset[lang] = {
            split: RetrievalSplitData(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                top_ranked=None,
            )
        }

    return dataset


class XFlickr30kCoT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="XFlickr30kCoT2IRetrieval",
        description="Retrieve images based on multilingual descriptions.",
        reference="https://proceedings.mlr.press/v162/bugliarello22a/bugliarello22a.pdf",
        dataset={
            "path": "mteb/XFlickr30kCoT2IRetrieval",
            "revision": "0b0e192defceb4db8df0e7094ee8ad807a73d992",
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

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = _load_xflickrco_i2t_data(
            path=self.metadata.dataset["path"],
            langs=self.hf_subsets,
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True
