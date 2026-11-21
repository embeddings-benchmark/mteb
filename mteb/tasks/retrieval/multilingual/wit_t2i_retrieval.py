from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
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


def _load_wit_i2t_data(
    path: str, langs: list[str], splits: list[str], revision: str | None = None
) -> dict[str, dict[str, RetrievalSplitData]]:
    split = splits[0]
    dataset: dict[str, dict[str, RetrievalSplitData]] = {}

    for lang in langs:
        lang_data = load_dataset(path, split=lang, revision=revision)
        source_ids = list(lang_data["image_id"])
        valid_indices = []
        query_ids = []
        corpus_ids = []
        corpus_texts = []
        relevant_docs = {}

        for index, (source_id, raw_captions) in enumerate(
            zip(source_ids, lang_data["captions"], strict=True)
        ):
            captions = [caption for caption in raw_captions if caption.strip()]
            if not captions:
                continue

            valid_indices.append(index)
            query_id = f"query-{source_id}"
            query_ids.append(query_id)
            relevant_docs[query_id] = {}
            for caption_index, caption in enumerate(captions):
                corpus_id = f"corpus-{source_id}-{caption_index}"
                corpus_ids.append(corpus_id)
                corpus_texts.append(caption)
                relevant_docs[query_id][corpus_id] = 1

        queries = (
            lang_data.select(valid_indices)
            .select_columns(["image"])
            .add_column("id", query_ids)
        )
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


class WITT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WITT2IRetrieval",
        description="Retrieve images based on multilingual descriptions.",
        reference="https://proceedings.mlr.press/v162/bugliarello22a/bugliarello22a.pdf",
        dataset={
            "path": "mteb/WITT2IRetrieval",
            "revision": "2efdca4d8cbe9c815102f0f06eae6e7ed799c011",
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

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = _load_wit_i2t_data(
            path=self.metadata.dataset["path"],
            langs=self.hf_subsets,
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True
