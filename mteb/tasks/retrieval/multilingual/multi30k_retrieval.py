from datasets import Dataset, DatasetDict, Image, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "en": ["eng-Latn"],
    "cs": ["ces-Latn"],
    "de": ["deu-Latn"],
    "fr": ["fra-Latn"],
}

_DATASET_PATH = "romrawinjp/multi30k"
_DATASET_REVISION = "110e827dac7d6aabe6201d13bbdbc7413630390d"

_BIBTEX = r"""
@inproceedings{barrault2018findings,
  address = {Belgium, Brussels},
  author = {Barrault, Lo{\"i}c and Bougares, Fethi and Specia, Lucia and Lala, Chiraag and Elliott, Desmond and Frank, Stella},
  booktitle = {Proceedings of the Third Conference on Machine Translation: Shared Task Papers},
  pages = {304--323},
  publisher = {Association for Computational Linguistics},
  title = {Findings of the Third Shared Task on Multimodal Machine Translation},
  year = {2018},
}

@inproceedings{elliott2016multi30k,
  address = {Berlin, Germany},
  author = {Elliott, Desmond and Frank, Stella and Sima'an, Khalil and Specia, Lucia},
  booktitle = {Proceedings of the 5th Workshop on Vision and Language},
  pages = {70--74},
  publisher = {Association for Computational Linguistics},
  title = {Multi30K: Multilingual English-German Image Descriptions},
  year = {2016},
}
"""


def _load_multi30k_data(
    path: str,
    langs: list[str],
    splits: list[str],
    direction: str,
    revision: str | None = None,
):
    """Load Multi30k into MTEB retrieval format.

    Multi30k stores one row per image with parallel captions in four columns
    (en/cs/de/fr), so the image side is identical across every language subset
    and only the caption side changes. The image side is therefore built once
    per split and shared. `direction` selects which side is the query: "t2i"
    retrieves images from captions, "i2t" retrieves captions from images.
    """
    corpus = {lang: dict.fromkeys(splits) for lang in langs}
    queries = {lang: dict.fromkeys(splits) for lang in langs}
    relevant_docs = {lang: dict.fromkeys(splits) for lang in langs}

    for split in splits:
        data = load_dataset(path, split=split, revision=revision)
        row_ids = [str(i) for i in range(len(data))]

        # Built once and reused across languages. add_column is used rather
        # than map so the image bytes are never decoded and re-encoded.
        image_side = data.select_columns(["image"])
        image_side = image_side.add_column("id", [f"image-{i}" for i in row_ids])
        image_side = image_side.add_column("modality", ["image"] * len(row_ids))
        image_side = image_side.add_column("text", [None] * len(row_ids))
        image_side = image_side.cast_column("image", Image())

        for lang in langs:
            text_side = Dataset.from_dict(
                {
                    "id": [f"text-{i}" for i in row_ids],
                    "text": data[lang],
                    "modality": ["text"] * len(row_ids),
                    "image": [None] * len(row_ids),
                }
            )

            if direction == "t2i":
                query_side, doc_side = text_side, image_side
                query_prefix, doc_prefix = "text", "image"
            else:
                query_side, doc_side = image_side, text_side
                query_prefix, doc_prefix = "image", "text"

            corpus[lang][split] = doc_side
            queries[lang][split] = query_side
            relevant_docs[lang][split] = {
                f"{query_prefix}-{i}": {f"{doc_prefix}-{i}": 1} for i in row_ids
            }

    corpus = DatasetDict({lang: DatasetDict(s) for lang, s in corpus.items()})
    queries = DatasetDict({lang: DatasetDict(s) for lang, s in queries.items()})
    relevant_docs = DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class Multi30kT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Multi30kT2IRetrieval",
        description=(
            "Retrieve Flickr30k images from parallel English, Czech, German and "
            "French descriptions of the same image."
        ),
        reference="https://aclanthology.org/W16-3210/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyMultilingualRetrieval",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2016-01-01", "2018-12-31"),
        domains=["Scene", "Web", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_multi30k_data(
            path=self.metadata.dataset["path"],
            langs=self.hf_subsets,
            splits=self.metadata.eval_splits,
            direction="t2i",
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class Multi30kI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Multi30kI2TRetrieval",
        description=(
            "Retrieve English, Czech, German and French descriptions from a "
            "Flickr30k image, testing cross-lingual image-to-text alignment."
        ),
        reference="https://aclanthology.org/W16-3210/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyMultilingualRetrieval",
        category="i2t",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2016-01-01", "2018-12-31"),
        domains=["Scene", "Web", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_multi30k_data(
            path=self.metadata.dataset["path"],
            langs=self.hf_subsets,
            splits=self.metadata.eval_splits,
            direction="i2t",
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True
