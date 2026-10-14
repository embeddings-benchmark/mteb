from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import load_dataset

from mteb.abstasks.analogy import AbsTaskAnalogy
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

_VISLINGA_METADATA: dict[str, Any] = {
    "reference": "https://huggingface.co/datasets/YGideoni/VisLingA",
    "dataset": {
        "path": "YGideoni/VisLingA",
        "revision": "428b306543b1ab693516a33c49ec825171c4f0da",
        "name": "analogies",
    },
    "type": "Analogy",
    "eval_splits": ["test"],
    "eval_langs": ["eng-Latn"],
    "main_score": "hit_rate_at_10",
    "date": ("2013-01-16", "2026-10-14"),
    "domains": ["Encyclopaedic"],
    "task_subtypes": [],
    "license": "multiple",
    "annotations_creators": "derived",
    "dialect": [],
    "sample_creation": "found",
    "bibtex_citation": "",  # TODO: add once the paper is on arXiv
}


class VisLingAText(AbsTaskAnalogy):
    metadata = TaskMetadata(
        name="VisLingAText",
        description=(
            "Word analogies (A is to B as C is to D) over 746 concepts from the analogy dataset of Mikolov et al. "
            "(2013), e.g. capitals, currencies and grammatical forms. Each concept is represented by its embedding,"
            " and D is ranked relative to all other concepts by cosine similarity to B - A + C."
        ),
        category="t2t",
        modalities=["text"],
        **_VISLINGA_METADATA,
    )


class VisLingAImage(AbsTaskAnalogy):
    input_column_name = "image"

    metadata = TaskMetadata(
        name="VisLingAImage",
        description=(
            "Visual analogies (A is to B as C is to D) over 746 concepts from the analogy dataset of Mikolov et al. "
            "(2013), e.g. capitals, currencies and grammatical forms. Each concept is represented by the mean "
            "embedding of several Wikimedia Commons images of that concept. D is ranked among all concepts by cosine "
            "similarity to B - A + C."
        ),
        category="i2i",
        modalities=["image"],
        **_VISLINGA_METADATA,
    )

    concepts: DatasetDict

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        self.concepts = load_dataset(
            self.metadata.dataset["path"],
            "images",
            revision=self.metadata.dataset["revision"],
            num_proc=num_proc,
        ).select_columns([self.concept_column_name, self.input_column_name])

    def get_concepts(self, analogies: Dataset, split: str) -> Dataset:
        return self.concepts[split]
