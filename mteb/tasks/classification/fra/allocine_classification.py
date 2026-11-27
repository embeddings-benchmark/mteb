from __future__ import annotations

from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AllocineClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AllocineClassification",
        dataset={
            "path": "tblard/allocine",
            "revision": "a4654f4896408912913a62ace89614879a549287",
        },
        description="Allociné is a dataset of French movie reviews labeled for binary sentiment classification (positive / negative).",
        reference="https://huggingface.co/datasets/allocine",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2020-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        from datasets import DatasetDict

        if not isinstance(self.dataset, DatasetDict):
            raise TypeError("Dataset must be a DatasetDict")

        # Rename input column to 'text' expected by MTEB
        self.dataset = self.dataset.rename_columns({"review": "text"})

        def _has_min_length(example: dict[str, Any]) -> bool:
            return len(example["text"].strip()) >= 10

        # Filter out overly short texts (MTEB requires min_text_length > 3)
        self.dataset = self.dataset.filter(
            _has_min_length,
            num_proc=num_proc,
        )
        # Subsample to keep evaluation fast and standardized
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train", "test"]
        )
