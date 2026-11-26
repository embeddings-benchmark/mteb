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
        bibtex_citation="""
        @article{theophile2020allocine,
          title={Allocin{é}: a french dataset for sentiment analysis},
          author={Bluche, Th{é}ophile},
          journal={Hugging Face datasets},
          year={2020}
        }
        """,
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        from datasets import DatasetDict

        if not isinstance(self.dataset, DatasetDict):
            raise TypeError("Dataset must be a DatasetDict")

        # Renommer la colonne d'entrée vers 'text' attendue par MTEB
        self.dataset = self.dataset.rename_columns({"review": "text"})
        # Échantillonnage pour que l'évaluation reste rapide et standardisée
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train", "test"]
        )
