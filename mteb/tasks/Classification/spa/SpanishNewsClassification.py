from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SpanishNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SpanishNewsClassification",
        description="A Spanish dataset for news classification. The dataset includes articles from reputable Spanish news sources spanning 12 different categories.",
        reference="https://huggingface.co/datasets/MarcOrfilaCarreras/spanish-news",
        dataset={
            "path": "MarcOrfilaCarreras/spanish-news",
            "revision": "0086c197b914690a9dace258a19398890a05299a",
        },
        type="Classification",
        category="s2s",
        date=("2023-05-01", "2024-05-01"),
        eval_splits=["train"],
        eval_langs=["spa-Latn"],
        main_score="accuracy",
        form=["written"],
        domains=["News"],
        task_subtypes=[],
        license="MIT",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        """,
        n_samples={"train": 2048},
        avg_character_length={"train": 4218.2},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"category": "label"})
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
