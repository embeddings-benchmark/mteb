from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SpanishNewsClassification(AbsTaskClassification):
    superseded_by = "SpanishNewsClassification.v2"
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
        modalities=["text"],
        date=("2023-05-01", "2024-05-01"),
        eval_splits=["train"],
        eval_langs=["spa-Latn"],
        main_score="accuracy",
        domains=["News", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        """,
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"category": "label"})
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )


class SpanishNewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SpanishNewsClassification.v2",
        description="""A Spanish dataset for news classification. The dataset includes articles from reputable Spanish news sources spanning 12 different categories.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)""",
        reference="https://huggingface.co/datasets/MarcOrfilaCarreras/spanish-news",
        dataset={
            "path": "mteb/spanish_news",
            "revision": "345aa68ec44052d28828c6f88e7a2aafaf74be5a",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2023-05-01", "2024-05-01"),
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
        main_score="accuracy",
        domains=["News", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        """,
        adapted_from=["SpanishNewsClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
