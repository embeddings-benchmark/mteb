from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SpanishNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SpanishNewsClassification",
        description="A Spanish dataset for news classification. The dataset includes articles from reputable Spanish news sources spanning 12 different categories.",
        reference="https://huggingface.co/datasets/mteb/SpanishNewsClassification",
        dataset={
            "path": "mteb/SpanishNewsClassification",
            "revision": "9f568e396857395bd803c63452b83316124d31c9",
        },
        type="Classification",
        category="t2c",
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
        bibtex_citation=None,
        superseded_by="SpanishNewsClassification.v2",
    )


class SpanishNewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SpanishNewsClassification.v2",
        description="A Spanish dataset for news classification. The dataset includes articles from reputable Spanish news sources spanning 12 different categories. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/MarcOrfilaCarreras/spanish-news",
        dataset={
            "path": "mteb/spanish_news",
            "revision": "345aa68ec44052d28828c6f88e7a2aafaf74be5a",
        },
        type="Classification",
        category="t2c",
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

    def dataset_transform(
        self,
        num_proc: int | None = None,
    ):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
