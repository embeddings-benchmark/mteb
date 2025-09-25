from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class GeoreviewClassification(AbsTaskClassification):
    superseded_by = "GeoreviewClassification.v2"
    metadata = TaskMetadata(
        name="GeoreviewClassification",
        dataset={
            "path": "ai-forever/georeview-classification",
            "revision": "3765c0d1de6b7d264bc459433c45e5a75513839c",
            "trust_remote_code": True,
        },
        description="Review classification (5-point scale) based on Yandex Georeview dataset",
        reference="https://github.com/yandex/geo-reviews-dataset-2023",
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("2023-01-01", "2023-08-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt="Classify the organization rating based on the reviews",
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, n_samples=2048, splits=["test"]
        )


class GeoreviewClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="GeoreviewClassification.v2",
        dataset={
            "path": "mteb/georeview",
            "revision": "5194395f82217bc31212fd6a275002fb405f9dfb",
        },
        description="""Review classification (5-point scale) based on Yandex Georeview dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)""",
        reference="https://github.com/yandex/geo-reviews-dataset-2023",
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("2023-01-01", "2023-08-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt="Classify the organization rating based on the reviews",
        adapted_from=["GeoreviewClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, n_samples=2048, splits=["test"]
        )
