from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class GeoreviewClassification(AbsTaskClassification):
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
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("2023-01-01", "2023-08-01"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="mit",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": 2048},
        avg_character_length={"test": 409.0},
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, n_samples=2048, splits=["test"]
        )
