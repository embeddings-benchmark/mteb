from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class RuSciBenchGRNTIClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuSciBenchGRNTIClassification",
        dataset={
            "path": "ai-forever/ru-scibench-grnti-classification",
            "revision": "673a610d6d3dd91a547a0d57ae1b56f37ebbf6a1",
        },
        description="Classification of scientific papers (title+abstract) by rubric",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench/",
        type="Classification",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("1999-01-01", "2024-01-01"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Topic classification"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": 2048},
        avg_character_length={"test": 890.1},
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, n_samples=2048, splits=["test"]
        )
