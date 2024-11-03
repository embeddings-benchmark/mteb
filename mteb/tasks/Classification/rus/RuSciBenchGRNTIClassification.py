from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


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
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("1999-01-01", "2024-01-01"),
        domains=["Academic", "Written"],
        task_subtypes=["Topic classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt="Classify the category of scientific papers based on the titles and abstracts",
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, n_samples=2048, splits=["test"]
        )
