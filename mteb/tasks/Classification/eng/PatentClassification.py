from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PatentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PatentClassification",
        description="Classification Dataset of Patents and Abstract",
        dataset={
            "path": "ccdv/patent-classification",
            "revision": "2f38a1dfdecfacee0184d74eaeafd3c0fb49d2a6",
        },
        reference="https://aclanthology.org/P19-1212.pdf",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-11-05", "2022-10-22"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=["Topic classification"],
        license="Not specified",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={"test": 5000},
        avg_character_length={"test": 18620.44},
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
