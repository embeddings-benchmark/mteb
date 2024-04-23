from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SanskritShlokasClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SanskritShlokasClassification",
        description="This data set contains ~500 Shlokas  ",
        reference="https://github.com/goru001/nlp-for-sanskrit",
        dataset={
            "path": "bpHigh/iNLTK_Sanskrit_Shlokas_Dataset",
            "revision": "772c6ef188a4ffd2bd2e115fcefe5957f2a1dc1f",
        },
        type="Classification",
        category="s2s",
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["san-Guru"],
        main_score="accuracy",
        form=["written"],
        domains=["Religious Text"],
        task_subtypes=["Topic classification"],
        license="CC BY-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={"train": 383, "test": 96},
        avg_character_length={"train":276 , "test":170 },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("Sloka", "text")
        self.dataset = self.dataset.rename_column("Class", "label")