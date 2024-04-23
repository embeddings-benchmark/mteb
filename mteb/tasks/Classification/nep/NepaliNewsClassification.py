from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class NepaliNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NepaliNewsClassification",
        description="A Nepali dataset for 7500 news articles ",
        reference="https://github.com/goru001/nlp-for-nepali",
        dataset={
            "path": "bpHigh/iNLTK_Nepali_News_Dataset",
            "revision": "3f799428a7c403ccc5c14d6ddb6558ae9075b344",
        },
        type="Classification",
        category="s2s",
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["pan-Guru"],
        main_score="accuracy",
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="CC BY-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={"train": 5975, "test": 1495},
        avg_character_length={"train": , "test": },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("paras", "text")
        self.dataset = self.dataset.rename_column("label", "label")