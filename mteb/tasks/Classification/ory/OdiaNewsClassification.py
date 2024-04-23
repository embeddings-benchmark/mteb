from __future__ import annotations


import random
from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata
 

TEST_SAMPLES = 2048


class OdiaNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OdiaNewsClassification",
        description="A Odia dataset for 3-class classification of Odia news articles",
        reference="https://github.com/goru001/nlp-for-odia",
        dataset={
            "path": "mlexplorer008/odia_news_classification",
            "revision": "ffb8a34c9637fb20256e8c7be02504d16af4bd6b",
        },
        type="Classification",
        category="s2s",
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["ory-Orya"],
        main_score="accuracy",
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="MIT",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={"train": 15200, "test": 157},
        avg_character_length={"train": 4222.22, "test": 4115.14},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("headings", "text")
        self.dataset = self.dataset.class_encode_column("label")
        self.dataset["test"] = self.dataset["test"].train_test_split(
            test_size=TEST_SAMPLES, seed=42, stratify_by_column="label"
        )["test"]
       
        