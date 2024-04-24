from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PunjabiNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PunjabiNewsClassification",
        description="A Punjabi dataset for 2-class classification of Punjabi news articles",
        reference="https://github.com/goru001/nlp-for-punjabi/",
        dataset={
            "path": "mlexplorer008/punjabi_news_classification",
            "revision": "cec3923e16519efe51d535497e711932b8f1dc44",
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
        license="MIT",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"train": 627, "test": 157},
        avg_character_length={"train": 4222.22, "test": 4115.14},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"article": "text", "is_about_politics": "label"}
        )
