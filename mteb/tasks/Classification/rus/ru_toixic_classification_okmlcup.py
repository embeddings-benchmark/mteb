from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class RuToxicOKMLCUPClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuToxicOKMLCUPClassification",
        dataset={
            "path": "mteb/RuToxicOKMLCUPClassification",
            "revision": "13722b7320ef4b6a471f9e8b379f3f49167d0517",
        },
        description="On the Odnoklassniki social network, users post a huge number of comments of various directions and nature every day.",
        reference="https://cups.online/ru/contests/okmlcup2020",
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("2015-01-01", "2020-01-01"),
        domains=[],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("toxic", "label")
