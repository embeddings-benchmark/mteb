from __future__ import annotations

import random

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

TEST_SAMPLES = 2048


class MalayalamNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MalayalamNewsClassification",
        description="A Malayalam dataset for 3-class classification of Malayalam news articles",
        reference="https://github.com/goru001/nlp-for-malyalam",
        dataset={
            "path": "mlexplorer008/malayalam_news_classification",
            "revision": "666f63bba2387456d8f846ea4d0565181bd47b81",
        },
        type="Classification",
        category="s2s",
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["mal-Mlym"],
        main_score="accuracy",
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="MIT",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation=None,
        n_samples={"train": 5036, "test": 1260},
        avg_character_length={"train": 79.48, "test": 80.44},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"headings": "text"})
