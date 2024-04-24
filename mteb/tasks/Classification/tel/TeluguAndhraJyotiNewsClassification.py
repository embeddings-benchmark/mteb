from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata



class TeluguAndhraJyotiNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TeluguAndhraJyotiNewsClassification",
        description="A Telugu dataset for 5-class classification of Tamil news articles",
        reference="https://github.com/AnushaMotamarri/Telugu-Newspaper-Article-Dataset",
        dataset={
            "path": "mlexplorer008/telugu_news_classification",
            "revision": "3821aa93aa461c9263071e0897234e8d775ad616",
        },
        type="Classification",
        category="s2s",
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["tel-Telu"],
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
        n_samples={"train": 17312, "test": TEST_SAMPLES},
        avg_character_length={"train": 1435.53, "test": 1428.28},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column({"body": "text", "topic": "label"})
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )

       