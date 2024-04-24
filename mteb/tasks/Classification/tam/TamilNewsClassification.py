from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TamilNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TamilNewsClassification",
        description="A Tamil dataset for 6-class classification of Tamil news articles",
        reference="https://github.com/vanangamudi/tamil-news-classification",
        dataset={
            "path": "mlexplorer008/tamil_news_classification",
            "revision": "bb34dd6690cf17aa731d75d45388c5801b8c4e4b",
        },
        type="Classification",
        category="s2s",
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["tam-Taml"],
        main_score="f1",
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="MIT",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation=None,
        n_samples={"train": 14521, "test": 3631},
        avg_character_length={"train": 56.50, "test": 56.52},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"NewsInTamil": "text", "Category": "label"}
        )
        self.dataset = self.stratified_subsampling(self.dataset, seed=self.seed)
