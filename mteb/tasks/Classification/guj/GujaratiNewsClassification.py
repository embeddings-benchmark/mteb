from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class GujaratiNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="GujaratiNewsClassification",
        description="A Gujarati dataset for 3-class classification of Gujarati news articles",
        reference="https://github.com/goru001/nlp-for-gujarati",
        dataset={
            "path": "mlexplorer008/gujarati_news_classification",
            "revision": "1a5f2fa2914bfeff4fcdc6fff4194fa8ec8fa19e",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["guj-Gujr"],
        main_score="accuracy",
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="MIT",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",  # none found
        descriptive_stats={
            "n_samples": {"train": 5269, "test": 1318},
            "avg_character_length": {"train": 61.95, "test": 61.91},
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("headline", "text")
