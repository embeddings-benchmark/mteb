from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class OnlineStoreReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OnlineStoreReviewSentimentClassification",
        dataset={
            "path": "Ruqiya/Arabic_Reviews_of_SHEIN",
            "revision": "fb63ba1255f57054d411fe02bb5cec25cd6b150c",
        },
        description="This dataset contains Arabic reviews of products from the SHEIN online store.",
        reference="https://huggingface.co/datasets/Ruqiya/Arabic_Reviews_of_SHEIN",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["ara-Arab"],
        main_score="accuracy",
        date=("2024-05-01", "2024-05-15"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=["ara-Arab-SA"],
        sample_creation="found",
        bibtex_citation="",
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
