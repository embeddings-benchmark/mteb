from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

N_SAMPLES = 1297

class OnlineStoreReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OnlineStoreReviewSentimentClassification",
        dataset={
            "path": "Ruqiya/Arabic_Reviews_of_SHEIN",
            "revision": "65cc3759ee6bf2506b497484e33347255b0f499d",
        },
        description="This dataset contains Arabic reviews of products from the SHEIN online store.",
        reference="https://huggingface.co/datasets/Ruqiya/Arabic_Reviews_of_SHEIN",
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["ara-Arab"],
        main_score="accuracy",
        date=("2024-05-01", "2024-05-15"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="apache-2.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=["ara-arab-EG", "ara-arab-JO", "ara-arab-LB", "ara-arab-SA"],
        text_creation="found",
        bibtex_citation="""
@misc{Ruqiya/Arabic_Reviews_of_SHEIN,
title = {Arabic_Reviews_of_SHEIN},
author = {Ruqiya Bin Safi},
howpublished = {\\url{https://huggingface.co/datasets/Ruqiya/Arabic_Reviews_of_SHEIN}},
year = {2024},
publisher = {Hugging Face}
}
""",
        n_samples={"train": N_SAMPLES},
        avg_character_length={"train": 137.2},
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"], test_size=100
        )
