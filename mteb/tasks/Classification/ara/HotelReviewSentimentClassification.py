from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class HotelReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HotelReviewSentimentClassification",
        dataset={
            "path": "Elnagara/hard",
            "revision": "b108d2c32ee4e1f4176ea233e1a5ac17bceb9ef9",
            "trust_remote_code": True,
        },
        description="HARD is a dataset of Arabic hotel reviews collected from the Booking.com website.",
        reference="https://link.springer.com/chapter/10.1007/978-3-319-67056-0_3",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["ara-Arab"],
        main_score="accuracy",
        date=("2016-06-01", "2016-07-31"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=["ara-arab-EG", "ara-arab-JO", "ara-arab-LB", "ara-arab-SA"],
        sample_creation="found",
        bibtex_citation=r"""
@article{elnagar2018hotel,
  author = {Elnagar, Ashraf and Khalifa, Yasmin S and Einea, Anas},
  journal = {Intelligent natural language processing: Trends and applications},
  pages = {35--52},
  publisher = {Springer},
  title = {Hotel Arabic-reviews dataset construction for sentiment analysis applications},
  year = {2018},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
