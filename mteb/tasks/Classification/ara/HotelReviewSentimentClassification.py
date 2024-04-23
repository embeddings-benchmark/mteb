from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

N_SAMPLES = 2048


class HotelReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HotelReviewSentimentClassification",
        dataset={
            "path": "hard",
            "revision": "b108d2c32ee4e1f4176ea233e1a5ac17bceb9ef9",
        },
        description="HARD is a dataset of Arabic hotel reviews collected from the Booking.com website.",
        reference="https://link.springer.com/chapter/10.1007/978-3-319-67056-0_3",
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["ara-Arab"],
        main_score="accuracy",
        date=("2016-06-01", "2016-07-31"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license=None,
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=["ara-arab-EG", "ara-arab-JO", "ara-arab-LB", "ara-arab-SA"],
        text_creation="found",
        bibtex_citation="""
@article{elnagar2018hotel,
  title={Hotel Arabic-reviews dataset construction for sentiment analysis applications},
  author={Elnagar, Ashraf and Khalifa, Yasmin S and Einea, Anas},
  journal={Intelligent natural language processing: Trends and applications},
  pages={35--52},
  year={2018},
  publisher={Springer}
}
""",
        n_samples={"train": N_SAMPLES},
        avg_character_length={"train": 137.2},
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(self.dataset, splits=["train"])
