from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.abs_text_classification import AbsTextClassification


class HotelReviewSentimentClassification(AbsTextClassification):
    metadata = TaskMetadata(
        name="HotelReviewSentimentClassification",
        dataset={
            "path": "mteb/HotelReviewSentimentClassification",
            "revision": "273d5105974460d3979149e29e88c06a8214c541",
        },
        description="HARD is a dataset of Arabic hotel reviews collected from the Booking.com website.",
        reference="https://link.springer.com/chapter/10.1007/978-3-319-67056-0_3",
        type="Classification",
        category="t2t",
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
    )
