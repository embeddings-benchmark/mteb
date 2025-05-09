from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class HotelReviewSentimentClassification(AbsTaskClassification):
    superseded_by = "HotelReviewSentimentClassification.v2"
    metadata = TaskMetadata(
        name="HotelReviewSentimentClassification",
        dataset={
            "path": "mteb/HotelReviewSentimentClassification",
            "revision": "273d5105974460d3979149e29e88c06a8214c541",
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


class HotelReviewSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HotelReviewSentimentClassification.v2",
        dataset={
            "path": "mteb/hotel_review_sentiment",
            "revision": "9dd15890ce97da846d75dc329b70d3b8398c5542",
        },
        description="""HARD is a dataset of Arabic hotel reviews collected from the Booking.com website.
        This version corrects errors found in the original data. For details, see https://github.com/embeddings-benchmark/mteb/pull/2632""",
        reference="https://link.springer.com/chapter/10.1007/978-3-319-67056-0_3",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
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
