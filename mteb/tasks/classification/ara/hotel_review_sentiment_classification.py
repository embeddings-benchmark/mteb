from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class HotelReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HotelReviewSentimentClassification",
        dataset={
            "path": "mteb/HotelReviewSentimentClassification",
            "revision": "273d5105974460d3979149e29e88c06a8214c541",
        },
        description="HARD is a dataset of Arabic hotel reviews collected from the Booking.com website.",
        reference="https://link.springer.com/chapter/10.1007/978-3-319-67056-0_3",
        type="Classification",
        category="t2c",
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
        superseded_by="HotelReviewSentimentClassification.v2",
    )


class HotelReviewSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HotelReviewSentimentClassification.v2",
        dataset={
            "path": "mteb/HotelReviewSentimentClassification",
            "revision": "f5e6a24acbed4182114ffdf46747090b3f51e836",
        },
        description="HARD is a dataset of Arabic hotel reviews collected from the Booking.com website. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)",
        reference="https://link.springer.com/chapter/10.1007/978-3-319-67056-0_3",
        type="Classification",
        category="t2c",
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
