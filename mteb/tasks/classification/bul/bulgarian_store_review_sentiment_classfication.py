from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class BulgarianStoreReviewSentimentClassfication(AbsTaskClassification):
    metadata = TaskMetadata(
        name="BulgarianStoreReviewSentimentClassfication",
        description="Bulgarian online store review dataset for sentiment classification.",
        reference="https://doi.org/10.7910/DVN/TXIK9P",
        dataset={
            "path": "mteb/BulgarianStoreReviewSentimentClassfication",
            "revision": "0d00595ed48ba8b802da579231a078557c6e9bc4",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2018-05-14", "2018-05-14"),
        eval_splits=["test"],
        eval_langs=["bul-Cyrl"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@data{DVN/TXIK9P_2018,
  author = {Georgieva-Trifonova, Tsvetanka and Stefanova, Milena and Kalchev, Stefan},
  doi = {10.7910/DVN/TXIK9P},
  publisher = {Harvard Dataverse},
  title = {{Dataset for ``Customer Feedback Text Analysis for Online Stores Reviews in Bulgarian''}},
  url = {https://doi.org/10.7910/DVN/TXIK9P},
  version = {V1},
  year = {2018},
}
""",
    )
