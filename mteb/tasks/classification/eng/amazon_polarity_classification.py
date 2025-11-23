from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AmazonPolarityClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AmazonPolarityClassification",
        description="Amazon Polarity Classification Dataset.",
        reference="https://huggingface.co/datasets/amazon_polarity",
        dataset={
            "path": "mteb/amazon_polarity",
            "revision": "e2d317d38cd51312af73b3d32a06d1a08b442046",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2012-01-01",
            "2015-12-31",
        ),  # Estimated range for the collection of reviews
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        annotations_creators="derived",
        license="apache-2.0",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{McAuley2013HiddenFA,
  author = {Julian McAuley and Jure Leskovec},
  journal = {Proceedings of the 7th ACM conference on Recommender systems},
  title = {Hidden factors and hidden topics: understanding rating dimensions with review text},
  url = {https://api.semanticscholar.org/CorpusID:6440341},
  year = {2013},
}
""",
        prompt="Classify Amazon reviews into positive or negative sentiment",
        superseded_by="AmazonPolarityClassification.v2",
    )


class AmazonPolarityClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AmazonPolarityClassification.v2",
        description="Amazon Polarity Classification Dataset. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/amazon_polarity",
        dataset={
            "path": "mteb/amazon_polarity",
            "revision": "ec149c1fe36043668a50804214d4597804001f6f",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2012-01-01",
            "2015-12-31",
        ),  # Estimated range for the collection of reviews
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        annotations_creators="derived",
        license="apache-2.0",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{McAuley2013HiddenFA,
  author = {Julian McAuley and Jure Leskovec},
  journal = {Proceedings of the 7th ACM conference on Recommender systems},
  title = {Hidden factors and hidden topics: understanding rating dimensions with review text},
  url = {https://api.semanticscholar.org/CorpusID:6440341},
  year = {2013},
}
""",
        prompt="Classify Amazon reviews into positive or negative sentiment",
        adapted_from=["AmazonPolarityClassification"],
    )
