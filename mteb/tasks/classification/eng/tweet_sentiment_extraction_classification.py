from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class TweetSentimentExtractionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TweetSentimentExtractionClassification",
        description="",
        reference="https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview",
        dataset={
            "path": "mteb/tweet_sentiment_extraction",
            "revision": "d604517c81ca91fe16a244d1248fc021f9ecee7a",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-01-01",
            "2020-12-31",
        ),  # Estimated range for the collection of tweets
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{tweet-sentiment-extraction,
  author = {Maggie, Phil Culliton, Wei Chen},
  publisher = {Kaggle},
  title = {Tweet Sentiment Extraction},
  url = {https://kaggle.com/competitions/tweet-sentiment-extraction},
  year = {2020},
}
""",
        prompt="Classify the sentiment of a given tweet as either positive, negative, or neutral",
        superseded_by="TweetSentimentExtractionClassification.v2",
    )

    samples_per_label = 32


class TweetSentimentExtractionClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TweetSentimentExtractionClassification.v2",
        description="This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview",
        dataset={
            "path": "mteb/tweet_sentiment_extraction",
            "revision": "7261898ee3b9a739595e8dbf41df6b2332f429bb",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-01-01",
            "2020-12-31",
        ),  # Estimated range for the collection of tweets
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{tweet-sentiment-extraction,
  author = {Maggie, Phil Culliton, Wei Chen},
  publisher = {Kaggle},
  title = {Tweet Sentiment Extraction},
  url = {https://kaggle.com/competitions/tweet-sentiment-extraction},
  year = {2020},
}
""",
        prompt="Classify the sentiment of a given tweet as either positive, negative, or neutral",
        adapted_from=["TweetSentimentExtractionClassification"],
    )

    samples_per_label = 32
