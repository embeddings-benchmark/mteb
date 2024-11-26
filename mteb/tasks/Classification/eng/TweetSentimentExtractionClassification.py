from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


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
        category="s2s",
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
        bibtex_citation="""@misc{tweet-sentiment-extraction,
    author = {Maggie, Phil Culliton, Wei Chen},
    title = {Tweet Sentiment Extraction},
    publisher = {Kaggle},
    year = {2020},
    url = {https://kaggle.com/competitions/tweet-sentiment-extraction}
}""",
        prompt="Classify the sentiment of a given tweet as either positive, negative, or neutral",
    )

    samples_per_label = 32
