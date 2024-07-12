from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


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
        license="Not specified",
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
        descriptive_stats={
            "n_samples": {"test": 3534},
            "avg_character_length": {"test": 67.8},
        },
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 32
        return metadata_dict
