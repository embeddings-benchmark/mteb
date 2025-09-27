from __future__ import annotations

from datasets import DatasetDict, load_dataset

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TweetSentimentExtractionClassificationHumanSubset(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TweetSentimentExtractionClassificationHumanSubset",
        description="Human evaluation subset of Tweet Sentiment Extraction dataset.",
        reference="https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview",
        dataset={
            "path": "mteb/mteb-human-tweet-sentiment-classification",
            "revision": "9e8b4c52157ee3d3e78ab802ea441abd819e0569",
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
    )

    samples_per_label = 32

    def load_data(self, **kwargs):
        """Load human test subset + full original training data"""
        # Load human evaluation subset
        human_dataset = load_dataset(
            self.metadata_dict["dataset"]["path"],
            revision=self.metadata_dict["dataset"]["revision"],
        )

        # Load full original training data
        original_dataset = load_dataset(
            "mteb/tweet_sentiment_extraction",
            revision="d604517c81ca91fe16a244d1248fc021f9ecee7a",
        )

        # Combine: original train + human test
        self.dataset = DatasetDict(
            {"train": original_dataset["train"], "test": human_dataset["test"]}
        )
