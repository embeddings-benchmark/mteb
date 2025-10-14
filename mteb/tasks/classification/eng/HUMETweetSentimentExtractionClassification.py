from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class HUMETweetSentimentExtractionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HUMETweetSentimentExtractionClassification",
        description="Human evaluation subset of Tweet Sentiment Extraction dataset.",
        reference="https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview",
        dataset={
            "path": "mteb/HUMETweetSentimentExtractionClassification",
            "revision": "264bce01a98dfaf3581b53dcaa0fd5e2d44aa589",
        },
        type="Classification",
        category="t2t",
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
