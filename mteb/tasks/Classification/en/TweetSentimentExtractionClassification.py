from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class TweetSentimentExtractionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TweetSentimentExtractionClassification",
        description="",
        reference="https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview",
        hf_hub_name="mteb/tweet_sentiment_extraction",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="accuracy",
        revision="d604517c81ca91fe16a244d1248fc021f9ecee7a",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 3534},
        avg_character_length={"test": 67.8},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 32
        return metadata_dict
