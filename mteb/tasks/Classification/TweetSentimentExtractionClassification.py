from ...abstasks import AbsTaskClassification


class TweetSentimentExtractionClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "TweetSentimentExtractionClassification",
            "hf_hub_name": "mteb/tweet_sentiment_extraction",
            "description": "",
            "reference": "https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview",
            "category": "s2s",
            "type": "Classification",
            "available_splits": ["train", "test"],
            "available_langs": ["en"],
            "main_score": "accuracy",
            "n_splits": 10,
            "samples_per_label": 32,
        }
