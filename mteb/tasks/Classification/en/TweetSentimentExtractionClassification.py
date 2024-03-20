from ....abstasks import AbsTaskClassification


class TweetSentimentExtractionClassification(AbsTaskClassification):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "TweetSentimentExtractionClassification",
            "hf_hub_name": "mteb/tweet_sentiment_extraction",
            "description": "",
            "reference": "https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 32,
            "revision": "d604517c81ca91fe16a244d1248fc021f9ecee7a",
        }
