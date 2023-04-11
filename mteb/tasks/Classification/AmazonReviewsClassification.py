from ...abstasks import AbsTaskClassification, MultilingualTask

_LANGUAGES = ["en", "de", "es", "fr", "ja", "zh"]


class AmazonReviewsClassification(MultilingualTask, AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "AmazonReviewsClassification",
            "hf_hub_name": "mteb/amazon_reviews_multi",
            "description": (
                "A collection of Amazon reviews specifically designed to aid research in multilingual text"
                " classification."
            ),
            "reference": "https://arxiv.org/abs/2010.02573",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation", "test"],
            "eval_langs": _LANGUAGES,
            "main_score": "accuracy",
            "revision": "1399c76144fd37290681b995c656ef9b2e06e26d",
        }
