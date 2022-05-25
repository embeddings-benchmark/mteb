from ...abstasks import AbsTaskKNNClassification, MultilingualTask

_LANGUAGES = ["en", "de", "es", "fr", "ja", "zh"]


class AmazonReviewsClassification(MultilingualTask, AbsTaskKNNClassification):
    @property
    def description(self):
        return {
            "name": "AmazonReviewsClassification",
            "hf_hub_name": "mteb/amazon_reviews_multi",
            "description": "A collection of Amazon reviews specifically designed to aid research in multilingual text classification.",
            "reference": "https://arxiv.org/abs/2010.02573",
            "category": "s2s",
            "type": "kNNClassification",
            "available_splits": ["train", "validation", "test"],
            "available_langs": _LANGUAGES,
            "main_score": "accuracy",
        }
