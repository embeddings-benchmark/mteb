from ...abstasks import AbsTaskClassification


class AmazonPolarityClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "AmazonPolarityClassification",
            "hf_hub_name": "mteb/amazon_polarity",
            "description": "Amazon Polarity Classification Dataset.",
            "reference": "https://dl.acm.org/doi/10.1145/2507157.2507163",
            "category": "s2s",
            "type": "Classification",
            "available_splits": ["train", "test"],
            "available_langs": ["en"],
            "main_score": "accuracy",
        }
