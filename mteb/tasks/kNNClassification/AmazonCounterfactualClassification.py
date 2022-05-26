from ...abstasks import AbsTaskKNNClassification, MultilingualTask

_LANGUAGES = ["en", "de", "en-ext", "ja"]


class AmazonCounterfactualClassification(MultilingualTask, AbsTaskKNNClassification):
    @property
    def description(self):
        return {
            "name": "AmazonCounterfactualClassification",
            "hf_hub_name": "mteb/amazon_counterfactual",
            "description": "A collection of Amazon customer reviews annotated for counterfactual detection binary classification.",
            "reference": "https://arxiv.org/abs/2104.06893",
            "category": "s2s",
            "type": "kNNClassification",
            "available_splits": ["train", "validation", "test"],
            "available_langs": _LANGUAGES,
            "main_score": "accuracy",
        }
