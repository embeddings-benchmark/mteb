from mteb.abstasks.AbsTaskClassification import AbsTaskClassification


class AngryTweetsClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "AngryTweetsClassification",
            "hf_hub_name": "DDSC/angry-tweets",
            "description": "A sentiment dataset with 3 classes (positiv, negativ, neutral) for Danish tweets",
            "reference": "https://aclanthology.org/2021.nodalida-main.53/",
            "eval_splits": ["test"],
            "eval_langs": ["da"],
            "type": "Classification",
            "category": "s2s",
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "20b0e6081892e78179356fada741b7afa381443d",
        }
