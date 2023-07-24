from mteb.abstasks import AbsTaskClassification


class NoRecClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "NoRecClassification",
            "hf_hub_name": "ScandEval/norec-mini",  # Using the mini version to keep results ~comparable to the ScandEval benchmark
            "description": "A Norwegian dataset for sentiment classification on review",
            "reference": "https://aclanthology.org/L18-1661/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["no"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "07b99ab3363c2e7f8f87015b01c21f4d9b917ce3",
        }
