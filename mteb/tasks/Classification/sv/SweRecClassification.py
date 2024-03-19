from mteb.abstasks import AbsTaskClassification


class SweRecClassification(AbsTaskClassification):
    @property
    def metadata_dict(self):
        return {
            "name": "SweRecClassification",
            "hf_hub_name": "ScandEval/swerec-mini",  # using the mini version to keep results ~comparable to ScandEval
            "description": "A Swedish dataset for sentiment classification on review",
            "reference": "https://aclanthology.org/2023.nodalida-1.20/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["sv"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "3c62f26bafdc4c4e1c16401ad4b32f0a94b46612",
        }
