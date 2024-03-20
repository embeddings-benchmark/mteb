from mteb.abstasks import AbsTaskClassification


class NorwegianParliamentClassification(AbsTaskClassification):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "NorwegianParliament",
            "hf_hub_name": "NbAiLab/norwegian_parliament",
            "description": "Norwegian parliament speeches annotated for sentiment",
            "reference": "https://huggingface.co/datasets/NbAiLab/norwegian_parliament",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test", "validation"],
            "eval_langs": ["nb"],  # assumed to be bokmål
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "f7393532774c66312378d30b197610b43d751972",
        }
