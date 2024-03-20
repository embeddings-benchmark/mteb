from ....abstasks import AbsTaskClassification


class Banking77Classification(AbsTaskClassification):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "Banking77Classification",
            "hf_hub_name": "mteb/banking77",
            "description": "Dataset composed of online banking queries annotated with their corresponding intents.",
            "reference": "https://arxiv.org/abs/2003.04807",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "accuracy",
            "revision": "0fd18e25b25c072e09e0d92ab615fda904d66300",
        }
