from ...abstasks import AbsTaskClassification


class Banking77Classification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "Banking77Classification",
            "hf_hub_name": "mteb/banking77",
            "description": "Dataset composed of online banking queries annotated with their corresponding intents.",
            "reference": "https://arxiv.org/abs/2003.04807",
            "category": "s2s",
            "type": "Classification",
            "available_splits": ["train", "test"],
            "eval_langs": ["en"],
            "main_score": "accuracy",
        }
